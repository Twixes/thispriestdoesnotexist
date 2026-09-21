"""Fixed-offset variant of adaptation_masks.py; old helper is unchanged.

API: FixedOffsetsAdaptationMasks(model, selection), then fresh Adam/bind, and
controller.step after every backward. Noise and synthesis activation-bias
additive offsets stay zero and excluded; their originals stay exact. G mapping,
affine and ToRGB and D policies otherwise follow the parent. No main runner.
"""
"""Fixed output-row main-adaptation masks, separate from probing mechanics.

API: controller = FixedOffsetsAdaptationMasks(model, selection); build a FRESH torch.optim.Adam
from controller.parameters(); controller.bind_optimizer(optimizer). After EACH
backward (adversarial, R1, path, etc.), call controller.step(optimizer), or call
mask_gradients(), optimizer.step(), verify_invariants(optimizer) explicitly.
Use controller.set_enabled() for alternating G/D; never set_probing_grad here.

Selection is select_rows()'s one-component primitive result. No threshold is
chosen here. Input must have output_rank1 modulation installed. Source-style
exclusions and mask rules follow importance-source-audit.md. Only the native
skip-G/resnet-D topology is supported. Caller must establish fresh source/model,
factor and optimizer provenance; this helper cannot authenticate checkpoint
history. It never carries probing optimizer state into main adaptation.

Protected ORIGINAL rows stay exact; important EFFECTIVE rows may change via
modulation. Low v/selected bias offsets are zeroed once then must stay exactly
zero, including Adam moments. Shared u stays trainable. Any attempted violation
raises; no post-step clamping or momentum repair hides it. This helper neither
saves/resumes a runner nor proves visual quality. All original parameters remain
registered, so normal state_dict and the existing fold helper still work.
"""
import copy
import re

import torch
from fixed_offset_policy import capture_fixed_reference, verify_fixed_offsets, POLICY

try:
    from .modulation import describe_modulation, modulation_named_parameters
except ImportError:
    from modulation import describe_modulation, modulation_named_parameters


def adaptation_inventory(model):
    """Independently enumerate the exact selectable native modules and names."""
    inventory = describe_modulation(model)
    if inventory['conv_layout'] != 'output_rank1':
        raise ValueError('Main adaptation row masks require output_rank1 probing/selection')
    component = inventory['component']
    if component == 'G':
        blocks = model.synthesis.block_resolutions
        if any(model.get_submodule(f'synthesis.b{r}').architecture != 'skip' for r in blocks):
            raise ValueError('Only native skip generator topology is supported')
        selected = {f'synthesis.b{r}.{conv}{affine}'
                    for r in blocks if r > 4 for conv in ('conv0', 'conv1')
                    for affine in ('', '.affine')}
        allowed = re.compile(r'(mapping\.fc\d+|synthesis\.b\d+\.(conv[01]|torgb)(\.affine)?)')
    elif component == 'D':
        blocks = model.block_resolutions
        if any(model.get_submodule(f'b{r}').architecture != 'resnet' for r in blocks):
            raise ValueError('Only native resnet discriminator topology is supported')
        selected = {f'b{r}.{conv}' for r in blocks for conv in ('conv0', 'conv1', 'skip')}
        allowed = re.compile(r'b\d+\.(fromrgb|conv[01]|skip)')
    else:
        raise ValueError('Unknown component')
    targets = {(r['module'], r['tensor']): r for r in inventory['targets']}
    weights = {r['module'] for r in inventory['targets'] if r['tensor'] == 'weight'}
    if not selected <= weights or any(not allowed.fullmatch(path) for path in weights):
        raise ValueError('Native modulation inventory does not match supported architecture')
    modules = {}
    for path in sorted(selected):
        record = targets[(path, 'weight')]
        module = model.get_submodule(path)
        adapter = module.parametrizations.weight[0]
        weight = module.parametrizations.weight.original
        if record['layout'] != 'output_rank1' or adapter.layout != 'output_rank1':
            raise ValueError(f'{path}: actual factor layout is not output_rank1')
        if tuple(record['shape']) != tuple(weight.shape):
            raise ValueError(f'{path}: stale weight metadata')
        rows = weight.shape[0]
        if adapter.v_vector.shape != (rows,) or adapter.u_vector.numel() != weight.numel() // rows:
            raise ValueError(f'{path}: factor dimensions do not match output rows')
        masks_bias = path.endswith('.affine') or (component == 'D' and (path, 'bias') in targets)
        bias = f'{path}.parametrizations.bias.original' if masks_bias else None
        offset = f'{path}.parametrizations.bias.0.b_vector' if masks_bias else None
        if masks_bias and (model.get_parameter(bias).shape != (rows,) or model.get_parameter(offset).shape != (rows,)):
            raise ValueError(f'{path}: selected bias must have one entry per output row')
        prefix = f'{path}.parametrizations.weight'
        modules[path] = {'pool': ('g_affine' if path.endswith('.affine') else 'g_conv') if component == 'G' else 'd_conv',
                         'weight_parameter': prefix + '.original',
                         'u_parameter': prefix + '.0.u_vector', 'v_parameter': prefix + '.0.v_vector',
                         'bias_parameter': bias, 'b_parameter': offset, 'weight_shape': list(weight.shape)}
    return {'component': component, 'conv_layout': 'output_rank1', 'modules': modules}


class FixedOffsetsAdaptationMasks:
    def __init__(self, model, selection):
        if '_adam_native_adaptation_mask_metadata' in model.__dict__:
            raise ValueError('Adaptation masks already installed')
        if selection.get('fixed_offset_policy') != POLICY:
            raise ValueError('Require explicitly fixed-offset selection policy')
        fixed_reference = capture_fixed_reference(model)
        if selection.get('fixed_offset_names') != sorted(fixed_reference):
            raise ValueError('Fixed-offset selection inventory mismatch')
        expected = adaptation_inventory(model)
        if selection['component'] != expected['component'] or selection['conv_layout'] != 'output_rank1':
            raise ValueError('Selection component/layout mismatch')
        modules = expected['modules']
        if set(selection['masks']) != set(modules) or set(selection['modules']) != set(modules):
            raise ValueError('Selection must cover every selected native module exactly once')
        masks = {}
        for path, record in modules.items():
            for key, value in record.items():
                if selection['modules'][path][key] != value:
                    raise ValueError(f'{path}: selection {key} does not match actual native parameter mapping')
            mask = selection['masks'][path]
            if (not isinstance(mask, list) or len(mask) != record['weight_shape'][0]
                    or any(type(v) is not bool for v in mask)):
                raise ValueError(f'{path}: mask must be one explicit boolean per output row')
            masks[path] = torch.tensor(mask, dtype=torch.bool, device=model.get_parameter(record['weight_parameter']).device)
        # Everything above validates without changing parameters or requires_grad.
        self.model = model
        self.selection = copy.deepcopy(selection)
        self.enabled = True
        self._optimizer = None
        self._all = dict(model.named_parameters())
        self._trainable = {n for n, _ in modulation_named_parameters(model)}
        metadata = describe_modulation(model)
        self._trainable.update(metadata['ordinary_trainable_names'])
        self._fixed_reference = fixed_reference
        self._trainable.difference_update(fixed_reference)
        self._allowed_rows = {}
        zero_entries = {}
        for path, record in modules.items():
            high = masks[path]
            self._allowed_rows[record['weight_parameter']] = ~high
            self._allowed_rows[record['v_parameter']] = high
            zero_entries[record['v_parameter']] = ~high
            self._trainable.add(record['weight_parameter'])
            if record['bias_parameter'] is not None:
                self._allowed_rows[record['bias_parameter']] = ~high
                self._allowed_rows[record['b_parameter']] = high
                zero_entries[record['b_parameter']] = ~high
                self._trainable.add(record['bias_parameter'])
        for name, rows in self._allowed_rows.items():
            if not bool(rows.any()):
                self._trainable.discard(name)
        with torch.no_grad():
            for name, low in zero_entries.items():
                self._all[name][low] = 0
        # Snapshot only protected original/adapter slices. Unselected source
        # tensors stay fully frozen; low-row originals are intentionally mutable.
        self._protected = {}
        for name, p in self._all.items():
            rows = self._allowed_rows.get(name)
            if rows is not None:
                self._protected[name] = (~rows, p.detach()[~rows].clone())
            elif name not in self._trainable:
                self._protected[name] = (None, p.detach().clone())
        self._buffers = {n: t.detach().clone() for n, t in model.named_buffers()}
        model.__dict__['_adam_native_adaptation_mask_metadata'] = {
            'component': expected['component'], 'conv_layout': 'output_rank1',
            'masks': copy.deepcopy(selection['masks']), 'fresh_adam_required': True,
            'fixed_offset_policy': POLICY, 'fixed_offsets': sorted(fixed_reference)}
        self.set_enabled(True)

    def named_parameters(self):
        return [(n, p) for n, p in self._all.items() if n in self._trainable]

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def set_enabled(self, enabled):
        if type(enabled) is not bool:
            raise ValueError('enabled must be bool')
        self.enabled = enabled
        self.model.requires_grad_(False)
        for _, p in self.named_parameters():
            p.requires_grad_(enabled)

    def _validate_optimizer(self, optimizer):
        if type(optimizer) is not torch.optim.Adam:
            raise ValueError('Only plain Adam is supported; no AdamW/weight decay')
        params = [p for group in optimizer.param_groups for p in group['params']]
        if len(params) != len(self.parameters()) or {id(p) for p in params} != {id(p) for p in self.parameters()}:
            raise ValueError('Optimizer must contain each adaptation parameter exactly once')
        if any(group['weight_decay'] != 0 for group in optimizer.param_groups):
            raise ValueError('Weight decay could move protected slices')
        if self._optimizer is not optimizer:
            raise ValueError('Bind a fresh optimizer before use')

    def bind_optimizer(self, optimizer):
        if self._optimizer is not None:
            raise ValueError('Optimizer already bound; implicit optimizer replacement/resume forbidden')
        if optimizer.state:
            raise ValueError('Main adaptation requires a fresh optimizer with empty state')
        self._optimizer = optimizer
        try:
            self._validate_optimizer(optimizer)
            self.verify_invariants(optimizer)
        except BaseException:
            self._optimizer = None
            raise

    def verify_invariants(self, optimizer=None):
        verify_fixed_offsets(self.model, self._fixed_reference, optimizer)
        current = dict(self.model.named_parameters())
        if current.keys() != self._all.keys() or any(current[n] is not p for n, p in self._all.items()):
            raise AssertionError('Registered parameter identity changed')
        for name, p in self._all.items():
            if p.requires_grad != (self.enabled and name in self._trainable):
                raise AssertionError(f'{name}: unexpected requires_grad policy')
            if not bool(torch.isfinite(p).all()):
                raise AssertionError(f'{name}: nonfinite parameter')
        for name, (rows, original) in self._protected.items():
            actual = self._all[name].detach()
            if rows is not None:
                actual = actual[rows]
            if not torch.equal(actual, original):
                raise AssertionError(f'{name}: protected parameter entries changed')
        current_buffers = dict(self.model.named_buffers())
        if current_buffers.keys() != self._buffers.keys() or any(not torch.equal(t, current_buffers[n]) for n, t in self._buffers.items()):
            raise AssertionError('Source buffers changed; preserve mapping w_avg and synthesis noise buffers')
        if optimizer is not None:
            self._validate_optimizer(optimizer)
            for name, p in self.named_parameters():
                state = optimizer.state.get(p, {})
                if set(state) - {'step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'}:
                    raise AssertionError(f'{name}: unsupported Adam state')
                if state:
                    if not {'step', 'exp_avg', 'exp_avg_sq'} <= set(state):
                        raise AssertionError(f'{name}: incomplete Adam state')
                    counter = state['step']
                    if not isinstance(counter, torch.Tensor) or counter.numel() != 1 or not bool(torch.isfinite(counter).all()) or counter.item() < 0:
                        raise AssertionError(f'{name}: invalid Adam step count')
                rows = self._allowed_rows.get(name)
                for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                    if key not in state:
                        continue
                    value = state[key]
                    if value.shape != p.shape or not bool(torch.isfinite(value).all()):
                        raise AssertionError(f'{name}: invalid Adam {key}')
                    if rows is not None and bool(torch.count_nonzero(value[~rows])):
                        raise AssertionError(f'{name}: protected Adam {key} entries are not zero')
        return True

    def mask_gradients(self, optimizer):
        self.verify_invariants(optimizer)
        if not self.enabled:
            raise ValueError('Cannot step a disabled component')
        # Validate all gradients before modifying any gradient tensor.
        for name, p in self.named_parameters():
            if p.grad is not None and (p.grad.layout != torch.strided or not bool(torch.isfinite(p.grad).all())):
                raise ValueError(f'{name}: invalid gradient')
        with torch.no_grad():
            for name, rows in self._allowed_rows.items():
                p = self._all[name]
                if p.grad is not None:
                    p.grad[~rows] = 0

    def step(self, optimizer):
        self.mask_gradients(optimizer)
        optimizer.step()
        self.verify_invariants(optimizer)
