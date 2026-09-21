"""AdAM probing mechanics for the existing NVIDIA StyleGAN2 networks.

Independent implementation of the pinned AdAM rank-factor equations. Original
NVIDIA forward methods/scales are untouched. Probing is not Fisher selection or
main adaptation, and provides no evidence of image quality.

Source: yunqing-me/AdAM@6428e99cfb36bc8bda3506350824f0c7dac9a5ad,
gan_training/models/model_adam.py and AdAM_importance_probing.py. The default
conv layout deliberately follows its outer(u_flat,v_out).reshape(O,I,k,k),
which is rank one BEFORE reshape, not generally in the O-by-Ikk unfolding.
The optional output_rank1 layout is an explicit mathematical variant.
"""
import copy
import math
import torch
from torch import nn
from torch.nn.utils import parametrize

UPSTREAM_REVISION = "6428e99cfb36bc8bda3506350824f0c7dac9a5ad"
CONV_LAYOUTS = ("source_flattened", "output_rank1")


class RankOneWeight(nn.Module):
    """W_eff = W * (1 + rank-one-factor outer product, explicitly laid out)."""

    def __init__(self, weight: torch.Tensor, *, layout: str, generator: torch.Generator,
                 init_scale: float = 1e-7):
        super().__init__()
        if weight.ndim not in (2, 4) or not weight.is_floating_point():
            raise ValueError("Expected floating FC2D or convolution4D weights")
        if layout not in CONV_LAYOUTS or not math.isfinite(init_scale) or init_scale <= 0:
            raise ValueError("Invalid layout or positive initialization scale")
        self.layout = "output_rank1" if weight.ndim == 2 else layout
        self.weight_shape = tuple(weight.shape)
        self.init_scale = init_scale
        output_channels = weight.shape[0]
        flattened_input = weight.numel() // output_channels
        # A separate CPU generator prevents changing model/training RNG state.
        self.u_vector = nn.Parameter((torch.randn(flattened_input, generator=generator,
            dtype=torch.float32) * init_scale).to(device=weight.device, dtype=weight.dtype))
        self.v_vector = nn.Parameter((torch.randn(output_channels, generator=generator,
            dtype=torch.float32) * init_scale).to(device=weight.device, dtype=weight.dtype))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.layout == "source_flattened":
            modulation = torch.outer(self.u_vector, self.v_vector).reshape(self.weight_shape)
        else:
            modulation = torch.outer(self.v_vector, self.u_vector).reshape(self.weight_shape)
        return weight * (torch.ones_like(weight) + modulation)


class AdditiveOffset(nn.Module):
    """Bias/noise offset remains active during eval or disabled parameter gradients."""

    def __init__(self, original: torch.Tensor):
        super().__init__()
        self.b_vector = nn.Parameter(torch.zeros_like(original))

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        return original + self.b_vector


def _installed(model: nn.Module) -> bool:
    return "_adam_native_metadata" in model.__dict__


def _require_installed(model: nn.Module) -> dict:
    if not _installed(model):
        raise ValueError("Call install_modulation first")
    return model.__dict__["_adam_native_metadata"]


def _targets(model: nn.Module, component: str) -> list[tuple[str, nn.Module, str]]:
    """Discover actual network tensors, independent of resolution/block counts."""
    targets = []
    for path, module in model.named_modules():
        if component == "G":
            if not path.startswith(("mapping.", "synthesis.")):
                continue
        elif not path or path.startswith("b4.") or path == "b4":
            continue
        weight = module._parameters.get("weight")
        if weight is None or weight.ndim not in (2, 4):
            continue
        if component == "D" and weight.ndim != 4:
            raise ValueError("Unexpected conditional/linear discriminator prefix: " + path)
        targets.append((path, module, "weight"))
        if module._parameters.get("bias") is not None:
            targets.append((path, module, "bias"))
        if component == "G" and module._parameters.get("noise_strength") is not None:
            targets.append((path, module, "noise_strength"))
    return targets


def install_modulation(model: nn.Module, *, component: str,
                       conv_layout: str = "source_flattened", seed: int = 20260922,
                       init_scale: float = 1e-7) -> dict:
    """Install source-layout probing adapters IN PLACE; return primitive metadata.

    G: modulate mapping, synthesis, affine and ToRGB weights/biases plus noise
    strengths. Freeze every original parameter (including learned constant).
    D: modulate fromRGB and residual convolution/skip weights/biases. Original
    b4 conv/fc/out remain fully trainable as in official importance probing.

    Exact initial effective tensors are checked before mutation. The source's
    nonzero1e-7 factors round 1+outer to exactly1 in FP32 while retaining nonzero
    factor gradients; no centered/rank-two or dead both-zero initialization.
    Unsupported precision/scales fail instead of approximating equality.
    """
    if component not in ("G", "D") or conv_layout not in CONV_LAYOUTS:
        raise ValueError("Expected component G/D and a documented conv layout")
    if _installed(model) or any(parametrize.is_parametrized(m) for m in model.modules()):
        raise ValueError("Refuse duplicate or pre-existing parametrizations")
    if model.c_dim != 0:
        raise ValueError("This experiment supports unconditional networks only")
    if component == "G" and "synthesis" not in model._modules:
        raise ValueError("Expected NVIDIA generator mapping+synthesis structure")
    if component == "D" and "b4" not in model._modules:
        raise ValueError("Expected NVIDIA discriminator with b4 epilogue")
    targets = _targets(model, component)
    if not targets:
        raise ValueError("No modulation targets discovered")
    originals = [(name, tuple(value.shape)) for name, value in model.named_parameters()]
    ordinary_trainable = [name for name, _ in originals if component == "D" and name.startswith("b4.")]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pending = []
    records = []
    for path, module, tensor_name in targets:
        original = module._parameters[tensor_name]
        if original.device.type == "meta":
            raise ValueError("Identity verification requires materialized small/native source tensors")
        adapter = (RankOneWeight(original, layout=conv_layout, generator=generator, init_scale=init_scale)
                   if tensor_name == "weight" else AdditiveOffset(original))
        with torch.no_grad():
            if not torch.equal(adapter(original), original):
                raise ValueError(f"Initialization is not exactly identical: {path}.{tensor_name}; dtype/scale unsupported")
        pending.append((module, tensor_name, adapter))
        records.append({"module": path, "tensor": tensor_name, "shape": list(original.shape),
            "dtype": str(original.dtype), "mechanism": "multiplicative_rank_factors" if tensor_name == "weight" else "additive_offset",
            "layout": adapter.layout if tensor_name == "weight" else None,
            "base_parameter": f"{path}.parametrizations.{tensor_name}.original",
            "initial_effective_tensor_equal": True})
    # Preflight above leaves the model untouched on identity/shape failure.
    model.requires_grad_(False)
    for module, tensor_name, adapter in pending:
        parametrize.register_parametrization(module, tensor_name, adapter)
    model.__dict__["_adam_native_metadata"] = {"component": component,
        "upstream_revision": UPSTREAM_REVISION, "stage": "importance_probing_only",
        "conv_layout": conv_layout, "init_scale": init_scale, "initialization_seed": seed,
        "initialization": "Source nonzero small factors, exact effective-weight equality checked in actual dtype",
        "native_equalized_learning_rate_and_forward_unchanged": True,
        "bias_offsets_active_independently_of_requires_grad": True,
        "original_parameter_names": [name for name, _ in originals],
        "ordinary_trainable_names": ordinary_trainable, "targets": records}
    set_probing_grad(model, True)
    return describe_modulation(model)


def modulation_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    _require_installed(model)
    return [(name, parameter) for name, parameter in model.named_parameters()
            if ".parametrizations." in name and not name.endswith(".original")]


def modulation_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for _, parameter in modulation_named_parameters(model)]


def probing_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    metadata = _require_installed(model)
    return modulation_named_parameters(model) + [(name, model.get_parameter(name))
                                                for name in metadata["ordinary_trainable_names"]]


def probing_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Only these parameters belong in the probing optimizer (D includes epilogue)."""
    return [parameter for _, parameter in probing_named_parameters(model)]


def frozen_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    trainable = {id(parameter) for parameter in probing_parameters(model)}
    return [(name, parameter) for name, parameter in model.named_parameters() if id(parameter) not in trainable]


def set_probing_grad(model: nn.Module, enabled: bool) -> None:
    """Use instead of model.requires_grad_(True), which unfreezes source weights."""
    _require_installed(model)
    model.requires_grad_(False)
    for parameter in probing_parameters(model):
        parameter.requires_grad_(enabled)


def describe_modulation(model: nn.Module) -> dict:
    metadata = copy.deepcopy(_require_installed(model))
    trainable = probing_named_parameters(model)
    frozen = frozen_parameters(model)
    metadata.update(modulation_parameter_count=sum(p.numel() for p in modulation_parameters(model)),
        probing_parameter_count=sum(p.numel() for _, p in trainable),
        probing_parameter_names=[name for name, _ in trainable],
        frozen_parameter_names=[name for name, _ in frozen],
        currently_enabled_names=[name for name, p in model.named_parameters() if p.requires_grad])
    return metadata


def fold_modulation(model: nn.Module, *, inplace: bool = False) -> nn.Module:
    """Return ordinary NVIDIA weights with static modulation folded in, no adapters.

    Copies by default. Root should compare fixed outputs before exporting the
    standard state_dict. This function makes no claim about trained image quality.
    """
    _require_installed(model)
    folded = model if inplace else copy.deepcopy(model)
    for module in list(folded.modules()):
        if parametrize.is_parametrized(module):
            for name in list(module.parametrizations.keys()):
                parametrize.remove_parametrizations(module, name, leave_parametrized=True)
    del folded.__dict__["_adam_native_metadata"]
    folded.requires_grad_(False)
    return folded
