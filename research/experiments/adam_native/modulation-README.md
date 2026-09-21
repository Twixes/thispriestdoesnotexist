# Native AdAM probing mechanics

`modulation.py` implements importance-probing parameterization for the existing NVIDIA StyleGAN2 G/D. It does not implement Fisher estimation, target-aware kernel selection or main adaptation yet. No photographic-quality result follows from these tests.

```python
install_modulation(G, component="G", seed=20260922)
install_modulation(D, component="D", seed=20260923)
g_optimizer = torch.optim.Adam(probing_parameters(G), lr=0.002, betas=(0.0, 0.99))
set_probing_grad(G, True)
set_probing_grad(D, False)
# Run the experiment's separately reviewed losses and schedule.
metadata = describe_modulation(G)
folded_G = fold_modulation(G)  # Copy by default; original remains intact.
```

The optimizer line illustrates the API, not the official lazy-regularization schedule. The root-owned smoke specifies actual optimizer settings, losses, RNG, noise and resource guards. Use `set_probing_grad`, not blanket `requires_grad_(True)`, which would unfreeze source parameters. `probing_named_parameters` and `frozen_parameters` expose the exact invariant sets. `modulation_parameters` excludes D's ordinary trainable epilogue and therefore is not the complete D probing optimizer set.

G receives multiplicative weight factors and additive bias offsets for mapping FCs, all synthesis convolutions and their style affines, and all ToRGBs and their affines. Learned noise strengths receive scalar additive offsets. Every original G parameter, including the learned constant, stays frozen. D receives modulation on fromRGB and all residual convolutions/skips; original `b4.conv`, `b4.fc` and `b4.out` parameters are fully trainable, matching upstream probing. The loaded network's buffers, forward methods, equalized scales, resampling and minibatch-statistics behavior are retained. Caller controls mapping-average updates and training/eval modes explicitly.

Native1024 discovery finds17 synthesis convolutions,9ToRGBs and25 discriminator prefix convolutions without hardcoded12-layer loops. With the standard eight-layer mapping, G has197 probing parameter tensors; D has67 modulation tensors plus6 ordinary epilogue tensors. The lightweight1024 fixture uses narrow channels and no native forward/backward.

Source basis is [AdAM revision6428e99](https://github.com/yunqing-me/AdAM/tree/6428e99cfb36bc8bda3506350824f0c7dac9a5ad), archived by the independent audit task under `upstream/`. Differences and non-obvious choices are explicit:

- **Convolution layout:** default `source_flattened` reproduces upstream `outer(u_flat, v_out).reshape(O,I,k,k)`. That matrix is rank one before reshape; the output-channel unfolding is not generally rank one. `output_rank1` is an optional, explicitly different layout. FCs use `outer(v_out,u_in)` as upstream.
- **Initialization:** retain upstream nonzero factors scaled by1e-7. In actual FP32, `1 + outer` rounds to exactly1 while autograd factor derivatives remain nonzero. Every initial effective tensor is checked for bit equality before installation; unsupported precision/scales fail instead of making an approximate-identity claim. No both-zero factors or centered rank-two modification is used. A separate seeded CPU generator preserves global RNG state.
- **Forward scaling:** original NVIDIA FC/convolution scaling stays untouched. The upstream KML FC's different fan-out scale is not copied into a loaded native checkpoint.
- **Offsets remain functional in eval:** unlike upstream activation code that conditionally includes its learned offset based on `requires_grad`, learned offsets stay applied when gradients are disabled. Otherwise switching G/D phases or exporting would change the function.
- **Folding:** PyTorch parametrization removal deletes properties from a generated class. Ordinary deepcopy shares that class, so naive removal broke the original model. The corrected fold gives each copied module a separate property-owning class before removal. Tests explicitly rerender original G/D after folding and compare both folded outputs and strict ordinary-state reload outputs exactly. The failed first regression and archived source are recorded in `mechanics-tests.json`.

Save adapted `state_dict` plus metadata; PyTorch does not support pickling parametrized model objects. Reconstruct the original architecture, reinstall the identical policy and strictly load the adapted state. Folded ordinary state dictionaries need no adapters or prompts at inference. Dynamic class isolation is tested on the pinned current Torch/NVIDIA runtime; full pretrained smoke/export checks remain necessary.

Run small tests with `research/.venv/bin/python research/experiments/adam_native/test_modulation.py`. Seven tests pass, including tiny32px actual G/D forward/backward, R1/path-length second derivatives, frozen-base checks, adapted reload, native layer enumeration and fold regression. No pretrained checkpoint or teacher dataset is loaded by these tests. Exact hashes, observed0.534-second unittest runtime and failure history are in `mechanics-tests.json`. Native pretrained runtime is a separate root-owned experiment.
