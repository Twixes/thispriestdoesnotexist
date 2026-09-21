# Native StyleGAN2 input audit

Use the original NVIDIA FFHQ1024 checkpoint, not any priest-adapted generator:

- `research/models/ffhq1024.pkl` — SHA256 `a205a346e86a9ddaae702e118097d014b7b8bd719491396a162cca438f2f524c`,381,624,121 bytes; original public NVIDIA URL recorded in its provenance.
- `research/vendor/stylegan2-ada-pytorch/legacy.py` returns `G`, `D`, `G_ema`; it also handles the original TensorFlow conversion. Current source hashes match the existing discriminator extraction pins.
- `research/.venv/bin/python` imports that loader and all relevant NVIDIA ops successfully: Python3.11.16, Torch2.14, NumPy2.4.6. No model was instantiated or loaded by this audit. The SG3 setuptools workaround is unnecessary for this SG2 path.

## Simplest raw G/D loader

In a fresh process, put only `research/vendor/stylegan2-ada-pytorch` ahead of other research packages on `sys.path`. Assert `legacy.__file__` and `training.networks.__file__` resolve there. Hash the fixed pickle **before** deserializing. Then:

```python
with checkpoint.open('rb') as stream:
    nets = legacy.load_network_pkl(stream)
G = nets.pop('G').cpu().eval().requires_grad_(False)
D = nets.pop('D').cpu().eval().requires_grad_(False)
del nets                         # release unused G_ema immediately
import gc
gc.collect()
assert G.img_resolution == D.img_resolution == 1024
assert G.c_dim == D.c_dim == 0
assert G.img_channels == D.img_channels == 3
```

Do not import StyleGAN3's or AdAM's separate `training` package under the same module names in this process. Do not load `ffhq256.pkl`, a reference625/paired/resume checkpoint, or a Turbo adapter.

There is also a lower-allocation, non-pickle alternative: construct NVIDIA `Generator(**model_json['init_kwargs'])` and load `research/runs/inference-cpu/ffhq1024/baseline-bundle/generator.safetensors` strictly; construct `Discriminator(**metadata['init_kwargs'])` and load `research/models/ffhq1024-discriminator/D.safetensors` strictly. Both hashes were checked. **That safe generator is original pretrained `G_ema`, not raw `G`.** Record that choice explicitly if it matches the audited AdAM starting point. Its manifest has `training_step:0` and adjacent original-source provenance, but a null `source_checkpoint_sha256`; this audit did not perform a new source-pickle tensor comparison. The D has a previously verified exact source reconstruction and safetensor roundtrip (82 state tensors), with full constructor preserved.

## CPU reference operations and one-image differentiation

Use FP32, CPU1/inter-op1, batch1, `force_fp32=True`, and explicitly `fused_modconv=False`. CPU `bias_act` and `upfirdn2d` already choose the upstream PyTorch reference operations; no CUDA extension compilation or resampling replacement is needed. Leave `conv2d_gradfix.enabled` and `grid_sample_gradfix.enabled` false. Retain original filters, clamps, resolution, D architecture and minibatch-standard-deviation settings.

`eval()` does **not** disable gradients. It is useful here because `G.train()` updates `mapping.w_avg` even if all original parameters are frozen. Enable only the intended AdAM modulation probes; preserve original parameters and buffers. Alternatively, call `G.mapping(z, None, skip_w_avg_update=True)` followed by `G.synthesis(...)`. Passing `skip_w_avg_update` into `G.forward` is wrong: that wrapper forwards extra kwargs to synthesis, not mapping.

For the first bounded derivative check:

1. Enable G probe gradients; freeze D parameters/probes. Generate one fresh512-dimensional Gaussian z at psi1 and native1024; deterministic `noise_mode='const'` makes this compatibility check repeatable. Backpropagate `softplus(-D(fake)).mean()` through D to G probes. D must not be under `no_grad`, and fake must not be detached.
2. Release that graph. Generate a detached fake under `no_grad`; freeze G probes and enable D probes. Backpropagate fake `softplus(D(fake)).mean()` and real `softplus(-D(real)).mean()` sequentially, preserving their intended sum. Avoid retaining both D graphs simultaneously.
3. Check finite outputs, losses and probe gradients, actual selected parameter gradients, and unchanged original weights **and buffers**. Save the latent, native output and resource/timing evidence.

This is a differentiability smoke, not a substitute for the audited AdAM objective. Fisher importance must follow the upstream per-example/squared-gradient convention; squaring an arbitrary accumulated gradient changes it. Do not add R1 double-backward, path-length regularization, a full EMA copy, another teacher G, or optimizer states for frozen weights to this first check. Batch1 makes original D's whole-minibatch std channel effectively constant; microbatch accumulation cannot reproduce a larger batch's std statistic.

A conservative initial supervisor can use12GiB process-group RSS,20% available-memory floor,512MiB swap-growth stop and600-second deadline. These are proposed bounds: native G-to-D backward has **not** been measured by this audit. The older G_ema-only CPU1 inference measured about0.64s warm forward and3.86GB peak RSS; that is not a training-memory or server-latency estimate. Preserve any failed native smoke. Consider per-block non-reentrant activation checkpointing only after a measured need, with a separately recorded correctness check; do not silently shrink1024 or loosen the guards.

## Reviewed train images

`research/data/flux-priest-domain-v1/manifest.json` SHA256 `6796be940a10610843c154c7063b3bb907d0ffab18e014737c6abc6f02ee340b` selects exactly20 entries with `split == 'train'`. All20 destination hashes and1024×1024 headers were checked; exact paths/hashes are in `input-audit.json`. Load from `research/data/flux-priest-domain-v1/train/`, convert RGB, keep native geometry, transpose NCHW, convert FP32 and normalize with `/127.5 - 1`. Do not use captions for this unconditional G/D. Do not resize to512 or read validation/test images. No held-out image or sealed generated test was opened.

The owner rejected the latest diffusion baseline's photographic realism. Existing independent labels stay frozen, but no successful gradient smoke or agent metric means owner approval. NVIDIA's source/model license is retained and remains research/evaluation-only; this audit authorizes no production release.
