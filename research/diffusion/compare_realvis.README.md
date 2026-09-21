# Bounded RealVisXL V5 Lightning prior comparison

Native1024 command, using optional tiled decoding to reduce the local VAE memory peak:

```sh
research/.venv-flux-train/bin/python research/diffusion/compare_realvis.py --resolution 1024 --tiled-vae --output research/diffusion/runs/realvis-v5-lightning-sde5-native1024-tiled-v1
```

No model run occurred during preparation. Remove `--tiled-vae` only for a separately named untiled experiment. Tiling retains the checkpoint's own VAE but uses512-pixel/64-latent tiles with25% overlap; it can alter pixels and timing. Every protocol/runtime file records whether it was enabled. No external VAE, restoration or selected outputs are used.

Fixed settings: **five steps, CFG1, stochastic DPM++ SDE Karras via Diffusers `DPMSolverSinglestepScheduler`**. Exact overrides:

```json
{
  "algorithm_type": "sde-dpmsolver++",
  "solver_order": 2,
  "solver_type": "midpoint",
  "lower_order_final": true,
  "use_karras_sigmas": true,
  "use_exponential_sigmas": false,
  "use_beta_sigmas": false,
  "final_sigmas_type": "zero",
  "prediction_type": "epsilon"
}
```

The author supplies the sampler family, five steps and CFG1–2. Hugging Face's current scheduler mapping lists `DPMSolverSinglestepScheduler` for DPM++ SDE/Karras. **Its default algorithm is deterministic `dpmsolver++`, so the evaluator explicitly selects stochastic `sde-dpmsolver++`.** Midpoint/order2/final lower order are explicit implementation choices. The separate `DPMSolverSDEScheduler` implements a Brownian k-diffusion variant and can have more model calls; this experiment does not claim bit-identical A1111 output. [Author card](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning/blob/f4454158cedaab9f0688c199561d6c92525f3a85/README.md), [official mapping](https://huggingface.co/docs/diffusers/main/api/schedulers/overview), [pinned implementation](https://github.com/huggingface/diffusers/blob/9f1246971270c84dcbe71233edb7a519596a5d02/src/diffusers/schedulers/scheduling_dpmsolver_singlestep.py).

A **scheduler-only CPU preflight**, with no weights loaded or denoising, produced timesteps `[999,786,427,59,0]` and solver orders `[1,2,1,2,1]`. Its exact config/source hash is recorded in `models/realvisxl-v5-lightning/scheduler-preflight.json`. No scipy/torchsde installation was required and the shared environment was not modified. The eventual run records actual UNet pre-forward calls; it does not substitute nominal sampler steps for measured work.

All eight original comparison prompts and seeds plus warmup are hash-bound to the historical result. Fresh CPU float32 noise uses native1024 shape `[1,4,128,128]`; its generator continues into the stochastic scheduler. Noise and before/after RNG states are saved. Text is encoded per request. Native PNG, grayscale WebP, full per-image records and contact sheet retain every output. No retry, crop, hires fix, face repair or image selector. Failed/native nonfinite artifacts remain available.

Timing includes fresh RNG/noise, device transfer, uncached text encoding, UNet/scheduler work, original VAE, PIL conversion and grayscale WebP encoding. Raw PNG/provenance/latent writes are excluded; WebP saving includes its disk write. Record warmup separately and report warm median/p95/max. These timings include per-step guard overhead and exclude HTTP/queueing; they do not prove server generation below500ms.

FP16 components load sequentially directly onto MPS via local-only Diffusers/Transformers calls. Before launch require35% system memory available. Limits remain RSS24GiB, MPS driver22GiB, available memory20%, swap growth512MiB and20minutes. A failed guard does not trigger a lower-resolution retry.512 is only a separate tradeoff after a completed native result is judged promising; use `--native-review` with the same hash-bound decision structure documented for the Juggernaut evaluator.

Validation: component/full-part hashes, Python compilation, tokenizer fit and scheduler-only construction. No model loading, inference or training was performed by the preparation task. Research acceptance does not imply production approval.
