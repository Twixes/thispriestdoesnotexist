# Full Juggernaut quality reference

```sh
research/.venv-flux-train/bin/python research/diffusion/compare_juggernaut_full.py --resolution 1024 --steps 35 --guidance-scale 3.0 --tiled-vae --cached-text --output research/diffusion/runs/juggernaut-full-v10-35step-cfg3-native1024-v1
```

Frozen full-base reference, not a latency candidate:35steps,CFG3,DPM++2M Karras,original eight comparison prompts/seeds plus retained warmup. No priest trigger, adapters, custom cohort, output selector or repairs are accepted by this reference. Author supports30–40steps/CFG3–7; square1024 is our site's framing choice instead of the card's portrait/landscape recommendations.

Scheduler: `DPMSolverMultistepScheduler`, deterministic `algorithm_type=dpmsolver++`,order2,midpoint,Karras sigmas,lower-order final step,zero terminal sigma,epsilon prediction. The separate CPU scheduler preflight confirms35timesteps. Actual UNet calls are counted during execution. This is the standard Diffusers DPM++2M mapping, not the stochastic SDE sampler and not an A1111 bit-identical claim. [Official scheduler mapping](https://huggingface.co/docs/diffusers/main/api/schedulers/overview).

Only verified full-model components are used. CLIPs load directly to MPS first; all positive/negative/pooled conditioning is saved on CPU and verified equal after reload. CLIPs are released before loading UNet/VAE directly to MPS. The own VAE uses512pixel/64latent tiles with25% overlap when `--tiled-vae` is selected. Final-step GC and unused MPS cache release reduce decoder memory pressure without modifying live tensors. All choices are recorded in protocol/runtime files.

Original CPU float32 seeded noise and RNG states, every native PNG and grayscale WebP, cached conditioning, file/tensor hashes, model/export/source provenance and resource records are retained. Timing includes device transfers and cache clearing, excludes fixed conditioning preparation, and separates warmup. It is local MPS timing, not deployment latency.

Guards remain35% available before start,20% runtime floor,24GiB RSS,22GiB MPS driver allocation,512MiB swap growth and20minutes. Failures are retained without automatic changes to quality settings. During preparation only syntax, frozen-case preflight, scheduler-only construction and component export/reload verification ran; full image generation is a separate root-owned operation.

The root-owned execution subsequently completed all eight cases plus warmup in `runs/juggernaut-full-v10-35step-cfg3-native1024-v1`. Each image recorded 35 actual UNet calls. Median warm generation plus encoding was 24.710 seconds; maximum was 28.078 seconds. See `../reviews/juggernaut-realism/` for the unfiltered paired preview and subjective review. This is not a production approval or a server benchmark.
