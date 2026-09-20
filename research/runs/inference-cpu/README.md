# CPU serving verification — 2026-09-20

These are plumbing/performance artifacts, **not production-quality approvals**.
The baseline is the unadapted FFHQ model; the step-250 pilot was explicitly rejected
for visual quality by the training reviewer. Both bundles remain unreviewed and
cannot start the production service without the explicit local-experiment bypass.

Native local Apple Silicon CPU, two PyTorch threads (MPS training concurrently):

| Bundle | Cold generation | Warm median | Warm max | Peak RSS | Unique outputs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Untrained baseline | 0.252 s | 0.108 s | 0.132 s | 751 MB | 4 / 4 |
| Pilot step 250 | 0.294 s | 0.142 s | 0.171 s | 769 MB | 8 / 8 |

`baseline-local/benchmark.json` and `pilot250-local/benchmark.json` contain exact
platform, model hashes, full random seeds, per-forward timings, and output hashes.
Every sample was produced by an actual generator forward and encoded as WebP.

`inference.test_server` passed all six tests natively and inside the Linux x86-64
container. These exercise real tiny-network forwards and entropy, no-store HTTP
semantics, token authorization, bounded overload and recovery, failed inference,
and checksum/review refusal. `docker-tests.log` is the container test output.

The Dockerfile built successfully for `linux/amd64` using the actual PyTorch
2.14.0 CPU wheel. Its build log is `docker-build.log`; image size reported by
Docker was 362,030,838 bytes. The image was run locally under Apple
Silicon emulation with a read-only root, 3 GB memory, two CPU limit, 128 MB tmpfs,
256 PID limit, all capabilities dropped, and no-new-privileges. Three private HTTP
requests loaded the full step-250 safetensors and produced distinct outputs and
256-bit seeds in 1.062 / 0.820 / 0.818 seconds. These emulation timings are **not**
native VPS measurements. `pilot250-linux-http/verification.json` preserves all
headers and hashes; the service was stopped and the validation container removed.

No public deployment was performed. Review and benchmark the eventual accepted
trained model and measure it again on the actual personal hosting machine before
promotion.
