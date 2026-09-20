# Real HTTP inference under a 3 GiB Linux limit

The existing 1024-pixel inference service passed this bounded local HTTP test with **3 GiB memory, no extra swap, one CPU, and one Torch thread**. The server and client both ran inside one `--network none` Linux/amd64 container, communicating only over its loopback interface; no host ports, external services, credentials, GPU work, or cloud charges were involved.

The model remains the **unadapted FFHQ baseline**, with training step zero and `reviewed: false`. The harness explicitly requires `ALLOW_UNREVIEWED=1` and passes the service's local-research `--allow-unreviewed` flag. Neither the model nor any generated image is approved for the priest site. Production configuration and candidate sizing were not changed.

The initial host check reported 28% free memory on a 48 GiB Mac. The captured check immediately before execution reported **23% free**, still above the 20% caution threshold used for these bounded local diagnostics. Existing MPS training continued independently. The reused image is Linux amd64 running under local Apple Silicon emulation, not a native Cloudflare deployment.

| Measurement | Observed result |
| --- | ---: |
| Enforced cgroup memory maximum | 3,221,225,472 bytes / 3 GiB |
| Enforced additional swap allowance | 0 bytes |
| Cgroup peak, including server and client | **1,740,255,232 bytes / 1.621 GiB** |
| Server process high-water RSS | 1,295,972 KiB / 1.236 GiB |
| Memory limit-hit / OOM / OOM-kill counters | **0 / 0 / 0** |
| Python server launch to successful health | 5.740 s |
| First HTTP generation | 5.175 s |
| Second sequential HTTP generation | 4.535 s |
| Three admitted concurrent requests, including queue wait | **4.534 / 9.134 / 13.754 s** |
| Overloaded generation / health response | 503 in 1.19 / 0.88 ms |
| Recovery HTTP generation | 4.727 s |
| Total bounded harness elapsed time | 34.513 s |
| Server exit / Docker exit / Docker OOMKilled | **0 / 0 / false** |
| Distinct decoded images and distinct 256-bit seeds | **6 / 6** |

Health reported the expected safetensors hash, CPU device, 1024 resolution, step zero, and unreviewed status. All six successful responses decoded as **1024 × 1024 RGB WebP**, carried the expected model hash, had different 64-character hexadecimal seeds and output hashes, and used `Cache-Control: no-store, max-age=0`. Full headers, seeds, hashes, sizes, and timings are retained in `verification.json` along with the original WebP responses.

For deterministic admission verification, the client opened three genuine HTTP sockets with their header terminators briefly withheld. The server's process thread count increased from one to four, proving three request handlers were admitted. A fourth generation request and a health probe both received empty, uncached **503** responses with **`Retry-After: 3`**. Completing the first three headers then allowed their real model requests to finish; no generation function, lock, or server code was mocked. The increasing completion times reflect the service's single-generation lock. After all three completed, both health and another generation succeeded, verifying slot release and recovery.

`invocation.sh` preserves the exact Docker command. It uses the existing `priest-inference:test` image, a read-only root and model mount, a 128 MiB temporary filesystem, disabled Docker health checks, dropped capabilities, no-new-privileges, and a 128-process limit. `run.py` is the supervisor and loopback client. It captures cgroup counters after the server exits, so a child OOM is distinguishable from a successful run. `provenance.json` records the image/model/harness hashes and confirms the image's server source matches the current workspace. `docker-inspect.json` and `docker-state.json` preserve the enforced limits and terminal state. The stopped test container was removed after evidence collection; `cleanup.json` confirms no matching container remains.

The service log contains NNPACK unsupported-hardware warnings under emulation; all assertions and requests nevertheless completed. This result supersedes concern based solely on native Mac RSS for **this tested Linux service path**: the 3 GiB allocation accommodated the complete HTTP workload with observed headroom. It does not establish the memory needs of different model architectures, restoration stages, or long-running traffic. Six generated requests are not a load/soak test.

The measured startup excludes image pulling and platform container provisioning. Timings under amd64 emulation and concurrent laptop training are not Cloudflare latency predictions. The eventual trained generator must still pass visual review and be measured on the chosen real hosting platform, with its actual cold starts, traffic, account limits, and cost. No production deployment occurred.
