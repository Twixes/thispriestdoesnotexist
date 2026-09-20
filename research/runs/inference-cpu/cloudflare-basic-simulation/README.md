# Local simulation of Cloudflare's basic resource limits

2026-09-20. Three real forwards through the existing `priest-inference:test` Linux amd64 image with the unreviewed `pilot256-250` model. This is a resource test only; the model is not visually approved and no Cloudflare resources were deployed.

| Measurement | Result |
| --- | ---: |
| CPU cap / application threads | 0.25 vCPU / 1 |
| Memory / swap cap | 1 GiB / no additional swap |
| Cgroup peak memory (including cache and emulator overhead) | 938,094,592 bytes / 894.64 MiB |
| Python peak RSS | 624,381,952 bytes / 595.46 MiB |
| Approximate Python process start to first image | 29.89 seconds |
| Model load, after Python dependencies imported | 9.29 seconds |
| First forward | 5.60 seconds |
| Two subsequent forwards | 3.50 and 4.62 seconds |
| Total CPU consumed by the container | 9.66 CPU-seconds |
| Unique output hashes | 3 / 3 |

The process completed successfully without an OOM at these limits, with about 129 MiB of cgroup memory headroom. The model has 256-pixel output; a larger model/resolution must be tested separately. This runs under Linux amd64 emulation on Apple Silicon, concurrent with local model training. It does not establish Cloudflare latency. The startup measurement begins inside Python and excludes VM/image provisioning. The image's Docker health check was left enabled and can add a small amount of CPU and memory overhead. The measured cgroup CPU time includes all container processes.

Run from the repository root after building the existing image:

```sh
docker run --rm --platform linux/amd64 \
  --memory 1g --memory-swap 1g --cpus 0.25 \
  --read-only --tmpfs /tmp:rw,nosuid,size=128m \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 \
  -v "$PWD/research/runs/inference-cpu/pilot250-bundle:/app/model:ro" \
  -v "$PWD/research/runs/inference-cpu/cloudflare-basic-simulation:/results" \
  -e CPU_THREADS=1 -e PYTHONPATH=/app \
  priest-inference:test python /results/smoke.py \
  > research/runs/inference-cpu/cloudflare-basic-simulation/run.log 2>&1
```

`benchmark.json` retains exact values, model hash, random seeds, output hashes, and cgroup counters. `sample-*.webp` are the actual outputs and must be stored in Git LFS. The wrapper explicitly allows an unreviewed model only for this local experiment. Production validation must continue to reject it.
