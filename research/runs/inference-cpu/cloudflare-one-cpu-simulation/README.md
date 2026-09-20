# Local simulation: one CPU, 3 GiB, one inference thread

2026-09-20. Same Linux amd64 image and unreviewed `pilot256-250` model as the basic-tier experiment. Four real forwards: one cold and three warm. No public deployment or model quality approval.

| Measurement | 0.25 CPU / 1 GiB | 1 CPU / 3 GiB |
| --- | ---: | ---: |
| Approximate Python start to first image | 29.89 s | 5.13 s |
| Model load after imports | 9.29 s | 1.88 s |
| First forward | 5.60 s | 1.00 s |
| Warm median | 4.06 s (2 samples) | 0.71 s (3 samples) |
| Warm maximum | 4.62 s | 0.91 s |
| Cgroup peak memory | 894.64 MiB | 890.61 MiB |
| Python peak RSS | 595.46 MiB | 585.18 MiB |
| Container CPU usage | 9.66 CPU-s / 3 images | 7.54 CPU-s / 4 images |
| Unique images | 3 / 3 | 4 / 4 |

The one-CPU option is preferred for an initial low-traffic Cloudflare trial: materially better local latency and plenty of memory headroom. Both tests ran under Apple Silicon amd64 emulation, with local training also active; this is not a controlled Cloudflare hardware benchmark or a guarantee of the same speedup. Startup begins inside Python and excludes image/VM provisioning. Cgroup memory includes page cache and emulator overhead. No extra swap was allowed.

Reproduce from the repository root after building `priest-inference:test`:

```sh
docker run --rm --platform linux/amd64 \
  --memory 3g --memory-swap 3g --cpus 1 \
  --read-only --tmpfs /tmp:rw,nosuid,size=128m \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 \
  -v "$PWD/research/runs/inference-cpu/pilot250-bundle:/app/model:ro" \
  -v "$PWD/research/runs/inference-cpu/cloudflare-one-cpu-simulation:/results" \
  -e CPU_THREADS=1 -e PYTHONPATH=/app \
  priest-inference:test python /results/smoke.py \
  > research/runs/inference-cpu/cloudflare-one-cpu-simulation/run.log 2>&1
```

The exact model hash, seeds, timings and cgroup counters are in `benchmark.json`; generated WebPs use Git LFS. Production must still require an approved model; the explicit unreviewed override is solely for this local resource experiment.
