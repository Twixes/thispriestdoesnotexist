# FFHQ 1024 Linux memory-cap experiment

2026-09-21 Europe/Warsaw (2026-09-20 22:35 UTC). One bounded local Docker run of the unadapted 1024-pixel generator. This is deployment-resource evidence, not priest adaptation or visual approval. The manifest reports step zero and `reviewed: false`; the explicit local-only override allowed the benchmark.

The host reported **27% free** via `memory_pressure` before launch, above the required 20% threshold. The container used Linux amd64 emulation on Apple Silicon, one CPU, one inference thread, 4 GiB memory with no extra swap, no network, a read-only root and model mount, a 128 MiB temporary filesystem, and no Docker health-check workload. Two local training runs were also active.

| Measurement | Result |
| --- | ---: |
| Completed forwards | 1 cold + 3 warm |
| Cgroup peak memory | 1,613,926,400 bytes / **1.503 GiB** |
| Python peak RSS | 1,232,060,416 bytes / 1.147 GiB |
| Cgroup OOM / OOM kill / limit-hit events | **0 / 0 / 0** |
| Docker `OOMKilled` / exit status | **false / 0** |
| Model load after Python imports | 2.72 seconds |
| Approximate Python process launch to first image | 11.91 seconds |
| Cold forward | 5.32 seconds |
| Three warm forwards | 4.44 / 4.48 / 4.53 seconds |
| Four-image elapsed time | 25.35 seconds |
| Container CPU time | 25.27 CPU-seconds |
| Distinct, decoded RGB WebPs | 4 / 4, all 1024 × 1024 |

The existing service fits this tested 4 GiB Linux cap with substantial memory headroom. This result is more relevant to a Linux host than the earlier native Mac process peak, but it is only a sequential inference test of this exact unadapted bundle. It does not establish production latency on native x86, account eligibility, cost, full HTTP concurrency memory, or the final adapted model's behavior. NNPACK unsupported-hardware warnings are recorded in the log; all outputs completed and decoded. No larger-container retry was needed, no configuration was promoted, and no paid service was created.

Reproduce from the repository root, with the existing local `priest-inference:test` image:

```sh
docker run --name priest-ffhq1024-4g --platform linux/amd64 \
  --memory 4g --memory-swap 4g --cpus 1 --network none --no-healthcheck \
  --read-only --tmpfs /tmp:rw,nosuid,size=128m \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 \
  -v "$PWD/research/runs/inference-cpu/ffhq1024/baseline-bundle:/app/model:ro" \
  -v "$PWD/research/runs/inference-cpu/ffhq1024/docker-4g:/results" \
  -e CPU_THREADS=1 -e PYTHONPATH=/app \
  priest-inference:test python /results/run.py \
  > research/runs/inference-cpu/ffhq1024/docker-4g/run.log 2>&1
```

The supervisor captures cgroup counters after the child exits, so a child OOM is distinguishable from successful completion. `docker-state.json` and the sanitized `docker-inspect.json` retain the final Docker state, image ID, enforced limits, and mount access. The exited container was removed after saving that evidence. `benchmark.json` contains exact hashes, seeds and timings; `image-verification.json` records decoded dimensions. Generated images use Git LFS.
