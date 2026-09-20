#!/usr/bin/env bash
# Research-only local Docker invocation; no production credentials or network.
set -euo pipefail
cd /Users/twixes/Developer/thispriestdoesnotexist
docker run --name priest-ffhq1024-http-3g --platform linux/amd64 \
  --memory 3g --memory-swap 3g --cpus 1 --network none --no-healthcheck \
  --read-only --tmpfs /tmp:rw,nosuid,size=128m \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 \
  -v "$PWD/research/runs/inference-cpu/ffhq1024/baseline-bundle:/app/model:ro" \
  -v "$PWD/research/runs/inference-cpu/ffhq1024/http-3g:/results" \
  -e CPU_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 -e PYTHONPATH=/app -e ALLOW_UNREVIEWED=1 \
  priest-inference:test python /results/run.py
