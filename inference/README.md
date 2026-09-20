# CPU inference service

Every successful `GET /generate` performs the trained generator's actual mapping
and synthesis forward passes. A fresh 256-bit OS-random seed creates an independent
normal latent vector. No input seed, fixed catalog, image blend, or response cache
is accepted. Constant synthesis noise makes identity depend on the latent, while
preserving the random latent's full entropy. Independent random sampling is not a
mathematical guarantee that two images can never coincide.

`GET /health` returns model version, SHA-256, training step, resolution, and review
state. Image responses carry `Cache-Control: no-store`, the generation seed, model
checksum, and measured generation time. If `INFERENCE_TOKEN` is set, `/generate`
requires `Authorization: Bearer …`; health is readable without the token. The
Cloudflare worker should hold this secret and proxy images without caching.

The default two CPU threads run one generator forward at a time. At most three
requests are admitted, including two waiting for the generator. Further requests
receive an immediate `503` with `Retry-After: 3`; socket timeouts also bound slow
clients. `REQUEST_CAPACITY`, `CPU_THREADS`, `PORT`, and `MODEL_DIR` override defaults.
Use a TLS reverse proxy or authenticated tunnel; this process binds plain HTTP on
port 8080. It is intentionally a small origin service, not an internet-facing TLS
server. Health requests share the bounded admission capacity.

## Export and review

From the repository root with `research/.venv` installed:

```sh
research/.venv/bin/python research/export_model.py \
  --checkpoint research/runs/pilot256/generator-000250.pt \
  --output research/models/candidate256 --version pilot256-250
research/.venv/bin/python -m inference.benchmark \
  --model-dir research/models/candidate256 \
  --output research/runs/inference-cpu/candidate256 --allow-unreviewed
```

The offline exporter loads the trusted NVIDIA base pickle for architecture and the
local training state with `torch.load(weights_only=True)`. It writes
`generator.safetensors`, `model.json`, and the NVIDIA `LICENSE.txt`. Serving loads
only safetensors and architecture JSON, with strict tensor shapes and checksum
verification; no pickle is loaded in the server.

An export is **unreviewed by default**. After inspecting sufficient random outputs
for realistic, varied adult male priests, visible white collars, and no hats,
record the review in research with `approved: true`, `weights_sha256` matching this
exact bundle, and evidence paths. Re-export with `--review <that-json>` and promote
the bundle to `models/production/` through Git LFS. The runtime refuses unreviewed,
untrained, or checksum-mismatched weights. `--allow-unreviewed` exists only for local
experiments and must never be used by the production service. Neither the review
gate nor the benchmark proves visual quality by itself.

## Run and test

```sh
research/.venv/bin/python -m unittest inference.test_server -v
research/.venv/bin/python -m inference.server --model-dir models/production
docker build -f inference/Dockerfile -t priest-inference .
docker run --rm -p 127.0.0.1:8080:8080 \
  --mount type=bind,src="$PWD/models/production",dst=/app/model,readonly \
  --env INFERENCE_TOKEN priest-inference
```

The image contains NVIDIA reference code and its complete original license. The
model and derivatives retain NVIDIA's research/evaluation-only use limitation;
the project's application code is MIT. Builds install the CPU-only PyTorch wheel.
Image building and promotion should run through the repository's CI workflow.
