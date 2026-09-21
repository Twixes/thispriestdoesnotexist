"""Offline SDXL-Turbo CUDA benchmark; no server, downloads or provisioning."""
import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
MODEL_REVISION = "71153311d3dbb46851df1931d3ca6e939de83304"
PROVENANCE_SHA256 = "3339ec6460b58e799ef2c323349f949387a8be1e53feca5324b5614e6ff62a09"
DIFFUSERS_COMMIT = "9f1246971270c84dcbe71233edb7a519596a5d02"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def distribution(values):
    ordered = sorted(values)
    return {"count": len(ordered), "median": (ordered[(len(ordered)-1)//2] + ordered[len(ordered)//2])/2,
            "p95_nearest_rank": ordered[math.ceil(.95 * len(ordered))-1], "max": ordered[-1],
            "strictly_below_500ms_count": sum(v < .5 for v in ordered)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/sdxl-turbo")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="Local Diffusers LoRA .safetensors file")
    parser.add_argument("--adapter-sha256", help="Required expected SHA-256 when using --adapter")
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--fuse-lora", action="store_true", help="Fuse optional adapter before conditioning/warmup")
    parser.add_argument("--steps", type=int, choices=[1, 4], default=1)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=202609214000)
    parser.add_argument("--prompts-json", type=Path, help="JSON array of nonempty prompts; cycled in order")
    parser.add_argument("--uncached-prompts", action="store_true", help="Time text encoding on each request")
    parser.add_argument("--compile-unet", action="store_true", help="torch.compile reduce-overhead; compilation is warmup")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.count < 1 or args.warmups < 1 or args.cpu_threads < 1:
        parser.error("count, warmups and CPU threads must be positive")
    if bool(args.adapter) != bool(args.adapter_sha256):
        parser.error("--adapter and --adapter-sha256 must be supplied together")
    if args.fuse_lora and not args.adapter:
        parser.error("--fuse-lora requires an adapter")
    if not math.isfinite(args.adapter_scale):
        parser.error("adapter scale must be finite")
    if args.output.exists():
        parser.error("output already exists; use a fresh directory")
    from compare import PROMPTS
    prompts = json.loads(args.prompts_json.read_text()) if args.prompts_json else PROMPTS
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
        parser.error("prompts must be a nonempty JSON array of nonempty strings")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    started = time.perf_counter()
    import torch
    import diffusers
    from PIL import features
    from restore import restore
    imports_seconds = time.perf_counter() - started
    diffusers_install = json.loads(importlib.metadata.distribution("diffusers").read_text("direct_url.json") or "{}")
    if diffusers_install.get("vcs_info", {}).get("commit_id") != DIFFUSERS_COMMIT:
        raise RuntimeError(f"Use pinned Diffusers git commit {DIFFUSERS_COMMIT}, matching the adapter evaluator")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; this script does not substitute CPU or MPS timings")
    if not features.check("webp"):
        raise RuntimeError("Pillow must support WebP")
    device = torch.device(f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    args.model_dir = args.model_dir.resolve()
    check_start = time.perf_counter()
    if sha(args.model_dir / "provenance.json") != PROVENANCE_SHA256:
        raise ValueError("Pinned SDXL-Turbo provenance manifest mismatch")
    provenance = restore(args.model_dir)  # Validates every file; restores verified LFS chunks if needed.
    if provenance["revision"] != MODEL_REVISION or provenance["model"] != "stabilityai/sdxl-turbo":
        raise ValueError("Unexpected model identity")
    adapter = None
    if args.adapter:
        args.adapter = args.adapter.resolve()
        if args.adapter.suffix != ".safetensors" or sha(args.adapter) != args.adapter_sha256.lower():
            raise ValueError("Adapter must be a safetensors file matching the expected SHA-256")
        adapter = {"path": str(args.adapter), "sha256": args.adapter_sha256.lower(), "scale": args.adapter_scale, "fused": args.fuse_lora}
    verification_seconds = time.perf_counter() - check_start
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output / "benchmark-source.py")
    shutil.copyfile(args.model_dir / "provenance.json", args.output / "model-provenance.json")
    driver = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=15, check=True).stdout.strip()
    versions = {}
    for package in ("torch", "diffusers", "transformers", "accelerate", "peft", "pillow", "safetensors", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata = {
        "complete": False, "model_revision": MODEL_REVISION, "model_provenance_sha256": PROVENANCE_SHA256,
        "adapter": adapter, "source_sha256": sha(__file__), "packages": versions, "diffusers_git_commit": DIFFUSERS_COMMIT,
        "platform": platform.platform(), "python": platform.python_version(), "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(), "nvidia_smi": driver, "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device), "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "resolution": 512, "batch_size": 1, "steps": args.steps, "guidance_scale": 0.0, "dtype": "float16",
        "compile_unet": args.compile_unet, "compile_mode": "reduce-overhead" if args.compile_unet else None,
        "prompt_embeddings_cached": not args.uncached_prompts, "prompts": prompts,
        "seed_base": args.seed_base, "count": args.count, "warmups": args.warmups, "cpu_threads": args.cpu_threads,
        "noise": "fresh CPU float32 torch.randn, CPU generator; converted to CUDA float16 inside timer; consumed generator passed on for scheduler noise",
        "watermarker": False, "tf32": False,
        "encoding": {"format": "WebP", "quality": 90, "method": 4, "grayscale_rgb": True},
        "quality_checks_and_retries": "none; all outputs retained for human review, no rejection or selection",
        "timing_scope": "CPU RNG + noise transfer + pipeline including VAE/PIL + grayscale + in-memory WebP encoding",
        "excluded_from_generation": ["artifact file writes and hashes", "model load", "cached prompt embedding setup", "HTTP/network", "queueing"],
        "http_latency_measured": False, "production_latency_proven": False,
        "imports_seconds": imports_seconds, "verification_and_reassembly_seconds": verification_seconds,
    }
    write(args.output / "result.json", metadata)
    torch.cuda.synchronize(device)
    load_start = time.perf_counter()
    pipe = diffusers.AutoPipelineForText2Image.from_pretrained(args.model_dir, torch_dtype=torch.float16,
                    variant="fp16", local_files_only=True, use_safetensors=True, add_watermarker=False)
    if args.adapter:
        pipe.load_lora_weights(str(args.adapter.parent), weight_name=args.adapter.name,
                               adapter_name="priest", local_files_only=True, use_safetensors=True)
        if args.fuse_lora:
            pipe.fuse_lora(lora_scale=args.adapter_scale, safe_fusing=True)
            pipe.unload_lora_weights()
        else:
            pipe.set_adapters(["priest"], [args.adapter_scale])
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    torch.cuda.synchronize(device)
    metadata["cold_model_and_adapter_load_seconds"] = time.perf_counter() - load_start
    metadata["scheduler"] = dict(pipe.scheduler.config)
    metadata["vae_force_upcast"] = pipe.vae.config.force_upcast
    setup_start = time.perf_counter()
    embeddings = []
    embedding_miss_seconds = []
    with torch.inference_mode():
        if not args.uncached_prompts:
            for prompt in prompts:
                torch.cuda.synchronize(device)
                embedding_start = time.perf_counter()
                embedded, _, pooled, _ = pipe.encode_prompt(prompt=prompt, device=device, do_classifier_free_guidance=False)
                torch.cuda.synchronize(device)
                embedding_miss_seconds.append(time.perf_counter() - embedding_start)
                embeddings.append({"prompt_embeds": embedded, "pooled_prompt_embeds": pooled})
        if args.compile_unet:
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    torch.cuda.synchronize(device)
    metadata["conditioning_and_compile_setup_seconds"] = time.perf_counter() - setup_start
    metadata["prompt_cache_miss_encoding_seconds"] = embedding_miss_seconds
    write(args.output / "result.json", metadata)
    records = []
    try:
        with torch.inference_mode():
            for index in range(-args.warmups, args.count):
                prompt_index = max(0, index) % len(prompts)
                seed = args.seed_base + index
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                conditioning = embeddings[prompt_index] if embeddings else {"prompt": prompts[prompt_index]}
                conditioning_done = time.perf_counter()
                generator = torch.Generator(device="cpu").manual_seed(seed)
                noise = torch.randn((1, 4, 64, 64), generator=generator, dtype=torch.float32, device="cpu")
                noise_done = time.perf_counter()
                image = pipe(**conditioning, height=512, width=512, num_inference_steps=args.steps,
                             guidance_scale=0.0, generator=generator, latents=noise.to(device=device, dtype=torch.float16)).images[0]
                torch.cuda.synchronize(device)
                pipeline_done = time.perf_counter()
                encoded = io.BytesIO()
                image.convert("L").convert("RGB").save(encoded, format="WEBP", quality=90, method=4)
                body = encoded.getvalue()
                torch.cuda.synchronize(device)
                end = time.perf_counter()
                directory = args.output / (f"warmup-{index + args.warmups:03}" if index < 0 else f"{index:03}")
                directory.mkdir()
                (directory / "display.webp").write_bytes(body)
                image.save(directory / "native.png")
                record = {"index": index, "warmup": index < 0, "seed": seed, "prompt_index": prompt_index,
                          "prompt_cache_hit": bool(embeddings), "conditioning_lookup_seconds": conditioning_done-start,
                          "cpu_noise_seconds": noise_done-conditioning_done, "pipeline_and_transfer_seconds": pipeline_done-noise_done,
                          "encoding_seconds": end-pipeline_done, "generation_seconds": end-start,
                          "cpu_noise_sha256": hashlib.sha256(noise.numpy().tobytes()).hexdigest(),
                          "webp_sha256": hashlib.sha256(body).hexdigest(), "native_png_sha256": sha(directory / "native.png"),
                          "webp_bytes": len(body), "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                          "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)}
                write(directory / "record.json", record)
                records.append(record)
                with (args.output / "records.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps({"index": index, "generation_seconds": end-start}), flush=True)
        measured = [r for r in records if not r["warmup"]]
        metadata.update(complete=True, measured_count=len(measured),
                        generation_seconds=distribution([r["generation_seconds"] for r in measured]),
                        cpu_noise_seconds=distribution([r["cpu_noise_seconds"] for r in measured]),
                        warmup_generation_seconds=[r["generation_seconds"] for r in records if r["warmup"]],
                        unique_webp_hashes=len({r["webp_sha256"] for r in measured}))
    finally:
        metadata["recorded_outputs"] = len(records)
        write(args.output / "result.json", metadata)


if __name__ == "__main__":
    main()
