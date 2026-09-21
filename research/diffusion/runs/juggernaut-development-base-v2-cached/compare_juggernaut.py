"""Uncurated offline Juggernaut X Hyper prior comparison: TCD4, CFG1, eta0.3.

Uses original eight comparison prompts/seeds plus the retained warmup. Default
1024 is the native-quality comparison. A separate512 tradeoff requires a reviewed
promising1024 run; no LoRA, cached embeddings, retries, repairs or selected output.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models/juggernaut-x-hyper"
REVISION = "42fee7475922d8de7246cdcbe5e3a5e22c0ecf61"
PROVENANCE_SHA256 = "da4b21ed1fb3ff4dcc3cbd9881afc1549351214643d1ffe324421611fcdbe330"
CHECKPOINT_SHA256 = "010be7341cd98a136da775330ba3eb4e87025c6cfd2f5455dc64daee2200ae98"
ORIGINAL_RESULTS = HERE / "runs/sdxl-turbo-1step-v1/result.json"
ORIGINAL_RESULTS_SHA256 = "cb605b4146090d2457d232dc4671f0c1c1beb2e57210a67a84f5a84500c83c21"
LIMIT_SECONDS = 1200
GIB = 1024 ** 3


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    temporary = Path(path).with_name(Path(path).name + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def preflight(args):
    require(sha(MODEL / "provenance.json") == PROVENANCE_SHA256, "Pinned model provenance changed")
    manifest = json.loads((MODEL / "provenance.json").read_text())
    require(manifest["complete"] and manifest["revision"] == REVISION, "Wrong/incomplete model")
    require(sha(ORIGINAL_RESULTS) == ORIGINAL_RESULTS_SHA256, "Original comparison record changed")
    original = json.loads(ORIGINAL_RESULTS.read_text())
    cases = [{"index": r["index"], "seed": r["seed"], "prompt": r["prompt"], "warmup": r["warmup"],
              "original_noise_path": str(ORIGINAL_RESULTS.parent / ("warmup" if r["warmup"] else f"{r['index']:03}") / "noise.npz"),
              "original_noise_sha256": r["files"]["noise.npz"]} for r in original["records"]]
    require([c["index"] for c in cases] == list(range(-1, 8)), "Expected warmup plus original eight cases")
    require([c["seed"] for c in cases] == list(range(202609213999, 202609214008)), "Original seeds differ")
    if args.case_manifest:
        custom = json.loads(args.case_manifest.read_text())
        require(custom["split"] == "development", "Only development cases can be opened by this evaluator")
        expected = custom["sampling"]
        require((args.resolution, args.steps, args.guidance_scale, args.tiled_vae, args.prompt_prefix) ==
                (expected["resolution"], expected["steps"], expected["guidance_scale"], expected["tiled_vae"], expected["prompt_prefix"]),
                "CLI settings differ from frozen development sampling")
        require(expected["eta"] == .3, "Development eta differs from evaluator")
        custom_cases = custom["cases"]
        require(1 <= len(custom_cases) <= 48, "Bounded development cohort required")
        require([c["index"] for c in custom_cases] == list(range(len(custom_cases))), "Case indices must be sequential")
        require(len({c["seed"] for c in custom_cases}) == len(custom_cases), "Seeds must be distinct")
        require(all(isinstance(c["prompt"], str) and c["prompt"] and isinstance(c["seed"], int) for c in custom_cases), "Bad development cases")
        require(args.resolution == 1024, "Custom cohort is frozen to native1024")
        cases = [cases[0]] + [{**c, "warmup": False} for c in custom_cases]
    cases = [{**c, "prompt": args.prompt_prefix + c["prompt"]} for c in cases]
    lora = None
    if args.lora:
        require(args.lora.is_file() and args.lora.suffix == ".safetensors", "Provide an exact local safetensors adapter")
        lora = {"path": str(args.lora.resolve()), "sha256": sha(args.lora), "strength": 1.0}
    decision = None
    # Test images from the priest pilot are never touched by this prior comparison.
    if args.resolution == 512:
        require(args.native_review is not None, "512 requires --native-review after a promising native1024 comparison")
    if args.native_review is not None:
        decision = json.loads(args.native_review.read_text())
        require(decision["native_1024_promising"] is True, "The recorded native1024 review is not promising")
        reference = Path(decision["source_result"])
        if not reference.is_absolute():
            reference = args.native_review.resolve().parent / reference
        require(sha(reference) == decision["source_result_sha256"], "Native review source hash mismatch")
        previous = json.loads(reference.read_text())
        require(previous["complete"] and previous["model"] == "juggernaut-x-hyper" and previous["revision"] == REVISION
                and previous["native_resolution"] == 1024 and previous["steps"] == 4,
                "512 review must refer to this completed native1024 base protocol")
    require((args.steps, args.guidance_scale) in ((4,1.0),(6,1.5)), "Use frozen4/CFG1 or6/CFG1.5 arm")
    require(args.resolution == 1024 or args.steps == 4, "512 arm frozen to4/CFG1")
    protocol = {"protocol_id": "juggernaut-hyper-native-prior-v1" if args.resolution == 1024 else "juggernaut-hyper-512-tradeoff-v1",
                "model": "juggernaut-x-hyper", "revision": REVISION, "model_provenance_sha256": PROVENANCE_SHA256,
                "checkpoint_sha256": CHECKPOINT_SHA256, "original_comparison_result_sha256": ORIGINAL_RESULTS_SHA256,
                "resolution": args.resolution, "steps": args.steps, "guidance_scale": args.guidance_scale, "negative_prompt": None,
                "scheduler": "TCDScheduler.from_config(pinned publisher scheduler config)", "eta": .3,
                "eta_basis": "Our frozen choice; publisher specifies TCD but does not prescribe eta",
                "native_reference_resolution": 1024, "non_native_tradeoff": args.resolution == 512,
                "cases": cases, "count": len(cases)-1,
                "case_manifest_sha256": sha(args.case_manifest) if args.case_manifest else None,
                "prompt_prefix": args.prompt_prefix, "lora": lora, "cached_text": args.cached_text, "warmup_count": 1, "all_outputs_retained": True,
                "noise": "Fresh CPU float32 randn [1,4,resolution/8,resolution/8]; same numeric seeds, different512/1024 latent shapes, not matched identities",
                "timing_scope": "CPU RNG/fresh noise + transfer + text encoding or cached-embedding transfer + TCD/UNet + originalVAE + PIL + grayscaleWebP; fixed conditioning preparation excluded in cached mode; excludes other artifact writes, model load, warmup, HTTP, queueing and hypothetical retries",
                "load_strategy": "Single-file mmap+meta low_cpu_mem_usage FP16 CPU assembly; transfer components toMPS sequentially after return/GC",
                "vae_tiling": {"enabled": args.tiled_vae, "sample_tile": 512, "latent_tile": 64, "overlap": .25},
                "post_trained": lora is not None, "production_approved": False, "server_latency_proven": False,
                "source_sha256": sha(Path(__file__)),
                "native_review_sha256": sha(args.native_review) if args.native_review else None}
    return manifest, protocol


def worker(args):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import numpy as np
    import psutil
    import torch
    from PIL import Image
    from diffusers import StableDiffusionXLPipeline, TCDScheduler
    from diffusers.loaders import single_file, single_file_model, single_file_utils
    from diffusers.models import model_loading_utils
    from diffusers.schedulers import scheduling_tcd

    output = args.output
    protocol = json.loads((output / "protocol.json").read_text())
    require(sha(Path(__file__)) == protocol["source_sha256"], "Live evaluator changed after protocol freeze")
    manifest = json.loads((output / "model-provenance.json").read_text())
    require(sha(output / "model-provenance.json") == PROVENANCE_SHA256, "Model snapshot mismatch")
    require(torch.backends.mps.is_available(), "This bounded comparison is MPS-only, not a server benchmark")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.mps.set_per_process_memory_fraction(.55)
    started = time.monotonic()
    initial_swap = psutil.swap_memory().used

    def guard(phase):
        torch.mps.synchronize()
        vm = psutil.virtual_memory()
        record = {"phase": phase, "elapsed_seconds": time.monotonic() - started,
                  "rss_bytes": psutil.Process().memory_info().rss, "mps_driver_bytes": torch.mps.driver_allocated_memory(),
                  "mps_allocated_bytes": torch.mps.current_allocated_memory(), "available_fraction": vm.available / vm.total,
                  "swap_growth_bytes": psutil.swap_memory().used - initial_swap}
        with (output / "resources.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if (record["rss_bytes"] > 24 * GIB or record["mps_driver_bytes"] > 22 * GIB or
                record["available_fraction"] < .20 or record["swap_growth_bytes"] > 512 * 1024 ** 2 or
                record["elapsed_seconds"] > LIMIT_SECONDS):
            write(output / "guard-failure.json", record)
            raise MemoryError("Resource guard failed: " + phase)
        return record

    guard("before_model_file_verification")
    for entry in manifest["files"]:
        path = MODEL / entry["path"]
        require(path.is_file(), "Missing local file; run model reassemble.py separately: " + str(path))
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], "Model file mismatch: " + str(path))
    require(sha(MODEL / manifest["checkpoint_name"]) == CHECKPOINT_SHA256, "Checkpoint hash mismatch")
    guard("verified_all_upstream_files")
    load_started = time.perf_counter()
    # Installed loader uses a mapped safetensors state dictionary and meta models;
    # explicit dtype prevents allocating a full randomly initialized FP32 pipeline.
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(MODEL / manifest["checkpoint_name"]), config=str(MODEL), local_files_only=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, disable_mmap=False, add_watermarker=False)
    guard("single_file_cpu_assembly_complete")
    if protocol["lora"]:
        adapter = Path(protocol["lora"]["path"])
        require(sha(adapter) == protocol["lora"]["sha256"], "Adapter changed after protocol freeze")
        pipe.load_lora_weights(str(adapter.parent), weight_name=adapter.name)
        guard("lora_loaded")
    gc.collect()
    component_timings = {}
    for name, component in (("text_encoder", pipe.text_encoder), ("text_encoder_2", pipe.text_encoder_2),
                            ("unet", pipe.unet), ("vae", pipe.vae)):
        begin = time.perf_counter()
        component.eval().requires_grad_(False)
        component.to(device="mps", dtype=torch.float16)
        torch.mps.synchronize()
        gc.collect()
        component_timings[name] = time.perf_counter() - begin
        guard(name + "_transferred")
    if protocol["vae_tiling"]["enabled"]:
        pipe.vae.enable_tiling()
        pipe.vae.tile_sample_min_size = 512
        pipe.vae.tile_latent_min_size = 64
        pipe.vae.tile_overlap_factor = .25
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    require(pipe.scheduler.config.prediction_type == "epsilon", "Unexpected TCD prediction type")
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - load_started
    packages = {n: importlib.metadata.version(n) for n in
                ("torch", "diffusers", "transformers", "accelerate", "safetensors", "numpy", "Pillow", "psutil")}
    runtime = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
               "device": "mps", "packages": packages, "load_seconds": load_seconds,
               "component_transfer_seconds": component_timings, "scheduler": dict(pipe.scheduler.config),
               "dtype": "float16", "vae_force_upcast": pipe.vae.config.force_upcast,
               "vae_precision_policy": "original checkpoint VAE with pipeline automatic upcast; no substitute",
               "prompt_embeddings_cached": False, "watermark": False, "eta": .3,
               "vae_tiling": protocol["vae_tiling"], "lora": protocol["lora"],
               "source_hashes": {"single_file": sha(single_file.__file__), "single_file_model": sha(single_file_model.__file__),
                                 "single_file_utils": sha(single_file_utils.__file__), "model_loading_utils": sha(model_loading_utils.__file__),
                                 "tcd_scheduler": sha(scheduling_tcd.__file__)},
               "server_latency_proven": False, "generation_includes_per_step_guard_overhead": True}
    write(output / "runtime.json", runtime)
    cached_embeddings = {}
    conditioning_seconds = None
    if protocol["cached_text"]:
        from safetensors.torch import save_file
        begin = time.perf_counter()
        cache_dir = output / "conditioning"
        cache_dir.mkdir()
        cache_records = []
        for case in protocol["cases"]:
            with torch.inference_mode():
                encoded = pipe.encode_prompt(prompt=case["prompt"], device="mps", num_images_per_prompt=1,
                    do_classifier_free_guidance=protocol["guidance_scale"] > 1, negative_prompt=None)
            values = {name: tensor.detach().cpu().contiguous() for name, tensor in zip(
                ("prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds"), encoded) if tensor is not None}
            cached_embeddings[case["index"]] = values
            path = cache_dir / f"{case['index']}.safetensors"
            save_file(values, str(path))
            cache_records.append({"index": case["index"], "prompt": case["prompt"], "file_sha256": sha(path),
                "tensor_value_sha256": {name: hashlib.sha256(t.numpy().tobytes()).hexdigest() for name,t in values.items()}})
            del encoded
            guard("cached_prompt_" + str(case["index"]))
        pipe.text_encoder = None
        pipe.text_encoder_2 = None
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()
        conditioning_seconds = time.perf_counter() - begin
        write(cache_dir / "manifest.json", {"complete": True, "seconds": conditioning_seconds,
            "timing_note": "Fixed prompt preparation excluded from image time; CPU-to-MPS embedding transfer included",
            "records": cache_records})
        guard("text_encoders_released")
    runtime.update(prompt_embeddings_cached=protocol["cached_text"], conditioning_seconds=conditioning_seconds)
    write(output / "runtime.json", runtime)
    counter = {"calls": 0}

    def count_unet_call(module, positional_inputs):
        counter["calls"] += 1

    handle = pipe.unet.register_forward_pre_hook(count_unet_call)
    records = []
    try:
        for case in protocol["cases"]:
            directory = output / ("warmup" if case["warmup"] else f"{case['index']:03}")
            directory.mkdir()
            record = {**case, "status": "started", "steps": protocol["steps"], "eta": .3, "guidance_scale": protocol["guidance_scale"],
                      "resolution": args.resolution, "protocol_sha256": sha(output / "protocol.json"),
                      "model_provenance_sha256": PROVENANCE_SHA256, "runtime_sha256": sha(output / "runtime.json")}
            write(directory / "record.json", record)
            try:
                tokenization = {}
                for name, tokenizer in (("tokenizer", pipe.tokenizer), ("tokenizer_2", pipe.tokenizer_2)):
                    ids = tokenizer(case["prompt"], truncation=False)["input_ids"]
                    tokenization[name] = {"token_count": len(ids), "maximum_length": tokenizer.model_max_length, "input_ids": ids}
                    require(len(ids) <= tokenizer.model_max_length, "Original prompt would be truncated by this model tokenizer")
                record["tokenization"] = tokenization
                guard(f"case-{case['index']}-before")
                noise_start = time.perf_counter()
                rng = torch.Generator(device="cpu").manual_seed(case["seed"])
                noise = torch.randn((1, 4, args.resolution // 8, args.resolution // 8), generator=rng, dtype=torch.float32)
                noise_seconds = time.perf_counter() - noise_start
                np.savez(directory / "noise.npz", noise=noise.numpy())
                before_state = rng.get_state().numpy().tobytes()
                (directory / "generator-after-noise.bin").write_bytes(before_state)
                if args.resolution == 512:
                    require(sha(case["original_noise_path"]) == case["original_noise_sha256"], "Historical512 latent changed")
                    with np.load(case["original_noise_path"], allow_pickle=False) as original:
                        require(np.array_equal(noise.numpy(), original["noise"]), "Current CPU RNG differs from original512 comparison")
                counter["calls"] = 0

                def step_guard(pipeline, step, timestep, values):
                    if not bool(torch.isfinite(values["latents"]).all().item()):
                        raise FloatingPointError("Nonfinite denoised latent")
                    guard(f"case-{case['index']}-step-{step}")
                    return values

                torch.mps.synchronize()
                begin = time.perf_counter()
                with torch.inference_mode():
                    prompt_args = ({name: value.to("mps") for name, value in cached_embeddings[case["index"]].items()}
                                   if protocol["cached_text"] else {"prompt": case["prompt"]})
                    pixels = pipe(**prompt_args, height=args.resolution, width=args.resolution,
                                  num_inference_steps=protocol["steps"], guidance_scale=protocol["guidance_scale"], negative_prompt=None, eta=.3,
                                  generator=rng, latents=noise.to(device="mps", dtype=torch.float16),
                                  callback_on_step_end=step_guard, callback_on_step_end_tensor_inputs=["latents"],
                                  output_type="np").images
                if not np.isfinite(pixels).all():
                    np.savez(directory / "nonfinite-decoded.npz", pixels=pixels)
                    raise FloatingPointError("Nonfinite pixels retained without repair/retry")
                image = pipe.image_processor.numpy_to_pil(pixels)[0]
                torch.mps.synchronize()
                pipeline_seconds = time.perf_counter() - begin
                image.save(directory / "native.png")
                (directory / "generator-after-inference.bin").write_bytes(rng.get_state().numpy().tobytes())
                begin = time.perf_counter()
                image.convert("L").convert("RGB").save(directory / "display.webp", quality=90, method=4)
                encoding_seconds = time.perf_counter() - begin
                generation_seconds = noise_seconds + pipeline_seconds
                record.update(status="complete", noise_generation_seconds=noise_seconds, pipeline_seconds=pipeline_seconds,
                              generation_seconds=generation_seconds, encoding_seconds=encoding_seconds,
                              generation_plus_encoding_seconds=generation_seconds + encoding_seconds,
                              unet_forward_calls=counter["calls"], scheduler_timesteps=pipe.scheduler.timesteps.detach().cpu().tolist(),
                              noise_values_sha256=hashlib.sha256(noise.numpy().tobytes()).hexdigest(),
                              generator_after_noise_sha256=hashlib.sha256(before_state).hexdigest(),
                              actual_cfg_enabled=pipe.do_classifier_free_guidance, prompt_embedding_cache_used=protocol["cached_text"])
                require(counter["calls"] == protocol["steps"], "TCD did not execute expected UNet calls; raw images retained")
                require(pipe.do_classifier_free_guidance == (protocol["guidance_scale"] > 1), "Unexpected CFG branch state")
                record["files"] = {p.name: sha(p) for p in directory.iterdir() if p.name != "record.json"}
                write(directory / "record.json", record)
                records.append(record)
                with (output / "records.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps({"index": case["index"], "generation_seconds": generation_seconds,
                                  "encoding_seconds": encoding_seconds, "unet_calls": counter["calls"]}), flush=True)
                del pixels, image, noise
                guard(f"case-{case['index']}-after")
            except BaseException as error:
                record.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                              unet_forward_calls=counter["calls"],
                              files={p.name: sha(p) for p in directory.iterdir() if p.name != "record.json"})
                write(directory / "record.json", record)
                raise
    finally:
        handle.remove()
    require(len(records) == protocol["count"] + 1, "Expected all cases plus warmup")
    contact = Image.new("RGB", (4 * 512, math.ceil(protocol["count"] / 4) * 512))
    for i in range(protocol["count"]):
        with Image.open(output / f"{i:03}" / "display.webp") as image:
            contact.paste(image.resize((512, 512), Image.Resampling.LANCZOS), ((i % 4) * 512, (i // 4) * 512))
    contact.save(output / "contact.png")
    timings = {}
    for key in ("noise_generation_seconds", "pipeline_seconds", "generation_seconds", "encoding_seconds", "generation_plus_encoding_seconds"):
        values = sorted(r[key] for r in records if not r["warmup"])
        timings[key] = {"median": statistics.median(values), "p95_nearest_rank": values[math.ceil(.95 * len(values)) - 1], "max": max(values)}
    write(output / "result.json", {"complete": True, "model": "juggernaut-x-hyper", "revision": REVISION,
          "steps": protocol["steps"], "eta": .3, "guidance_scale": protocol["guidance_scale"], "native_resolution": args.resolution,
          "device": "Apple MPS", "load_seconds": load_seconds, "scheduler": dict(pipe.scheduler.config),
          "runtime_sha256": sha(output / "runtime.json"), "protocol_sha256": sha(output / "protocol.json"),
          "source_sha256": sha(output / "compare_juggernaut.py"), "model_provenance_sha256": PROVENANCE_SHA256,
          "contact_sha256": sha(output / "contact.png"), "warm_timings": timings,
          "all_outputs_included": True, "count": protocol["count"], "warmup_retained": True, "post_trained": protocol["post_trained"], "lora": protocol["lora"],
          "production_approved": False, "server_latency_proven": False, "records": records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, choices=(4,6), default=4)
    parser.add_argument("--guidance-scale", type=float, choices=(1.0,1.5), default=1.0)
    parser.add_argument("--resolution", type=int, choices=(512, 1024), default=1024)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-review", type=Path)
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--prompt-prefix", default="")
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--cached-text", action="store_true")
    parser.add_argument("--tiled-vae", action="store_true", help="Decode overlapping512 tiles; changes pixels and timing")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        try:
            worker(args)
        except BaseException as error:
            write(args.output / "worker-failure.json", {"type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
            raise
        return
    import psutil
    require(not args.output.exists(), "Never overwrite a comparison run")
    manifest, protocol = preflight(args)
    vm = psutil.virtual_memory()
    require(vm.available / vm.total >= .35, "At least35% available system memory required")
    args.output.mkdir(parents=True)
    shutil.copy2(Path(__file__), args.output / "compare_juggernaut.py")
    shutil.copy2(MODEL / "provenance.json", args.output / "model-provenance.json")
    if args.case_manifest:
        shutil.copy2(args.case_manifest, args.output / "case-manifest.json")
    if args.native_review:
        shutil.copy2(args.native_review, args.output / "native-review-decision.json")
    write(args.output / "protocol.json", protocol)
    started = time.monotonic()
    initial_swap = psutil.swap_memory().used
    peak = 0
    failure = None
    with (args.output / "worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--resolution", str(args.resolution),
                                    "--output", str(args.output), "--worker"],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                vm = psutil.virtual_memory()
                try:
                    peak = max(peak, psutil.Process(process.pid).memory_info().rss)
                except psutil.NoSuchProcess:
                    break
                if peak > 24 * GIB:
                    failure = "worker RSS exceeds24GiB"
                elif vm.available / vm.total < .20:
                    failure = "system available memory below20%"
                elif psutil.swap_memory().used - initial_swap > 512 * 1024 ** 2:
                    failure = "system swap grew more than512MiB"
                elif time.monotonic() - started > LIMIT_SECONDS:
                    failure = "20minute deadline"
                if failure:
                    break
                time.sleep(.5)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
            write(args.output / "supervisor.json", {"exit_code": code, "failure": failure, "seconds": time.monotonic() - started,
                  "peak_sampled_rss_bytes": peak})
    if failure or code:
        raise SystemExit(f"Comparison failed: {failure or code}; all partial artifacts retained")


if __name__ == "__main__":
    main()
