"""Bounded matched SDXL-Turbo inference; no training, selection, repairs or retries.

Run with research/.venv-flux-train/bin/python. Examples:
  evaluate_sdxl_pilot.py --arms base --split development --steps 1 --output NEW_DIR
  evaluate_sdxl_pilot.py --arms adapter --lora CHECKPOINT --split development --steps 1 --output NEW_DIR

Four-step/test runs also require --decision JSON, containing protocol_sha256,
primary_one_step_development_failed=true (four-step), and selected_steps plus
selected_adapter_sha256 (test). Keep that decision beside the reviewed development
results. The evaluator snapshots it; this is a recorded research gate, not approval.
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
REPO = HERE.parents[1]
PROTOCOL = HERE / "sdxl-pilot-protocol.json"
PROTOCOL_SHA256 = "ac8448d8facd5910d0e512486522a436a8e5360903ac8867bf63d04e7abfb22d"
MODEL = HERE / "models/sdxl-turbo"
GIB = 1024 ** 3
LIMIT_SECONDS = 1200


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, data):
    temporary = Path(path).with_name(Path(path).name + ".partial")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validated_inputs(args):
    require(sha(PROTOCOL) == PROTOCOL_SHA256, "Frozen protocol hash changed")
    protocol = json.loads(PROTOCOL.read_text())
    require(sha(MODEL / "provenance.json") == protocol["base"]["provenance_sha256"], "Base provenance changed")
    require(sha(REPO / protocol["dataset"]["manifest_path"]) == protocol["dataset"]["manifest_sha256"], "Reviewed dataset manifest changed")
    require(sha(REPO / protocol["base"]["scheduler_config_path"]) == protocol["base"]["scheduler_config_sha256"], "Scheduler config changed")
    require((args.arms == "adapter") == (args.lora is not None), "Use --lora exactly when --arms adapter")
    lora = None
    if args.lora is not None:
        path = args.lora.resolve()
        if path.is_dir():
            path = path / "pytorch_lora_weights.safetensors"
        require(path.is_file() and path.suffix == ".safetensors", "Expected an existing Diffusers LoRA safetensors file/directory")
        lora = {"source": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
    decision = None
    if args.decision is not None:
        decision = json.loads(args.decision.read_text())
        require(decision["protocol_sha256"] == PROTOCOL_SHA256, "Decision belongs to another protocol")
    if args.steps == 4:
        require(decision is not None and decision.get("primary_one_step_development_failed") is True,
                "Four-step fallback requires a recorded failure of the primary development gate")
    if args.split == "test":
        require(decision is not None and decision.get("selected_steps") == args.steps,
                "Test requires a development decision freezing checkpoint and step count")
        selected = decision.get("selected_adapter_sha256", "")
        require(len(selected) == 64 and all(c in "0123456789abcdef" for c in selected), "Test decision must identify selected adapter SHA256, also for base arms")
        if lora:
            require(selected == lora["sha256"], "Test adapter differs from the frozen selected checkpoint")
    return protocol, lora, decision


def package_versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "diffusers", "transformers", "accelerate", "peft", "safetensors", "numpy", "Pillow", "psutil")}


def timing_summary(records):
    def stats(values):
        if not values:
            return None
        ordered = sorted(values)
        return {"count": len(values), "median_seconds": statistics.median(values),
                "p95_seconds_nearest_rank": ordered[math.ceil(.95 * len(ordered)) - 1],
                "max_seconds": ordered[-1]}
    return {arm: {name: stats([r[name] for r in records if r["arm"] == arm and not r["first_measured_call"]])
                  for name in ("generation_seconds", "encoding_seconds", "generation_plus_encoding_seconds")}
            for arm in sorted({r["arm"] for r in records})}


def worker(args):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import numpy as np
    import psutil
    import torch
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    from safetensors import safe_open
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    output = args.output
    launch = json.loads((output / "launch.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    require(sha(output / "protocol.json") == PROTOCOL_SHA256, "Snapshot protocol hash changed")
    require(torch.backends.mps.is_available(), "This bounded evaluator requires MPS; it is not a server benchmark")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.mps.set_per_process_memory_fraction(.55)
    started = time.monotonic()
    initial_swap = psutil.swap_memory().used

    def guard(phase):
        torch.mps.synchronize()
        vm = psutil.virtual_memory()
        record = {"phase": phase, "elapsed_seconds": time.monotonic() - started,
                  "rss_bytes": psutil.Process().memory_info().rss,
                  "mps_driver_bytes": torch.mps.driver_allocated_memory(),
                  "mps_allocated_bytes": torch.mps.current_allocated_memory(),
                  "available_fraction": vm.available / vm.total,
                  "swap_growth_bytes": psutil.swap_memory().used - initial_swap}
        with (output / "resources.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if (record["rss_bytes"] > 24 * GIB or record["mps_driver_bytes"] > 22 * GIB or
                record["available_fraction"] < .20 or record["swap_growth_bytes"] > 512 * 1024 ** 2 or
                record["elapsed_seconds"] > LIMIT_SECONDS):
            write(output / "guard-failure.json", record)
            raise MemoryError("Inference resource/time guard failed: " + phase)
        return record

    guard("before_file_verification")
    provenance = json.loads((output / "model-provenance.json").read_text())
    require(provenance["revision"] == protocol["base"]["revision"], "Wrong model revision")
    for entry in provenance["files"]:
        path = MODEL / entry["path"]
        require(path.is_file(), f"Missing restored local file: {path}; restore weights separately")
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], f"Local model file mismatch: {path}")
    guard("base_files_verified")
    cases = protocol[args.split + "_cases"]
    arms = [a for a in protocol["inference"]["arms"] if a["adapter"] == (args.arms == "adapter")]
    require(len(cases) == (24 if args.split == "development" else 32) and len(arms) == 2, "Unexpected case/arm count")
    shape = tuple(protocol["inference"]["noise"]["shape"])
    prepared = []
    # Prepare and persist all case randomness before model load or any model output.
    for case in cases:
        directory = output / case["case_id"]
        directory.mkdir()
        rng = torch.Generator(device="cpu").manual_seed(case["seed"])
        regenerated = torch.randn(shape, generator=rng, dtype=torch.float32)
        if "reuse_initial_noise" in case:
            source = REPO / case["reuse_initial_noise"]
            require(sha(source) == case["initial_noise_file_sha256"], "Legacy latent artifact changed")
            shutil.copy2(source, directory / "noise.npz")
            with np.load(source, allow_pickle=False) as data:
                require(data.files == ["noise"], "Unexpected legacy latent keys")
                array = data["noise"].copy()
            require(array.dtype == np.float32 and array.shape == shape, "Invalid legacy latent dtype/shape")
            noise = torch.from_numpy(array)
            require(torch.equal(noise, regenerated), "Current CPU RNG cannot reconstruct legacy noise/state exactly; stop instead of guessing")
        else:
            noise = regenerated
            np.savez(directory / "noise.npz", noise=noise.numpy())
        state = rng.get_state()
        (directory / "generator-state-after-noise.bin").write_bytes(state.numpy().tobytes())
        evidence = {"case_id": case["case_id"], "seed": case["seed"], "shape": list(shape), "dtype": "float32",
                    "noise_file_sha256": sha(directory / "noise.npz"),
                    "noise_values_sha256": hashlib.sha256(noise.numpy().tobytes()).hexdigest(),
                    "generator_state_after_noise_sha256": sha(directory / "generator-state-after-noise.bin"),
                    "legacy_noise_reused": "reuse_initial_noise" in case,
                    "generator_state_reconstruction": "CPU manual_seed then exactly one float32 randn of [1,4,64,64]"}
        write(directory / "randomness.json", evidence)
        prepared.append((case, noise, state, evidence))
    guard("all_noise_saved_before_model_load")

    model_load_start = time.perf_counter()
    tokenizer = CLIPTokenizer.from_pretrained(MODEL / "tokenizer", local_files_only=True)
    tokenizer_2 = CLIPTokenizer.from_pretrained(MODEL / "tokenizer_2", local_files_only=True)
    # Sequential direct-device FP16 loads avoid an extra complete CPU/FP32 model.
    load_options = {"variant": "fp16", "torch_dtype": torch.float16, "local_files_only": True,
                    "use_safetensors": True, "low_cpu_mem_usage": True, "device_map": {"": "mps"}}
    text_encoder = CLIPTextModel.from_pretrained(MODEL / "text_encoder", **load_options).eval()
    guard("text_encoder_loaded")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(MODEL / "text_encoder_2", **load_options).eval()
    guard("text_encoder_2_loaded")
    vae = AutoencoderKL.from_pretrained(MODEL / "vae", **load_options).eval()
    guard("vae_loaded")
    unet = UNet2DConditionModel.from_pretrained(MODEL / "unet", **load_options).eval()
    guard("unet_loaded")
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(MODEL / "scheduler", local_files_only=True)
    require(scheduler.config.timestep_spacing == "trailing", "Expected pinned trailing scheduler")
    pipe = StableDiffusionXLPipeline(vae=vae, text_encoder=text_encoder, text_encoder_2=text_encoder_2,
                                    tokenizer=tokenizer, tokenizer_2=tokenizer_2, unet=unet, scheduler=scheduler,
                                    force_zeros_for_empty_prompt=True, add_watermarker=False)
    pipe.set_progress_bar_config(disable=True)
    for component in (text_encoder, text_encoder_2, vae, unet):
        component.requires_grad_(False)
    model_load_seconds = time.perf_counter() - model_load_start
    lora_load_seconds = 0.0
    if args.arms == "adapter":
        adapter_path = output / "pytorch_lora_weights.safetensors"
        require(sha(adapter_path) == launch["lora"]["sha256"], "Adapter snapshot changed")
        with safe_open(adapter_path, framework="pt", device="cpu") as checkpoint:
            keys = list(checkpoint.keys())
            require(keys and all(k.startswith("unet.") and "lora" in k for k in keys), "Expected UNet-only LoRA")
            for key in keys:
                require(bool(torch.isfinite(checkpoint.get_tensor(key)).all()), "Nonfinite saved LoRA tensor")
        begin = time.perf_counter()
        pipe.load_lora_weights(str(output), weight_name=adapter_path.name, adapter_name="priest",
                               use_safetensors=True, local_files_only=True, low_cpu_mem_usage=True)
        pipe.set_adapters(["priest"], adapter_weights=[1.0])
        pipe.unet.to(dtype=torch.float16)
        pipe.unet.requires_grad_(False)
        lora_load_seconds = time.perf_counter() - begin
        require(any("lora_" in n for n, _ in pipe.unet.named_parameters()), "LoRA load attached no matrices")
    gc.collect()
    torch.mps.synchronize()
    guard("pipeline_ready")
    from diffusers.pipelines.stable_diffusion_xl import pipeline_stable_diffusion_xl
    from diffusers.schedulers import scheduling_euler_ancestral_discrete
    runtime = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
               "machine": platform.machine(), "processor": platform.processor(), "packages": package_versions(),
               "device": "mps", "cpu_threads": 2, "mps_memory_fraction": .55, "watermark": False,
               "base_dtype": "float16", "inference_adapter_dtype": "float16" if args.arms == "adapter" else None,
               "vae_force_upcast": vae.config.force_upcast,
               "vae_precision_policy": "Original pipeline automatic VAE upcast to float32 and restore; no substituted VAE",
               "scheduler": dict(pipe.scheduler.config),
               "installed_source_sha256": {"pipeline": sha(pipeline_stable_diffusion_xl.__file__),
                                           "scheduler": sha(scheduling_euler_ancestral_discrete.__file__)},
               "model_load_seconds": model_load_seconds, "lora_load_seconds": lora_load_seconds,
               "server_latency_proven": False, "timings_include_per_step_guard_overhead": True}
    write(output / "runtime.json", runtime)
    records = []
    first_measured_call = True
    for case, noise, state, randomness in prepared:
        expected_after_state = None
        for arm in arms:
            directory = output / case["case_id"] / arm["id"]
            directory.mkdir()
            prompt = case["prompt"]
            if arm["prompt_transform"] == "prefix PR1EST_CAL. ":
                prompt = "PR1EST_CAL. " + prompt
            else:
                require(arm["prompt_transform"] == "identity", "Unknown prompt transform")
            tokenization = {}
            for name, tok in (("tokenizer", tokenizer), ("tokenizer_2", tokenizer_2)):
                ids = tok(prompt, truncation=False)["input_ids"]
                tokenization[name] = {"count": len(ids), "limit": tok.model_max_length,
                                      "would_truncate": len(ids) > tok.model_max_length, "input_ids": ids}
                require(len(ids) <= tok.model_max_length, "Frozen prompt unexpectedly exceeds tokenizer context")
            rng = torch.Generator(device="cpu")
            rng.set_state(state.clone())
            record = {"case_id": case["case_id"], "seed": case["seed"], "arm": arm["id"], "prompt": prompt,
                      "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "tokenization": tokenization,
                      "randomness": randomness, "steps": args.steps, "guidance_scale": 0.0, "resolution": 512,
                      "negative_prompt": None, "adapter_scale": 1.0 if arm["adapter"] else None,
                      "protocol_sha256": PROTOCOL_SHA256, "lora_sha256": launch["lora"]["sha256"] if launch["lora"] else None,
                      "model_provenance_sha256": protocol["base"]["provenance_sha256"],
                      "runtime_sha256": sha(output / "runtime.json"), "first_measured_call": first_measured_call,
                      "server_latency_proven": False, "status": "started"}
            write(directory / "record.json", record)
            guard(case["case_id"] + "/" + arm["id"] + "/before")

            def check_step(pipeline, index, timestep, callback_values):
                if not bool(torch.isfinite(callback_values["latents"]).all().item()):
                    raise FloatingPointError("Nonfinite denoised latents")
                guard(case["case_id"] + "/" + arm["id"] + f"/step-{index}")
                return callback_values

            try:
                torch.mps.synchronize()
                begin = time.perf_counter()
                with torch.inference_mode():
                    pixels = pipe(prompt=prompt, height=512, width=512, num_inference_steps=args.steps,
                                  guidance_scale=0.0, negative_prompt=None, num_images_per_prompt=1,
                                  generator=rng, latents=noise.to(device="mps", dtype=torch.float16),
                                  callback_on_step_end=check_step, callback_on_step_end_tensor_inputs=["latents"],
                                  output_type="np").images
                if not np.isfinite(pixels).all():
                    np.savez(directory / "nonfinite-decoded.npz", pixels=pixels)
                    raise FloatingPointError("Nonfinite decoded pixels; raw array retained without repair/retry")
                image = pipe.image_processor.numpy_to_pil(pixels)[0]
                torch.mps.synchronize()
                generation_seconds = time.perf_counter() - begin
                image.save(directory / "native.png")
                end_state = rng.get_state().numpy().tobytes()
                end_hash = hashlib.sha256(end_state).hexdigest()
                (directory / "generator-state-after-inference.bin").write_bytes(end_state)
                require(expected_after_state is None or expected_after_state == end_hash,
                        "Matched arms consumed different generator streams")
                expected_after_state = end_hash
                begin = time.perf_counter()
                image.convert("L").convert("RGB").save(directory / "display.webp", quality=90, method=4)
                encoding_seconds = time.perf_counter() - begin
                record.update(status="complete", generation_seconds=generation_seconds, encoding_seconds=encoding_seconds,
                              generation_plus_encoding_seconds=generation_seconds + encoding_seconds,
                              generator_state_after_inference_sha256=end_hash,
                              files={p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                                     for p in directory.iterdir() if p.name != "record.json"})
                write(directory / "record.json", record)
                records.append(record)
                with (output / "records.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps({"case": case["case_id"], "arm": arm["id"], "generation_seconds": generation_seconds,
                                  "encoding_seconds": encoding_seconds}), flush=True)
                first_measured_call = False
                del pixels, image
                guard(case["case_id"] + "/" + arm["id"] + "/after")
            except BaseException as error:
                record.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
                record["files"] = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                                   for p in directory.iterdir() if p.name != "record.json"}
                write(directory / "record.json", record)
                raise
    require(len(records) == len(cases) * 2, "Not every scheduled output completed")
    write(output / "result.json", {"complete": True, "protocol_sha256": PROTOCOL_SHA256, "arms": args.arms,
          "split": args.split, "steps": args.steps, "native_resolution": 512, "count": len(records),
          "all_outputs_retained": True, "production_approved": False, "server_latency_proven": False,
          "post_trained": args.arms == "adapter", "first_measured_call": records[0],
          "warm_timings_exclude_first_measured_call": timing_summary(records),
          "runtime_sha256": sha(output / "runtime.json"), "launch_sha256": sha(output / "launch.json"),
          "records_jsonl_sha256": sha(output / "records.jsonl"), "records": records})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--arms", choices=("base", "adapter"), required=True)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--steps", type=int, choices=(1, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision", type=Path)
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
    protocol, lora, decision = validated_inputs(args)
    require(not args.output.exists(), "Refusing to overwrite an existing evaluation directory")
    vm = psutil.virtual_memory()
    require(vm.available / vm.total >= .35, "At least35% available system memory required before starting")
    args.output.mkdir(parents=True)
    shutil.copy2(PROTOCOL, args.output / "protocol.json")
    shutil.copy2(MODEL / "provenance.json", args.output / "model-provenance.json")
    shutil.copy2(Path(__file__), args.output / "evaluate-source.py")
    if lora:
        shutil.copy2(lora["source"], args.output / "pytorch_lora_weights.safetensors")
        require(sha(args.output / "pytorch_lora_weights.safetensors") == lora["sha256"], "Adapter changed during snapshot")
    if decision:
        shutil.copy2(args.decision, args.output / "decision.json")
    write(args.output / "launch.json", {"argv": sys.argv, "protocol_sha256": PROTOCOL_SHA256, "source_sha256": sha(Path(__file__)),
          "lora": lora, "arms": args.arms, "split": args.split, "steps": args.steps,
          "decision_sha256": sha(args.output / "decision.json") if decision else None,
          "limits": {"seconds": LIMIT_SECONDS, "rss_gib": 24, "mps_driver_gib": 22, "available_fraction": .20, "swap_growth_mib": 512},
          "production_approved": False, "server_latency_proven": False})
    command = [sys.executable, str(Path(__file__).resolve()), "--arms", args.arms, "--split", args.split,
               "--steps", str(args.steps), "--output", str(args.output), "--worker"]
    started = time.monotonic()
    initial_swap = psutil.swap_memory().used
    failure = None
    peak = 0
    with (args.output / "worker.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
                    failure = "20minute evaluation deadline"
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
            write(args.output / "supervisor.json", {"exit_code": code, "failure": failure,
                  "seconds": time.monotonic() - started, "peak_sampled_rss_bytes": peak})
    if failure or code:
        raise SystemExit(f"Evaluation failed: {failure or code}; partial artifacts retained in {args.output}")


if __name__ == "__main__":
    main()
