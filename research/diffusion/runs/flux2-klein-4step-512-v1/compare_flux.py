"""Unfiltered FLUX.2-klein comparison with reusable fixed-prompt embeddings."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from compare import PROMPTS, ROOT, sha, write


def worker(args):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import save_file
    from diffusers import Flux2KleinPipeline

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.mps.set_per_process_memory_fraction(.55)
    def check_driver_memory():
        if torch.mps.driver_allocated_memory() > 22 * 1024 ** 3:
            raise MemoryError("MPS driver allocation exceeds22GiB")
    model = ROOT / "models/flux2-klein-4b"
    provenance = json.loads((model / "provenance.json").read_text())
    start = time.perf_counter()
    # Encoding is deployment initialization for a fixed prompt bank, never an image cache.
    encoder = Flux2KleinPipeline.from_pretrained(model, transformer=None, vae=None,
        torch_dtype=torch.bfloat16, local_files_only=True, use_safetensors=True).to("mps")
    torch.mps.synchronize()
    check_driver_memory()
    encoder_load_seconds = time.perf_counter() - start
    embeddings = []
    encoding_times = []
    with torch.inference_mode():
        for i, prompt in enumerate(PROMPTS):
            torch.mps.synchronize()
            check_driver_memory()
            start = time.perf_counter()
            embeds, _ = encoder.encode_prompt(prompt, device="mps")
            torch.mps.synchronize()
            encoding_times.append(time.perf_counter() - start)
            embeddings.append(embeds.cpu().contiguous())
            save_file({"prompt_embeds": embeddings[-1]}, args.output / f"prompt-{i}.safetensors")
    del encoder, embeds
    gc.collect()
    torch.mps.empty_cache()
    start = time.perf_counter()
    pipe = Flux2KleinPipeline.from_pretrained(model, text_encoder=None, tokenizer=None,
        torch_dtype=torch.bfloat16, local_files_only=True, use_safetensors=True).to("mps")
    embeddings = [e.to("mps") for e in embeddings]
    torch.mps.synchronize()
    check_driver_memory()
    load_seconds = time.perf_counter() - start
    if not pipe.config.is_distilled:
        raise ValueError("This recipe requires the official distilled checkpoint")
    write(args.output / "initialization.json", {"encoder_load_seconds": encoder_load_seconds,
        "prompt_encoding_seconds": encoding_times, "generator_load_seconds": load_seconds,
        "cached_prompts": PROMPTS, "images_cached": False, "prompt_embeddings_cached": True})
    records = []
    for index in range(-1, 8):
        seed = 202609214000 + index
        prompt_index = max(0, index) // 2
        directory = args.output / ("warmup" if index == -1 else f"{index:03}")
        directory.mkdir()
        rng = torch.Generator(device="cpu").manual_seed(seed)
        latent_channels = pipe.transformer.config.in_channels
        side = args.resolution // (pipe.vae_scale_factor * 2)
        noise_start = time.perf_counter()
        noise = torch.randn((1, latent_channels, side, side), generator=rng, dtype=torch.float32)
        noise_seconds = time.perf_counter() - noise_start
        np.savez(directory / "noise.npz", noise=noise.numpy())
        def check_resources(pipeline, step, timestep, values):
            check_driver_memory()
            return values
        torch.mps.synchronize()
        start = time.perf_counter()
        image = pipe(prompt_embeds=embeddings[prompt_index], height=args.resolution, width=args.resolution,
            num_inference_steps=4, guidance_scale=1.0,
            latents=noise.to(device="mps", dtype=torch.bfloat16), generator=rng,
            callback_on_step_end=check_resources).images[0]
        torch.mps.synchronize()
        generation_seconds = time.perf_counter() - start + noise_seconds
        check_driver_memory()
        image.save(directory / "native.png")
        start = time.perf_counter()
        image.convert("L").convert("RGB").save(directory / "display.webp", quality=90, method=4)
        encoding_seconds = time.perf_counter() - start
        record = {"index": index, "seed": seed, "prompt": PROMPTS[prompt_index], "warmup": index == -1,
            "generation_seconds": generation_seconds, "encoding_seconds": encoding_seconds,
            "noise_generation_seconds": noise_seconds, "noise_included_in_generation_seconds": True,
            "generation_plus_encoding_seconds": generation_seconds + encoding_seconds,
            "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
            "prompt_embedding_cache_used": True, "files": {p.name: sha(p) for p in directory.iterdir()}}
        write(directory / "record.json", record)
        records.append(record)
        print(json.dumps({"index": index, "generation_seconds": generation_seconds,
            "encoding_seconds": encoding_seconds}), flush=True)
    contact = Image.new("RGB", (4 * 512, 2 * 512))
    for i in range(8):
        with Image.open(args.output / f"{i:03}/display.webp") as im:
            contact.paste(im.resize((512, 512), Image.Resampling.LANCZOS), ((i % 4) * 512, (i // 4) * 512))
    contact.save(args.output / "contact.png")
    write(args.output / "result.json", {"complete": True, "model": "flux2-klein-4b",
        "revision": provenance["revision"], "steps": 4, "native_resolution": args.resolution,
        "device": "Apple M5 Pro MPS", "dtype": "bfloat16", "scheduler": dict(pipe.scheduler.config),
        "server_latency_proven": False, "post_trained": False, "all_outputs_included": True,
        "prompt_embeddings_cached": True, "load_seconds": load_seconds, "records": records})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, choices=[512, 1024], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        return worker(args)
    import psutil
    if args.output.exists():
        raise ValueError("Never overwrite an experiment")
    if psutil.virtual_memory().available / psutil.virtual_memory().total < .35:
        raise MemoryError("At least35% available memory required")
    from reassemble_flux2_klein import verified_model_dir
    verified_model_dir()
    args.output.mkdir(parents=True)
    for source in [Path(__file__), ROOT / "compare.py"]:
        (args.output / source.name).write_bytes(source.read_bytes())
    write(args.output / "protocol.json", {"model": "flux2-klein-4b", "steps": 4, "count": 8,
        "prompts": PROMPTS, "seed_base": 202609214000, "warmup_seed": 202609213999,
        "guidance_scale": 1.0, "resolution": args.resolution, "all_outputs_retained": True,
        "prompt_embeddings_cached": True, "images_cached": False,
        "timing_scope": "Fresh noise to PIL image plus monochrome WebP; excludes fixed-prompt initialization, load, warmup, HTTP, queueing and any future quality gates",
        "source_sha256": sha(Path(__file__)),
        "provenance_sha256": sha(ROOT / "models/flux2-klein-4b/provenance.json")})
    start = time.monotonic()
    peak = 0
    failure = None
    with (args.output / "worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__)), "--resolution", str(args.resolution),
            "--output", str(args.output), "--worker"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                try:
                    peak = max(peak, psutil.Process(process.pid).memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                if peak > 24 * 1024 ** 3:
                    failure = "worker RSS exceeds24GiB"
                elif psutil.virtual_memory().available / psutil.virtual_memory().total < .15:
                    failure = "available memory below15%"
                elif time.monotonic() - start > 1200:
                    failure = "20minute timeout"
                if failure:
                    os.killpg(process.pid, signal.SIGTERM)
                    break
                time.sleep(.5)
        finally:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
            write(args.output / "supervisor.json", {"exit_code": code, "failure": failure,
                "seconds": time.monotonic()-start, "peak_rss_bytes": peak})
    if failure or code:
        raise SystemExit(f"Comparison failed: {failure or code}")


if __name__ == "__main__":
    main()
