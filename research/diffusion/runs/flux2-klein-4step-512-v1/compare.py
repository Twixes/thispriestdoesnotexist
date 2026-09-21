"""Bounded, uncurated local comparison of pinned one/four-step pretrained bases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PROMPTS = [
    "A candid photographic head and shoulders portrait of a handsome 35 year old Italian Catholic priest, short dark hair, bareheaded, black clerical shirt with a small white Roman collar tab clearly visible below his chin, natural skin texture, gentle smile, softly blurred stone courtyard, black and white portrait photography.",
    "A photographic head and shoulders portrait of a handsome 45 year old Black Catholic priest, short curly hair, bareheaded, black clerical shirt with a small white Roman collar tab clearly visible below his chin, natural skin texture, relaxed expression, softly blurred neutral background, black and white portrait photography.",
    "A photographic head and shoulders portrait of a handsome 40 year old East Asian Catholic priest, short neat hair, bareheaded, black clerical shirt with a small white Roman collar tab clearly visible below his chin, natural skin texture, warm smile, softly blurred neutral background, black and white portrait photography.",
    "A candid photographic head and shoulders portrait of a handsome 50 year old Catholic priest, salt and pepper hair and short beard, bareheaded, black clerical shirt with a small white Roman collar tab clearly visible below his chin, natural skin texture, thoughtful expression, softly blurred stone courtyard, black and white portrait photography.",
]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def worker(args):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoPipelineForText2Image
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.mps.set_per_process_memory_fraction(0.55)
    model = ROOT / "models" / args.model
    from restore import restore
    provenance = restore(model)
    begin = time.perf_counter()
    if args.model == "sdxl-lightning-2step":
        from load_lightning import load_pipeline
        pipe = load_pipeline(device="mps", dtype=torch.float16, model_dir=model)
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=torch.float16, variant="fp16", local_files_only=True, use_safetensors=True)
        pipe.to("mps")
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - begin
    records = []
    for index in range(-1, 8):
        seed = 202609214000 + index
        prompt = PROMPTS[max(0, index) // 2]
        rng = torch.Generator(device="cpu").manual_seed(seed)
        latent = torch.randn((1, 4, args.resolution // 8, args.resolution // 8), generator=rng)
        directory = args.output / ("warmup" if index == -1 else f"{index:03}")
        directory.mkdir()
        np.savez(directory / "noise.npz", noise=latent.numpy())
        def check_resources(pipeline, step, timestep, callback_kwargs):
            if torch.mps.driver_allocated_memory() > 22 * 1024 ** 3:
                raise MemoryError("MPS driver allocation exceeds 22GiB")
            return callback_kwargs
        torch.mps.synchronize()
        start = time.perf_counter()
        image = pipe(prompt=prompt, height=args.resolution, width=args.resolution, num_inference_steps=args.steps, guidance_scale=0.0,
                     generator=rng, latents=latent.to(device="mps", dtype=torch.float16),
                     callback_on_step_end=check_resources).images[0]
        torch.mps.synchronize()
        generation_seconds = time.perf_counter() - start
        image.save(directory / "native.png")
        encoding_start = time.perf_counter()
        gray = image.convert("L").convert("RGB")
        gray.save(directory / "display.webp", quality=90, method=4)
        encoding_seconds = time.perf_counter() - encoding_start
        record = {"index": index, "seed": seed, "prompt": prompt, "warmup": index == -1,
                  "generation_seconds": generation_seconds, "encoding_seconds": encoding_seconds,
                  "generation_plus_encoding_seconds": generation_seconds + encoding_seconds,
                  "mps_allocated_bytes": torch.mps.current_allocated_memory(),
                  "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
                  "files": {p.name: sha(p) for p in directory.iterdir()}}
        write(directory / "record.json", record)
        records.append(record)
        print(json.dumps({"index": index, "generation_seconds": generation_seconds, "encoding_seconds": encoding_seconds}), flush=True)
    contact = Image.new("RGB", (4 * 512, 2 * 512))
    for i in range(8):
        with Image.open(args.output / f"{i:03}/display.webp") as im:
            contact.paste(im.resize((512, 512), Image.Resampling.LANCZOS), ((i % 4) * 512, (i // 4) * 512))
    contact.save(args.output / "contact.png")
    write(args.output / "result.json", {"complete": True, "model": args.model, "revision": provenance["revision"],
          "steps": args.steps, "native_resolution": args.resolution, "load_seconds": load_seconds, "device": "Apple M5 Pro MPS",
          "scheduler": dict(pipe.scheduler.config),
          "server_latency_proven": False, "post_trained": False, "all_outputs_included": True, "records": records})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sd-turbo", "sdxl-turbo", "sdxl-lightning-2step"], required=True)
    parser.add_argument("--steps", type=int, choices=[1, 2, 4], required=True)
    parser.add_argument("--resolution", type=int, choices=[512, 1024], default=512)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if (args.model == "sdxl-lightning-2step") != (args.steps == 2):
        raise ValueError("Lightning requires exactly two steps; Turbo comparison uses one or four")
    args.output = args.output.resolve()
    if args.worker:
        return worker(args)
    import psutil
    if args.output.exists():
        raise ValueError("Output already exists; never overwrite runs")
    if psutil.virtual_memory().available / psutil.virtual_memory().total < .35:
        raise MemoryError("At least 35% available memory required")
    args.output.mkdir(parents=True)
    (args.output / "compare-source.py").write_bytes(Path(__file__).read_bytes())
    write(args.output / "protocol.json", {"model": args.model, "steps": args.steps, "count": 8, "prompts": PROMPTS,
          "seed_base": 202609214000, "warmup_seed": 202609213999, "guidance_scale": 0, "resolution": args.resolution,
          "all_outputs_retained": True, "purpose": "Matched pretrained base prior comparison, no priest post-training yet",
          "source_sha256": sha(Path(__file__)), "model_provenance_sha256": sha(ROOT / "models" / args.model / "provenance.json")})
    start = time.monotonic()
    peak = 0
    failure = None
    with (args.output / "worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__)), "--model", args.model, "--steps", str(args.steps), "--resolution", str(args.resolution), "--output", str(args.output), "--worker"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                try:
                    peak = max(peak, psutil.Process(process.pid).memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                if peak > 24 * 1024 ** 3:
                    failure = "worker RSS exceeds 24GiB"
                elif psutil.virtual_memory().available / psutil.virtual_memory().total < .15:
                    failure = "available system memory below15%"
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
            exit_code = process.wait()
            write(args.output / "supervisor.json", {"exit_code": exit_code, "failure": failure, "seconds": time.monotonic()-start, "peak_rss_bytes": peak})
    if failure or exit_code:
        raise SystemExit(f"Comparison failed: {failure or exit_code}")


if __name__ == "__main__":
    main()
