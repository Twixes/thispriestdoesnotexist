"""Precompute upstream training inputs with text encoder and VAE loaded sequentially."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
from compare import ROOT, sha, write


def worker(output):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import torch
    from diffusers import Flux2KleinPipeline, AutoencoderKLFlux2
    from datasets import load_dataset, Image as DatasetImage
    from safetensors.torch import save_file
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.mps.set_per_process_memory_fraction(.55)
    def guard(phase):
        torch.mps.synchronize()
        vm = psutil.virtual_memory()
        record = {"phase": phase, "rss_bytes": psutil.Process().memory_info().rss,
            "mps_driver_bytes": torch.mps.driver_allocated_memory(), "available_fraction": vm.available/vm.total}
        with (output / "worker-resources.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if record["mps_driver_bytes"] > 22*1024**3 or record["rss_bytes"] > 24*1024**3 or record["available_fraction"] < .25:
            raise MemoryError(record)
    dataset_root = ROOT.parent / "data/flux-priest-domain-v1"
    model = ROOT / "models/flux2-klein-base-4b"
    spec = importlib.util.spec_from_file_location("upstream_flux_trainer", ROOT / "vendor/flux-train/upstream.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    upstream.args = upstream.parse_args(["--pretrained_model_name_or_path", str(model),
        "--dataset_name", str(dataset_root / "train"), "--image_column", "image", "--caption_column", "text",
        "--instance_prompt", "PR1EST_CAL. A portrait of an adult male priest.", "--resolution", "512", "--center_crop"])
    dataset = upstream.DreamBoothDataset(instance_data_root=None,
        instance_prompt=upstream.args.instance_prompt, class_prompt=None, size=512, repeats=1, center_crop=True)
    raw = load_dataset(str(dataset_root / "train"), split="train").cast_column("image", DatasetImage(decode=False))
    if len(dataset) != 20 or len(raw) != 20:
        raise ValueError("Expected exactly20 reviewed training images")
    encoder = Flux2KleinPipeline.from_pretrained(model, transformer=None, vae=None,
        torch_dtype=torch.float16, local_files_only=True).to("mps")
    guard("encoder_loaded")
    encoded = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            caption = dataset.custom_instance_prompts[index]
            if raw[index]["text"] != caption:
                raise ValueError("Caption order differs")
            embeds, ids = encoder.encode_prompt(caption, device="mps", max_sequence_length=512,
                text_encoder_out_layers=(9,18,27))
            if not torch.isfinite(embeds).all().item() or not torch.isfinite(ids).all().item():
                raise FloatingPointError(f"Nonfinite encoder output at{index}")
            encoded.append({"prompt_embeds": embeds.cpu().contiguous(), "text_ids": ids.cpu().contiguous()})
            guard(f"encoded_{index}")
            print(f"Encoded prompt {index+1}/20", flush=True)
    del encoder, embeds, ids
    gc.collect()
    torch.mps.empty_cache()
    vae = AutoencoderKLFlux2.from_pretrained(model, subfolder="vae", torch_dtype=torch.float32,
        local_files_only=True).to(device="mps", dtype=torch.float16)
    guard("vae_loaded_after_encoder_released")
    entries = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            pixels = dataset[index]["instance_images"].unsqueeze(0).contiguous().float()
            latents = vae.encode(pixels.to(device="mps", dtype=torch.float16)).latent_dist.mode()
            if not torch.isfinite(latents).all().item():
                raise FloatingPointError(f"Nonfinite VAE output at{index}")
            tensors = {**encoded[index], "latents": latents.cpu().contiguous()}
            filename = Path(raw[index]["image"]["path"]).name
            path = output / (Path(filename).stem + ".safetensors")
            save_file(tensors, path)
            entries.append({"file_name": filename, "caption": dataset.custom_instance_prompts[index],
                "path": path.name, "sha256": sha(path), "image_sha256": sha(dataset_root / "train" / filename),
                "tensor_shapes": {k:list(v.shape) for k,v in tensors.items()},
                "tensor_dtypes": {k:str(v.dtype) for k,v in tensors.items()}})
            guard(f"latent_{index}")
            print(f"Encoded image {index+1}/20", flush=True)
    write(output / "manifest.json", {"complete": True, "entries": entries,
        "dataset_manifest_sha256": sha(dataset_root / "manifest.json"), "resolution":512,
        "max_sequence_length":512, "text_encoder_out_layers":[9,18,27], "dtype":"float16",
        "latents":"Raw VAE encoder mode before patchification and BN normalization",
        "image_transform":"Unmodified pinned upstream DreamBoothDataset with center_crop and no random_flip",
        "upstream_source_sha256":sha(ROOT / "vendor/flux-train/upstream.py"),
        "model_provenance_sha256":sha(model / "provenance.json"), "source_sha256":sha(Path(__file__))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        return worker(output)
    if output.exists():
        raise ValueError("Never overwrite a cache experiment")
    if psutil.virtual_memory().available/psutil.virtual_memory().total < .35:
        raise MemoryError("At least35% available memory required")
    output.mkdir(parents=True)
    (output / "cache-source.py").write_bytes(Path(__file__).read_bytes())
    start = time.monotonic()
    peak = 0
    failure = None
    with (output / "worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__)), "--output", str(output), "--worker"],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                vm = psutil.virtual_memory()
                try:
                    peak = max(peak, psutil.Process(process.pid).memory_info().rss)
                except psutil.NoSuchProcess:
                    break
                if peak > 24*1024**3:
                    failure = "RSS above24GiB"
                elif vm.available/vm.total < .25:
                    failure = "Available memory below25%"
                elif time.monotonic()-start > 1200:
                    failure = "20minute timeout"
                if failure:
                    os.killpg(process.pid,signal.SIGTERM)
                    break
                time.sleep(.5)
        finally:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
            code = process.wait()
            write(output / "supervisor.json", {"exit_code":code,"failure":failure,
                "seconds":time.monotonic()-start,"peak_sampled_rss_bytes":peak})
    if failure or code:
        raise SystemExit(f"Cache preparation failed: {failure or code}")


if __name__ == "__main__":
    main()
