"""Bounded original-RealVis VAE CPU/MPS reconstruction diagnostic, no UNet."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models/realvisxl-v5-lightning"
SOURCE = HERE.parent / "data/flux-priest-domain-v1/train/141.png"
SOURCE_SHA = "751d3492a88e8b2b348ad56012d2de89001cf8256b6bd82e9fe0aeb7b29ed22b"
WEIGHT_SHA = "6353737672c94b96174cb590f711eac6edf2fcce5b6e91aa9d73c5adc589ee48"
GIB = 1024 ** 3


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temp.replace(path)


def worker(output):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import numpy as np
    import psutil
    import torch
    from diffusers import AutoencoderKL
    from diffusers.models.autoencoders import autoencoder_kl
    from PIL import Image
    assert sha(Path(__file__)) == json.loads((output / "protocol.json").read_text())["script_sha256"]
    assert torch.backends.mps.is_available()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.mps.set_per_process_memory_fraction(.55)
    initial_swap = psutil.swap_memory().used
    started = time.monotonic()
    timings = {}

    def guard(phase):
        torch.mps.synchronize()
        vm = psutil.virtual_memory()
        row = {"phase": phase, "rss_bytes": psutil.Process().memory_info().rss,
               "mps_driver_bytes": torch.mps.driver_allocated_memory(),
               "available_fraction": vm.available / vm.total,
               "swap_growth_bytes": psutil.swap_memory().used - initial_swap,
               "elapsed_seconds": time.monotonic() - started}
        with (output / "resources.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        assert row["rss_bytes"] <= 24 * GIB and row["mps_driver_bytes"] <= 22 * GIB
        assert row["available_fraction"] >= .20 and row["swap_growth_bytes"] <= 512 * 1024 ** 2
        assert row["elapsed_seconds"] <= 300

    def operation(name, function):
        guard(name + "-before")
        write(output / "phase.json", {"phase": name, "started_unix": time.time(), "active": True})
        begin = time.perf_counter()
        with torch.inference_mode():
            result = function()
        torch.mps.synchronize()
        timings[name] = time.perf_counter() - begin
        write(output / "phase.json", {"phase": name, "started_unix": time.time(), "active": False,
                                     "seconds": timings[name]})
        assert timings[name] <= 60, name + " exceeded60seconds"
        guard(name + "-after")
        return result

    vae = operation("load_vae_cpu_fp32", lambda: AutoencoderKL.from_pretrained(
        MODEL / "vae", variant="fp16", torch_dtype=torch.float32, local_files_only=True,
        low_cpu_mem_usage=True, use_safetensors=True).eval().requires_grad_(False))
    vae.enable_tiling()
    vae.tile_sample_min_size = 512
    vae.tile_latent_min_size = 64
    vae.tile_overlap_factor = .25
    write(output / "runtime.json", {"packages": {n: importlib.metadata.version(n) for n in
        ("torch", "diffusers", "transformers", "safetensors", "numpy", "Pillow", "psutil")},
        "python": sys.version, "cpu_threads": 2, "dtype": "float32",
        "vae_config": dict(vae.config), "vae_source_sha256": sha(autoencoder_kl.__file__),
        "tiling": {"sample": 512, "latent": 64, "overlap": .25}})
    source = np.asarray(Image.open(SOURCE).convert("RGB"), dtype=np.float32) / 255
    assert source.shape == (1024, 1024, 3)
    tensor = torch.from_numpy(source.copy()).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    latent = operation("encode_cpu_fp32_posterior_mean", lambda: vae.encode(tensor).latent_dist.mean)
    assert torch.isfinite(latent).all()
    np.save(output / "latent-unscaled-fp32.npy", latent.numpy(), allow_pickle=False)
    decoded_cpu = operation("decode_cpu_fp32", lambda: vae.decode(latent).sample).numpy().copy()
    np.save(output / "decoded-cpu-raw-fp32.npy", decoded_cpu, allow_pickle=False)
    cpu_pixels = np.clip(decoded_cpu[0].transpose(1, 2, 0) / 2 + .5, 0, 1)
    Image.fromarray((cpu_pixels * 255).round().astype(np.uint8)).save(output / "decoded-cpu.png")
    vae = operation("transfer_vae_to_mps_fp32", lambda: vae.to("mps"))
    latent_mps = operation("transfer_matched_latent_to_mps", lambda: latent.to("mps"))
    assert torch.equal(latent, latent_mps.cpu()), "Latent values changed in transfer"
    decoded_mps = operation("decode_mps_fp32", lambda: vae.decode(latent_mps).sample).cpu().numpy().copy()
    np.save(output / "decoded-mps-raw-fp32.npy", decoded_mps, allow_pickle=False)
    mps_pixels = np.clip(decoded_mps[0].transpose(1, 2, 0) / 2 + .5, 0, 1)
    Image.fromarray((mps_pixels * 255).round().astype(np.uint8)).save(output / "decoded-mps.png")
    assert np.isfinite(decoded_cpu).all() and np.isfinite(decoded_mps).all()

    def stats(a, b):
        difference = a.astype(np.float64) - b.astype(np.float64)
        absolute = np.abs(difference)
        mse = float(np.mean(difference ** 2))
        return {"mae": float(absolute.mean()), "rmse": mse ** .5, "max_absolute": float(absolute.max()),
                "p99_absolute": float(np.quantile(absolute, .99)),
                "psnr_db_for_unit_range": float(-10 * np.log10(mse)) if mse else None}

    result = {"complete": True, "timings_seconds": timings,
        "raw_decoder_mps_vs_cpu_range_minus1_plus1": stats(decoded_mps, decoded_cpu),
        "display_pixel_mps_vs_cpu_range0_1": stats(mps_pixels, cpu_pixels),
        "cpu_reconstruction_vs_source_range0_1": stats(cpu_pixels, source),
        "mps_reconstruction_vs_source_range0_1": stats(mps_pixels, source),
        "fraction_rounded_rgb_channels_differ": float(np.mean(np.round(cpu_pixels * 255) != np.round(mps_pixels * 255))),
        "latent_scaling": "Raw posterior mean, no scaling factor applied; same tensor decoded on both backends",
        "scope": "One teacher reconstruction; isolates decoder backend numerics, not UNet or generated-latent behavior",
        "files": {p.name: sha(p) for p in output.iterdir() if p.is_file() and p.suffix in (".png", ".npy")}}
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        try:
            worker(output)
        except BaseException as error:
            write(output / "failure.json", {"type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
            raise
        return
    import psutil
    assert not output.exists(), "Never overwrite a probe"
    vm = psutil.virtual_memory()
    assert vm.available / vm.total >= .35, "Need35% available memory"
    assert sha(SOURCE) == SOURCE_SHA
    weights = MODEL / "vae/diffusion_pytorch_model.fp16.safetensors"
    assert sha(weights) == WEIGHT_SHA
    output.mkdir(parents=True)
    shutil.copy2(SOURCE, output / "source.png")
    shutil.copy2(__file__, output / Path(__file__).name)
    write(output / "protocol.json", {"source_path": str(SOURCE), "source_sha256": SOURCE_SHA,
        "weight_path": str(weights), "weight_sha256": WEIGHT_SHA,
        "model_revision": "f4454158cedaab9f0688c199561d6c92525f3a85",
        "config_sha256": sha(MODEL / "vae/config.json"), "script_sha256": sha(__file__),
        "source_selection": "First accepted train image141; viewed native before diagnostic",
        "dtype": "float32 on CPU and MPS from same publisherFP16 weights",
        "posterior": "CPU encoder mean, deterministic, no sampling",
        "tiling": {"sample": 512, "latent": 64, "overlap": .25},
        "total_deadline_seconds": 300, "operation_deadline_seconds": 60, "production_approved": False})
    initial_swap = psutil.swap_memory().used
    started = time.monotonic()
    failure = None
    with (output / "worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(output)],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                vm = psutil.virtual_memory()
                try:
                    rss = psutil.Process(process.pid).memory_info().rss
                except psutil.NoSuchProcess:
                    break
                if rss > 24 * GIB or vm.available / vm.total < .20 or psutil.swap_memory().used - initial_swap > 512 * 1024 ** 2:
                    failure = "Memory guard"
                elif time.monotonic() - started > 300:
                    failure = "Five-minute total deadline"
                phase_path = output / "phase.json"
                if phase_path.exists():
                    phase = json.loads(phase_path.read_text())
                    if phase["active"] and time.time() - phase["started_unix"] > 60:
                        failure = "60second operation deadline: " + phase["phase"]
                if failure:
                    break
                time.sleep(.25)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
            write(output / "supervisor.json", {"exit_code": code, "failure": failure, "seconds": time.monotonic() - started})
    if failure or code:
        raise SystemExit(f"Diagnostic stopped: {failure or code}; partial artifacts retained")


if __name__ == "__main__":
    main()
