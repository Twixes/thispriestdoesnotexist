"""Run one bounded20-update local adapter compatibility experiment, not a quality claim."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
from compare import ROOT, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Never overwrite an experiment")
    if psutil.virtual_memory().available / psutil.virtual_memory().total < .35:
        raise MemoryError("At least35% available system memory required")
    repo = ROOT.parents[1]
    dataset = repo / "research/data/flux-priest-domain-v1/train"
    model = ROOT / "models/flux2-klein-base-4b"
    trainer = ROOT / "vendor/flux-train/run_training.py"
    manifest = repo / "research/data/flux-priest-domain-v1/manifest.json"
    # These inputs must already have passed their separate assembly/review steps.
    for path in [trainer, manifest, dataset / "metadata.jsonl", model / "provenance.json"]:
        if not path.is_file():
            raise FileNotFoundError(path)
    command = [sys.executable, str(trainer),
        "--pretrained_model_name_or_path", str(model),
        "--dataset_name", str(dataset), "--image_column", "image", "--caption_column", "text",
        "--instance_prompt", "PR1EST_CAL. A portrait of an adult male priest.",
        "--output_dir", str(output), "--resolution", "512", "--center_crop", "--repeats", "1",
        "--train_batch_size", "1", "--gradient_accumulation_steps", "1", "--gradient_checkpointing",
        "--cache_latents", "--offload", "--mixed_precision", "fp16", "--optimizer", "AdamW",
        "--rank", "16", "--lora_alpha", "16", "--learning_rate", "5e-5", "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0", "--max_train_steps", "20", "--checkpointing_steps", "10",
        "--dataloader_num_workers", "0", "--seed", "2026092181", "--skip_final_inference",
        "--report_to", "tensorboard"]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        PYTHONUNBUFFERED="1", PYTORCH_ENABLE_MPS_FALLBACK="0")
    output.mkdir(parents=True)
    for path in [Path(__file__), trainer]:
        (output / path.name).write_bytes(path.read_bytes())
    write(output / "launch.json", {"command": command, "purpose": "20-update MPS compatibility test",
        "production_approved": False, "automatic_retry": False, "maximum_updates": 20,
        "maximum_seconds": 1800, "maximum_sampled_rss_bytes": 24 * 1024**3,
        "minimum_available_memory_fraction": .25, "source_sha256": sha(trainer),
        "dataset_manifest_sha256": sha(manifest), "model_provenance_sha256": sha(model / "provenance.json"),
        "environment_lock_sha256": sha(ROOT / "flux-train-requirements-lock.txt")})
    start = time.monotonic()
    peak = 0
    failure = None
    with (output / "worker.log").open("w") as log, (output / "resources.jsonl").open("w") as monitor:
        process = subprocess.Popen(command, cwd=repo, env=environment, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                vm = psutil.virtual_memory()
                try:
                    rss = psutil.Process(process.pid).memory_info().rss
                except psutil.NoSuchProcess:
                    break
                peak = max(peak, rss)
                elapsed = time.monotonic() - start
                monitor.write(json.dumps({"elapsed_seconds": elapsed, "rss_bytes": rss,
                    "available_system_bytes": vm.available, "system_total_bytes": vm.total}) + "\n")
                monitor.flush()
                if rss > 24 * 1024**3:
                    failure = "WorkerRSS exceeds24GiB"
                elif vm.available / vm.total < .25:
                    failure = "Available system memory below25%"
                elif elapsed > 1800:
                    failure = "30minute compatibility-test deadline"
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
            write(output / "supervisor.json", {"exit_code": code, "failure": failure,
                "seconds": time.monotonic() - start, "peak_sampled_rss_bytes": peak,
                "quality_not_established": True, "production_approved": False})
    if failure or code:
        raise SystemExit(f"Smoke failed: {failure or code}")


if __name__ == "__main__":
    main()
