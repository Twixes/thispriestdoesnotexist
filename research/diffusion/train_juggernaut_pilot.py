"""Supervise a fresh exact-base Juggernaut20/100 pilot; no Turbo adapter or resume."""
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
from juggernaut_pilot import PROTOCOL, MODEL, DATASET, preflight, validate_continuation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, choices=(20, 100), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--continue-decision", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="Validate artifacts and print exact command without starting compute")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Never overwrite an experiment")
    if not args.prepare_only and psutil.virtual_memory().available / psutil.virtual_memory().total < .35:
        raise MemoryError("At least35% available system memory required before launch")
    repo = ROOT.parents[1]
    protocol_path = PROTOCOL
    protocol, components = preflight()
    decision = validate_continuation(args.continue_decision, protocol) if args.updates == 100 else None
    recipe = protocol["training"]
    if not 0 <= recipe["seed"] < 2**32:
        raise ValueError("Upstream Accelerate/NumPy requires a32bit training seed")
    model = MODEL
    dataset = repo / protocol["dataset"]["training_directory"]
    manifest = repo / protocol["dataset"]["manifest_path"]
    cache = args.cache.resolve()
    trainer = ROOT / "vendor/sdxl-train/run_training.py"
    if sha(manifest) != protocol["dataset"]["manifest_sha256"]:
        raise ValueError("Dataset manifest changed after protocol freeze")
    cached = json.loads((cache / "manifest.json").read_text())
    if not cached["complete"] or cached["dataset_manifest_sha256"] != sha(manifest) or cached["model_provenance_sha256"] != sha(model/"provenance.json") or cached["protocol_sha256"] != sha(PROTOCOL):
        raise ValueError("Incomplete or wrong training cache")
    command = [sys.executable, str(trainer),
        "--pretrained_model_name_or_path", str(model), "--variant", "fp16",
        "--dataset_name", str(dataset), "--priest_cached_data", str(cache),
        "--output_dir", str(output), "--resolution", "512", "--center_crop",
        "--image_interpolation_mode", "lanczos", "--train_batch_size", "1",
        "--dataloader_num_workers", "0", "--max_train_steps", str(args.updates),
        "--gradient_accumulation_steps", "1", "--gradient_checkpointing",
        "--mixed_precision", "fp16", "--rank", str(recipe["rank"]),
        "--learning_rate", str(recipe["learning_rate"]), "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0", "--checkpointing_steps", str(args.updates),
        "--seed", str(recipe["seed"]), "--report_to", "tensorboard",
        "--adam_beta1", str(recipe["optimizer_betas"][0]),
        "--adam_beta2", str(recipe["optimizer_betas"][1]),
        "--adam_weight_decay", str(recipe["optimizer_weight_decay"]),
        "--adam_epsilon", str(recipe["optimizer_epsilon"]),
        "--max_grad_norm", str(recipe["max_grad_norm"])]
    if args.prepare_only:
        print(json.dumps({"command":command,"protocol_sha256":sha(PROTOCOL),"cache_manifest_sha256":sha(cache/"manifest.json")},indent=2))
        return
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        PYTHONUNBUFFERED="1", PYTORCH_ENABLE_MPS_FALLBACK="0")
    output.mkdir(parents=True)
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (output / "juggernaut_pilot.py").write_bytes((ROOT/"juggernaut_pilot.py").read_bytes())
    if decision is not None:
        (output/"continue-decision.json").write_bytes(args.continue_decision.read_bytes())
    (output / protocol_path.name).write_bytes(protocol_path.read_bytes())
    vendor_hashes = {p.name: sha(p) for p in trainer.parent.iterdir() if p.is_file()}
    write(output / "launch.json", {"command": command,
        "purpose": "Juggernaut Hyper512-training/native1024-inference pilot; ordinary denoising can harm Hyper",
        "continue_decision_sha256":sha(args.continue_decision) if decision is not None else None,
        "upstream_checkpoint_sha256":components["upstream_checkpoint_sha256"],
        "production_approved": False, "automatic_retry": False,
        "maximum_updates": args.updates, "fresh_base_run": True,
        "maximum_seconds": 1800, "maximum_sampled_rss_bytes": 24 * 1024**3,
        "minimum_available_memory_fraction": .20, "maximum_swap_growth_bytes": 512 * 1024**2,
        "source_sha256": sha(Path(__file__)), "vendor_source_hashes": vendor_hashes,
        "protocol_sha256": sha(protocol_path), "cache_manifest_sha256": sha(cache / "manifest.json"),
        "dataset_manifest_sha256": sha(manifest), "model_provenance_sha256": sha(model / "provenance.json"),
        "environment_lock_sha256": sha(ROOT / "flux-train-requirements-lock.txt")})
    start = time.monotonic()
    initial_swap = psutil.swap_memory().used
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
                swap_growth = psutil.swap_memory().used - initial_swap
                monitor.write(json.dumps({"elapsed_seconds": elapsed, "rss_bytes": rss,
                    "available_system_bytes": vm.available, "system_total_bytes": vm.total,
                    "swap_growth_bytes": swap_growth}) + "\n")
                monitor.flush()
                if rss > 24 * 1024**3:
                    failure = "WorkerRSS exceeds24GiB"
                elif vm.available / vm.total < .20:
                    failure = "Available system memory below20%"
                elif swap_growth > 512 * 1024**2:
                    failure = "System swap grew more than512MiB"
                elif elapsed > 1800:
                    failure = "30minute bounded-run deadline"
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
        raise SystemExit(f"Training failed: {failure or code}")


if __name__ == "__main__":
    main()
