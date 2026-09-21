"""Offline CPU export of the proven Juggernaut single-file FP16 load, with tensor equality checks."""
import argparse
import gc
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
SOURCE = HERE / "models/juggernaut-x-hyper"
DESTINATION = HERE / "models/juggernaut-x-hyper-components"
REVISION = "42fee7475922d8de7246cdcbe5e3a5e22c0ecf61"
CHECKPOINT_SHA = "010be7341cd98a136da775330ba3eb4e87025c6cfd2f5455dc64daee2200ae98"
SOURCE_PROVENANCE_SHA = "da4b21ed1fb3ff4dcc3cbd9881afc1549351214643d1ffe324421611fcdbe330"
GIB = 1024 ** 3
BUFFER = 8 * 1024 ** 2
DEADLINE = 1200


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(BUFFER), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temp.replace(path)


def fingerprint(component):
    """Digest each CPU tensor without a full weight-file-sized byte copy."""
    result = {}
    for name, tensor in sorted(component.state_dict().items()):
        assert tensor.device.type == "cpu", "Conversion must remain CPU-only"
        contiguous = tensor.detach().contiguous()
        data = memoryview(contiguous.numpy()).cast("B")
        digest = hashlib.sha256()
        for offset in range(0, len(data), BUFFER):
            digest.update(data[offset:offset + BUFFER])
        result[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "value_sha256": digest.hexdigest()}
    return result


def split_weight(path, output):
    chunks = []
    if path.stat().st_size > GIB:
        # Ignore the reassembled original before making any LFS parts.
        with (output / ".gitignore").open("a") as stream:
            stream.write("/" + str(path.relative_to(output)) + "\n")
        combined = hashlib.sha256()
        with path.open("rb") as source:
            while source.tell() < path.stat().st_size:
                size = min(GIB, path.stat().st_size - source.tell())
                chunk = path.with_name(path.name + f".weights-part-{len(chunks):03}")
                temporary = chunk.with_name(chunk.name + ".partial")
                remaining = size
                digest = hashlib.sha256()
                with temporary.open("wb") as target:
                    while remaining:
                        block = source.read(min(BUFFER, remaining))
                        assert block, "Unexpected EOF"
                        target.write(block)
                        digest.update(block)
                        combined.update(block)
                        remaining -= len(block)
                temporary.replace(chunk)
                assert sha(chunk) == digest.hexdigest()
                chunks.append({"path": str(chunk.relative_to(output)), "bytes": size, "sha256": digest.hexdigest()})
        assert combined.hexdigest() == sha(path), "Parts differ from saved component file"
    return chunks


def worker(output):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import psutil
    import torch
    from diffusers import StableDiffusionXLPipeline
    from diffusers.loaders import single_file, single_file_utils
    from diffusers.models import model_loading_utils
    protocol = json.loads((output / "export-protocol.json").read_text())
    assert sha(__file__) == protocol["script_sha256"], "Exporter changed after launch"
    assert sha(SOURCE / "provenance.json") == SOURCE_PROVENANCE_SHA
    manifest = json.loads((SOURCE / "provenance.json").read_text())
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    initial_swap = psutil.swap_memory().used

    def guard(phase):
        vm = psutil.virtual_memory()
        row = {"phase": phase, "elapsed_seconds": time.monotonic() - started,
               "rss_bytes": psutil.Process().memory_info().rss, "available_fraction": vm.available / vm.total,
               "swap_growth_bytes": psutil.swap_memory().used - initial_swap}
        with (output / "resources.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        assert row["rss_bytes"] <= 24 * GIB and row["available_fraction"] >= .20
        assert row["swap_growth_bytes"] <= 512 * 1024 ** 2 and row["elapsed_seconds"] <= DEADLINE

    for entry in manifest["files"]:
        path = SOURCE / entry["path"]
        assert path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], str(path)
    checkpoint = SOURCE / manifest["checkpoint_name"]
    assert sha(checkpoint) == CHECKPOINT_SHA
    original = output / "sources/original"
    original.mkdir(parents=True)
    # Copy every original small config/tokenizer/license/card byte-exact. The full
    # checkpoint remains in its existing hash-bound source archive, not duplicated.
    for entry in manifest["files"]:
        path = SOURCE / entry["path"]
        if path.name == manifest["checkpoint_name"]:
            continue
        assert path.stat().st_size < GIB, "Unexpected large source support file"
        target = original / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(SOURCE / "provenance.json", output / "sources/upstream-provenance.json")
    shutil.copy2(SOURCE / "LICENSE.txt", output / "LICENSE.txt")
    guard("source_verified")
    begin = time.perf_counter()
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(checkpoint), config=str(SOURCE), local_files_only=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, disable_mmap=False, add_watermarker=False)
    load_seconds = time.perf_counter() - begin
    gc.collect()
    guard("single_file_cpu_assembly_complete")
    # Preserve loader-resolved pipeline metadata and the publisher's scheduler
    # byte-exact at the standard path. No TCD inference replacement enters training.
    pipe.save_config(output)
    for name in ("scheduler", "tokenizer", "tokenizer_2"):
        shutil.copytree(SOURCE / name, output / name)
    write(output / "sources/loader-resolved-scheduler.json", dict(pipe.scheduler.config))
    verifications = []
    for name, component in (("text_encoder", pipe.text_encoder), ("text_encoder_2", pipe.text_encoder_2),
                            ("unet", pipe.unet), ("vae", pipe.vae)):
        begin = time.perf_counter()
        guard(name + "_before_export")
        component.eval().requires_grad_(False)
        before = fingerprint(component)
        component.save_pretrained(output / name, safe_serialization=True, variant="fp16", max_shard_size="10GB")
        # Reload saved tensors as a fresh CPU module, not the original reference.
        reloaded = type(component).from_pretrained(output / name, local_files_only=True,
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True, low_cpu_mem_usage=True)
        after = fingerprint(reloaded)
        report = {"component": name, "tensor_count": len(before), "equal": before == after,
                  "before": before, "after": after, "seconds": time.perf_counter() - begin}
        write(output / f"verification-{name}.json", report)
        assert before == after, "Saved/reloaded tensor values, shapes or dtypes changed: " + name
        del reloaded, before, after
        gc.collect()
        guard(name + "_reload_verified")
        verifications.append({"component": name, "tensor_count": report["tensor_count"],
            "equal": True, "report_sha256": sha(output / f"verification-{name}.json")})
        print(json.dumps(verifications[-1]), flush=True)
    del pipe, component
    gc.collect()
    guard("all_modules_released")
    files = []
    payload = [output / "model_index.json", output / "LICENSE.txt"]
    for directory in ("text_encoder", "text_encoder_2", "unet", "vae", "scheduler", "tokenizer", "tokenizer_2", "sources"):
        payload.extend(p for p in (output / directory).rglob("*") if p.is_file())
    for path in sorted(payload):
        chunks = split_weight(path, output) if path.suffix == ".safetensors" else []
        files.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                      "sha256": sha(path), "chunks": chunks})
        guard("archived_" + str(path.relative_to(output)))
    runtime = {"python": sys.version, "device": "cpu", "load_seconds": load_seconds,
        "packages": {name: importlib.metadata.version(name) for name in
            ("torch", "diffusers", "transformers", "accelerate", "safetensors", "numpy", "psutil")},
        "loader_source_hashes": {"single_file": sha(single_file.__file__),
            "single_file_utils": sha(single_file_utils.__file__), "model_loading_utils": sha(model_loading_utils.__file__)}}
    write(output / "runtime.json", runtime)
    write(output / "provenance.json", {"complete": True, "model": "RunDiffusion/Juggernaut-X-Hyper",
        "revision": REVISION, "kind": "Local verified FP16 Diffusers conversion of pinned single-file checkpoint",
        "upstream_checkpoint_sha256": CHECKPOINT_SHA, "upstream_provenance_sha256": SOURCE_PROVENANCE_SHA,
        "license": "creativeml-openrail-m", "license_caveat": "Publisher also requires explicit licensing for paid API deployment; see original model card.",
        "tensor_equality_scope": "Each tensor in the proven FP16 single-file loader output compared to fresh CPU FP16 component reload; original mixed-precision checkpoint may be cast by the loader, which is preserved.",
        "verifications": verifications, "files": files, "runtime_sha256": sha(output / "runtime.json"),
        "script_sha256": protocol["script_sha256"], "model_loading_performed": True,
        "inference_or_training_performed": False, "production_approved": False})
    write(output / "verification.json", {"complete": True, "all_component_tensors_equal": True,
        "components": verifications, "all_part_sizes_at_most_1gib": True,
        "all_parts_match_saved_files": True, "provenance_sha256": sha(output / "provenance.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        try:
            worker(output)
        except BaseException as error:
            write(output / "failure.json", {"error": str(error), "type": type(error).__name__, "traceback": traceback.format_exc()})
            raise
        return
    import psutil
    assert not (output / "export-protocol.json").exists(), "Never overwrite an export attempt; use a fresh directory"
    if output.exists():
        assert set(p.name for p in output.iterdir()) <= {"README.md", ".gitignore", ".gitattributes"}
    vm = psutil.virtual_memory()
    assert vm.available / vm.total >= .35, "Need35% available memory before starting"
    assert shutil.disk_usage(output.parent).free >= 34 * GIB, "Need34GiB free disk including archive headroom"
    assert sha(SOURCE / "provenance.json") == SOURCE_PROVENANCE_SHA
    output.mkdir(parents=True, exist_ok=True)
    (output / ".gitignore").write_text("*.partial\n")
    (output / ".gitattributes").write_text("*.safetensors filter=lfs diff=lfs merge=lfs -text\n*.weights-part-* filter=lfs diff=lfs merge=lfs -text\n")
    shutil.copy2(__file__, output / "export-source.py")
    write(output / "export-protocol.json", {"script_sha256": sha(__file__), "source_provenance_sha256": SOURCE_PROVENANCE_SHA,
        "source_checkpoint_sha256": CHECKPOINT_SHA, "device": "cpu", "dtype": "float16",
        "max_rss_bytes": 24 * GIB, "start_available_fraction": .35, "stop_available_fraction": .20,
        "max_swap_growth_bytes": 512 * 1024 ** 2, "deadline_seconds": DEADLINE})
    started = time.monotonic()
    initial_swap = psutil.swap_memory().used
    failure = None
    peak = 0
    with (output / "worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(output)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                vm = psutil.virtual_memory()
                try:
                    peak = max(peak, psutil.Process(process.pid).memory_info().rss)
                except psutil.NoSuchProcess:
                    break
                if peak > 24 * GIB or vm.available / vm.total < .20 or psutil.swap_memory().used - initial_swap > 512 * 1024 ** 2:
                    failure = "Memory guard"
                elif time.monotonic() - started > DEADLINE:
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
            write(output / "supervisor.json", {"exit_code": code, "failure": failure,
                "seconds": time.monotonic() - started, "peak_sampled_rss_bytes": peak})
    if failure or code:
        raise SystemExit(f"Export stopped: {failure or code}; partial attempt retained")


if __name__ == "__main__":
    main()
