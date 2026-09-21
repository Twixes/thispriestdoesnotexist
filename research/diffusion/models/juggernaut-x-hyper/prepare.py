"""Fetch a pinned Juggernaut Hyper checkpoint/config/license; no model execution."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

DEST = Path(__file__).resolve().parent
MODEL = "RunDiffusion/Juggernaut-X-Hyper"
REVISION = "42fee7475922d8de7246cdcbe5e3a5e22c0ecf61"
WEIGHT = "JuggernautXRundiffusion_Hyper.safetensors"
WEIGHT_SHA256 = "010be7341cd98a136da775330ba3eb4e87025c6cfd2f5455dc64daee2200ae98"
WEIGHT_BYTES = 7105348616
LICENSE_REPO = "CompVis/stable-diffusion-license"
LICENSE_REVISION = "14d42d09bffd871b1666a084fc954a50cff72ac0"
PART_BYTES = 1024 ** 3
BUFFER_BYTES = 8 * 1024 ** 2


def digest(path, git_blob=False):
    h = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(BUFFER_BYTES), b""):
            h.update(data)
    return h.hexdigest()


def write(path, data):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def verify(path, entry):
    if path.stat().st_size != entry["size"]:
        raise ValueError(f"Size mismatch: {path}")
    actual = digest(path)
    expected = entry.get("lfs", {}).get("sha256")
    if expected:
        if actual != expected:
            raise ValueError(f"Upstream SHA256 mismatch: {path}")
    elif digest(path, git_blob=True) != entry["blobId"]:
        raise ValueError(f"Upstream Git blob mismatch: {path}")
    return actual


def fetch(spec):
    entry, url, name = spec
    path = DEST / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(path.name + ".partial")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as target:
            for data in iter(lambda: response.read(BUFFER_BYTES), b""):
                target.write(data)
        verify(temporary, entry)
        temporary.replace(path)
    actual = verify(path, entry)
    print(f"Verified {name}: {path.stat().st_size} bytes", flush=True)
    return {"path": name, "bytes": path.stat().st_size, "sha256": actual,
            "source_url": url, "upstream_entry": entry, "chunks": []}


def split_weight(path):
    chunks = []
    full = hashlib.sha256()
    with path.open("rb") as source:
        index = 0
        while source.tell() < WEIGHT_BYTES:
            size = min(PART_BYTES, WEIGHT_BYTES - source.tell())
            remaining = size
            part = path.with_name(path.name + f".weights-part-{index:03}")
            temporary = part.with_name(part.name + ".partial")
            h = hashlib.sha256()
            with temporary.open("wb") as target:
                while remaining:
                    data = source.read(min(BUFFER_BYTES, remaining))
                    if not data:
                        raise ValueError("Unexpected checkpoint EOF")
                    h.update(data)
                    full.update(data)
                    target.write(data)
                    remaining -= len(data)
            temporary.replace(part)
            chunks.append({"path": part.name, "bytes": size, "sha256": h.hexdigest()})
            index += 1
    if full.hexdigest() != WEIGHT_SHA256:
        raise ValueError("Checkpoint changed during splitting")
    return chunks


def verify_saved():
    manifest = json.loads((DEST / "provenance.json").read_text())
    if not manifest["complete"] or manifest["revision"] != REVISION:
        raise ValueError("Unexpected manifest")
    for entry in manifest["files"]:
        path = DEST / entry["path"]
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Artifact mismatch: {path}")
        if entry["chunks"]:
            joined = hashlib.sha256()
            total = 0
            for chunk in entry["chunks"]:
                part = DEST / chunk["path"]
                if part.stat().st_size != chunk["bytes"] or digest(part) != chunk["sha256"] or chunk["bytes"] > PART_BYTES:
                    raise ValueError(f"LFS chunk mismatch: {part}")
                with part.open("rb") as stream:
                    for data in iter(lambda: stream.read(BUFFER_BYTES), b""):
                        joined.update(data)
                        total += len(data)
            if total != entry["bytes"] or joined.hexdigest() != entry["sha256"]:
                raise ValueError("Concatenated LFS parts differ from full upstream checkpoint")
    result = {"complete": True, "verified_files": len(manifest["files"]),
              "full_checkpoint_sha256": WEIGHT_SHA256, "parts_concatenate_to_upstream_sha256": True,
              "maximum_part_bytes": PART_BYTES, "provenance_sha256": digest(DEST / "provenance.json"),
              "model_load_or_inference_performed": False}
    write(DEST / "verification.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        return verify_saved()
    metadata_url = f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(metadata_url, timeout=30) as response:
        metadata = json.load(response)
    if metadata["sha"] != REVISION or metadata.get("gated"):
        raise ValueError("Unexpected model revision/access state")
    write(DEST / "upstream-metadata.json", metadata)
    selected = [e for e in metadata["siblings"] if e["rfilename"] == WEIGHT or
                (e["rfilename"].endswith((".json", ".txt", ".md")) and not e["rfilename"].startswith("assets/"))]
    weight_entry = next(e for e in selected if e["rfilename"] == WEIGHT)
    if weight_entry["size"] != WEIGHT_BYTES or weight_entry["lfs"]["sha256"] != WEIGHT_SHA256:
        raise ValueError("Upstream checkpoint identity differs from pinned research record")
    specs = [(e, f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{e['rfilename']}",
              "sources/publisher-model-card.md" if e["rfilename"] == "README.md" else e["rfilename"])
             for e in selected]
    license_api = f"https://huggingface.co/api/spaces/{LICENSE_REPO}/revision/{LICENSE_REVISION}?blobs=true"
    with urllib.request.urlopen(license_api, timeout=30) as response:
        license_metadata = json.load(response)
    if license_metadata["sha"] != LICENSE_REVISION:
        raise ValueError("Unexpected license revision")
    write(DEST / "license-upstream-metadata.json", license_metadata)
    license_entry = next(e for e in license_metadata["siblings"] if e["rfilename"] == "license.txt")
    specs.append((license_entry, f"https://huggingface.co/spaces/{LICENSE_REPO}/resolve/{LICENSE_REVISION}/license.txt", "LICENSE.txt"))
    download_bytes = sum(e[0]["size"] for e in specs)
    needed = download_bytes + WEIGHT_BYTES + 20 * PART_BYTES
    free = shutil.disk_usage(DEST).free
    if free < needed:
        raise RuntimeError(f"Need {needed} bytes including20GiB headroom, free={free}")
    write(DEST / "download-plan.json", {"model": MODEL, "revision": REVISION, "source_api": metadata_url,
          "download_bytes": download_bytes, "additional_part_bytes": WEIGHT_BYTES, "free_bytes_before": free,
          "license_source_api": license_api, "selected_paths": [s[2] for s in specs], "maximum_network_threads": 3})
    print(f"Downloading {len(specs)} files, {download_bytes} bytes, no model execution", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(fetch, specs))
    weight = next(e for e in files if e["path"] == WEIGHT)
    weight["chunks"] = split_weight(DEST / WEIGHT)
    with (DEST / WEIGHT).open("rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        if length > 20 * 1024 ** 2:
            raise ValueError("Unreasonable safetensors header length")
        header = json.loads(stream.read(length))
    tensors = [v for k, v in header.items() if k != "__metadata__"]
    manifest = {"complete": True, "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "model": MODEL, "revision": REVISION, "checkpoint_name": WEIGHT,
          "checkpoint_kind": "publisher single-file complete SDXL Hyper checkpoint; no conversion performed",
          "license": "creativeml-openrail-m", "license_notice": "Publisher card additionally requires licensing for paid API deployment; preserve both card and LICENSE.txt. Not MIT.",
          "license_origin": {"repo": LICENSE_REPO, "revision": LICENSE_REVISION, "path": "license.txt", "linked_by_publisher_card": True},
          "files": files, "upstream_metadata_sha256": digest(DEST / "upstream-metadata.json"),
          "license_metadata_sha256": digest(DEST / "license-upstream-metadata.json"), "prepare_source_sha256": digest(Path(__file__)),
          "header": {"tensor_count": len(tensors), "dtypes": sorted({t['dtype'] for t in tensors}), "bytes": length},
          "inference_reference": {"author_steps": [4, 8], "author_start_steps": 6,
              "author_samplers": ["DPM++ SDE", "TCD"], "author_cfg_range": [1.0, 2.0], "native_resolution": 1024,
              "candidate": "TCD4steps,CFG1,1024; exact scheduler config/eta to freeze in comparison protocol",
              "resolution512": "separate non-native quality/latency experiment, not publisher reference",
              "lora": "SDXL structure compatible; priest adaptation effectiveness/distillation preservation unproven"},
          "oversized_weights": "Full checkpoint ignored locally; concatenate <=1GiB manifest chunks in order and verify upstream SHA256 before loading.",
          "model_load_or_inference_performed": False, "production_approved": False}
    write(DEST / "provenance.json", manifest)
    verify_saved()


if __name__ == "__main__":
    main()
