"""Fetch a pinned public SDXL teacher; retain oversized weights as LFS chunks."""
import concurrent.futures
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
SPECS = {
    "sdxl-inpaint": ("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", "115134f363124c53c7d878647567d04daf26e41e"),
    "sdxl-turbo": ("stabilityai/sdxl-turbo", "71153311d3dbb46851df1931d3ca6e939de83304"),
    "sd-turbo": ("stabilityai/sd-turbo", "b261bac6fd2cf515557d5d0707481eafa0485ec2"),
}
CHUNK = 1024 ** 3


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 ** 2), b""):
            h.update(b)
    return h.hexdigest()


def fetch(entry):
    name = entry["rfilename"]
    path = DEST / name
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = entry.get("lfs", {}).get("sha256")
    if not path.exists():
        url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
        temporary = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as f:
            for data in iter(lambda: response.read(8 * 1024 ** 2), b""):
                f.write(data)
        temporary.rename(path)
    sha = digest(path)
    if path.stat().st_size != entry["size"] or (expected and sha != expected):
        raise ValueError(f"Upstream size/hash mismatch: {name}")
    chunks = []
    if name.startswith("unet/") and name.endswith(".safetensors"):
        with path.open("rb") as f:
            i = 0
            while data := f.read(CHUNK):
                part = path.with_name(path.name + f".weights-part-{i:03}")
                part.write_bytes(data)
                chunks.append({"path": str(part.relative_to(DEST)), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
                i += 1
    print(f"Verified {name}: {path.stat().st_size} bytes", flush=True)
    return {"path": name, "bytes": path.stat().st_size, "sha256": sha, "chunks": chunks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=list(SPECS), default="sdxl-inpaint")
    args = parser.parse_args()
    MODEL, REVISION = SPECS[args.name]
    DEST = ROOT / "models" / args.name
    DEST.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    if metadata["sha"] != REVISION:
        raise ValueError("Model revision mismatch")
    (DEST / "upstream-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    entries = [s for s in metadata["siblings"] if (s["rfilename"].endswith((".json", ".txt", ".md")) or s["rfilename"] in ("LICENSE", "LICENSE.TXT") or ("/" in s["rfilename"] and s["rfilename"].endswith(".fp16.safetensors"))) and not s["rfilename"].startswith(("onnx/", "openvino/"))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(fetch, entries))
    (DEST / "provenance.json").write_text(json.dumps({"model": MODEL, "revision": REVISION, "files": files, "license": metadata.get("cardData", {}).get("license"), "oversized_weights": "Concatenate chunks in listed order and verify full SHA256 to reconstruct ignored local weight."}, indent=2) + "\n")
