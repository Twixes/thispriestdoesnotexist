"""Rebuild the ignored original checkpoint from verified LFS parts, without loading it."""
import hashlib
import json
from pathlib import Path
import shutil

DEST = Path(__file__).resolve().parent
EXPECTED_REVISION = "42fee7475922d8de7246cdcbe5e3a5e22c0ecf61"
EXPECTED_SHA256 = "010be7341cd98a136da775330ba3eb4e87025c6cfd2f5455dc64daee2200ae98"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            h.update(data)
    return h.hexdigest()


def main():
    manifest = json.loads((DEST / "provenance.json").read_text())
    if not manifest["complete"] or manifest["revision"] != EXPECTED_REVISION:
        raise ValueError("Unexpected checkpoint provenance")
    entry = next(e for e in manifest["files"] if e["path"] == manifest["checkpoint_name"])
    if entry["sha256"] != EXPECTED_SHA256:
        raise ValueError("Unexpected checkpoint hash")
    path = DEST / entry["path"]
    if not path.exists():
        if shutil.disk_usage(DEST).free < entry["bytes"] + 5 * 1024 ** 3:
            raise RuntimeError("Insufficient disk headroom to reconstruct checkpoint")
        temporary = path.with_name(path.name + ".partial")
        with temporary.open("wb") as output:
            for chunk in entry["chunks"]:
                part = DEST / chunk["path"]
                if part.stat().st_size != chunk["bytes"] or digest(part) != chunk["sha256"]:
                    raise ValueError(f"Corrupt or missing LFS part: {part}")
                with part.open("rb") as source:
                    shutil.copyfileobj(source, output, length=8 * 1024 ** 2)
        if temporary.stat().st_size != entry["bytes"] or digest(temporary) != EXPECTED_SHA256:
            raise ValueError("Reassembled checkpoint differs from upstream")
        temporary.replace(path)
    if path.stat().st_size != entry["bytes"] or digest(path) != EXPECTED_SHA256:
        raise ValueError("Local checkpoint differs from upstream")
    print(json.dumps({"path": str(path), "sha256": EXPECTED_SHA256, "verified": True, "model_loaded": False}))


if __name__ == "__main__":
    main()
