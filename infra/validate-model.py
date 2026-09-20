#!/usr/bin/env python3
"""Fail closed before CI sends a promoted model to the personal origin."""
import hashlib
import json
from pathlib import Path
import sys

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "models/production")
metadata = json.loads((folder / "model.json").read_text())
with (folder / "generator.safetensors").open("rb") as stream:
    hasher = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        hasher.update(chunk)
    digest = hasher.hexdigest()
review = metadata.get("review", {})
if not (
    metadata.get("schema_version") == 1
    and metadata.get("training_step", 0) > 0
    and metadata.get("weights_sha256") == digest
    and review.get("approved") is True
    and review.get("weights_sha256") == digest
):
    raise SystemExit("Refusing untrained, unreviewed, or mismatched production model")
print(digest)
