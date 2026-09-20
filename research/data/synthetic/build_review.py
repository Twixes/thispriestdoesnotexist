"""Build review sheets and a checksum manifest; run from any directory.

Visual acceptance is recorded separately in review/accepted.json. This script
does not turn unreviewed image files into accepted training data.
"""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
images = sorted(path for path in (ROOT / "images").glob("*.webp")
                if path.with_suffix(".json").exists())
accepted_path = ROOT / "review" / "accepted.json"
accepted = json.loads(accepted_path.read_text()) if accepted_path.exists() else {}
entries = []
for path in images:
    record = json.loads(path.with_suffix(".json").read_text())
    dimensions = subprocess.check_output(
        ["magick", "identify", "-format", "%w %h", str(path)], text=True
    ).split()
    record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    record["width"], record["height"] = map(int, dimensions)
    record["bytes"] = path.stat().st_size
    record["visual_review"] = accepted.get(path.stem, "pending")
    entries.append(record)

for start in range(0, len(images), 6):
    group = images[start:start + 6]
    sheet = ROOT / "review" / f"contact-{group[0].stem}-{group[-1].stem}.jpg"
    subprocess.run(
        ["magick", "montage", "-font", FONT, "-label", "%f", *map(str, group),
         "-thumbnail", "480x480", "-tile", "3x2", "-geometry", "+8+8",
         "-background", "#202020", "-fill", "white", "-pointsize", "20", str(sheet)],
        check=True,
    )

manifest = {
    "schema_version": 1,
    "generator": "OpenAI built-in image_gen",
    "intended_use": "Additional synthetic training data; not production static portraits",
    "requested_count": 60,
    "saved_count": len(entries),
    "reviewed_accepted_count": sum(e["visual_review"] != "pending" for e in entries),
    "unique_file_hashes": len({e["sha256"] for e in entries}),
    "entries": entries,
}
(ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({key: value for key, value in manifest.items() if key != "entries"}, indent=2))
