#!/usr/bin/env python3
"""Materialize the reviewed imagefolder without modifying any image bytes."""
import hashlib
import json
import shutil
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "research/data/plain-background36-whole-image"
EXPECTED_SOURCE_MANIFEST = "5a9e6558f5bdb4a2fe53d86c19f34cbb4221b3f1030255f43d3d8415b51961bd"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def build():
    source_manifest = SOURCE / "manifest.json"
    assert digest(source_manifest) == EXPECTED_SOURCE_MANIFEST, "Source manifest changed"
    source = {e["id"]: e for e in json.loads(source_manifest.read_text())["entries"]}
    annotations = json.loads((HERE / "annotations.json").read_text())
    records = annotations["records"]
    assert len(records) == 34 and len({r["id"] for r in records}) == 34
    excluded = set(annotations["additional_excluded_ids"])
    validation = set(annotations["validation_ids"])
    test = set(annotations["test_ids"])
    assert not validation & test
    family = {n: f["group"] for f in annotations["family_groups"] for n in f["members"]}
    accepted = [r for r in records if r["id"] not in excluded]
    assert len(accepted) == 30 and len({r["caption"] for r in accepted}) == 30
    assert not excluded & (validation | test)
    prepared = []
    for record in records:
        entry = source[record["id"]]
        native = SOURCE / entry["outputs"]["1024"]["path"]
        original = ROOT / entry["source_path"]
        assert digest(native) == entry["outputs"]["1024"]["sha256"]
        assert digest(original) == entry["source_sha256"]
        data = native.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", data[16:24]) == (1024, 1024)
        assert record["caption"].startswith("PR1EST_CAL. ")
        split = None if record["id"] in excluded else (
            "validation" if record["id"] in validation else
            "test" if record["id"] in test else "train")
        prepared.append({
            **record, "split": split,
            "family_group": family.get(record["id"], f"individual-{record['id']}"),
            "native_source": str(native.relative_to(ROOT)),
            "native_sha256": digest(native),
            "original_source": str(original.relative_to(ROOT)),
            "original_sha256": digest(original),
            "caption_sha256": hashlib.sha256(record["caption"].encode()).hexdigest(),
        })
    included = [r for r in prepared if r["split"]]
    assert len({r["native_sha256"] for r in included}) == 30
    counts = {s: sum(r["split"] == s for r in included) for s in ("train", "validation", "test")}
    assert counts == {"train": 20, "validation": 5, "test": 5}
    for group in {r["family_group"] for r in included}:
        assert len({r["split"] for r in included if r["family_group"] == group}) == 1
    for split in counts:
        assert not (HERE / split).exists(), f"Refusing to overwrite existing {split}"
    for output in ("manifest.json", "native-review.json"):
        assert not (HERE / output).exists(), f"Refusing to overwrite existing {output}"
    metadata_hashes = {}
    for split in counts:
        folder = HERE / split
        folder.mkdir()
        metadata = []
        for record in included:
            if record["split"] != split:
                continue
            filename = f"{record['id']}.png"
            dest = folder / filename
            shutil.copy2(ROOT / record["native_source"], dest)
            assert digest(dest) == record["native_sha256"]
            record["destination"] = str(dest.relative_to(HERE))
            record["destination_sha256"] = digest(dest)
            metadata.append({"file_name": filename, "text": record["caption"]})
        metadata_path = folder / "metadata.jsonl"
        metadata_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata))
        metadata_hashes[split] = digest(metadata_path)
    review = {
        "review_scope": annotations["review_scope"],
        "method": "Manual individual native1024 visual inspection; no face recognizer or biometric uniqueness claim.",
        "reviewer": annotations["reviewer"],
        "annotations_sha256": digest(HERE / "annotations.json"),
        "dataset_accepted_for_bounded_pilot": True,
        "production_approved": False,
        "subject_scope": annotations["subject_scope"],
        "family_groups": annotations["family_groups"],
        "inherited_excluded_ids_not_newly_reviewed": annotations["prior_excluded_ids"],
        "records": [{
            **r, "accepted": r["split"] is not None,
            "rejection_reason": None if r["split"] else "Conservative near-identity-family redundancy exclusion",
            "quality_observations": {
                "photographic": True, "coherent_eyes": True, "recognizable_clerical_collar": True,
                "adult_male_appearance": True, "bareheaded": True, "single_person": True,
            },
        } for r in prepared],
    }
    write_json(HERE / "native-review.json", review)
    write_json(HERE / "manifest.json", {
        "dataset": "flux-priest-domain-v1", "trigger": "PR1EST_CAL", "counts": counts,
        "image_size": [1024, 1024], "image_method": "shutil.copy2 byte-exact copy; no image editing or re-encoding",
        "source_manifest": str(source_manifest.relative_to(ROOT)),
        "source_manifest_sha256": digest(source_manifest),
        "annotations_sha256": digest(HERE / "annotations.json"),
        "build_script_sha256": digest(Path(__file__)),
        "native_review_sha256": digest(HERE / "native-review.json"),
        "metadata_sha256": metadata_hashes,
        "production_approved": False,
        "train_directory_only": str((HERE / "train").relative_to(ROOT)),
        "excluded_ids": sorted(excluded | set(annotations["prior_excluded_ids"])),
        "family_groups": annotations["family_groups"], "entries": included,
    })
    print(json.dumps({"counts": counts, "manifest_sha256": digest(HERE / "manifest.json")}, indent=2))


if __name__ == "__main__":
    build()
