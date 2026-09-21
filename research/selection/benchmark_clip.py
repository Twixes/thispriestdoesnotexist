"""One fixed CPU-only CLIP experiment on the recorded 32 FFHQ source images.

No generated samples, learned thresholds, calibrated probabilities, or approvals.
"""
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research/selection"
OUT = BASE / "clip-ffhq32"
SOURCE = ROOT / "research/data/ffhq-paired-sources32"
WEIGHTS = BASE / "models/ViT-B-32.pt"
WEIGHTS_SHA = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
VENDOR = BASE / f"vendor/CLIP-{COMMIT}"
# Fixed before seeing any scores. Each category score is the arithmetic mean
# of its three individual image/text cosine similarities; no softmax.
PROMPTS = {
    "adult_male": [
        "a photograph of an adult man",
        "a photograph of a middle-aged man",
        "a portrait photograph of a mature adult male",
    ],
    "adult_woman": [
        "a photograph of an adult woman",
        "a photograph of a middle-aged woman",
        "a portrait photograph of a mature adult female",
    ],
    "child": [
        "a photograph of a child",
        "a portrait photograph of a young boy",
        "a portrait photograph of a young girl",
    ],
    "headwear": [
        "a portrait photograph of a person wearing a hat",
        "a portrait photograph of a person wearing a cap",
        "a portrait photograph of a person wearing headwear",
    ],
    "bare_head": [
        "a portrait photograph of a bareheaded person",
        "a portrait photograph of a person with an uncovered head",
        "a portrait photograph of a person without a hat",
    ],
}
LABELS = {
    **{i: {"appearance": "child"} for i in ["008", "019", "026"]},
    **{i: {"headwear": True} for i in ["021", "023", "027"]},
    **{i: {"appearance": "adult_male", "headwear": False}
       for i in ["000", "004", "025", "030"]},
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    preflight = subprocess.check_output(["memory_pressure"], text=True)
    (OUT / "host-memory-before.txt").write_text(preflight)
    free = int(re.search(r"System-wide memory free percentage: (\d+)%", preflight)[1])
    if free < 25:
        raise SystemExit(f"DEFER: {free}% free is below the required 25%; model not loaded")
    assert sha(WEIGHTS) == WEIGHTS_SHA, "official weight hash mismatch"
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    assert len(manifest["entries"]) == 32
    for e in manifest["entries"]:
        assert sha(SOURCE / e["source_path"]) == e["image_sha256"]
    started = time.perf_counter()
    import torch
    import clip
    from PIL import Image
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    loaded_at = time.perf_counter()
    # Only official hash-verified OpenAI archive and official architecture.
    # No Hugging Face remote code, custom model loader, or unsafe pickle override.
    model, preprocess = clip.load(str(WEIGHTS), device="cpu", jit=False)
    model.eval().requires_grad_(False)
    load_seconds = time.perf_counter() - loaded_at
    prompts = [p for ps in PROMPTS.values() for p in ps]
    rows = []
    with torch.inference_mode():
        text_started = time.perf_counter()
        # Text and image batches both respect the requested maximum of four.
        text = torch.cat([model.encode_text(clip.tokenize(prompts[i:i+4]))
                          for i in range(0, len(prompts), 4)])
        text /= text.norm(dim=-1, keepdim=True)
        text_seconds = time.perf_counter() - text_started
        for entry in manifest["entries"]:
            row_start = time.perf_counter()
            with Image.open(SOURCE / entry["source_path"]) as im:
                image = preprocess(im).unsqueeze(0)
            encode_start = time.perf_counter()
            features = model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
            scores = (features @ text.T).squeeze(0).tolist()
            encode_seconds = time.perf_counter() - encode_start
            category = {k: statistics.mean(scores[i*3:i*3+3])
                        for i, k in enumerate(PROMPTS)}
            rows.append({
                "id": entry["id"], "image_sha256": entry["image_sha256"],
                "manual_labels": LABELS.get(entry["id"], {}),
                "prompt_cosines": dict(zip(prompts, scores)),
                "category_mean_cosines": category,
                "adult_male_margin": category["adult_male"] - max(category["adult_woman"], category["child"]),
                "headwear_margin": category["headwear"] - category["bare_head"],
                "appearance_argmax": max(["adult_male", "adult_woman", "child"], key=category.__getitem__),
                "encode_seconds": encode_seconds,
                "preprocess_encode_seconds": time.perf_counter() - row_start,
            })
            print(json.dumps({k: rows[-1][k] for k in ["id", "appearance_argmax", "adult_male_margin", "headwear_margin", "preprocess_encode_seconds"]}), flush=True)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Fixed exploratory zero-shot image/text comparison, not calibrated age, attractiveness, or production approval",
        "production_approved": False, "count": 32, "device": "cpu",
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "image_batch_size": 1, "text_batch_max": 4,
        "memory_preflight_free_percent": free,
        "platform": platform.platform(), "python": sys.version,
        "packages": {n: importlib.metadata.version(n) for n in ["torch", "torchvision", "clip", "pillow", "numpy", "ftfy", "regex"]},
        "source_manifest_sha256": sha(SOURCE / "manifest.json"),
        "manual_label_source_sha256": sha(SOURCE / "README.md"),
        "benchmark_sha256": sha(Path(__file__)),
        "clip_commit": COMMIT, "clip_weights_sha256": sha(WEIGHTS),
        "clip_source_hashes": {str(p.relative_to(VENDOR)): sha(p) for p in sorted(VENDOR.rglob("*.py"))},
        "prompts": PROMPTS, "score_definition": "mean of three individual cosine similarities per category; margins are differences, not probabilities",
        "preprocess": str(preprocess),
        "load_seconds": load_seconds, "text_encoding_seconds": text_seconds,
        "import_load_text_images_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": peak if sys.platform == "darwin" else peak*1024,
        "image_encode_seconds_total": sum(r["encode_seconds"] for r in rows),
        "first_image_preprocess_encode_seconds": rows[0]["preprocess_encode_seconds"],
        "warm_image_preprocess_encode_median_seconds": statistics.median(r["preprocess_encode_seconds"] for r in rows[1:]),
        "entries": rows,
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    with (OUT / "scores.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=["id", "appearance_argmax", "adult_male_margin", "headwear_margin", *PROMPTS, "preprocess_encode_seconds"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ["id", "appearance_argmax", "adult_male_margin", "headwear_margin", "preprocess_encode_seconds"]} | r["category_mean_cosines"])
    print(json.dumps({k:v for k,v in result.items() if k not in ["entries", "clip_source_hashes", "prompts", "preprocess"]}, indent=2))


if __name__ == "__main__":
    main()
