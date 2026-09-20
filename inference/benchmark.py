"""Record actual CPU forwards and generated outputs in a research run directory."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import time

import torch

from inference.server import Model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--allow-unreviewed", action="store_true")
    args = parser.parse_args()
    if args.count < 2:
        parser.error("Use at least two samples")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model = Model(args.model_dir, args.threads, args.allow_unreviewed)
    loaded = time.monotonic() - started
    generations = []
    for index in range(args.count):
        body, seed, seconds = model.generate()
        target = args.output / f"sample-{index:03}.webp"
        target.write_bytes(body)
        record = {"sample": target.name, "seed": seed, "seconds": seconds,
                  "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        generations.append(record)
        print(json.dumps(record), flush=True)
    warm = [record["seconds"] for record in generations[1:]]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        rss *= 1024
    report = {"model": model.health(), "platform": platform.platform(), "torch": torch.__version__,
              "threads": args.threads, "model_load_seconds": loaded, "peak_rss_bytes": rss,
              "median_warm_seconds": statistics.median(warm), "max_warm_seconds": max(warm),
              "unique_output_hashes": len({record["sha256"] for record in generations}),
              "generations": generations,
              "interpretation": "Local CPU measurement, not a production-host benchmark or visual quality approval."}
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "generations"}), flush=True)


if __name__ == "__main__":
    main()
