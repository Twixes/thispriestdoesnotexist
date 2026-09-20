"""Export a trusted local training snapshot to a pickle-free serving bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from safetensors.torch import save_file
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor/stylegan2-ada-pytorch"))
import legacy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Own trainer's generator-*.pt")
    parser.add_argument("--base", type=Path, default=ROOT / "models/ffhq256.pkl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--truncation", type=float, default=.7)
    parser.add_argument("--review", type=Path, help="JSON review with approved and weights_sha256")
    parser.add_argument("--baseline-only", action="store_true", help="Untrained plumbing benchmark; never release")
    args = parser.parse_args()
    if bool(args.checkpoint) == args.baseline_only:
        parser.error("Choose a trained --checkpoint OR --baseline-only")
    if not 0 < args.truncation <= 1:
        parser.error("--truncation must be greater than zero and at most one")
    # Pickle is used only at this trusted offline conversion boundary. The
    # service imports reviewed source and loads only safetensors + JSON.
    with args.base.open("rb") as source:
        generator = legacy.load_network_pkl(source)["G_ema"].cpu().eval()
    step = 0
    source_checksum = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        generator.load_state_dict(checkpoint["G_ema"], strict=True)
        step = checkpoint["step"]
        if step <= 0:
            raise ValueError("A trained checkpoint must have a positive training step")
        with args.checkpoint.open("rb") as source:
            source_checksum = hashlib.file_digest(source, "sha256").hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "generator.safetensors"
    save_file({key: value.contiguous() for key, value in generator.state_dict().items()}, str(target))
    with target.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    review = json.loads(args.review.read_text()) if args.review else {"approved": False}
    if review.get("approved") and (step == 0 or review.get("weights_sha256") != digest):
        raise ValueError("Review does not match this trained model")
    metadata = {"schema_version": 1, "model_version": args.version, "training_step": step,
                "init_kwargs": dict(generator.init_kwargs), "truncation_psi": args.truncation,
                "weights_sha256": digest, "review": review,
                "source_checkpoint": str(args.checkpoint) if args.checkpoint else "untrained baseline",
                "source_checkpoint_sha256": source_checksum,
                "license": "NVIDIA Source Code License — research/evaluation only; see LICENSE.txt"}
    (args.output / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "LICENSE.txt").write_bytes((ROOT / "vendor/stylegan2-ada-pytorch/LICENSE.txt").read_bytes())
    print(json.dumps({"output": str(args.output), "weights_sha256": digest, "training_step": step}))


if __name__ == "__main__":
    main()
