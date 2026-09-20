"""Bounded CPU inference; no pickle loading, catalog, or pre-generated responses."""
import argparse
import hashlib
import hmac
import io
import json
import logging
import os
from pathlib import Path
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch

VENDOR = Path(__file__).resolve().parents[1] / "research/vendor/stylegan2-ada-pytorch"
sys.path.insert(0, str(VENDOR))
from training.networks import Generator

LOG = logging.getLogger(__name__)


class Model:
    def __init__(self, directory, threads=2, allow_unreviewed=False):
        directory = Path(directory)
        self.metadata = json.loads((directory / "model.json").read_text())
        weights = directory / "generator.safetensors"
        with weights.open("rb") as source:
            self.sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        if self.metadata["weights_sha256"] != self.sha256:
            raise ValueError("Model checksum does not match manifest")
        if self.metadata.get("schema_version") != 1:
            raise ValueError("Unsupported model manifest schema")
        review = self.metadata.get("review", {})
        self.reviewed = review.get("approved") is True and review.get("weights_sha256") == self.sha256
        if not allow_unreviewed and (not self.reviewed or self.metadata["training_step"] <= 0):
            raise ValueError("Production requires a trained, visually reviewed model matching this checksum")
        if threads < 1:
            raise ValueError("CPU thread count must be positive")
        torch.set_num_threads(threads)
        self.generator = Generator(**self.metadata["init_kwargs"]).cpu().eval().requires_grad_(False)
        if self.generator.c_dim != 0 or self.generator.img_channels != 3:
            raise ValueError("Only unconditional RGB generators are supported")
        self.generator.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
        self.lock = threading.Lock()

    def health(self):
        return {"status": "ready", "model_version": self.metadata["model_version"],
                "model_sha256": self.sha256, "training_step": self.metadata["training_step"],
                "reviewed": self.reviewed, "device": "cpu", "resolution": self.generator.img_resolution}

    def generate(self):
        # PCG64 consumes the full 256-bit entropy through SeedSequence; torch's
        # manual_seed accepts only 64 bits. Each request gets a distinct latent.
        seed = secrets.token_bytes(32)
        latent = np.random.default_rng(int.from_bytes(seed, "big")).standard_normal(
            (1, self.generator.z_dim), dtype=np.float32)
        started = time.monotonic()
        with self.lock, torch.inference_mode():
            pixels = self.generator(torch.from_numpy(latent), None,
                                    truncation_psi=self.metadata["truncation_psi"],
                                    noise_mode="const", force_fp32=True)
            pixels = ((pixels[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().numpy()
            encoded = io.BytesIO()
            Image.fromarray(pixels).save(encoded, format="WEBP", quality=90, method=4)
        return encoded.getvalue(), seed.hex(), time.monotonic() - started


class Server(ThreadingHTTPServer):
    # One socket/request per admitted thread; slow clients cannot create an
    # unbounded thread pool or inference queue. Responses always close sockets.
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, model, capacity=3, token=None):
        if capacity < 1:
            raise ValueError("Capacity must be positive")
        self.model = model
        self.token = token
        self.slots = threading.BoundedSemaphore(capacity)
        super().__init__(address, Handler)

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(10)
        return request, address

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            try:
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n"
                                b"Cache-Control: no-store\r\nRetry-After: 3\r\nConnection: close\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class Handler(BaseHTTPRequestHandler):
    server_version = "PriestInference/1"

    def reply(self, status, body=b"", content_type="application/json", headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.close_connection = True
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/health":
            self.reply(200, json.dumps(self.server.model.health()).encode())
            return
        if path != "/generate":
            self.reply(404)
            return
        if self.server.token and not hmac.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + self.server.token):
            self.reply(401)
            return
        try:
            body, seed, elapsed = self.server.model.generate()
        except Exception:
            LOG.exception("Generation failed")
            self.reply(503, headers={"Retry-After": "3"})
            return
        self.reply(200, body, "image/webp", {
            "X-Generation-Seed": seed,
            "X-Model-SHA256": self.server.model.sha256,
            "Server-Timing": f"inference;dur={elapsed * 1000:.1f}",
        })

    def log_message(self, format, *args):
        LOG.info("%s %s", self.client_address[0], format % args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=os.getenv("MODEL_DIR", "/app/model"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")))
    parser.add_argument("--threads", type=int, default=int(os.getenv("CPU_THREADS", "2")))
    parser.add_argument("--capacity", type=int, default=int(os.getenv("REQUEST_CAPACITY", "3")))
    parser.add_argument("--allow-unreviewed", action="store_true", help="Local research only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    model = Model(args.model_dir, args.threads, args.allow_unreviewed)
    server = Server((args.host, args.port), model, args.capacity, os.getenv("INFERENCE_TOKEN"))
    LOG.info("Model loaded: %s", json.dumps(model.health()))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
