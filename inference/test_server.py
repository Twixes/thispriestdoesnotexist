"""Exercise real tiny-generator forwards plus HTTP admission/error semantics."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from PIL import Image
from safetensors.torch import save_file
import torch

from inference.server import Generator, Model, Server


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name)
        kwargs = {"z_dim": 8, "c_dim": 0, "w_dim": 8, "img_resolution": 8, "img_channels": 3,
                  "mapping_kwargs": {"num_layers": 2},
                  "synthesis_kwargs": {"channel_base": 64, "channel_max": 16}}
        generator = Generator(**kwargs).eval().requires_grad_(False)
        save_file(generator.state_dict(), str(cls.path / "generator.safetensors"))
        digest = hashlib.sha256((cls.path / "generator.safetensors").read_bytes()).hexdigest()
        cls.metadata = {"schema_version": 1, "model_version": "unit-fixture-not-a-priest-model",
                        "training_step": 1, "init_kwargs": kwargs, "truncation_psi": .7,
                        "weights_sha256": digest,
                        "review": {"approved": True, "weights_sha256": digest}}
        (cls.path / "model.json").write_text(json.dumps(cls.metadata))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def model(self):
        return Model(self.path, threads=1)

    def serve(self, model, capacity=3, token=None):
        server = Server(("127.0.0.1", 0), model, capacity, token)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server, f"http://127.0.0.1:{server.server_port}"

    def test_actual_model_forward_each_request_with_full_entropy(self):
        model = self.model()
        with patch.object(model.generator, "forward", wraps=model.generator.forward) as forward:
            with patch("inference.server.secrets.token_bytes", side_effect=[bytes(32), bytes([1]) * 32]) as entropy:
                first, first_seed, _ = model.generate()
                second, second_seed, _ = model.generate()
        self.assertEqual(forward.call_count, 2)
        self.assertEqual(entropy.call_args_list[0].args, (32,))
        self.assertNotEqual(first_seed, second_seed)
        self.assertNotEqual(first, second)
        self.assertEqual(Image.open(io.BytesIO(first)).size, (8, 8))
        self.assertEqual(Image.open(io.BytesIO(first)).format, "WEBP")

    def test_http_images_are_not_cached_or_selected(self):
        model = self.model()
        _, url = self.serve(model)
        with patch.object(model.generator, "forward", wraps=model.generator.forward) as forward:
            with urllib.request.urlopen(url + "/generate?nonce=one") as first:
                first_body, first_seed = first.read(), first.headers["X-Generation-Seed"]
                self.assertIn("no-store", first.headers["Cache-Control"])
                self.assertEqual(first.headers["Content-Type"], "image/webp")
                self.assertEqual(first.headers["X-Model-SHA256"], model.sha256)
            with urllib.request.urlopen(url + "/generate?nonce=two") as second:
                self.assertNotEqual(first_body, second.read())
                self.assertNotEqual(first_seed, second.headers["X-Generation-Seed"])
        self.assertEqual(forward.call_count, 2)
        with urllib.request.urlopen(url + "/health") as response:
            self.assertEqual(json.load(response)["model_sha256"], model.sha256)

    def test_origin_token_and_unknown_routes_do_not_run_model(self):
        model = self.model()
        _, url = self.serve(model, token="test-only-token")
        with patch.object(model, "generate", wraps=model.generate) as generate:
            for path, expected in [("/generate", 401), ("/unknown", 404)]:
                with self.assertRaises(urllib.error.HTTPError) as response:
                    urllib.request.urlopen(url + path)
                self.assertEqual(response.exception.code, expected)
            self.assertEqual(generate.call_count, 0)
            request = urllib.request.Request(url + "/generate", headers={"Authorization": "Bearer test-only-token"})
            with urllib.request.urlopen(request) as response:
                self.assertEqual(response.status, 200)
            self.assertEqual(generate.call_count, 1)

    def test_bounded_capacity_rejects_overload_and_recovers(self):
        model = self.model()
        entered, release = threading.Event(), threading.Event()
        real_generate = model.generate

        def blocking_generate():
            entered.set()
            release.wait(timeout=5)
            return real_generate()

        server, url = self.serve(model, capacity=1)
        results = []
        with patch.object(model, "generate", side_effect=blocking_generate):
            def first_request():
                with urllib.request.urlopen(url + "/generate", timeout=10) as response:
                    results.append(response.status)
            thread = threading.Thread(target=first_request)
            thread.start()
            self.assertTrue(entered.wait(timeout=5))
            try:
                with self.assertRaises(urllib.error.HTTPError) as response:
                    urllib.request.urlopen(url + "/generate", timeout=2)
                self.assertEqual(response.exception.code, 503)
                self.assertEqual(response.exception.headers["Retry-After"], "3")
                self.assertIn("no-store", response.exception.headers["Cache-Control"])
            finally:
                release.set()
                thread.join(timeout=10)
        self.assertEqual(results, [200])
        # Wait for the admitted request's thread to release the bounded slot.
        self.assertTrue(server.slots.acquire(timeout=3))
        server.slots.release()
        with urllib.request.urlopen(url + "/generate") as response:
            self.assertEqual(response.status, 200)

    def test_failure_is_uncached_503_and_slot_is_released(self):
        model = self.model()
        server, url = self.serve(model, capacity=1)
        with patch.object(model, "generate", side_effect=RuntimeError("test inference failure")):
            with self.assertLogs("inference.server", level="ERROR"):
                with self.assertRaises(urllib.error.HTTPError) as response:
                    urllib.request.urlopen(url + "/generate")
                self.assertEqual(response.exception.code, 503)
                self.assertIn("no-store", response.exception.headers["Cache-Control"])
        self.assertTrue(server.slots.acquire(timeout=3))
        server.slots.release()

    def test_checksum_and_review_fail_closed(self):
        for change in [{"weights_sha256": "bad"}, {"review": {"approved": False}},
                       {"training_step": 0}, {"review": {"approved": True, "weights_sha256": "other"}}]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path / "generator.safetensors").write_bytes((self.path / "generator.safetensors").read_bytes())
                (path / "model.json").write_text(json.dumps({**self.metadata, **change}))
                with self.assertRaises(ValueError):
                    Model(path, threads=1)


if __name__ == "__main__":
    unittest.main()
