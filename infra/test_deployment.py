"""Exercise the promotion and restricted-command gates without a cloud server."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        data = b"fixture; runtime independently validates safetensors structure"
        (self.folder / "generator.safetensors").write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        self.manifest = {"schema_version": 1, "training_step": 1,
                         "weights_sha256": digest,
                         "review": {"approved": True, "weights_sha256": digest}}

    def validate(self):
        (self.folder / "model.json").write_text(json.dumps(self.manifest))
        return subprocess.run(["python3", str(ROOT / "infra/validate-model.py"), str(self.folder)],
                              capture_output=True, text=True)

    def test_valid_review(self):
        result = self.validate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), self.manifest["weights_sha256"])

    def test_review_cannot_be_reused_for_different_weights(self):
        self.manifest["review"]["weights_sha256"] = "0" * 64
        self.assertNotEqual(self.validate().returncode, 0)

    def test_weight_corruption_is_rejected(self):
        (self.folder / "generator.safetensors").write_bytes(b"corrupted")
        self.assertNotEqual(self.validate().returncode, 0)

    def test_untrained_and_unreviewed_models_are_rejected(self):
        for patch in ({"training_step": 0}, {"review": {"approved": False}}, {"schema_version": 2}):
            with self.subTest(patch=patch):
                original = self.manifest.copy()
                self.manifest.update(patch)
                self.assertNotEqual(self.validate().returncode, 0)
                self.manifest = original

    def test_ssh_wrapper_dispatches_only_the_validated_revision(self):
        executable = self.folder / "sudo"
        executable.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
        executable.chmod(0o755)
        revision = "a" * 40
        result = subprocess.run(["bash", str(ROOT / "infra/ssh-command.sh")],
                                env={**os.environ, "PATH": str(self.folder) + ":" + os.environ["PATH"],
                                     "SSH_ORIGINAL_COMMAND": "deploy " + revision},
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["-n", "/usr/local/sbin/priest-deploy", revision])

    def test_ssh_wrapper_rejects_shell_and_invalid_revisions(self):
        for command in ("", "bash", "deploy main", "deploy " + "a" * 40 + "; id",
                        "deploy " + "a" * 40 + " extra"):
            with self.subTest(command=command):
                result = subprocess.run(["bash", str(ROOT / "infra/ssh-command.sh")],
                                        env={**os.environ, "SSH_ORIGINAL_COMMAND": command},
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
                self.assertIn("Only deploy", result.stderr)


if __name__ == "__main__":
    unittest.main()
