from __future__ import annotations

import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_jetson_t4.sh"


class JetsonModelProfileTests(unittest.TestCase):
    def test_runtime_configs_compile_and_eadom_delegates_to_b0(self) -> None:
        b0 = (
            REPO_ROOT
            / "configs"
            / "adom"
            / "runtime"
            / "segformer_b0_640x384_rellis3d.py"
        )
        eadom = b0.with_name("segformer_b0_640x384_eadom.py")
        compile(b0.read_text(encoding="utf-8"), str(b0), "exec")
        eadom_text = eadom.read_text(encoding="utf-8")
        compile(eadom_text, str(eadom), "exec")
        self.assertIn("segformer_b0_640x384_rellis3d.py", eadom_text)
        self.assertIn("eadom-b0-seed42-iter26000", eadom_text)

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_profile_argument_is_required(self) -> None:
        result = subprocess.run(
            ["bash", LAUNCHER.relative_to(REPO_ROOT).as_posix()],
            capture_output=True,
            check=False,
            cwd=REPO_ROOT,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("{b0-e0|eadom}", result.stderr)

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_checkpoint_hash_mismatch_fails_before_ros_launch(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as directory:
            checkpoint = Path(directory) / "checkpoint.pth"
            checkpoint.write_bytes(b"not-an-adom-checkpoint")
            config = (
                REPO_ROOT
                / "configs"
                / "adom"
                / "runtime"
                / "segformer_b0_640x384_eadom.py"
            ).relative_to(REPO_ROOT).as_posix()
            command = " ".join(
                (
                    f"ADOM_MODEL_CONFIG={shlex.quote(config)}",
                    "ADOM_CHECKPOINT="
                    + shlex.quote(checkpoint.relative_to(REPO_ROOT).as_posix()),
                    "ADOM_EXPECTED_CHECKPOINT_SHA256=" + "0" * 64,
                    "bash",
                    shlex.quote(LAUNCHER.relative_to(REPO_ROOT).as_posix()),
                    "eadom",
                )
            )
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                check=False,
                cwd=REPO_ROOT,
                text=True,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("checkpoint SHA256 mismatch", result.stderr)
        self.assertNotIn("ros2 launch", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
