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
            command = " ".join(
                (
                    "ADOM_MODEL_CONFIG=/tmp/stale-export-config.py",
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
        self.assertIn("ignoring ADOM_MODEL_CONFIG", result.stderr)
        self.assertNotIn("model config", result.stderr.splitlines()[-1])
        self.assertIn("checkpoint SHA256 mismatch", result.stderr)
        self.assertNotIn("ros2 launch", result.stdout + result.stderr)

    def test_canonical_hash_enables_trusted_mmengine_compatibility(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        hash_check = launcher.index(
            'if [[ "$adom_actual_checkpoint_sha" != "$adom_expected_checkpoint_sha" ]]'
        )
        compatibility = launcher.index(
            'if [[ "$adom_actual_checkpoint_sha" == "$adom_checkpoint_sha_default" ]]'
        )
        self.assertLess(hash_check, compatibility)
        self.assertIn("unset TORCH_FORCE_WEIGHTS_ONLY_LOAD", launcher)
        self.assertIn("export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1", launcher)
        self.assertIn('adom_model_config="$adom_model_config_default"', launcher)
        self.assertNotIn(
            'adom_model_config="${ADOM_MODEL_CONFIG:-',
            launcher,
        )


if __name__ == "__main__":
    unittest.main()
