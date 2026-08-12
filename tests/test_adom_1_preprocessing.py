from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "src" / "data" / "adom_1" / "scripts"


def save_png(path: Path, size: tuple[int, int] = (4, 3)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(path)


class Adom1PreprocessingTests(unittest.TestCase):
    def test_sync_uses_session_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            masks = root / "masks"
            output = root / "output"
            for session in ("session-a", "session-b"):
                save_png(raw / session / "frame_000001.png")
                save_png(masks / session / "frame_000001.png")
            save_png(raw / "session-a" / "frame_000002.png")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_ROOT / "sync_raw_masks.py"),
                    "--raw",
                    str(raw),
                    "--masks",
                    str(masks),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Raw without mask: 1", result.stdout)
            for session in ("session-a", "session-b"):
                self.assertTrue(output.joinpath("raw", session, "frame_000001.png").is_file())
                self.assertTrue(output.joinpath("masks", session, "frame_000001.png").is_file())
            self.assertFalse(output.joinpath("raw", "session-a", "frame_000002.png").exists())

    def test_upload_manifest_contains_only_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = source / "upload"
            date = source / "260811_1"
            relative = Path("session-a") / "frame_000001.png"
            save_png(date / "normalized" / "raw" / relative)
            save_png(date / "normalized" / "masks" / relative)
            (date / "labelmap.txt").write_text("background:0,0,0::\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_ROOT / "build_upload_package.py"),
                    "--source-root",
                    str(source),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest_text = (output / "manifest.json").read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)
            record = manifest["dates"]["260811_1"]["files"][0]
            self.assertEqual(record["relative_path"], "session-a/frame_000001.png")
            self.assertNotIn(str(root), manifest_text)
            self.assertTrue((output / "260811_1" / "labelmap.txt").is_file())


if __name__ == "__main__":
    unittest.main()
