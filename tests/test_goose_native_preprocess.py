from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "src" / "data" / "goose" / "scripts" / "01_materialize_native.py"


def png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


class GooseNativePreprocessTests(unittest.TestCase):
    def test_preserves_label_bytes_uses_relative_manifest_and_excludes_nir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_root = root / "archives"
            archive_root.mkdir()
            rgb = np.full((4, 5, 3), 127, dtype=np.uint8)
            mask = np.array(
                [
                    [0, 2, 19, 33, 54],
                    [63, 2, 19, 33, 54],
                    [0, 2, 19, 33, 54],
                    [63, 2, 19, 33, 54],
                ],
                dtype=np.uint8,
            )
            label_bytes = png_bytes(mask)
            with zipfile.ZipFile(archive_root / "goose_2d_val.zip", "w") as archive:
                archive.writestr("images/val/scene/sample_windshield_vis.png", png_bytes(rgb))
                archive.writestr("images/val/scene/sample_windshield_nir.png", png_bytes(rgb))
                archive.writestr("labels/val/scene/sample_labelids.png", label_bytes)
                archive.writestr("labels/val/scene/sample_color.png", png_bytes(rgb))

            output_root = root / "output"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--archive-root",
                    str(archive_root),
                    "--output-root",
                    str(output_root),
                    "--splits",
                    "val",
                    "--skip-archive-verification",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                (output_root / "labels/val/scene/sample_labelids.png").read_bytes(),
                label_bytes,
            )
            self.assertFalse((output_root / "images/val/scene/sample_windshield_nir.png").exists())
            with (output_root / "metadata/pair_manifest.csv").open(
                "r", encoding="utf-8", newline=""
            ) as file:
                row = next(csv.DictReader(file))
            self.assertFalse(Path(row["output_image"]).is_absolute())
            self.assertFalse(Path(row["output_label"]).is_absolute())
            self.assertEqual(row["source_archive"], "goose_2d_val.zip")
            summary = json.loads(
                (output_root / "metadata/preprocess_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["archive_verification"], "SKIPPED")
            self.assertNotIn(str(root), json.dumps(summary))


if __name__ == "__main__":
    unittest.main()
