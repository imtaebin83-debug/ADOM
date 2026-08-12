from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
ADOM_DATA_ROOT = REPO_ROOT / "src" / "data" / "adom_data"
SCRIPT_ROOT = ADOM_DATA_ROOT / "scripts"


def save_png(path: Path, size: tuple[int, int] = (4, 3)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(path)


def save_mask(path: Path, colors: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (len(colors), 1))
    image.putdata(colors)
    image.save(path)


def write_split_config(path: Path, values: dict[str, list[str]]) -> None:
    path.write_text(
        json.dumps({"format_version": 1, "splits": values}),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_upload_manifest(source: Path) -> None:
    dates: dict[str, object] = {}
    for date_root in sorted(path for path in source.iterdir() if path.is_dir()):
        records = []
        for raw_path in sorted((date_root / "raw").rglob("*.png")):
            relative_path = raw_path.relative_to(date_root / "raw")
            mask_path = date_root / "masks" / relative_path
            records.append(
                {
                    "relative_path": relative_path.as_posix(),
                    "raw_sha256": file_sha256(raw_path),
                    "mask_sha256": file_sha256(mask_path),
                }
            )
        dates[date_root.name] = {"pairs": len(records), "files": records}
    (source / "manifest.json").write_text(
        json.dumps({"format_version": 1, "dates": dates}),
        encoding="utf-8",
    )


class AdomDataPreprocessingTests(unittest.TestCase):
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

    def test_semantic20_conversion_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "semantic20"
            split_config = root / "splits.json"
            sequences = {
                "train": ["260810/train-session"],
                "val": ["260811_1/val-session"],
                "test": ["260811_3/test-session"],
            }
            write_split_config(split_config, sequences)
            fixtures = {
                ("260810", "train-session"): [
                    (0, 0, 0),
                    (204, 153, 51),
                    (50, 183, 250),
                ],
                ("260811_1", "val-session"): [
                    (0, 0, 0),
                    (202, 88, 23),
                    (36, 179, 83),
                ],
                ("260811_3", "test-session"): [
                    (61, 61, 245),
                    (178, 80, 80),
                    (0, 0, 0),
                ],
            }
            for (date, session), colors in fixtures.items():
                relative = Path(session) / "frame_000001.png"
                save_png(source / date / "raw" / relative, (3, 1))
                save_mask(source / date / "masks" / relative, colors)
            write_upload_manifest(source)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_ROOT / "convert_semantic20.py"),
                    "--input-root",
                    str(source),
                    "--output-root",
                    str(output),
                    "--splits",
                    str(split_config),
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Observed target IDs: [0, 10, 11, 13, 18, 255]", result.stdout)

            expected = {
                "260810/train-session/frame_000001.png": {0, 10, 255},
                "260811_1/val-session/frame_000001.png": {13, 18, 255},
                "260811_3/test-session/frame_000001.png": {11, 18, 255},
            }
            for relative, values in expected.items():
                with Image.open(output / "masks" / relative) as mask:
                    self.assertEqual(mask.mode, "L")
                    self.assertEqual(set(mask.getdata()), values)

            manifest_text = (output / "manifest.csv").read_text(encoding="utf-8")
            self.assertNotIn(str(root), manifest_text)
            validation = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_ROOT / "validate_semantic20_package.py"),
                    "--input-root",
                    str(output),
                    "--write-success-marker",
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(validation.returncode, 0, validation.stderr)
            self.assertIn("ADOM SEMANTIC20 PACKAGE VALID", validation.stdout)
            self.assertTrue((output / "_SUCCESS").is_file())
            self.assertTrue(
                (output / "results" / "validation_report.json").is_file()
            )

    def test_unknown_mask_color_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            split_config = root / "splits.json"
            write_split_config(
                split_config,
                {
                    "train": ["date/train"],
                    "val": ["date/val"],
                    "test": ["date/test"],
                },
            )
            for session in ("train", "val", "test"):
                relative = Path(session) / "frame_000001.png"
                save_png(source / "date" / "raw" / relative, (1, 1))
                color = (1, 2, 3) if session == "train" else (0, 0, 0)
                save_mask(source / "date" / "masks" / relative, [color])
            write_upload_manifest(source)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_ROOT / "convert_semantic20.py"),
                    "--input-root",
                    str(source),
                    "--output-root",
                    str(output),
                    "--splits",
                    str(split_config),
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Unknown RGB colors", result.stderr)
            self.assertFalse(output.exists())

    def test_upload_manifest_checksum_mismatch_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            split_config = root / "splits.json"
            write_split_config(
                split_config,
                {
                    "train": ["date/train"],
                    "val": ["date/val"],
                    "test": ["date/test"],
                },
            )
            for session in ("train", "val", "test"):
                relative = Path(session) / "frame.png"
                save_png(source / "date" / "raw" / relative, (1, 1))
                save_mask(source / "date" / "masks" / relative, [(204, 153, 51)])
            write_upload_manifest(source)
            manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
            manifest["dates"]["date"]["files"][0]["raw_sha256"] = "0" * 64
            (source / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_ROOT / "convert_semantic20.py"),
                    "--input-root",
                    str(source),
                    "--output-root",
                    str(output),
                    "--splits",
                    str(split_config),
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("checksum mismatch", result.stderr)
            self.assertFalse(output.exists())

    def test_all_ignore_train_mask_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            split_config = root / "splits.json"
            write_split_config(
                split_config,
                {
                    "train": ["date/train"],
                    "val": ["date/val"],
                    "test": ["date/test"],
                },
            )
            for session in ("train", "val", "test"):
                relative = Path(session) / "frame.png"
                save_png(source / "date" / "raw" / relative, (1, 1))
                color = (0, 0, 0) if session == "train" else (204, 153, 51)
                save_mask(source / "date" / "masks" / relative, [color])
            write_upload_manifest(source)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_ROOT / "convert_semantic20.py"),
                    "--input-root",
                    str(source),
                    "--output-root",
                    str(output),
                    "--splits",
                    str(split_config),
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Train contains all-ignore masks", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
