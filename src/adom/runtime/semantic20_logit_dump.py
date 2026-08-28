from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from adom.runtime.semantic20_tensorrt import EXPECTED_INPUT_SHAPE, TensorRTRunner


def input_tensors(root: Path) -> list[Path]:
    paths = sorted(root.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy input tensors found in {root}")
    return paths


def dump_logits(args: argparse.Namespace) -> dict[str, object]:
    paths = input_tensors(args.input_dir)
    if args.maximum_inputs is not None:
        paths = paths[: args.maximum_inputs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    with TensorRTRunner(args.engine) as runner:
        for path in paths:
            tensor = np.load(path, allow_pickle=False)
            if tensor.shape != EXPECTED_INPUT_SHAPE:
                raise ValueError(
                    f"Expected input {EXPECTED_INPUT_SHAPE}, got {tensor.shape}: {path}"
                )
            logits, timing = runner.infer(tensor)
            if not np.isfinite(logits).all():
                raise RuntimeError(f"TensorRT produced non-finite logits for {path}")
            stem = path.stem.removesuffix("_input")
            logits_path = args.output_dir / f"{stem}_logits.npy"
            mask_path = args.output_dir / f"{stem}_mask.png"
            if not args.overwrite and (logits_path.exists() or mask_path.exists()):
                raise FileExistsError(f"Refusing to overwrite outputs for {stem}")
            np.save(logits_path, logits, allow_pickle=False)
            prediction = np.argmax(logits, axis=1)[0].astype(np.uint8)
            Image.fromarray(prediction, mode="L").save(mask_path)
            records.append(
                {
                    "sample_id": stem,
                    "input_tensor": path.name,
                    "logits_path": logits_path.name,
                    "mask_path": mask_path.name,
                    "shape": list(logits.shape),
                    "finite_logits": bool(np.isfinite(logits).all()),
                    "timing": timing,
                }
            )
    report = {
        "schema_version": "semantic20-tensorrt-logit-dump-v1",
        "engine": args.engine.name,
        "num_samples": len(records),
        "warning": "Raw logits are large generated artifacts and must not be committed.",
        "samples": records,
    }
    report_path = args.output_dir / "logit-dump-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"report": str(report_path), "num_samples": len(records)}, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Dump raw Semantic20 TensorRT logits for frozen input tensors"
    )
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--maximum-inputs", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.engine.is_file():
        parser.error(f"engine does not exist: {args.engine}")
    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")
    if args.maximum_inputs is not None and args.maximum_inputs < 1:
        parser.error("maximum-inputs must be positive")
    dump_logits(args)


if __name__ == "__main__":
    main()
