from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np


EXPECTED_INPUT_NAME = "input"
EXPECTED_INPUT_SHAPE = (1, 3, 384, 640)
EXPECTED_OUTPUT_NAME = "output"
EXPECTED_OUTPUT_SHAPE = (1, 19, 384, 640)
DEFAULT_MINIMUM_IMAGES = 10
DEFAULT_MINIMUM_AGREEMENT = 0.99
DEFAULT_MAXIMUM_AREA_DIFFERENCE_PP = 0.2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_pairs(reference_io: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for input_path in sorted(reference_io.glob("*_input.npy")):
        prefix = input_path.name[: -len("_input.npy")]
        mask_path = reference_io / f"{prefix}_onnx_mask.png"
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"ONNX reference mask is missing for {input_path.name}: {mask_path}"
            )
        pairs.append((input_path, mask_path))
    return pairs


def valid_region_mask(
    shape: tuple[int, int], region: Sequence[int] | None
) -> np.ndarray:
    height, width = shape
    top, left, valid_height, valid_width = (
        (0, 0, height, width) if region is None else tuple(int(x) for x in region)
    )
    if (
        top < 0
        or left < 0
        or valid_height <= 0
        or valid_width <= 0
        or top + valid_height > height
        or left + valid_width > width
    ):
        raise ValueError(f"Invalid valid region {region} for mask shape {shape}")
    valid = np.zeros(shape, dtype=bool)
    valid[top : top + valid_height, left : left + valid_width] = True
    return valid


def compare_masks(
    onnx_mask: np.ndarray,
    tensorrt_mask: np.ndarray,
    *,
    valid_region: Sequence[int] | None,
    num_classes: int = 19,
) -> dict[str, Any]:
    if onnx_mask.shape != tensorrt_mask.shape:
        raise ValueError(
            f"Mask shape mismatch: ONNX={onnx_mask.shape}, TRT={tensorrt_mask.shape}"
        )
    if onnx_mask.ndim != 2:
        raise ValueError(f"Expected 2D masks, got {onnx_mask.shape}")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    for name, mask in (("ONNX", onnx_mask), ("TRT", tensorrt_mask)):
        if mask.size and (int(mask.min()) < 0 or int(mask.max()) >= num_classes):
            raise ValueError(f"{name} mask contains an ID outside 0..{num_classes - 1}")

    valid = valid_region_mask(tuple(onnx_mask.shape), valid_region)
    full_matches = int(np.count_nonzero(onnx_mask == tensorrt_mask))
    valid_matches = int(np.count_nonzero((onnx_mask == tensorrt_mask) & valid))
    class_reports: list[dict[str, Any]] = []
    for class_id in range(num_classes):
        onnx_ratio = float(np.mean(onnx_mask[valid] == class_id))
        tensorrt_ratio = float(np.mean(tensorrt_mask[valid] == class_id))
        class_reports.append(
            {
                "class_id": class_id,
                "onnx_area_ratio": onnx_ratio,
                "tensorrt_area_ratio": tensorrt_ratio,
                "absolute_difference_percentage_points": abs(
                    onnx_ratio - tensorrt_ratio
                )
                * 100.0,
            }
        )
    return {
        "full_pixel_argmax_agreement": full_matches / onnx_mask.size,
        "valid_pixel_argmax_agreement": valid_matches / int(np.count_nonzero(valid)),
        "matching_pixels": full_matches,
        "total_pixels": int(onnx_mask.size),
        "valid_matching_pixels": valid_matches,
        "valid_total_pixels": int(np.count_nonzero(valid)),
        "maximum_class_area_difference_percentage_points": max(
            item["absolute_difference_percentage_points"] for item in class_reports
        ),
        "class_area_ratios": class_reports,
    }


def percentile_summary(values_ms: Sequence[float]) -> dict[str, float]:
    values = np.asarray(values_ms, dtype=np.float64)
    if values.size == 0:
        raise ValueError("At least one latency sample is required")
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "minimum_ms": float(np.min(values)),
        "maximum_ms": float(np.max(values)),
    }


class TensorRTRunner:
    def __init__(self, engine_path: Path) -> None:
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        self._trt = trt
        self._cudart = cudart
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"TensorRT could not deserialize {engine_path}")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("TensorRT could not create an execution context")

        inputs: list[str] = []
        outputs: list[str] = []
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                inputs.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                outputs.append(name)
        if inputs != [EXPECTED_INPUT_NAME] or outputs != [EXPECTED_OUTPUT_NAME]:
            raise RuntimeError(
                f"Unexpected engine I/O names: inputs={inputs}, outputs={outputs}"
            )

        input_shape = tuple(int(x) for x in engine.get_tensor_shape(inputs[0]))
        output_shape = tuple(int(x) for x in engine.get_tensor_shape(outputs[0]))
        input_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(inputs[0])))
        output_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(outputs[0])))
        if input_shape != EXPECTED_INPUT_SHAPE or output_shape != EXPECTED_OUTPUT_SHAPE:
            raise RuntimeError(
                f"Unexpected engine shapes: input={input_shape}, output={output_shape}"
            )
        if input_dtype != np.float32 or output_dtype != np.float32:
            raise RuntimeError(
                f"Expected FP32 engine I/O, got input={input_dtype}, output={output_dtype}"
            )

        self._runtime = runtime
        self._engine = engine
        self._context = context
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self._output = np.empty(output_shape, dtype=output_dtype)
        self._stream = self._cuda_value(cudart.cudaStreamCreate(), "create stream")
        self._input_device = self._cuda_value(
            cudart.cudaMalloc(int(np.prod(input_shape)) * input_dtype.itemsize),
            "allocate input",
        )
        self._output_device = self._cuda_value(
            cudart.cudaMalloc(self._output.nbytes), "allocate output"
        )
        if not context.set_tensor_address(inputs[0], int(self._input_device)):
            raise RuntimeError("Failed to bind TensorRT input address")
        if not context.set_tensor_address(outputs[0], int(self._output_device)):
            raise RuntimeError("Failed to bind TensorRT output address")
        self.contract = {
            "inputs": [
                {
                    "name": inputs[0],
                    "shape": list(input_shape),
                    "dtype": str(input_dtype),
                }
            ],
            "outputs": [
                {
                    "name": outputs[0],
                    "shape": list(output_shape),
                    "dtype": str(output_dtype),
                }
            ],
        }

    def _cuda_value(self, result: tuple[Any, ...], action: str) -> Any:
        error, *values = result
        if error != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA failed to {action}: {error}")
        if len(values) != 1:
            raise RuntimeError(f"Unexpected CUDA result while trying to {action}")
        return values[0]

    def _cuda_ok(self, result: tuple[Any, ...], action: str) -> None:
        error = result[0]
        if error != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA failed to {action}: {error}")

    def infer(self, input_tensor: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        tensor = np.ascontiguousarray(input_tensor, dtype=self.input_dtype)
        if tensor.shape != self.input_shape:
            raise ValueError(
                f"Expected input shape {self.input_shape}, got {tensor.shape}"
            )
        kind = self._cudart.cudaMemcpyKind
        total_start = time.perf_counter_ns()
        start = time.perf_counter_ns()
        self._cuda_ok(
            self._cudart.cudaMemcpyAsync(
                int(self._input_device),
                int(tensor.ctypes.data),
                tensor.nbytes,
                kind.cudaMemcpyHostToDevice,
                self._stream,
            ),
            "copy input to device",
        )
        self._cuda_ok(
            self._cudart.cudaStreamSynchronize(self._stream), "synchronize input copy"
        )
        h2d_ms = (time.perf_counter_ns() - start) / 1e6

        start = time.perf_counter_ns()
        if not self._context.execute_async_v3(self._stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        self._cuda_ok(
            self._cudart.cudaStreamSynchronize(self._stream),
            "synchronize inference",
        )
        engine_ms = (time.perf_counter_ns() - start) / 1e6

        start = time.perf_counter_ns()
        self._cuda_ok(
            self._cudart.cudaMemcpyAsync(
                int(self._output.ctypes.data),
                int(self._output_device),
                self._output.nbytes,
                kind.cudaMemcpyDeviceToHost,
                self._stream,
            ),
            "copy output to host",
        )
        self._cuda_ok(
            self._cudart.cudaStreamSynchronize(self._stream), "synchronize output copy"
        )
        d2h_ms = (time.perf_counter_ns() - start) / 1e6
        total_ms = (time.perf_counter_ns() - total_start) / 1e6
        return self._output.copy(), {
            "h2d_ms": h2d_ms,
            "engine_ms": engine_ms,
            "d2h_ms": d2h_ms,
            "runtime_total_ms": total_ms,
        }

    def close(self) -> None:
        if getattr(self, "_input_device", None) is not None:
            self._cuda_ok(self._cudart.cudaFree(self._input_device), "free input")
            self._input_device = None
        if getattr(self, "_output_device", None) is not None:
            self._cuda_ok(self._cudart.cudaFree(self._output_device), "free output")
            self._output_device = None
        if getattr(self, "_stream", None) is not None:
            self._cuda_ok(
                self._cudart.cudaStreamDestroy(self._stream), "destroy stream"
            )
            self._stream = None

    def __enter__(self) -> TensorRTRunner:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _load_valid_regions(parity_report: Path, count: int) -> list[list[int] | None]:
    report = json.loads(parity_report.read_text(encoding="utf-8"))
    images = report.get("images", [])
    if len(images) != count:
        raise ValueError(
            f"Source parity report has {len(images)} images; expected {count}"
        )
    return [
        item.get("preprocess", {}).get("valid_region_top_left_height_width")
        for item in images
    ]


def _load_mask(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image, dtype=np.uint8)


def _load_palette(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if [item.get("id") for item in entries] != list(range(19)):
        raise ValueError("Palette must contain ordered Semantic20 IDs 0..18")
    return np.asarray([item["rgb"] for item in entries], dtype=np.uint8)


def _write_visualization(
    output_dir: Path,
    prefix: str,
    input_tensor: np.ndarray,
    mask: np.ndarray,
    palette: np.ndarray,
) -> list[str]:
    from PIL import Image

    rgb = input_tensor[0].transpose(1, 2, 0)
    rgb = np.clip(
        rgb * np.asarray([58.395, 57.12, 57.375])
        + np.asarray([123.675, 116.28, 103.53]),
        0,
        255,
    ).astype(np.uint8)
    color = palette[mask]
    overlay = np.rint(rgb * 0.55 + color * 0.45).astype(np.uint8)
    paths = {
        "mask": output_dir / f"{prefix}_tensorrt_mask.png",
        "color": output_dir / f"{prefix}_tensorrt_color.png",
        "overlay": output_dir / f"{prefix}_tensorrt_overlay.png",
    }
    Image.fromarray(mask.astype(np.uint8)).save(paths["mask"])
    Image.fromarray(color, mode="RGB").save(paths["color"])
    Image.fromarray(overlay, mode="RGB").save(paths["overlay"])
    return [path.name for path in paths.values()]


def validate_reference_io(args: argparse.Namespace) -> dict[str, Any]:
    pairs = reference_pairs(args.reference_io)
    if len(pairs) < args.minimum_images:
        raise ValueError(
            f"At least {args.minimum_images} reference tensors are required; "
            f"found {len(pairs)}"
        )
    regions = _load_valid_regions(args.source_parity_report, len(pairs))
    palette = _load_palette(args.palette)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "onnx-tensorrt-parity.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {report_path}")

    with TensorRTRunner(args.engine) as runner:
        first_tensor = np.load(pairs[0][0], allow_pickle=False)
        for _ in range(args.warmup):
            runner.infer(first_tensor)

        image_reports: list[dict[str, Any]] = []
        full_matches = full_pixels = valid_matches = valid_pixels = 0
        visualization_files: list[str] = []
        for index, ((input_path, mask_path), region) in enumerate(zip(pairs, regions)):
            tensor = np.load(input_path, allow_pickle=False)
            logits, timing = runner.infer(tensor)
            if logits.shape != EXPECTED_OUTPUT_SHAPE or not np.isfinite(logits).all():
                raise RuntimeError(
                    f"Invalid TensorRT logits for {input_path.name}: {logits.shape}"
                )
            tensorrt_mask = np.argmax(logits, axis=1)[0].astype(np.uint8)
            comparison = compare_masks(
                _load_mask(mask_path),
                tensorrt_mask,
                valid_region=region,
            )
            full_matches += comparison["matching_pixels"]
            full_pixels += comparison["total_pixels"]
            valid_matches += comparison["valid_matching_pixels"]
            valid_pixels += comparison["valid_total_pixels"]
            prefix = input_path.name[: -len("_input.npy")]
            generated: list[str] = []
            if index < args.visualization_count:
                generated = _write_visualization(
                    args.output_dir, prefix, tensor, tensorrt_mask, palette
                )
                visualization_files.extend(generated)
            image_reports.append(
                {
                    "index": index,
                    "input": input_path.name,
                    "onnx_reference_mask": mask_path.name,
                    "valid_region_top_left_height_width": region,
                    "finite_logits": True,
                    **comparison,
                    "timing": timing,
                    "generated_files": generated,
                }
            )

        benchmark_samples: list[dict[str, float]] = []
        for _ in range(args.benchmark_iterations):
            _, timing = runner.infer(first_tensor)
            benchmark_samples.append(timing)
        benchmark = {
            key: percentile_summary([sample[key] for sample in benchmark_samples])
            for key in ("h2d_ms", "engine_ms", "d2h_ms", "runtime_total_ms")
        }
        benchmark["iterations"] = args.benchmark_iterations
        benchmark["derived_runtime_fps_from_mean"] = 1000.0 / benchmark[
            "runtime_total_ms"
        ]["mean_ms"]

        overall_full = full_matches / full_pixels
        overall_valid = valid_matches / valid_pixels
        minimum_valid = min(
            item["valid_pixel_argmax_agreement"] for item in image_reports
        )
        maximum_area_difference = max(
            item["maximum_class_area_difference_percentage_points"]
            for item in image_reports
        )
        passed = (
            overall_valid >= args.minimum_agreement
            and minimum_valid >= args.minimum_agreement
            and maximum_area_difference <= args.maximum_area_difference_pp
        )
        report = {
            "schema_version": "semantic20-onnx-tensorrt-parity-v1",
            "status": "PASS" if passed else "FAIL",
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "tensorrt": runner._trt.__version__,
            },
            "engine": {
                "filename": args.engine.name,
                "size_bytes": args.engine.stat().st_size,
                "sha256": sha256_file(args.engine),
                "precision": "FP16 internal, FP32 I/O",
            },
            "contract": runner.contract,
            "reference": {
                "kind": "frozen ONNX argmax masks from hand-off package",
                "source_parity_report": args.source_parity_report.name,
                "num_images": len(image_reports),
                "reported_class_ids": list(range(19)),
                "roi_evaluated": False,
            },
            "thresholds": {
                "minimum_images": args.minimum_images,
                "minimum_valid_pixel_argmax_agreement": args.minimum_agreement,
                "maximum_class_area_difference_percentage_points": (
                    args.maximum_area_difference_pp
                ),
            },
            "summary": {
                "overall_full_pixel_argmax_agreement": overall_full,
                "overall_valid_pixel_argmax_agreement": overall_valid,
                "minimum_per_image_valid_pixel_argmax_agreement": minimum_valid,
                "maximum_class_area_difference_percentage_points": (
                    maximum_area_difference
                ),
                "all_finite_logits": True,
                "visualization_files": visualization_files,
            },
            "file_inference_benchmark": benchmark,
            "images": image_reports,
        }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"report": str(report_path), **report["summary"]}, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a target-Jetson TensorRT engine against frozen Semantic20 "
            "ONNX reference I/O"
        )
    )
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--reference-io", required=True, type=Path)
    parser.add_argument("--source-parity-report", required=True, type=Path)
    parser.add_argument("--palette", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-images", type=int, default=DEFAULT_MINIMUM_IMAGES)
    parser.add_argument(
        "--minimum-agreement", type=float, default=DEFAULT_MINIMUM_AGREEMENT
    )
    parser.add_argument(
        "--maximum-area-difference-pp",
        type=float,
        default=DEFAULT_MAXIMUM_AREA_DIFFERENCE_PP,
    )
    parser.add_argument("--visualization-count", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--benchmark-iterations", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    for path in (args.engine, args.source_parity_report, args.palette):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")
    if not args.reference_io.is_dir():
        parser.error(f"reference directory does not exist: {args.reference_io}")
    if args.minimum_images < 1 or args.warmup < 0 or args.benchmark_iterations < 1:
        parser.error("minimum-images/benchmark-iterations must be positive; warmup >= 0")
    report = validate_reference_io(args)
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
