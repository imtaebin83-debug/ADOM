from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_MINIMUM_IMAGES = 10
DEFAULT_EXPECTED_NUM_CLASSES = 19
DEFAULT_VISUALIZATION_COUNT = 3
MAX_ABSOLUTE_ERROR = 1e-3
MINIMUM_ARGMAX_AGREEMENT = 0.999


def _as_input_tensor(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise RuntimeError(f"Expected one input tensor, got {len(value)}")
        value = value[0]
    return value


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _preprocess_metadata(data_samples: Any) -> dict[str, Any]:
    if not data_samples:
        return {}
    sample = data_samples[0] if isinstance(data_samples, (list, tuple)) else data_samples
    metainfo = getattr(sample, "metainfo", {})
    keys = (
        "ori_shape",
        "img_shape",
        "pad_shape",
        "padding_size",
        "scale_factor",
        "flip",
        "flip_direction",
    )
    return {key: _jsonable(metainfo[key]) for key in keys if key in metainfo}


def parse_normalized_polygon(value: str | None) -> list[tuple[float, float]] | None:
    if value is None:
        return None
    try:
        vertices = [
            tuple(float(component) for component in item.split(","))
            for item in value.split(";")
        ]
    except ValueError as error:
        raise ValueError(
            "ROI polygon must use normalized 'x,y;x,y;...' coordinates"
        ) from error
    if len(vertices) < 3 or any(len(vertex) != 2 for vertex in vertices):
        raise ValueError("ROI polygon must contain at least three x,y vertices")
    if any(not 0.0 <= coordinate <= 1.0 for vertex in vertices for coordinate in vertex):
        raise ValueError("ROI polygon coordinates must be within [0, 1]")
    return [(vertex[0], vertex[1]) for vertex in vertices]


def normalized_polygon_mask(
    shape: tuple[int, int],
    vertices: Sequence[tuple[float, float]],
    valid_region: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid mask shape: {shape}")
    top, left, valid_height, valid_width = valid_region or (0, 0, height, width)
    if (
        top < 0
        or left < 0
        or valid_height <= 0
        or valid_width <= 0
        or top + valid_height > height
        or left + valid_width > width
    ):
        raise ValueError(f"Invalid valid region {valid_region} for shape {shape}")
    polygon = np.asarray(
        [
            (left + x * (valid_width - 1), top + y * (valid_height - 1))
            for x, y in vertices
        ],
        dtype=np.float64,
    )
    grid_y, grid_x = np.mgrid[:height, :width]
    inside = np.zeros((height, width), dtype=bool)
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        crosses = (y1 > grid_y) != (y2 > grid_y)
        intersection_x = (x2 - x1) * (grid_y - y1) / (y2 - y1 + 1e-12) + x1
        inside ^= crosses & (grid_x < intersection_x)
        previous = current
    return inside


def keep_ratio_valid_region(
    original_shape: tuple[int, int], output_shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Return the top-left valid image region for keep-ratio + bottom/right pad."""
    original_height, original_width = original_shape
    output_height, output_width = output_shape
    if min(original_height, original_width, output_height, output_width) <= 0:
        raise ValueError(
            f"Invalid keep-ratio shapes: original={original_shape}, output={output_shape}"
        )
    scale = min(output_width / original_width, output_height / original_height)
    valid_width = min(output_width, int(original_width * scale + 0.5))
    valid_height = min(output_height, int(original_height * scale + 0.5))
    return (0, 0, valid_height, valid_width)


def _class_area_report(
    torch_prediction: np.ndarray,
    onnx_prediction: np.ndarray,
    class_ids: Sequence[int],
    valid_region: tuple[int, int, int, int],
    roi_polygon: Sequence[tuple[float, float]] | None,
) -> list[dict[str, Any]]:
    top, left, valid_height, valid_width = valid_region
    valid = np.zeros(torch_prediction.shape, dtype=bool)
    valid[top : top + valid_height, left : left + valid_width] = True
    roi = (
        normalized_polygon_mask(torch_prediction.shape, roi_polygon, valid_region)
        if roi_polygon is not None
        else None
    )
    reports: list[dict[str, Any]] = []
    for class_id in class_ids:
        full_torch = float(np.mean(torch_prediction[valid] == class_id))
        full_onnx = float(np.mean(onnx_prediction[valid] == class_id))
        value: dict[str, Any] = {
            "class_id": class_id,
            "valid_image": {
                "pytorch_area_ratio": full_torch,
                "onnx_area_ratio": full_onnx,
                "absolute_difference_percentage_points": abs(
                    full_torch - full_onnx
                )
                * 100.0,
            },
        }
        if roi is None:
            value["roi"] = {"status": "NOT_EVALUATED"}
        else:
            torch_ratio = float(np.mean(torch_prediction[roi] == class_id))
            onnx_ratio = float(np.mean(onnx_prediction[roi] == class_id))
            value["roi"] = {
                "status": "EVALUATED",
                "pixel_count": int(np.count_nonzero(roi)),
                "pytorch_area_ratio": torch_ratio,
                "onnx_area_ratio": onnx_ratio,
                "absolute_difference_percentage_points": abs(
                    torch_ratio - onnx_ratio
                )
                * 100.0,
            }
        reports.append(value)
    return reports


class ParityRunner:
    def __init__(
        self,
        deploy_config: Path,
        model_config: Path,
        checkpoint: Path,
        onnx_path: Path,
        device: str,
        expected_num_classes: int,
    ) -> None:
        import onnx
        import onnxruntime as ort
        from mmdeploy.apis.utils import build_task_processor
        from mmdeploy.utils import load_config

        self._torch = __import__("torch")
        self._expected_num_classes = expected_num_classes
        deploy_cfg, model_cfg = load_config(str(deploy_config), str(model_config))
        self._deploy_cfg = deploy_cfg
        self._task_processor = build_task_processor(model_cfg, deploy_cfg, device)
        self._pytorch_model = self._task_processor.build_pytorch_model(str(checkpoint))
        self._pytorch_model.eval()

        available_providers = set(ort.get_available_providers())
        providers = []
        if device.startswith("cuda") and "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self._session = ort.InferenceSession(str(onnx_path), providers=providers)

        graph = onnx.load(str(onnx_path), load_external_data=False)
        self.onnx_contract = {
            "opsets": [
                {"domain": item.domain or "ai.onnx", "version": int(item.version)}
                for item in graph.opset_import
            ],
            "inputs": [
                {"name": item.name, "shape": _jsonable(item.shape), "dtype": item.type}
                for item in self._session.get_inputs()
            ],
            "outputs": [
                {"name": item.name, "shape": _jsonable(item.shape), "dtype": item.type}
                for item in self._session.get_outputs()
            ],
            "providers": self._session.get_providers(),
        }

    def compare_image(
        self,
        image: Path,
        *,
        class_ids: Sequence[int],
        roi_polygon: Sequence[tuple[float, float]] | None,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        from PIL import Image

        with Image.open(image) as source:
            original_shape = (source.height, source.width)
        model_inputs, _ = self._task_processor.create_input(
            str(image),
            input_shape=self._deploy_cfg.onnx_config.input_shape,
        )
        processed = self._pytorch_model.data_preprocessor(model_inputs, training=False)
        tensor = _as_input_tensor(processed["inputs"])
        with self._torch.no_grad():
            torch_logits = self._pytorch_model(
                tensor,
                data_samples=processed.get("data_samples"),
                mode="tensor",
            )
        if isinstance(torch_logits, (tuple, list)):
            torch_logits = torch_logits[0]
        torch_array = torch_logits.detach().cpu().numpy()

        input_name = self._session.get_inputs()[0].name
        onnx_array = self._session.run(
            None, {input_name: tensor.detach().cpu().numpy()}
        )[0]
        if (
            torch_array.ndim == 4
            and onnx_array.ndim == 4
            and torch_array.shape[:2] == onnx_array.shape[:2]
            and torch_array.shape[2:] != onnx_array.shape[2:]
        ):
            torch_array = (
                self._torch.nn.functional.interpolate(
                    self._torch.from_numpy(torch_array),
                    size=onnx_array.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                .numpy()
            )
        if torch_array.shape != onnx_array.shape:
            raise RuntimeError(
                f"Output shape mismatch for {image.name}: "
                f"torch={torch_array.shape}, onnx={onnx_array.shape}"
            )
        expected_prefix = (1, self._expected_num_classes)
        if torch_array.ndim != 4 or tuple(torch_array.shape[:2]) != expected_prefix:
            raise RuntimeError(
                f"Expected raw batch-1 logits with {self._expected_num_classes} "
                f"channels for {image.name}, got {torch_array.shape}"
            )
        if not np.isfinite(torch_array).all() or not np.isfinite(onnx_array).all():
            raise RuntimeError(f"Non-finite logits detected for {image.name}")

        absolute_error = np.abs(torch_array - onnx_array)
        torch_prediction = np.argmax(torch_array, axis=1)[0]
        onnx_prediction = np.argmax(onnx_array, axis=1)[0]
        valid_region = keep_ratio_valid_region(
            original_shape, tuple(torch_prediction.shape)
        )
        matching_pixels = int(np.count_nonzero(torch_prediction == onnx_prediction))
        total_pixels = int(torch_prediction.size)
        agreement = matching_pixels / total_pixels
        passed = (
            float(np.max(absolute_error)) <= MAX_ABSOLUTE_ERROR
            and agreement >= MINIMUM_ARGMAX_AGREEMENT
        )
        report = {
            "image": image.name,
            "status": "PASS" if passed else "FAIL",
            "preprocess": {
                "input_shape_nchw": list(tensor.shape),
                "input_dtype": str(tensor.detach().cpu().numpy().dtype),
                "data_sample_metainfo": _preprocess_metadata(
                    processed.get("data_samples")
                ),
                "padding_policy": "right_and_bottom",
                "valid_region_top_left_height_width": list(valid_region),
            },
            "torch_output_shape": list(torch_array.shape),
            "onnx_output_shape": list(onnx_array.shape),
            "finite_logits": True,
            "max_absolute_error": float(np.max(absolute_error)),
            "mean_absolute_error": float(np.mean(absolute_error)),
            "matching_pixels": matching_pixels,
            "total_pixels": total_pixels,
            "pixel_argmax_agreement": agreement,
            "class_area_ratios": _class_area_report(
                torch_prediction,
                onnx_prediction,
                class_ids,
                valid_region,
                roi_polygon,
            ),
            "roi": {
                "status": "EVALUATED" if roi_polygon is not None else "NOT_EVALUATED",
                "coordinate_space": "normalized_unpadded_source_image",
                "normalized_polygon": (
                    [[x, y] for x, y in roi_polygon]
                    if roi_polygon is not None
                    else None
                ),
            },
        }
        return report, torch_prediction, onnx_prediction, tensor.detach().cpu().numpy()


def _write_reference_outputs(
    output_dir: Path,
    index: int,
    image: Path,
    torch_prediction: np.ndarray,
    onnx_prediction: np.ndarray,
    input_tensor: np.ndarray,
    valid_region: tuple[int, int, int, int],
    expected_num_classes: int,
    write_visualization: bool,
) -> None:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{index:03d}_{image.stem}"
    Image.fromarray(torch_prediction.astype(np.uint8)).save(
        output_dir / f"{stem}_pytorch_mask.png"
    )
    Image.fromarray(onnx_prediction.astype(np.uint8)).save(
        output_dir / f"{stem}_onnx_mask.png"
    )
    np.save(output_dir / f"{stem}_input.npy", input_tensor, allow_pickle=False)
    if not write_visualization:
        return

    if expected_num_classes == 19:
        from adom.mmseg.dataset import SEMANTIC20_PALETTE

        palette = np.asarray(SEMANTIC20_PALETTE, dtype=np.uint8)
    else:
        from adom.data.schema import COST4_PALETTE

        palette = np.asarray(
            [COST4_PALETTE[class_id] for class_id in range(expected_num_classes)],
            dtype=np.uint8,
        )
    top, left, valid_height, valid_width = valid_region
    unpadded = onnx_prediction[
        top : top + valid_height, left : left + valid_width
    ].astype(np.uint8)
    with Image.open(image) as source_file:
        source = source_file.convert("RGB")
    resized_mask = Image.fromarray(unpadded).resize(source.size, resample=Image.NEAREST)
    mask_array = np.asarray(resized_mask, dtype=np.uint8)
    color = Image.fromarray(palette[mask_array], mode="RGB")
    overlay = Image.blend(source, color, alpha=0.45)
    color.save(output_dir / f"{stem}_onnx_color.png")
    overlay.save(output_dir / f"{stem}_onnx_overlay.png")


def compare_many(
    deploy_config: Path,
    model_config: Path,
    checkpoint: Path,
    onnx_path: Path,
    images: Sequence[Path],
    device: str,
    *,
    minimum_images: int = DEFAULT_MINIMUM_IMAGES,
    class_ids: Sequence[int] | None = None,
    expected_num_classes: int = DEFAULT_EXPECTED_NUM_CLASSES,
    roi_polygon: Sequence[tuple[float, float]] | None = None,
    reference_output_dir: Path | None = None,
    visualization_count: int = DEFAULT_VISUALIZATION_COUNT,
) -> dict[str, Any]:
    if minimum_images < 1:
        raise ValueError("minimum_images must be positive")
    if expected_num_classes < 1:
        raise ValueError("expected_num_classes must be positive")
    if visualization_count < 0:
        raise ValueError("visualization_count must be non-negative")
    selected_class_ids = (
        list(range(expected_num_classes)) if class_ids is None else list(class_ids)
    )
    if not selected_class_ids or any(
        class_id < 0 or class_id >= expected_num_classes
        for class_id in selected_class_ids
    ):
        raise ValueError(
            f"class_ids must be within 0..{expected_num_classes - 1}: "
            f"{selected_class_ids}"
        )
    resolved_images = [path.resolve() for path in images]
    if len(resolved_images) < minimum_images:
        raise ValueError(
            f"At least {minimum_images} reference images are required, "
            f"got {len(resolved_images)}"
        )
    missing = [str(path) for path in resolved_images if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Reference images not found: {missing}")

    runner = ParityRunner(
        deploy_config,
        model_config,
        checkpoint,
        onnx_path,
        device,
        expected_num_classes,
    )
    image_reports: list[dict[str, Any]] = []
    matching_pixels = 0
    total_pixels = 0
    for index, image in enumerate(resolved_images):
        report, torch_prediction, onnx_prediction, input_tensor = runner.compare_image(
            image,
            class_ids=selected_class_ids,
            roi_polygon=roi_polygon,
        )
        image_reports.append(report)
        matching_pixels += report["matching_pixels"]
        total_pixels += report["total_pixels"]
        if reference_output_dir is not None:
            _write_reference_outputs(
                reference_output_dir,
                index,
                image,
                torch_prediction,
                onnx_prediction,
                input_tensor,
                tuple(report["preprocess"]["valid_region_top_left_height_width"]),
                expected_num_classes,
                index < visualization_count,
            )

    overall_agreement = matching_pixels / total_pixels
    passed = (
        all(report["status"] == "PASS" for report in image_reports)
        and overall_agreement >= MINIMUM_ARGMAX_AGREEMENT
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "num_images": len(image_reports),
        "onnx_contract": runner.onnx_contract,
        "thresholds": {
            "minimum_images": minimum_images,
            "expected_num_classes": expected_num_classes,
            "max_absolute_error": MAX_ABSOLUTE_ERROR,
            "pixel_argmax_agreement": MINIMUM_ARGMAX_AGREEMENT,
        },
        "summary": {
            "overall_pixel_argmax_agreement": overall_agreement,
            "minimum_per_image_argmax_agreement": min(
                report["pixel_argmax_agreement"] for report in image_reports
            ),
            "maximum_absolute_error": max(
                report["max_absolute_error"] for report in image_reports
            ),
            "all_finite_logits": True,
            "reported_class_ids": selected_class_ids,
            "roi_evaluated": roi_polygon is not None,
        },
        "images": image_reports,
    }


def _load_images(image_args: Sequence[Path], image_list: Path | None) -> list[Path]:
    images = list(image_args)
    if image_list is not None:
        base = image_list.resolve().parent
        for raw_line in image_list.read_text(encoding="utf-8").splitlines():
            value = raw_line.strip()
            if not value or value.startswith("#"):
                continue
            path = Path(value)
            images.append(path if path.is_absolute() else base / path)
    return images


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and ONNX raw logits on a reference image set"
    )
    parser.add_argument("--deploy-config", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        type=Path,
        help="Reference image; repeat at least 10 times unless --image-list is used",
    )
    parser.add_argument(
        "--image-list",
        type=Path,
        help="UTF-8 file containing one image path per line",
    )
    parser.add_argument("--minimum-images", type=int, default=DEFAULT_MINIMUM_IMAGES)
    parser.add_argument(
        "--target-class-id",
        action="append",
        type=int,
        help=(
            "Class ID whose valid-image/ROI area ratio is reported; repeat as "
            "needed. By default every output class is reported."
        ),
    )
    parser.add_argument(
        "--expected-num-classes",
        type=int,
        default=DEFAULT_EXPECTED_NUM_CLASSES,
    )
    parser.add_argument(
        "--roi-polygon",
        help="Frozen normalized ROI polygon as 'x,y;x,y;...'",
    )
    parser.add_argument("--reference-output-dir", type=Path)
    parser.add_argument(
        "--visualization-count",
        type=int,
        default=DEFAULT_VISUALIZATION_COUNT,
        help="Write color mask and overlay for the first N reference images",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    images = _load_images(args.image, args.image_list)
    if not images:
        parser.error("provide --image at least once or use --image-list")
    report = compare_many(
        args.deploy_config,
        args.model_config,
        args.checkpoint,
        args.onnx,
        images,
        args.device,
        minimum_images=args.minimum_images,
        class_ids=args.target_class_id,
        expected_num_classes=args.expected_num_classes,
        roi_polygon=parse_normalized_polygon(args.roi_polygon),
        reference_output_dir=args.reference_output_dir,
        visualization_count=args.visualization_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
