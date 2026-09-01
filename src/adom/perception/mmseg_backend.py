from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


COST4_PALETTE_BGR = np.asarray(
    [
        [128, 128, 128],  # paved_low_cost
        [75, 180, 60],    # natural_low_cost
        [0, 165, 255],    # medium_cost
        [60, 20, 220],    # high_cost_or_obstacle
    ],
    dtype=np.uint8,
)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a Cost4 class-ID mask to a BGR image."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be HxW, got {mask.shape}")
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(COST4_PALETTE_BGR))
    output[valid] = COST4_PALETTE_BGR[mask[valid]]
    return output


class MmsegBackend:
    """Thin, lazy-imported MMSegmentation inference backend.

    Keeping the OpenMMLab imports inside construction lets ROS tooling inspect
    and test the package on machines that do not contain the training runtime.
    """

    def __init__(self, config: str | Path, checkpoint: str | Path, device: str) -> None:
        config_path = Path(config).expanduser().resolve()
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"MMSeg config not found: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"MMSeg checkpoint not found: {checkpoint_path}")

        try:
            from mmseg.apis import inference_model, init_model
        except ImportError as error:
            raise RuntimeError(
                "MMSegmentation runtime is unavailable. Install the repository's "
                "pinned OpenMMLab environment (mmsegmentation 1.2.2, mmcv 2.1.0)."
            ) from error

        self._inference_model = inference_model
        self._model = init_model(
            str(config_path), str(checkpoint_path), device=str(device)
        )

    def infer(
        self, bgr_image: np.ndarray, *, include_confidence: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
            raise ValueError(f"image must be HxWx3 BGR, got {bgr_image.shape}")

        result: Any = self._inference_model(self._model, bgr_image)
        mask = result.pred_sem_seg.data.squeeze().detach().cpu().numpy().astype(np.uint8)
        if mask.shape != bgr_image.shape[:2]:
            import cv2

            mask = cv2.resize(
                mask,
                (bgr_image.shape[1], bgr_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        if not include_confidence:
            return mask, None

        logits = getattr(result, "seg_logits", None)
        if logits is None:
            confidence = np.full(mask.shape, 255, dtype=np.uint8)
        else:
            tensor = logits.data
            probabilities = tensor.softmax(dim=0)
            confidence = (
                probabilities.max(dim=0).values.mul(255.0).clamp(0, 255)
                .detach().cpu().numpy().astype(np.uint8)
            )
            if confidence.shape != mask.shape:
                import cv2

                confidence = cv2.resize(
                    confidence,
                    (mask.shape[1], mask.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
        return mask, confidence
