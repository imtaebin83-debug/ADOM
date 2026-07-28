from __future__ import annotations

import re
from pathlib import Path


BEST_PATTERN = re.compile(r"^best_mIoU_iter_(\d+)\.pth$")


def resolve_single_best_checkpoint(work_dir: Path) -> Path:
    """Return the one CheckpointHook-managed best mIoU checkpoint."""
    candidates = sorted(
        path
        for path in work_dir.glob("best_mIoU_iter_*.pth")
        if BEST_PATTERN.match(path.name)
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one best mIoU checkpoint in {work_dir}, "
            f"found {[path.name for path in candidates]}"
        )
    return candidates[0].resolve()
