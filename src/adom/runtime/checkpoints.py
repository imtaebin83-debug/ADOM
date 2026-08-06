from __future__ import annotations

import re
from pathlib import Path


BEST_PATTERN = re.compile(r"^best_clean_selection_iter_(\d+)\.pth$")


def resolve_single_best_checkpoint(work_dir: Path) -> Path:
    """Return the one Clean v1 constrained-selection checkpoint."""
    candidates = sorted(
        path
        for path in work_dir.glob("best_clean_selection_iter_*.pth")
        if BEST_PATTERN.match(path.name)
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one Clean v1 selected checkpoint in {work_dir}, "
            f"found {[path.name for path in candidates]}"
        )
    return candidates[0].resolve()
