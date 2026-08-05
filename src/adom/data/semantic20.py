from __future__ import annotations

import importlib
from importlib.resources import files
from pathlib import Path


def resource_path(*parts: str) -> Path:
    """Return a filesystem path to an installed canonical preprocessing asset."""
    resource = files("data")
    for part in parts:
        resource = resource.joinpath(part)
    path = Path(str(resource))
    if not path.exists():
        raise FileNotFoundError(f"Semantic20 package resource is missing: {path}")
    return path


def main(argv: list[str] | None = None) -> None:
    converter = importlib.import_module(
        "data.semantic_20.scripts.01_convert_bridge_sources"
    )
    converter.main(argv)


if __name__ == "__main__":
    main()
