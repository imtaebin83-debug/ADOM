from __future__ import annotations

import argparse
import json
from pathlib import Path

from adom.data.io import write_json
from adom.runtime.semantic20_cycle import validate_semantic20_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate every file in an E0/E1/E2 Semantic20 processed dataset"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--experiment", choices=("e0", "e1", "e2"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = validate_semantic20_dataset(args.dataset.resolve(), args.experiment)
    if args.output:
        write_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
