from __future__ import annotations

import argparse
from pathlib import Path

from _common import read_json, utc_now, validate_metadata, write_csv, write_json


def validate_experiment(root: Path) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    trials_root = root / "trials"
    if not trials_root.is_dir():
        rows.append({"trial_id": "*", "status": "ERROR", "message": "missing trials directory"})
    else:
        for trial_dir in sorted(path for path in trials_root.iterdir() if path.is_dir()):
            metadata = trial_dir / "metadata.json"
            if not metadata.is_file():
                rows.append({"trial_id": trial_dir.name, "status": "ERROR", "message": "missing metadata.json"})
                continue
            try:
                errors = validate_metadata(read_json(metadata))
            except Exception as error:  # malformed JSON is reported, not hidden
                errors = [f"invalid metadata JSON: {error}"]
            if errors:
                rows.extend({"trial_id": trial_dir.name, "status": "ERROR", "message": error} for error in errors)
            else:
                rows.append({"trial_id": trial_dir.name, "status": "PASS", "message": "metadata valid"})
    return {
        "schema_version": "adom-rc-eval-validation-v1",
        "generated_at_utc": utc_now(),
        "experiment_root": str(root.resolve()),
        "status": "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "ERROR",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an RC trial experiment directory")
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = validate_experiment(args.experiment_root)
    output = args.output_dir or args.experiment_root / "validation"
    write_json(output / "validation_summary.json", result)
    write_csv(output / "validation_rows.csv", result["rows"], ("trial_id", "status", "message"))
    print(result["status"])
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
