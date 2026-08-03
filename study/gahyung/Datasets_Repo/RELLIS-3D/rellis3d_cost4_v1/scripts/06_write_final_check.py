from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "rellis3d_cost4_v1"
)

DEFAULT_RESULTS_DIR = ROOT / "results"


def resolve_path(
    cli_value: Path | None,
    env_name: str,
    default: Path,
) -> Path:
    """
    Resolve a path in the following order:
    CLI argument -> environment variable -> repository-relative default.
    """
    value = cli_value or os.getenv(env_name)

    if value is None:
        return default.resolve()

    return Path(value).expanduser().resolve()


def main(
    output_root: Path,
    results_dir: Path,
) -> None:
    qc_report = output_root / "qc_report.csv"
    class_statistics = output_root / "class_statistics.csv"
    output_file = results_dir / "final_check.txt"

    if not qc_report.is_file():
        raise FileNotFoundError(
            f"qc_report.csv was not found: {qc_report}"
        )

    with qc_report.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames or []

    required_columns = {
        "sample_id",
        "status",
        "details",
    }

    missing_columns = sorted(
        required_columns - set(columns)
    )

    status_counts = Counter(
        (row.get("status") or "").strip().lower()
        for row in rows
    )

    failure_count = sum(
        count
        for status, count in status_counts.items()
        if status != "ok"
    )

    checks = {
        "qc_rows_present": bool(rows),
        "required_qc_columns_present": not missing_columns,
        "qc_failures_zero": failure_count == 0,
        "class_statistics_exists": class_statistics.is_file(),
    }

    final_status = (
        "PASS"
        if all(checks.values())
        else "FAIL"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        "[RELLIS-3D FINAL CHECK]",
        "",
        f"QC report: {qc_report.relative_to(output_root).as_posix()}",
        f"QC rows: {len(rows)}",
        f"QC columns: {columns}",
        "",
        "[QC STATUS COUNTS]",
    ]

    if status_counts:
        for status, count in sorted(status_counts.items()):
            display_status = status or "<empty>"
            lines.append(
                f"{display_status}: {count}"
            )
    else:
        lines.append("No QC rows were found.")

    lines.extend(
        [
            "",
            "[VALIDATION]",
            f"qc_rows_present: {checks['qc_rows_present']}",
            (
                "required_qc_columns_present: "
                f"{checks['required_qc_columns_present']}"
            ),
            f"missing_qc_columns: {missing_columns}",
            f"qc_failure_count: {failure_count}",
            (
                "class_statistics_exists: "
                f"{checks['class_statistics_exists']}"
            ),
            (
                "class_statistics_path: "
                f"{class_statistics.relative_to(output_root).as_posix()}"
            ),
            "",
            f"FINAL STATUS: {final_status}",
        ]
    )

    output_file.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(output_file.read_text(encoding="utf-8"))
    print(f"[SAVE] {output_file}")

    if final_status != "PASS":
        raise RuntimeError(
            "RELLIS-3D final check failed. "
            f"See: {output_file}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory containing qc_report.csv and "
            "class_statistics.csv. "
            "Environment variable: RELLIS_OUTPUT_ROOT"
        ),
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Directory for final_check.txt. "
            "Environment variable: RELLIS_RESULTS_DIR"
        ),
    )

    args = parser.parse_args()

    resolved_output_root = resolve_path(
        args.output_root,
        "RELLIS_OUTPUT_ROOT",
        DEFAULT_OUTPUT_ROOT,
    )

    resolved_results_dir = resolve_path(
        args.results_dir,
        "RELLIS_RESULTS_DIR",
        DEFAULT_RESULTS_DIR,
    )

    main(
        output_root=resolved_output_root,
        results_dir=resolved_results_dir,
    )
