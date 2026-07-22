from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[1]

# 실제 파일 위치에 따라 수정
QC_REPORT = ROOT / "data" / "processed" / \
    "rellis_cost4_standard" / "qc_report.csv"

CLASS_STATISTICS = ROOT / "data" / "processed" / \
    "rellis_cost4_standard" / "class_statistics.csv"

OUTPUT_DIR = ROOT / "github_results"
OUTPUT_FILE = OUTPUT_DIR / "final_check.txt"


TRUE_VALUES = {"1", "true", "yes", "pass", "ok"}


def is_true(value: str) -> bool:
    return value.strip().lower() in TRUE_VALUES


def main() -> None:
    if not QC_REPORT.exists():
        raise FileNotFoundError(f"qc_report.csv 없음: {QC_REPORT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with QC_REPORT.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames or []

    check_columns = [
        "image_readable",
        "mask_readable",
        "size_match",
        "valid_ids",
        "pair_exists",
    ]

    available_checks = [
        column
        for column in check_columns
        if column in columns
    ]

    failure_counts = {}

    for column in available_checks:
        failure_counts[column] = sum(
            1
            for row in rows
            if not is_true(row.get(column, ""))
        )

    total_failures = sum(failure_counts.values())
    final_status = "PASS" if total_failures == 0 else "CHECK REQUIRED"

    lines = [
        "[RELLIS-3D FINAL CHECK]",
        "",
        f"QC report: {QC_REPORT}",
        f"QC rows: {len(rows)}",
        f"QC columns: {columns}",
        "",
        "[QC RESULTS]",
    ]

    if available_checks:
        for column in available_checks:
            lines.append(
                f"{column} failures: {failure_counts[column]}"
            )
    else:
        lines.append(
            "Known QC columns were not found. "
            "Check qc_report.csv column names manually."
        )

    lines.extend(
        [
            "",
            f"class statistics exists: {CLASS_STATISTICS.exists()}",
            f"class statistics path: {CLASS_STATISTICS}",
            "",
            f"FINAL STATUS: {final_status}",
        ]
    )

    OUTPUT_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(OUTPUT_FILE.read_text(encoding="utf-8"))
    print(f"[SAVE] {OUTPUT_FILE}")


if __name__ == "__main__":
    main()