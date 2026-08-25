from __future__ import annotations

import argparse
from pathlib import Path

from _common import read_json, write_csv


def _fmt(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a compact paper table from RC analysis")
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = read_json(args.analysis_dir / "summary.json")
    rows = []
    for metric in (
        "stop_success_rate",
        "miss_rate",
        "false_stop_rate",
        "trial_completion_rate",
        "human_intervention_rate",
        "perception_hazard_detection_rate",
    ):
        item = summary[metric]
        rows.append(
            {
                "Metric": metric,
                "Value (%)": _fmt(item["percent"]),
                "95% Wilson CI (%)": (
                    "N/A" if item["wilson95_lower_percent"] is None
                    else f"[{item['wilson95_lower_percent']:.1f}, {item['wilson95_upper_percent']:.1f}]"
                ),
                "n": item["denominator"],
                "Outcome basis": "physical stop when available; otherwise stop-command proxy",
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    write_csv(args.output_dir / "rc_paper_table.csv", rows, fields)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(str(row[field]) for field in fields) + " |" for row in rows)
    (args.output_dir / "rc_paper_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "rc_paper_table.md")


if __name__ == "__main__":
    main()
