from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from _common import (
    markdown_table,
    read_json,
    sha256_file,
    write_dict_csv,
    write_json,
)


DATASETS = ("rellis", "korean")
MODELS = ("b0_e0", "eadom")
DISPLAY_DATASET = {"rellis": "RELLIS test", "korean": "Korean held-out test"}
DISPLAY_MODEL = {"b0_e0": "B0-E0", "eadom": "E-ADOM"}


def _number(value: str) -> float | None:
    if value in {"", "N/A", "None", "null"}:
        return None
    return float(value)


def _per_class(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            output[row["class_name"]] = {
                **row,
                "iou": _number(row["iou"]),
                "recall": _number(row["recall"]),
                "gt_pixel_count": int(row["gt_pixel_count"]),
            }
    return output


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def _load_results(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    metrics_dir = root / "metrics"
    for dataset in DATASETS:
        for model in MODELS:
            prefix = f"{dataset}__{model}"
            summary_path = metrics_dir / f"{prefix}__summary.json"
            per_class_path = metrics_dir / f"{prefix}__per_class.csv"
            confusion_path = metrics_dir / f"{prefix}__confusion_matrix.npy"
            for path in (summary_path, per_class_path, confusion_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            summary = read_json(summary_path)
            if summary.get("schema_version") != "adom-paper-eval-metrics-v1":
                raise ValueError(f"Unexpected metric schema: {summary_path}")
            if summary.get("dataset") != dataset or summary.get("model") != model:
                raise ValueError(f"Metric identity mismatch: {summary_path}")
            output[(dataset, model)] = {
                "summary": summary,
                "classes": _per_class(per_class_path),
                "paths": {
                    "summary": summary_path,
                    "per_class": per_class_path,
                    "confusion": confusion_path,
                },
            }
    return output


def _validate_pairing(results: dict[tuple[str, str], dict[str, Any]]) -> None:
    contracts: set[str] = set()
    for dataset in DATASETS:
        baseline = results[(dataset, "b0_e0")]["summary"]
        eadom = results[(dataset, "eadom")]["summary"]
        for field in ("ordered_manifest_sha256", "manifest_csv_sha256"):
            if baseline["provenance"][field] != eadom["provenance"][field]:
                raise RuntimeError(f"{dataset} models were evaluated on different manifests")
        if baseline["metrics"]["common_classes"] != eadom["metrics"]["common_classes"]:
            raise RuntimeError(f"{dataset} models use different common class sets")
        baseline_gt = {
            name: row["gt_pixel_count"]
            for name, row in results[(dataset, "b0_e0")]["classes"].items()
        }
        eadom_gt = {
            name: row["gt_pixel_count"]
            for name, row in results[(dataset, "eadom")]["classes"].items()
        }
        if baseline_gt != eadom_gt:
            raise RuntimeError(f"{dataset} paired results do not have identical GT support")
        contracts.update(
            (
                baseline["provenance"]["evaluation_contract_sha256"],
                eadom["provenance"]["evaluation_contract_sha256"],
            )
        )
    if len(contracts) != 1:
        raise RuntimeError("The four evaluations did not share one canonical evaluation contract")


def _paper_rows(results: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for model in MODELS:
            item = results[(dataset, model)]
            metrics = item["summary"]["metrics"]
            classes = item["classes"]
            rows.append(
                {
                    "Evaluation dataset": DISPLAY_DATASET[dataset],
                    "Model": DISPLAY_MODEL[model],
                    "Native supported mIoU": metrics["dataset_native_supported_mIoU"],
                    "Common-supported mIoU": metrics["common_supported_mIoU"],
                    "log IoU": classes["log"]["iou"],
                    "log Recall": classes["log"]["recall"],
                    "rubble IoU": classes["rubble"]["iou"],
                    "rubble Recall": classes["rubble"]["recall"],
                    "summary_source": str(item["paths"]["summary"].resolve()),
                    "per_class_source": str(item["paths"]["per_class"].resolve()),
                    "metric_unit": "percent",
                }
            )
    return rows


def _delta_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        path = root / "metrics" / f"{dataset}__paired_bootstrap.json"
        value = read_json(path)
        if value.get("dataset") != dataset:
            raise ValueError(f"Bootstrap dataset mismatch: {path}")
        for result in value["results"]:
            lower = result["ci95_lower"]
            upper = result["ci95_upper"]
            ci = (
                f"[{lower:.2f}, {upper:.2f}]"
                if lower is not None and upper is not None
                else result["status"]
            )
            rows.append(
                {
                    "Dataset": "RELLIS" if dataset == "rellis" else "Korean",
                    "Metric": result["metric"],
                    "B0-E0": result["b0_e0"],
                    "E-ADOM": result["eadom"],
                    "Delta": result["delta_eadom_minus_b0_e0"],
                    "95% CI": ci,
                    "status": result["status"],
                    "resampling_unit": result["resampling_unit"],
                    "unit_count": result["unit_count"],
                    "positive_unit_count": result["positive_unit_count"],
                    "bootstrap_source": str(path.resolve()),
                }
            )
    return rows


def _delta_lookup(rows: list[dict[str, Any]], dataset: str, metric: str) -> dict[str, Any]:
    display = "RELLIS" if dataset == "rellis" else "Korean"
    return next(row for row in rows if row["Dataset"] == display and row["Metric"] == metric)


def _framing(
    results: dict[tuple[str, str], dict[str, Any]], delta_rows: list[dict[str, Any]]
) -> tuple[str, list[str]]:
    evidence: list[str] = []
    scores: dict[str, int] = {"log": 0, "rubble": 0}
    for class_name in ("log", "rubble"):
        korean_support = results[("korean", "b0_e0")]["classes"][class_name]["gt_pixel_count"]
        korean_delta = _delta_lookup(delta_rows, "korean", f"{class_name}/IoU")
        rellis_delta = _delta_lookup(delta_rows, "rellis", f"{class_name}/IoU")
        if korean_support > 0:
            scores[class_name] += 1
        if korean_delta["Delta"] is not None and korean_delta["Delta"] > 0:
            scores[class_name] += 1
        if rellis_delta["Delta"] is not None and rellis_delta["Delta"] > 0:
            scores[class_name] += 1
        if korean_delta["status"] == "PASS" and korean_delta["95% CI"].startswith("["):
            lower = float(korean_delta["95% CI"].split(",", 1)[0].strip("["))
            if lower > 0:
                scores[class_name] += 1
        evidence.append(
            f"{class_name}: Korean GT pixels={korean_support}, "
            f"Korean IoU delta={_fmt(korean_delta['Delta'])}, "
            f"RELLIS IoU delta={_fmt(rellis_delta['Delta'])}."
        )
    common_deltas = [
        _delta_lookup(delta_rows, dataset, "common_supported_mIoU")["Delta"]
        for dataset in DATASETS
    ]
    native_deltas = [
        results[(dataset, "eadom")]["summary"]["metrics"][
            "dataset_native_supported_mIoU"
        ]
        - results[(dataset, "b0_e0")]["summary"]["metrics"][
            "dataset_native_supported_mIoU"
        ]
        for dataset in DATASETS
    ]
    korean_class_ci_supported = all(
        _delta_lookup(delta_rows, "korean", f"{name}/IoU")["status"] == "PASS"
        for name in ("log", "rubble")
    )
    if any(value <= 0 for value in native_deltas) or not korean_class_ci_supported:
        recommendation = "D. system demonstration with class-specific trade-offs"
    elif all(value is not None and value > 0 for value in common_deltas):
        if scores["log"] >= 3 and scores["rubble"] >= 3:
            recommendation = "C. general rare-hazard data refinement"
        else:
            winner = max(scores, key=scores.get)
            recommendation = (
                "A. log-centered rare-hazard adaptation"
                if winner == "log"
                else "B. rubble-centered rare-hazard adaptation"
            )
    else:
        recommendation = "D. system demonstration with class-specific trade-offs"
    if recommendation.startswith("D.") and all(
        value is not None and value > 0 for value in common_deltas
    ):
        recommendation += (
            "; secondary wording: rare-hazard refinement with log as the deployment "
            "scenario and rubble as the strongest RELLIS offline gain"
        )
    return recommendation, evidence


def _report(
    results: dict[tuple[str, str], dict[str, Any]],
    paper_markdown: str,
    delta_markdown: str,
    delta_rows: list[dict[str, Any]],
    dataset_audit: dict[str, Any],
) -> str:
    recommendation, evidence = _framing(results, delta_rows)
    lines = [
        "# ADOM canonical paper evaluation",
        "",
        "## Fresh 2×2 evaluation",
        "",
        paper_markdown.rstrip(),
        "",
        "## Paired deltas (E-ADOM − B0-E0)",
        "",
        delta_markdown.rstrip(),
        "",
        "## Interpretation",
        "",
        "- The fresh B0-E0 and E-ADOM results for each dataset use an identical ordered manifest and one canonical inference contract; `compare_results.py` refuses mismatches.",
        "- The historical B0-E0 run is documented as the 899-image canonical RELLIS test. The E-ADOM main config also inherits canonical RELLIS val/test, but old numeric artifacts cannot be declared manifest-identical unless their archived manifest hash is present in the audit evidence.",
        f"- Korean split audit: train-test image-hash overlap={len(dataset_audit['overlaps']['korean_train_vs_test']['image_sha256'])}, val-test image-hash overlap={len(dataset_audit['overlaps']['korean_val_vs_test']['image_sha256'])}, train-test sequence overlap={len(dataset_audit['overlaps']['korean_train_vs_test']['sequence'])}, val-test sequence overlap={len(dataset_audit['overlaps']['korean_val_vs_test']['sequence'])}.",
        f"- Korean train-val contains {len(dataset_audit['overlaps']['korean_train_vs_val']['image_sha256'])} byte-identical images under different sequence paths and different annotations. This does not overlap the held-out test, but it limits train/validation quality and must be disclosed.",
    ]
    for dataset in DATASETS:
        label = "RELLIS" if dataset == "rellis" else "Korean held-out"
        overall = _delta_lookup(delta_rows, dataset, "common_supported_mIoU")
        lines.append(
            f"- {label}: common-supported mIoU delta {_fmt(overall['Delta'])} pp "
            f"with CI {overall['95% CI']}."
        )
        native_delta = (
            results[(dataset, "eadom")]["summary"]["metrics"][
                "dataset_native_supported_mIoU"
            ]
            - results[(dataset, "b0_e0")]["summary"]["metrics"][
                "dataset_native_supported_mIoU"
            ]
        )
        lines.append(f"  - Native supported mIoU delta: {_fmt(native_delta)} pp.")
        for class_name in ("log", "rubble", "barrier", "mud"):
            item = _delta_lookup(delta_rows, dataset, f"{class_name}/IoU")
            lines.append(
                f"  - {class_name} IoU delta: {_fmt(item['Delta'])} pp ({item['95% CI']})."
            )
    lines.extend(
        [
            "",
            "## Headline recommendation",
            "",
            f"**{recommendation}**",
            "",
            *[f"- {value}" for value in evidence],
            "- A class with zero GT or insufficient independent positive sequences is not eligible as a headline result.",
            "- Any barrier or mud regression above is a reported trade-off, not hidden by the overall average.",
            "",
            "## Limitations",
            "",
            "- This is a single-seed checkpoint comparison; the paired CI quantifies evaluation-set sampling uncertainty, not training-seed uncertainty.",
            "- Korean target-only masks use ignore 255 outside labeled regions; predictions in those ignored pixels are intentionally excluded from false positives.",
            "- Historical metric values are not copied into these tables. Every displayed value traces to a newly accumulated confusion matrix or paired bootstrap artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def compare(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_dir.resolve()
    dataset_audit = read_json(root / "dataset_manifest_summary.json")
    results = _load_results(root)
    _validate_pairing(results)
    paper_rows = _paper_rows(results)
    delta_rows = _delta_rows(root)
    paper_fields = (
        "Evaluation dataset",
        "Model",
        "Native supported mIoU",
        "Common-supported mIoU",
        "log IoU",
        "log Recall",
        "rubble IoU",
        "rubble Recall",
        "summary_source",
        "per_class_source",
        "metric_unit",
    )
    delta_fields = (
        "Dataset",
        "Metric",
        "B0-E0",
        "E-ADOM",
        "Delta",
        "95% CI",
        "status",
        "resampling_unit",
        "unit_count",
        "positive_unit_count",
        "bootstrap_source",
    )
    outputs = {
        "paper_csv": root / "paper_table.csv",
        "paper_md": root / "paper_table.md",
        "delta_csv": root / "paired_deltas.csv",
        "delta_md": root / "paired_deltas.md",
        "report": root / "report.md",
        "comparison_manifest": root / "comparison_manifest.json",
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("Refusing to overwrite existing paper comparison outputs")
    write_dict_csv(outputs["paper_csv"], paper_rows, paper_fields)
    write_dict_csv(outputs["delta_csv"], delta_rows, delta_fields)
    paper_md = markdown_table(
        paper_fields[:8],
        (
            [row[paper_fields[0]], row[paper_fields[1]]]
            + [_fmt(row[field]) for field in paper_fields[2:8]]
            for row in paper_rows
        ),
    )
    delta_md = markdown_table(
        delta_fields[:6],
        (
            [row[delta_fields[0]], row[delta_fields[1]]]
            + [_fmt(row[field]) for field in delta_fields[2:5]]
            + [row[delta_fields[5]]]
            for row in delta_rows
        ),
    )
    outputs["paper_md"].write_text(paper_md, encoding="utf-8")
    outputs["delta_md"].write_text(delta_md, encoding="utf-8")
    outputs["report"].write_text(
        _report(results, paper_md, delta_md, delta_rows, dataset_audit),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "adom-paper-eval-comparison-v1",
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in outputs.items()
            if name != "comparison_manifest"
        },
        "metric_inputs": {
            f"{dataset}__{model}": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in results[(dataset, model)]["paths"].items()
            }
            for dataset in DATASETS
            for model in MODELS
        },
    }
    write_json(outputs["comparison_manifest"], manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the 2x2 comparison and generate traceable paper tables"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    result = compare(parse_args(argv))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
