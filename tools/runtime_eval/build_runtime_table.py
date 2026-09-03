#!/usr/bin/env python3
"""Build the paper's runtime table from Jetson TensorRT validation reports.

Every row comes from a `semantic20-onnx-tensorrt-parity-v1` report produced by
`scripts/validate_semantic20_tensorrt.sh`. Numbers are read out of the report,
never retyped, so all rows share one definition of "inference time" and "FPS".

Rows for a model whose engine was not built can be declared as an alias of a
model with the same architecture, input shape and precision. TensorRT latency
is set by the graph, not by the weight values, so an alias is a legitimate
report as long as the table says so - the tool marks those rows and emits the
footnote for you.

Read-only with respect to the reports. It only writes into --output-dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "semantic20-onnx-tensorrt-parity-v1"
BENCHMARK_KEY = "file_inference_benchmark"
STAGES = ("h2d_ms", "engine_ms", "d2h_ms", "runtime_total_ms")


def parse_assignment(value: str, flag: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"{flag} expects NAME=VALUE, got: {value!r}"
        )
    name, _, target = value.partition("=")
    name, target = name.strip(), target.strip()
    if not name or not target:
        raise argparse.ArgumentTypeError(f"{flag} expects NAME=VALUE, got: {value!r}")
    return name, target


def locate_report(path: Path) -> Path:
    """Accept either the report JSON itself or the directory holding it."""
    if path.is_file():
        return path
    if path.is_dir():
        candidates = []
        for candidate in sorted(path.rglob("*.json")):
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and payload.get("schema_version") == REPORT_SCHEMA:
                candidates.append(candidate)
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise FileNotFoundError(
                f"no {REPORT_SCHEMA} report found under {path}"
            )
        raise FileNotFoundError(
            f"{len(candidates)} reports found under {path}; pass the exact file. "
            + ", ".join(str(item) for item in candidates)
        )
    raise FileNotFoundError(f"no such file or directory: {path}")


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"{path}: report is not a JSON object")
    schema = report.get("schema_version")
    if schema != REPORT_SCHEMA:
        raise ValueError(
            f"{path}: expected schema_version {REPORT_SCHEMA!r}, found {schema!r}"
        )
    if BENCHMARK_KEY not in report:
        raise ValueError(f"{path}: report has no {BENCHMARK_KEY!r} block")
    return report


def input_shape(report: dict[str, Any]) -> str:
    inputs = report.get("contract", {}).get("inputs") or []
    if not inputs:
        return ""
    shape = inputs[0].get("shape") or []
    return "x".join(str(item) for item in shape)


def extract_row(name: str, report: dict[str, Any], source_path: Path) -> dict[str, Any]:
    benchmark = report[BENCHMARK_KEY]
    engine = report.get("engine", {})
    environment = report.get("environment", {})
    summary = report.get("summary", {})

    row: dict[str, Any] = {
        "model": name,
        "alias_of": "",
        "status": report.get("status", ""),
        "engine_filename": engine.get("filename", ""),
        "engine_bytes": engine.get("size_bytes"),
        "engine_mb": (
            round(engine["size_bytes"] / (1024 * 1024), 2)
            if isinstance(engine.get("size_bytes"), int)
            else None
        ),
        "engine_sha256": engine.get("sha256", ""),
        "precision": engine.get("precision", ""),
        "input_shape": input_shape(report),
        "iterations": benchmark.get("iterations"),
        "fps_from_mean": benchmark.get("derived_runtime_fps_from_mean"),
        "tensorrt": environment.get("tensorrt", ""),
        "platform": environment.get("platform", ""),
        "valid_pixel_agreement": summary.get("overall_valid_pixel_argmax_agreement"),
        "report_path": str(source_path),
    }
    for stage in STAGES:
        stats = benchmark.get(stage) or {}
        for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "maximum_ms"):
            row[f"{stage}.{key}"] = stats.get(key)
    return row


def alias_row(name: str, source: dict[str, Any]) -> dict[str, Any]:
    row = dict(source)
    row["model"] = name
    row["alias_of"] = source["model"]
    row["engine_sha256"] = ""
    row["engine_filename"] = ""
    return row


def check_consistency(rows: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    measured = [row for row in rows if not row["alias_of"]]

    shapes = {row["input_shape"] for row in measured if row["input_shape"]}
    if len(shapes) > 1:
        warnings.append(
            "Rows were measured at different input shapes ("
            + ", ".join(sorted(shapes))
            + "). The latencies are not comparable."
        )
    versions = {row["tensorrt"] for row in measured if row["tensorrt"]}
    if len(versions) > 1:
        warnings.append(
            "Rows were measured with different TensorRT versions ("
            + ", ".join(sorted(versions))
            + "). Say so in the paper or rebuild on one version."
        )
    platforms = {row["platform"] for row in measured if row["platform"]}
    if len(platforms) > 1:
        warnings.append(
            "Rows were measured on different platforms ("
            + "; ".join(sorted(platforms))
            + "). These are not the same hardware."
        )
    iterations = {row["iterations"] for row in measured if row["iterations"]}
    if len(iterations) > 1:
        warnings.append(
            "Rows used different benchmark iteration counts ("
            + ", ".join(str(item) for item in sorted(iterations))
            + ")."
        )
    for row in rows:
        if row["status"] and row["status"] != "PASS":
            warnings.append(
                f"{row['model']}: parity status is {row['status']}, not PASS. "
                "The engine does not reproduce the ONNX masks within contract."
            )
    seen_sha: dict[str, str] = {}
    for row in measured:
        sha = row["engine_sha256"]
        if not sha:
            continue
        if sha in seen_sha:
            warnings.append(
                f"{row['model']} and {seen_sha[sha]} point at the same engine SHA-256. "
                "One of them was measured against the wrong engine."
            )
        seen_sha[sha] = row["model"]
    return warnings


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]], warnings: list[str]) -> None:
    lines = [
        "# Jetson runtime, TensorRT FP16",
        "",
        "| Model | Engine (MB) | engine_ms p50 | engine_ms p95 | total_ms mean | total_ms p50 | total_ms p95 | FPS | Parity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        marker = " (*)" if row["alias_of"] else ""
        lines.append(
            "| {model}{marker} | {mb} | {ep50} | {ep95} | {tmean} | {tp50} | {tp95} | {fps} | {status} |".format(
                model=row["model"],
                marker=marker,
                mb=fmt(row["engine_mb"]),
                ep50=fmt(row["engine_ms.p50_ms"]),
                ep95=fmt(row["engine_ms.p95_ms"]),
                tmean=fmt(row["runtime_total_ms.mean_ms"]),
                tp50=fmt(row["runtime_total_ms.p50_ms"]),
                tp95=fmt(row["runtime_total_ms.p95_ms"]),
                fps=fmt(row["fps_from_mean"]),
                status=row["status"] or "n/a",
            )
        )

    aliases = [row for row in rows if row["alias_of"]]
    if aliases:
        lines += ["", "Footnote for the rows marked (*):", ""]
        for row in aliases:
            lines.append(
                f"- (*) {row['model']} shares {row['alias_of']}'s architecture, input shape "
                f"and precision, so its TensorRT engine size and latency are identical. "
                f"The measured row is {row['alias_of']}."
            )

    measured = [row for row in rows if not row["alias_of"]]
    lines += ["", "## Measurement conditions", ""]
    if measured:
        first = measured[0]
        lines += [
            f"- Input tensor: `{first['input_shape'] or 'FILL IN'}`, {first['precision'] or 'FILL IN'}",
            f"- TensorRT: {first['tensorrt'] or 'FILL IN'}",
            f"- Host platform string: `{first['platform'] or 'FILL IN'}`",
            f"- Benchmark: {fmt(first['iterations'], 0)} iterations after warmup",
        ]
    lines += [
        "- Hardware: **FILL IN** (Jetson Orin Nano 8GB or Orin NX - confirm the physical unit)",
        "- JetPack / L4T: **FILL IN**",
        "- Power mode (`nvpmodel -q`): **FILL IN**",
        "- Active cooling: **FILL IN**",
        "",
        "`engine_ms` is the GPU execution window. `runtime_total_ms` adds the host-to-device "
        "and device-to-host copies. Neither includes camera capture or ROS transport; measure "
        "that separately with `tools/runtime_eval/measure_camera_latency.py`.",
    ]

    if warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- {item}" for item in warnings]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="model name and its validation report (file or containing directory); repeatable",
    )
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        metavar="NAME=SOURCE_NAME",
        help="declare NAME as the same architecture as an already supplied SOURCE_NAME",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    if not args.report:
        print("ERROR: at least one --report NAME=PATH is required", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for item in args.report:
        name, raw_path = parse_assignment(item, "--report")
        try:
            report_path = locate_report(Path(raw_path))
            report = load_report(report_path)
        except (OSError, ValueError, FileNotFoundError) as error:
            print(f"ERROR: {name}: {error}", file=sys.stderr)
            return 2
        row = extract_row(name, report, report_path)
        rows.append(row)
        by_name[name] = row

    for item in args.alias:
        name, source_name = parse_assignment(item, "--alias")
        if source_name not in by_name:
            print(
                f"ERROR: --alias {name}={source_name}: {source_name} was not supplied with --report",
                file=sys.stderr,
            )
            return 2
        row = alias_row(name, by_name[source_name])
        rows.append(row)
        by_name[name] = row

    warnings = check_consistency(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_markdown(args.output_dir / "runtime_table.md", rows, warnings)
    write_csv(args.output_dir / "runtime_table.csv", rows)
    (args.output_dir / "runtime_rows.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print((args.output_dir / "runtime_table.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
