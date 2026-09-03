#!/usr/bin/env python3
"""Recover the camera-to-perception latency budget from recorded ADOM rosbags.

No Jetson, no ROS installation and no live vehicle are required. Everything is
read out of the recorded MCAP files.

Two independent sources are used, and they cross-check each other.

1. The perception node's own status topic. On the deployed Jetson it publishes
   a JSON object per frame carrying the whole budget, for example::

       capture_to_receive_ms       ZED capture -> node received the image
       queue_wait_ms               waiting before inference started
       capture_to_inference_start_ms
       inference_ms                TensorRT execution
       processing_ms               inference plus pre/post processing
       capture_to_perception_output_ms   the camera -> perception number
       average_fps, received_frames, overwritten_frames

   Every numeric field found is summarized, so this keeps working if the node
   gains or renames fields.

2. Header propagation on published messages. For any topic whose message
   begins with a std_msgs/Header, ``publish_time - header.stamp`` is the age of
   the camera frame when that message went out. If a publisher re-stamps its
   output instead of propagating the camera header, the value collapses to
   about zero and the summary says so rather than reporting a fake number.

Read-only. It never writes into a bag and never publishes a topic.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Iterator

# Messages whose first field is a std_msgs/Header, so the timestamp sits at a
# fixed offset right after the 4-byte CDR encapsulation header.
HEADER_FIRST_SCHEMAS = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/CompressedImage",
    "sensor_msgs/msg/CameraInfo",
    "sensor_msgs/msg/NavSatFix",
    "sensor_msgs/msg/PointCloud2",
    "nav_msgs/msg/OccupancyGrid",
    "nav_msgs/msg/Odometry",
    "ackermann_msgs/msg/AckermannDriveStamped",
}
STRING_SCHEMAS = {"std_msgs/msg/String"}

DEFAULT_TOPICS = [
    "/adom/perception/semantic20_mask_evidence",
    "/adom/navigation/semantic_costmap",
    "/drive",
]
DEFAULT_STATUS_TOPIC = "/adom/perception/status"

# Preferred display order for the latency budget. Fields not listed here are
# still reported, just after these.
BUDGET_ORDER = [
    "capture_to_receive_ms",
    "queue_wait_ms",
    "capture_to_inference_start_ms",
    "inference_ms",
    "processing_ms",
    "capture_to_perception_output_ms",
    "latency_ms",
]
RATE_FIELDS = ["average_fps", "target_fps", "evidence_mask_fps"]
# Cumulative counters: only the last value in a bag is meaningful.
COUNTER_FIELDS = ["received_frames", "overwritten_frames", "source_sequence"]

NS_PER_MS = 1_000_000.0


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile over an already sorted list."""
    if not sorted_values:
        raise ValueError("percentile of an empty sample")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def decode_header_stamp_ns(payload: bytes) -> int | None:
    """Read builtin_interfaces/Time from a CDR message that starts with a Header.

    Layout: 4-byte encapsulation, int32 sec, uint32 nanosec.
    """
    if len(payload) < 12:
        return None
    little_endian = payload[1] in (1, 3)
    fmt = "<iI" if little_endian else ">iI"
    seconds, nanoseconds = struct.unpack_from(fmt, payload, 4)
    if seconds < 0 or nanoseconds >= 1_000_000_000:
        return None
    return seconds * 1_000_000_000 + nanoseconds


def decode_string(payload: bytes) -> str | None:
    """Read std_msgs/String out of a CDR payload."""
    if len(payload) < 8:
        return None
    little_endian = payload[1] in (1, 3)
    fmt = "<I" if little_endian else ">I"
    (length,) = struct.unpack_from(fmt, payload, 4)
    start = 8
    end = start + length
    if length == 0 or end > len(payload):
        return None
    return payload[start:end].rstrip(b"\x00").decode("utf-8", errors="replace")


@dataclass
class TopicResult:
    topic: str
    schema: str = ""
    publish_latency_ms: list[float] = field(default_factory=list)
    log_latency_ms: list[float] = field(default_factory=list)
    first_log_ns: int | None = None
    last_log_ns: int | None = None
    undecodable: int = 0

    @property
    def message_count(self) -> int:
        return len(self.publish_latency_ms) + self.undecodable

    @property
    def duration_s(self) -> float:
        if self.first_log_ns is None or self.last_log_ns is None:
            return 0.0
        return (self.last_log_ns - self.first_log_ns) / 1e9

    @property
    def rate_hz(self) -> float | None:
        duration = self.duration_s
        if duration <= 0.0 or self.message_count < 2:
            return None
        return (self.message_count - 1) / duration


@dataclass
class BagResult:
    name: str
    path: str
    topics: dict[str, TopicResult] = field(default_factory=dict)
    status_values: dict[str, list[float]] = field(default_factory=dict)
    status_last: dict[str, float] = field(default_factory=dict)
    status_messages: int = 0
    status_undecodable: int = 0
    class_names: list[str] = field(default_factory=list)
    class_ratio_rows: list[dict[str, Any]] = field(default_factory=list)
    class_ratio_sums: dict[str, float] = field(default_factory=dict)
    class_ratio_samples: int = 0
    session: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def mean_class_ratios(self) -> dict[str, float]:
        if not self.class_ratio_samples:
            return {}
        return {
            name: total / self.class_ratio_samples
            for name, total in self.class_ratio_sums.items()
        }


def iter_mcap_messages(
    mcap_path: Path, topics: list[str]
) -> Iterator[tuple[str, str, int, int, bytes]]:
    """Yield (topic, schema_name, log_time_ns, publish_time_ns, payload)."""
    try:
        from mcap.reader import make_reader
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit(
            "This script needs the 'mcap' package to read .mcap bags.\n"
            "  pip install mcap\n"
            f"(import failed: {error})"
        ) from error

    with mcap_path.open("rb") as handle:
        reader = make_reader(handle)
        for schema, channel, message in reader.iter_messages(topics=topics or None):
            schema_name = schema.name if schema is not None else ""
            yield (
                channel.topic,
                schema_name,
                message.log_time,
                message.publish_time,
                message.data,
            )


def find_bags(
    root: Path, include: list[str] | None = None
) -> list[tuple[str, Path, list[Path]]]:
    """Group .mcap files by the bag directory a human would name.

    ``include`` is an optional list of fnmatch patterns tested against the bag
    directory name; a bag is kept when it matches any of them.
    """
    mcap_files = sorted(root.rglob("*.mcap"))
    grouped: dict[Path, list[Path]] = {}
    for mcap_path in mcap_files:
        # <bag_name>/rosbag/rosbag_0.mcap  ->  <bag_name>
        parent = mcap_path.parent
        anchor = parent.parent if parent.name == "rosbag" else parent
        grouped.setdefault(anchor, []).append(mcap_path)
    selected = [(anchor.name, anchor, files) for anchor, files in sorted(grouped.items())]
    if include:
        selected = [
            item
            for item in selected
            if any(fnmatch(item[0], pattern) for pattern in include)
        ]
    return selected


def absorb_status(
    result: BagResult, payload: bytes, publish_ns: int, dump_class_ratios: bool
) -> None:
    text = decode_string(payload)
    if not text:
        result.status_undecodable += 1
        return
    try:
        status = json.loads(text)
    except json.JSONDecodeError:
        result.status_undecodable += 1
        return
    if not isinstance(status, dict):
        result.status_undecodable += 1
        return

    result.status_messages += 1
    for key, value in status.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        result.status_values.setdefault(key, []).append(float(value))
        result.status_last[key] = float(value)

    names = status.get("class_names")
    if isinstance(names, list) and names and not result.class_names:
        result.class_names = [str(item) for item in names]

    ratios = status.get("class_pixel_ratios")
    if isinstance(ratios, list) and result.class_names:
        result.class_ratio_samples += 1
        row: dict[str, Any] = {
            "bag": result.name,
            "publish_time_ns": publish_ns,
            "source_stamp_ns": status.get("source_stamp_ns", ""),
        }
        for name, ratio in zip(result.class_names, ratios):
            if not isinstance(ratio, (int, float)):
                continue
            result.class_ratio_sums[name] = result.class_ratio_sums.get(name, 0.0) + float(ratio)
            row[name] = ratio
        if dump_class_ratios:
            result.class_ratio_rows.append(row)


def process_bag(
    name: str,
    anchor: Path,
    mcap_files: list[Path],
    topics: list[str],
    status_topic: str,
    dump_class_ratios: bool,
) -> BagResult:
    result = BagResult(name=name, path=str(anchor))

    session_path = anchor / "session.json"
    if session_path.is_file():
        try:
            result.session = json.loads(session_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            result.session = {"_error": str(error)}

    wanted = list(topics)
    if status_topic and status_topic not in wanted:
        wanted.append(status_topic)

    try:
        for mcap_path in mcap_files:
            for topic, schema, log_ns, publish_ns, payload in iter_mcap_messages(
                mcap_path, wanted
            ):
                if topic == status_topic and schema in STRING_SCHEMAS:
                    absorb_status(result, payload, publish_ns, dump_class_ratios)
                    if topic not in topics:
                        continue

                entry = result.topics.setdefault(topic, TopicResult(topic=topic))
                entry.schema = entry.schema or schema
                if entry.first_log_ns is None or log_ns < entry.first_log_ns:
                    entry.first_log_ns = log_ns
                if entry.last_log_ns is None or log_ns > entry.last_log_ns:
                    entry.last_log_ns = log_ns

                if schema not in HEADER_FIRST_SCHEMAS:
                    entry.undecodable += 1
                    continue
                header_ns = decode_header_stamp_ns(payload)
                if header_ns is None or header_ns == 0:
                    entry.undecodable += 1
                    continue
                entry.publish_latency_ms.append((publish_ns - header_ns) / NS_PER_MS)
                entry.log_latency_ms.append((log_ns - header_ns) / NS_PER_MS)
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001 - one bad bag must not kill the run
        result.error = f"{type(error).__name__}: {error}"
    return result


def ordered_status_fields(fields: set[str]) -> list[str]:
    known = [name for name in BUDGET_ORDER if name in fields]
    rates = [name for name in RATE_FIELDS if name in fields]
    counters = [name for name in COUNTER_FIELDS if name in fields]
    rest = sorted(fields - set(known) - set(rates) - set(counters))
    return known + rates + counters + rest


def build_report(bags: list[BagResult], topics: list[str]) -> dict[str, Any]:
    header_aggregate: dict[str, dict[str, Any]] = {}
    for topic in topics:
        publish_values: list[float] = []
        log_values: list[float] = []
        counted_bags = 0
        for bag in bags:
            entry = bag.topics.get(topic)
            if entry is None or not entry.publish_latency_ms:
                continue
            counted_bags += 1
            publish_values.extend(entry.publish_latency_ms)
            log_values.extend(entry.log_latency_ms)
        header_aggregate[topic] = {
            "bags_with_data": counted_bags,
            "publish_minus_header": summarize(publish_values),
            "log_minus_header": summarize(log_values),
        }

    all_fields: set[str] = set()
    for bag in bags:
        all_fields.update(bag.status_values)
    status_aggregate: dict[str, Any] = {}
    for name in ordered_status_fields(all_fields):
        values = [value for bag in bags for value in bag.status_values.get(name, [])]
        status_aggregate[name] = summarize(values)

    dropped_total = sum(bag.status_last.get("overwritten_frames", 0.0) for bag in bags)
    received_total = sum(bag.status_last.get("received_frames", 0.0) for bag in bags)
    drop_rate = (dropped_total / received_total) if received_total else None

    warnings: list[str] = []
    for topic, stats in header_aggregate.items():
        summary = stats["publish_minus_header"]
        if not summary.get("count"):
            warnings.append(f"{topic}: no decodable header timestamps were found.")
            continue
        if summary["minimum"] < -1.0:
            warnings.append(
                f"{topic}: negative latency observed (min {summary['minimum']:.2f} ms). "
                "The header stamp and the publish clock are not the same time base."
            )
        if summary["p50"] < 1.0:
            warnings.append(
                f"{topic}: median is {summary['p50']:.3f} ms, so this publisher re-stamps "
                "its output with the current time instead of propagating the camera "
                "header. That column is not camera-to-perception latency."
            )
    if not status_aggregate:
        warnings.append(
            "The perception status topic produced no numeric fields. Check --status-topic."
        )
    return {
        "schema_version": "adom-camera-latency-v2",
        "definition": {
            "status_fields": "numeric fields the perception node published about itself",
            "publish_minus_header": "message publish time minus the header.stamp it carries",
            "log_minus_header": "recorder write time minus header.stamp (adds recorder delay)",
            "unit": "milliseconds unless the field name says otherwise",
        },
        "bags_scanned": len(bags),
        "bags_failed": sum(1 for bag in bags if bag.error),
        "status_messages": sum(bag.status_messages for bag in bags),
        "perception_status_fields": status_aggregate,
        "frame_drop": {
            "received_frames_total": received_total,
            "overwritten_frames_total": dropped_total,
            "drop_rate": drop_rate,
            "note": "counters are cumulative per bag; the last value of each bag is summed",
        },
        "header_propagation": header_aggregate,
        "warnings": warnings,
    }


def write_per_bag_status_csv(path: Path, bags: list[BagResult]) -> None:
    columns = [
        "bag", "field", "count", "mean", "p50", "p90", "p95", "p99",
        "minimum", "maximum", "session_started_at", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for bag in bags:
            started_at = bag.session.get("started_at", "") if bag.session else ""
            for name in ordered_status_fields(set(bag.status_values)):
                stats = summarize(bag.status_values[name])
                row = {
                    "bag": bag.name,
                    "field": name,
                    "count": stats.get("count", 0),
                    "session_started_at": started_at,
                    "error": bag.error,
                }
                for key in ("mean", "p50", "p90", "p95", "p99", "minimum", "maximum"):
                    value = stats.get(key)
                    row[key] = "" if value is None else f"{value:.3f}"
                writer.writerow(row)


def write_per_bag_header_csv(path: Path, bags: list[BagResult], topics: list[str]) -> None:
    columns = [
        "bag", "topic", "schema", "messages", "decoded", "undecodable",
        "duration_s", "rate_hz",
        "mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "minimum_ms", "maximum_ms",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for bag in bags:
            if bag.error and not bag.topics:
                writer.writerow({"bag": bag.name, "error": bag.error})
                continue
            for topic in topics:
                entry = bag.topics.get(topic)
                if entry is None:
                    continue
                stats = summarize(entry.publish_latency_ms)
                row = {
                    "bag": bag.name,
                    "topic": topic,
                    "schema": entry.schema,
                    "messages": entry.message_count,
                    "decoded": stats.get("count", 0),
                    "undecodable": entry.undecodable,
                    "duration_s": f"{entry.duration_s:.3f}",
                    "rate_hz": "" if entry.rate_hz is None else f"{entry.rate_hz:.2f}",
                    "error": bag.error,
                }
                for source, target in (
                    ("mean", "mean_ms"), ("p50", "p50_ms"), ("p90", "p90_ms"),
                    ("p95", "p95_ms"), ("p99", "p99_ms"),
                    ("minimum", "minimum_ms"), ("maximum", "maximum_ms"),
                ):
                    value = stats.get(source)
                    row[target] = "" if value is None else f"{value:.3f}"
                writer.writerow(row)


def write_per_bag_class_ratio_csv(path: Path, bags: list[BagResult]) -> None:
    """Mean predicted pixel ratio per class, one row per bag.

    Useful for narrowing down which checkpoint a bag was recorded with when the
    session metadata does not record it: a model that never predicts a class
    leaves that column at zero across the whole bag.
    """
    names: list[str] = []
    for bag in bags:
        if bag.class_names:
            names = bag.class_names
            break
    if not names:
        return
    columns = ["bag", "samples", "session_started_at"] + names
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for bag in bags:
            means = bag.mean_class_ratios()
            if not means:
                continue
            row: dict[str, Any] = {
                "bag": bag.name,
                "samples": bag.class_ratio_samples,
                "session_started_at": (
                    bag.session.get("started_at", "") if bag.session else ""
                ),
            }
            for name in names:
                row[name] = f"{means.get(name, 0.0):.6f}"
            writer.writerow(row)


def write_class_ratio_csv(path: Path, bags: list[BagResult]) -> None:
    names: list[str] = []
    for bag in bags:
        if bag.class_names:
            names = bag.class_names
            break
    if not names:
        return
    columns = ["bag", "publish_time_ns", "source_stamp_ns"] + names
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for bag in bags:
            writer.writerows(bag.class_ratio_rows)


def write_summary_markdown(
    path: Path, report: dict[str, Any], topics: list[str], all_fields: bool = False
) -> None:
    lines = [
        "# Camera-to-perception latency, recovered offline from recorded bags",
        "",
        f"- Bags scanned: {report['bags_scanned']} (failed: {report['bags_failed']})",
        f"- Perception status messages parsed: {report['status_messages']}",
        "",
        "## Latency budget reported by the perception node",
        "",
        "| Field | Samples | mean | p50 | p95 | p99 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rendered = set(BUDGET_ORDER) | set(RATE_FIELDS)
    for name, stats in report["perception_status_fields"].items():
        if not stats.get("count"):
            continue
        if not all_fields and name not in rendered:
            # Cumulative counters and bookkeeping fields are kept in the CSV and
            # JSON but would only add noise to the table a human reads.
            continue
        lines.append(
            "| `{name}` | {count} | {mean:.2f} | {p50:.2f} | {p95:.2f} | {p99:.2f} | {maximum:.2f} |".format(
                name=name,
                count=stats["count"],
                mean=stats["mean"],
                p50=stats["p50"],
                p95=stats["p95"],
                p99=stats["p99"],
                maximum=stats["maximum"],
            )
        )

    drop = report["frame_drop"]
    if drop["drop_rate"] is not None:
        lines += [
            "",
            "## Frames dropped by the latest-frame policy",
            "",
            f"- received: {drop['received_frames_total']:.0f}, "
            f"overwritten: {drop['overwritten_frames_total']:.0f}, "
            f"drop rate: {100.0 * drop['drop_rate']:.1f}%",
            f"- {drop['note']}",
        ]

    lines += [
        "",
        "## Independent cross-check: header propagation",
        "",
        "Age of the camera frame when each message was published "
        "(`publish_time - header.stamp`).",
        "",
        "| Topic | Bags | Samples | mean (ms) | p50 | p95 | p99 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for topic in topics:
        stats = report["header_propagation"].get(topic, {})
        summary = stats.get("publish_minus_header", {})
        if not summary.get("count"):
            lines.append(
                f"| `{topic}` | {stats.get('bags_with_data', 0)} | 0 | - | - | - | - | - |"
            )
            continue
        lines.append(
            "| `{topic}` | {bags} | {count} | {mean:.2f} | {p50:.2f} | {p95:.2f} | {p99:.2f} | {maximum:.2f} |".format(
                topic=topic,
                bags=stats["bags_with_data"],
                count=summary["count"],
                mean=summary["mean"],
                p50=summary["p50"],
                p95=summary["p95"],
                p99=summary["p99"],
                maximum=summary["maximum"],
            )
        )

    if report["warnings"]:
        lines += ["", "## Warnings", ""]
        lines += [f"- {item}" for item in report["warnings"]]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--bags-root",
        required=True,
        type=Path,
        help="directory containing the downloaded bag folders",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--topic",
        action="append",
        dest="topics",
        default=None,
        help="header-bearing topic to cross-check; repeatable (default: mask, costmap, drive)",
    )
    parser.add_argument(
        "--status-topic",
        default=DEFAULT_STATUS_TOPIC,
        help="std_msgs/String topic carrying the node's own JSON status",
    )
    parser.add_argument(
        "--dump-class-ratios",
        action="store_true",
        help="also write the per-class pixel ratio time series to class_ratios.csv",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        metavar="GLOB",
        help="only process bags whose directory name matches this pattern; repeatable "
             "(e.g. --include 'autonomy_20260814_*')",
    )
    parser.add_argument(
        "--all-fields",
        action="store_true",
        help="render every numeric status field in summary.md, not just the budget",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="process at most N bags (0 = all)"
    )
    args = parser.parse_args(argv)

    topics = args.topics or list(DEFAULT_TOPICS)

    if not args.bags_root.is_dir():
        print(f"ERROR: --bags-root is not a directory: {args.bags_root}", file=sys.stderr)
        return 2

    discovered = find_bags(args.bags_root, args.include)
    if not discovered:
        print(
            f"ERROR: no .mcap files matched under {args.bags_root}", file=sys.stderr
        )
        return 2
    if args.limit:
        discovered = discovered[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    bags: list[BagResult] = []
    for index, (name, anchor, files) in enumerate(discovered, start=1):
        print(f"[{index}/{len(discovered)}] {name}", file=sys.stderr)
        bags.append(
            process_bag(
                name, anchor, files, topics, args.status_topic, args.dump_class_ratios
            )
        )

    report = build_report(bags, topics)
    write_per_bag_status_csv(args.output_dir / "per_bag_status.csv", bags)
    write_per_bag_header_csv(args.output_dir / "per_bag_header_latency.csv", bags, topics)
    write_summary_markdown(
        args.output_dir / "summary.md", report, topics, args.all_fields
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_per_bag_class_ratio_csv(args.output_dir / "per_bag_class_ratios.csv", bags)
    if args.dump_class_ratios:
        write_class_ratio_csv(args.output_dir / "class_ratios.csv", bags)

    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))
    for bag in bags:
        if bag.error:
            print(f"WARNING: {bag.name}: {bag.error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
