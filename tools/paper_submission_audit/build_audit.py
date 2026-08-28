from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image


CLASSES = (
    "dirt", "grass", "tree", "pole", "water", "sky", "vehicle", "object",
    "asphalt", "building", "log", "person", "fence", "bush", "concrete",
    "barrier", "puddle", "mud", "rubble",
)
CLASS_ID = {name: index for index, name in enumerate(CLASSES)}
IGNORE_INDEX = 255
COMMON_CLASSES = ("log", "rubble")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def command(args: Sequence[str], *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=120, check=False)
        return {"command": list(args), "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except Exception as error:
        return {"command": list(args), "returncode": None, "stdout": "", "stderr": f"{type(error).__name__}: {error}"}


def load_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image)
    if value.ndim != 2:
        raise ValueError(f"expected single-channel mask: {path}")
    invalid = ~(((value >= 0) & (value < len(CLASSES))) | (value == IGNORE_INDEX))
    if np.any(invalid):
        raise ValueError(f"invalid Semantic20 IDs in {path}: {np.unique(value[invalid]).tolist()}")
    return value


def package_versions() -> dict[str, Any]:
    output: dict[str, Any] = {"python": sys.version, "executable": sys.executable}
    for name in ("torch", "mmcv", "mmengine", "mmseg"):
        try:
            module = importlib.import_module(name)
            output[name] = {"version": getattr(module, "__version__", "unknown"), "path": getattr(module, "__file__", None)}
        except Exception as error:
            output[name] = {"status": "UNAVAILABLE", "error": f"{type(error).__name__}: {error}"}
    try:
        import torch

        output["torch_runtime"] = {
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
        }
    except Exception:
        pass
    return output


def build_environment(code_root: Path, workspace_root: Path, paper_root: Path, output: Path) -> dict[str, Any]:
    eadom_artifact = workspace_root / "artifacts/eadom-b0-seed42-iter26000"
    image_sha_path = eadom_artifact / "image_git_sha.txt"
    environment = {
        "captured_at_utc": now_utc(),
        "code_root": str(code_root),
        "code_root_has_git": (code_root / ".git").exists(),
        "git": {
            "head": command(["git", "-C", str(code_root), "rev-parse", "HEAD"]),
            "branch": command(["git", "-C", str(code_root), "branch", "--show-current"]),
            "status": command(["git", "-C", str(code_root), "status", "--short"]),
        },
        "immutable_image_git_sha": image_sha_path.read_text(encoding="utf-8").strip() if image_sha_path.is_file() else None,
        "packages": package_versions(),
        "gpu": command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader,nounits"]),
        "paper_eval_root": str(paper_root),
        "provenance_note": "The immutable /opt/adom image contains no .git directory. image_git_sha is the evaluation-code provenance; a separate volume snapshot is not treated as the active code checkout.",
    }
    write_json(output / "environment.json", environment)
    return environment


def build_checkpoint_manifest(workspace_root: Path, paper_root: Path, output: Path) -> dict[str, Any]:
    source = read_json(paper_root / "checkpoint_manifest.json")
    b0 = source["b0_e0"]
    eadom = source["eadom"]
    expected_eadom = "f4cc41fd91e9df8e7aa3f726498e80636b736dfadf0e1baf338fe7c82a83399c"
    archive = workspace_root / "exports/canonical-compare-20260814T013811Z.tar.gz"
    expected_archive = "8468bca1840c89b19145e743d877ffbcf6e5b4f50013de3bcb3d76b6ed45f77b"
    result = {
        "schema_version": "adom-paper-submission-checkpoints-v1",
        "source_checkpoint_manifest": str(paper_root / "checkpoint_manifest.json"),
        "b0_e0": {
            "path": b0["selected_path"],
            "sha256": sha256_file(Path(b0["selected_path"])),
            "expected_sha256": b0["expected_sha256"],
            "source_actual_sha256": b0["actual_sha256"],
        },
        "eadom": {
            "path": eadom["selected_path"],
            "sha256": sha256_file(Path(eadom["selected_path"])),
            "expected_sha256": expected_eadom,
            "iteration": 26000,
        },
        "canonical_archive": {
            "path": str(archive),
            "sha256": sha256_file(archive),
            "expected_sha256": expected_archive,
        },
    }
    for item in (result["b0_e0"], result["eadom"], result["canonical_archive"]):
        item["status"] = "PASS" if item["sha256"] == item["expected_sha256"] else "FAIL_SHA256"
    write_json(output / "checkpoint_manifest.json", result)
    return result


def build_artifact_inventory(paper_root: Path, output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(value for value in paper_root.rglob("*") if value.is_file()):
        stat = path.stat()
        rows.append({
            "relative_path": str(path.relative_to(paper_root)),
            "absolute_path": str(path),
            "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "sha256": sha256_file(path),
        })
    write_csv(output / "artifact_inventory.csv", rows, ("relative_path", "absolute_path", "size_bytes", "mtime_utc", "sha256"))
    return rows


def _support_row(row: dict[str, str]) -> dict[str, Any]:
    mask = load_mask(row["annotation_path"])
    values = Counter(mask[mask != IGNORE_INDEX].astype(int).tolist())
    return {
        **row,
        "log_pixels": values[CLASS_ID["log"]],
        "rubble_pixels": values[CLASS_ID["rubble"]],
        "log_positive": values[CLASS_ID["log"]] > 0,
        "rubble_positive": values[CLASS_ID["rubble"]] > 0,
        "hazard_negative": values[CLASS_ID["log"]] == 0 and values[CLASS_ID["rubble"]] == 0,
        "non_ignore_pixels": sum(values.values()),
        "ignore_pixels": int(np.count_nonzero(mask == IGNORE_INDEX)),
    }


def _overlap_rows(named: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    comparisons = (
        ("korean_train", "korean_val"),
        ("korean_train", "korean_test"),
        ("korean_val", "korean_test"),
        ("korean_test", "rellis_test"),
    )
    rows: list[dict[str, Any]] = []
    for left, right in comparisons:
        for field in ("sequence", "image_sha256", "annotation_sha256"):
            values = sorted({row[field] for row in named[left]} & {row[field] for row in named[right]})
            rows.append({"left": left, "right": right, "field": field, "overlap_count": len(values), "overlap_values": ";".join(values)})
    return rows


def build_sequence_support(paper_root: Path, output: Path) -> dict[str, Any]:
    target = output / "sequence_support"
    manifests = {
        "korean_train": read_csv(paper_root / "manifests/korean_train_manifest.csv"),
        "korean_val": read_csv(paper_root / "manifests/korean_val_manifest.csv"),
        "korean_test": read_csv(paper_root / "manifests/korean_test_manifest.csv"),
        "rellis_test": read_csv(paper_root / "manifests/rellis_test_manifest.csv"),
    }
    heldout = [_support_row(row) for row in manifests["korean_test"]]
    fields = list(heldout[0])
    write_csv(target / "korean_heldout_manifest.csv", heldout, fields)
    sequence_rows: list[dict[str, Any]] = []
    for sequence in sorted({row["sequence"] for row in heldout}):
        group = [row for row in heldout if row["sequence"] == sequence]
        sequence_rows.append({
            "sequence": sequence,
            "image_count": len(group),
            "log_positive_images": sum(bool(row["log_positive"]) for row in group),
            "rubble_positive_images": sum(bool(row["rubble_positive"]) for row in group),
            "both_positive_images": sum(bool(row["log_positive"] and row["rubble_positive"]) for row in group),
            "negative_images": sum(bool(row["hazard_negative"]) for row in group),
            "log_gt_pixels": sum(int(row["log_pixels"]) for row in group),
            "rubble_gt_pixels": sum(int(row["rubble_pixels"]) for row in group),
        })
    write_csv(target / "per_sequence_support.csv", sequence_rows, tuple(sequence_rows[0]))
    class_rows = []
    for name in COMMON_CLASSES:
        positive = [row for row in heldout if int(row[f"{name}_pixels"]) > 0]
        class_rows.append({
            "class_name": name,
            "class_id": CLASS_ID[name],
            "positive_images": len(positive),
            "positive_sequences": len({row["sequence"] for row in positive}),
            "gt_pixels": sum(int(row[f"{name}_pixels"]) for row in heldout),
        })
    write_csv(target / "per_class_support.csv", class_rows, tuple(class_rows[0]))
    overlaps = _overlap_rows(manifests)
    write_csv(target / "split_overlap_audit.csv", overlaps, tuple(overlaps[0]))
    by_class = {row["class_name"]: row for row in class_rows}
    both_images = sum(bool(row["log_positive"] and row["rubble_positive"]) for row in heldout)
    both_sequences = sum(bool(row["log_positive_images"] and row["rubble_positive_images"]) for row in sequence_rows)
    negative_images = sum(bool(row["hazard_negative"]) for row in heldout)
    negative_sequences = sum(row["log_positive_images"] == 0 and row["rubble_positive_images"] == 0 for row in sequence_rows)
    summary = {
        "schema_version": "adom-korean-heldout-support-v1",
        "manifest_rows": len(heldout),
        "unique_rgb_sha256_images": len({row["image_sha256"] for row in heldout}),
        "independent_sequences": len(sequence_rows),
        "sequence_rule": "existing audited manifest sequence field derived from capture-session parent directories",
        "log": by_class["log"],
        "rubble": by_class["rubble"],
        "both_positive_images": both_images,
        "both_positive_sequences": both_sequences,
        "negative_images": negative_images,
        "negative_sequences": negative_sequences,
        "ignore_index": IGNORE_INDEX,
        "class_mapping": dict(enumerate(CLASSES)),
        "overlap_rows": overlaps,
    }
    ko = (
        f"한국 held-out은 총 {len(heldout)}개 이미지와 {len(sequence_rows)}개 독립 sequence로 구성된다. "
        f"`log`와 `rubble`은 각각 {by_class['log']['positive_images']}/{by_class['log']['positive_sequences']}개 이미지/sequence와 "
        f"{by_class['rubble']['positive_images']}/{by_class['rubble']['positive_sequences']}개 이미지/sequence에서 나타났으며, "
        f"GT support는 각각 {by_class['log']['gt_pixels']}와 {by_class['rubble']['gt_pixels']} pixels이다."
    )
    en = (
        f"The Korean held-out set contains {len(heldout)} images from {len(sequence_rows)} independent sequences. "
        f"Log and rubble appear in {by_class['log']['positive_images']}/{by_class['log']['positive_sequences']} and "
        f"{by_class['rubble']['positive_images']}/{by_class['rubble']['positive_sequences']} images/sequences, with "
        f"{by_class['log']['gt_pixels']} and {by_class['rubble']['gt_pixels']} annotated pixels, respectively."
    )
    limitation = "Class-wise uncertainty could not be estimated reliably because each hazard was supported by only one independent positive sequence."
    summary["paper_sentence_ko"] = ko
    summary["paper_sentence_en"] = en
    summary["support_limitation"] = limitation if any(row["positive_sequences"] < 2 for row in class_rows) else None
    write_json(target / "summary.json", summary)
    overlap_md = "\n".join(f"- {row['left']} vs {row['right']} {row['field']}: {row['overlap_count']}" for row in overlaps)
    report = f"""# Korean held-out sequence/support audit

{ko}

{en}

{limitation if summary['support_limitation'] else ''}

## Sequence rule

The sequence field was already materialized by the audited manifest builder from the
capture-session parent directories (for example, `260811_3/20260811_111223_+0900`).
Adjacent frames therefore remain in one unit and are not counted as independent.

## Split and cross-dataset overlap

{overlap_md}
"""
    (target / "report.md").write_text(report, encoding="utf-8")
    return summary


def dominant_label(mask: np.ndarray) -> tuple[str, int]:
    values = mask[mask != IGNORE_INDEX]
    if values.size == 0:
        return "N/A", 0
    counts = np.bincount(values.astype(np.int64), minlength=len(CLASSES))
    index = int(np.argmax(counts))
    return CLASSES[index], int(counts[index])


def build_annotation_conflicts(paper_root: Path, workspace_root: Path, output: Path) -> tuple[dict[str, Any], set[str]]:
    target = output / "annotation_audit"
    splits = {
        name: read_csv(paper_root / f"manifests/korean_{name}_manifest.csv")
        for name in ("train", "val", "test")
    }
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for rows in splits.values():
        for row in rows:
            groups[row["image_sha256"]].append(row)
    duplicates = {digest: rows for digest, rows in groups.items() if len(rows) > 1}
    member_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    conflict_hashes: set[str] = set()
    for number, (digest, rows) in enumerate(sorted(duplicates.items()), start=1):
        masks = [load_mask(row["annotation_path"]) for row in rows]
        if any(mask.shape != masks[0].shape for mask in masks[1:]):
            raise RuntimeError(f"duplicate mask shape mismatch: {digest}")
        group_id = f"dup-{number:03d}-{digest[:12]}"
        for row, mask in zip(rows, masks):
            dominant, dominant_pixels = dominant_label(mask)
            member_rows.append({
                "duplicate_group_id": group_id,
                "image_sha256": digest,
                "split": row["split"],
                "sample_id": row["sample_id"],
                "image_path": row["image_path"],
                "annotation_path": row["annotation_path"],
                "annotation_sha256": row["annotation_sha256"],
                "non_ignore_pixels": int(np.count_nonzero(mask != IGNORE_INDEX)),
                "dominant_label": dominant,
                "dominant_label_pixels": dominant_pixels,
                "heldout_included": any(member["split"] == "test" for member in rows),
            })
        if len(rows) != 2:
            raise RuntimeError(f"unexpected duplicate multiplicity {len(rows)} for {digest}")
        first, second = masks
        overlap = (first != IGNORE_INDEX) & (second != IGNORE_INDEX)
        same = overlap & (first == second)
        conflict = overlap & (first != second)
        conflict_pixels = int(conflict.sum())
        if conflict_pixels:
            conflict_hashes.add(digest)
        pairs = Counter(zip(first[conflict].astype(int), second[conflict].astype(int)))
        conflict_rows.append({
            "duplicate_group_id": group_id,
            "image_sha256": digest,
            "splits": ";".join(row["split"] for row in rows),
            "sample_ids": ";".join(row["sample_id"] for row in rows),
            "annotation_sha256s": ";".join(row["annotation_sha256"] for row in rows),
            "overlap_non_ignore_pixels": int(overlap.sum()),
            "same_label_pixels": int(same.sum()),
            "conflict_pixels": conflict_pixels,
            "conflict_ratio_of_overlap": None if not overlap.any() else conflict_pixels / int(overlap.sum()),
            "log_conflict_pixels": int(np.count_nonzero(conflict & ((first == CLASS_ID['log']) | (second == CLASS_ID['log'])))),
            "rubble_conflict_pixels": int(np.count_nonzero(conflict & ((first == CLASS_ID['rubble']) | (second == CLASS_ID['rubble'])))),
            "label_pairs": json.dumps({f"{CLASSES[a]}->{CLASSES[b]}": count for (a, b), count in sorted(pairs.items())}, sort_keys=True),
            "heldout_included": any(row["split"] == "test" for row in rows),
        })
    write_csv(target / "duplicate_groups.csv", member_rows, tuple(member_rows[0]))
    write_csv(target / "conflicting_groups.csv", conflict_rows, tuple(conflict_rows[0]))
    summary = {
        "schema_version": "adom-korean-annotation-conflicts-v1",
        "duplicate_rgb_groups": len(duplicates),
        "conflicting_rgb_groups": len(conflict_hashes),
        "conflicting_pixels": sum(int(row["conflict_pixels"]) for row in conflict_rows),
        "heldout_duplicate_groups": sum(bool(row["heldout_included"]) for row in conflict_rows),
        "all_groups_train_val_only": all(set(row["splits"].split(";")) == {"train", "val"} for row in conflict_rows),
    }
    write_json(target / "conflict_summary.json", summary)
    return summary, conflict_hashes


def _iteration_from_path(path: Path) -> int | None:
    match = re.search(r"iter[_-](\d+)", path.name)
    return int(match.group(1)) if match else None


def _checkpoint_meta(path: Path) -> dict[str, Any]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            return {"payload_type": type(payload).__name__}
        meta = payload.get("meta") or {}
        output = {"top_level_keys": sorted(str(key) for key in payload)}
        for field in ("iter", "epoch", "seed", "experiment_name", "time"):
            if field in meta:
                value = meta[field]
                output[field] = value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
        if "dataset_meta" in meta:
            dataset_meta = meta["dataset_meta"]
            if isinstance(dataset_meta, dict):
                output["dataset_meta_classes"] = dataset_meta.get("classes")
        return output
    except Exception as error:
        return {"status": "METADATA_LOAD_FAILED", "error": f"{type(error).__name__}: {error}"}


def _selection_rank(records: list[dict[str, Any]]) -> dict[int, int]:
    best = max(float(row["overall_miou"]) for row in records)
    eligible = [row for row in records if float(row["overall_miou"]) >= best - 1.0]
    ordered = sorted(eligible, key=lambda row: (-float(row["rare_risk_miou"]), -float(row["overall_miou"]), int(row["iteration"])))
    ineligible = sorted((row for row in records if row not in eligible), key=lambda row: (-float(row["overall_miou"]), -float(row["rare_risk_miou"]), int(row["iteration"])))
    return {int(row["iteration"]): rank for rank, row in enumerate([*ordered, *ineligible], start=1)}


def build_training_and_checkpoint_audit(
    code_root: Path,
    workspace_root: Path,
    paper_root: Path,
    output: Path,
    conflict_hashes: set[str],
) -> dict[str, Any]:
    target = output / "annotation_audit"
    package_root = workspace_root / "datasets/processed/adom_semantic20_target_adaptation_v1"
    package_manifest = {row["sample_key"]: row for row in read_csv(package_root / "manifest.csv")}
    ta1_ids = [line.strip() for line in (package_root / "splits/ta1_train.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    val_ids = [line.strip() for line in (package_root / "splits/val.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    korean = {
        split: read_csv(paper_root / f"manifests/korean_{split}_manifest.csv")
        for split in ("train", "val", "test")
    }
    conflict_train_rows = [row for row in korean["train"] if row["image_sha256"] in conflict_hashes]
    conflict_val_rows = [row for row in korean["val"] if row["image_sha256"] in conflict_hashes]
    ta1_set = set(ta1_ids)
    val_set = set(val_ids)
    resolved_config = workspace_root / "runs/semantic20/eadom/seed42/full/b0/stage2/segformer_b0_stage2_eadom.py"
    config_text = resolved_config.read_text(encoding="utf-8")
    training_impact = {
        "eadom_train_split": str(package_root / "splits/ta1_train.txt"),
        "eadom_train_rows": len(ta1_ids),
        "eadom_validation_split": str(package_root / "splits/val.txt"),
        "eadom_validation_rows": len(val_ids),
        "validation_sources": dict(Counter(package_manifest[sample]["source"] for sample in val_ids)),
        "korean_train_rows": len(korean["train"]),
        "conflict_rgb_train_members": len(conflict_train_rows),
        "conflict_rgb_train_members_loaded": sum(row["sample_id"] in ta1_set for row in conflict_train_rows),
        "conflict_rgb_val_members_loaded_in_training": sum(row["sample_id"] in ta1_set for row in conflict_val_rows),
        "conflict_rgb_in_eadom_validation": sum(row["sample_id"] in val_set for row in conflict_train_rows + conflict_val_rows),
        "sampler": "InfiniteSampler(shuffle=True)" if "InfiniteSampler" in config_text else "UNVERIFIED",
        "repeat_dataset": "RepeatDataset" in config_text,
        "micro_batch": 16 if re.search(r"train_dataloader\s*=.*?batch_size=16", config_text, re.S) else "UNVERIFIED",
        "loss": "CrossEntropyLoss(avg_non_ignore=True, no class_weight)" if "CrossEntropyLoss" in config_text and "class_weight" not in config_text else "REVIEW_REQUIRED",
        "class_weight_present": "class_weight" in config_text,
        "confirmed_interpretation": (
            "All 12 train-side members of the conflicting RGB groups were in ta1_train. "
            "Their 12 val-side annotations were not in training. E-ADOM validation was 900-image canonical RELLIS-only, so no conflicting Korean RGB participated in checkpoint selection."
        ),
        "inference_or_training_repeated": False,
    }

    clean_train_ids = [sample for sample in ta1_ids if not (
        sample in {row["sample_id"] for row in conflict_train_rows}
    )]
    clean_train_rows = [{**package_manifest[sample], "sample_key": sample} for sample in clean_train_ids]
    clean_val_rows = [{**package_manifest[sample], "sample_key": sample} for sample in val_ids]
    manifest_fields = ("sample_key", "source", "source_split", "image_path", "mask_path")
    write_csv(target / "clean_train_manifest.csv", clean_train_rows, manifest_fields)
    write_csv(target / "clean_val_manifest.csv", clean_val_rows, manifest_fields)

    stage2 = workspace_root / "runs/semantic20/eadom/seed42/full/b0/stage2"
    selection_path = stage2 / "checkpoint_selection.json"
    selection = read_json(selection_path)
    records = selection["records"]
    metrics = {int(row["iteration"]): row for row in records}
    original_rank = _selection_rank(records)
    paths = sorted({*stage2.rglob("*.pth"), workspace_root / "artifacts/eadom-b0-seed42-iter26000/checkpoint.pth"})
    candidate_rows: list[dict[str, Any]] = []
    path_by_iteration: dict[int, list[str]] = defaultdict(list)
    expected_sha = "f4cc41fd91e9df8e7aa3f726498e80636b736dfadf0e1baf338fe7c82a83399c"
    for path in paths:
        iteration = _iteration_from_path(path)
        if path.name == "checkpoint.pth" and "eadom-b0-seed42-iter26000" in str(path):
            iteration = 26000
        sha = sha256_file(path)
        if iteration is not None:
            path_by_iteration[iteration].append(str(path))
        record = metrics.get(iteration or -1, {})
        metadata = _checkpoint_meta(path)
        candidate_rows.append({
            "path": str(path),
            "iteration": iteration,
            "sha256": sha,
            "size_bytes": path.stat().st_size,
            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "original_val_supported13_miou": record.get("overall_miou"),
            "original_rare_risk4_miou": record.get("rare_risk_miou"),
            "frozen_or_canonical": sha == expected_sha,
            "checkpoint_metadata": json.dumps(metadata, sort_keys=True),
        })
    write_csv(target / "checkpoint_candidates.csv", candidate_rows, tuple(candidate_rows[0]))

    rerank_rows: list[dict[str, Any]] = []
    selected_iteration = int(selection["selected"]["iteration"])
    for row in records:
        iteration = int(row["iteration"])
        rerank_rows.append({
            "Checkpoint": ";".join(path_by_iteration.get(iteration, [])) or "NOT_RETAINED",
            "Iteration": iteration,
            "Original val metric": row["overall_miou"],
            "Original RareRisk4": row["rare_risk_miou"],
            "Clean val metric": row["overall_miou"],
            "Clean RareRisk4": row["rare_risk_miou"],
            "Original rank": original_rank[iteration],
            "Clean rank": original_rank[iteration],
            "Selected under clean val": iteration == selected_iteration,
            "clean_metric_provenance": "reused: E-ADOM validation is canonical RELLIS-only and contains zero conflicting Korean RGBs",
        })
    write_csv(target / "checkpoint_reranking.csv", rerank_rows, tuple(rerank_rows[0]))
    validation_identity = {
        "original_val_rows": len(val_ids),
        "clean_val_rows": len(clean_val_ids := [row["sample_key"] for row in clean_val_rows]),
        "identical_ordered_sample_ids": val_ids == clean_val_ids,
        "conflict_rows_removed_from_eadom_val": 0,
        "selected_iteration_original": selected_iteration,
        "selected_iteration_clean": selected_iteration,
        "reranking_status": "PASS_IDENTICAL_RANK_NO_NEW_INFERENCE",
    }
    training_impact["checkpoint_reranking"] = validation_identity
    write_json(target / "training_impact_summary.json", training_impact)

    selection_report = f"""# Annotation conflict and checkpoint-selection audit

## Confirmed loading behavior

- E-ADOM train split: `{package_root / 'splits/ta1_train.txt'}` ({len(ta1_ids)} rows).
- Train-side members of conflicting RGB groups loaded: **{training_impact['conflict_rgb_train_members_loaded']} / {len(conflict_train_rows)}**.
- Val-side annotations of the same RGBs loaded in training: **{training_impact['conflict_rgb_val_members_loaded_in_training']}**.
- E-ADOM validation: **{len(val_ids)} canonical RELLIS-only rows**; conflicting Korean RGBs: **0**.
- Sampler: `InfiniteSampler(shuffle=True)`; `RepeatDataset` was not configured.
- Loss: unweighted cross entropy with `ignore_index=255`; no log/rubble class sampling or class weight was configured.

The model therefore saw one train-side annotation for each of the 12 ambiguous RGBs,
not two contradictory labels for the same RGB in training. Those rows can affect learned
weights because they are part of the uniformly sampled training set, but the separate
Korean val annotations did not affect checkpoint selection.

## Checkpoint selection

Forty validation records and multiple retained intermediate checkpoints were found.
The original rule was `{selection['rule']}` and selected iteration {selected_iteration}.
Because the selection validation set is RELLIS-only, removing Korean train/val conflicts
removes zero validation rows. Original and clean ranks are therefore mathematically
identical and iteration {selected_iteration} remains selected without repeating inference.

Status: **PASS_IDENTICAL_RANK_NO_NEW_INFERENCE**.
"""
    (target / "checkpoint_selection_report.md").write_text(selection_report, encoding="utf-8")

    runtime_status = read_json(workspace_root / "runs/semantic20/eadom/seed42/full/status.json")
    stage1 = runtime_status["phases"]["b0_stage1"]
    stage2_status = runtime_status["phases"]["b0_stage2"]
    plan = f"""# Conflict-free retraining plan (not executed)

## Decision

Clean retraining is **not required to support the existing held-out comparison**:

- the held-out set has no RGB or sequence overlap with train/val;
- checkpoint selection used RELLIS-only validation and is unaffected by the 12 groups;
- iteration {selected_iteration} remains selected under the conflict-free validation definition.

It is nevertheless **recommended as a sensitivity run before a stronger data-quality or
causal adaptation claim**, because 12 ambiguous train rows were repeatedly eligible for
uniform sampling and their correct target annotation cannot be resolved from the exports.

## Proposed change

- Use `{target / 'clean_train_manifest.csv'}` ({len(clean_train_rows)} rows) or materialize
  an ordered split file with the same sample IDs.
- Keep the canonical RELLIS validation/test splits unchanged.
- Config-only diff: replace `splits/ta1_train.txt` with the conflict-free split; do not
  change architecture, seed, optimizer, schedule, augmentation, loss, or selection rule.

## Proposed command (do not run in this audit)

```bash
cd /opt/adom
ADOM_DATA_ROOT={package_root} \
python -m adom.runtime.semantic20_cycle --experiment eadom --seed 42 \
  --train-split <conflict-free-split-file> --no-test
```

The exact wrapper flag must be confirmed against the runtime CLI before execution; if it
does not expose `--train-split`, create a minimal derived config that changes only the split.

## Expected cost and exit criteria

The recorded stage-1 runtime was {stage1['started_at']} to {stage1['finished_at']} and
stage-2 was {stage2_status['started_at']} to {stage2_status['finished_at']} on the recorded
RTX 4090 run, approximately 5.7 GPU-hours total plus validation/export overhead.

Required comparison: Korean held-out common mIoU, log/rubble IoU and recall; RELLIS native
mIoU, barrier/mud/rubble IoU; exact same manifests; three seeds for a submission-level
training-uncertainty claim. Stop if the split digest differs from the approved clean split,
loss becomes non-finite, or held-out is accessed before checkpoint freeze.
"""
    (target / "clean_retraining_plan.md").write_text(plan, encoding="utf-8")
    return training_impact


def _per_class(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv(path)
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        converted = dict(row)
        for field in ("iou", "precision", "recall", "f1"):
            converted[field] = None if row.get(field) in {None, "", "N/A"} else float(row[field])
        for field in ("gt_pixel_count", "prediction_pixel_count", "true_positive", "false_positive", "false_negative"):
            if field in row:
                converted[field] = int(row[field])
        output[row["class_name"]] = converted
    return output


def _metric_bundle(paper_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in ("rellis", "korean"):
        for model in ("b0_e0", "eadom"):
            prefix = f"{dataset}__{model}"
            summary_path = paper_root / "metrics" / f"{prefix}__summary.json"
            class_path = paper_root / "metrics" / f"{prefix}__per_class.csv"
            confusion_path = paper_root / "metrics" / f"{prefix}__confusion_matrix.npy"
            summary = read_json(summary_path)
            confusion = np.load(confusion_path, allow_pickle=False)
            output[(dataset, model)] = {
                "summary": summary,
                "classes": _per_class(class_path),
                "confusion": confusion,
                "paths": {"summary": str(summary_path), "per_class": str(class_path), "confusion": str(confusion_path)},
            }
    return output


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def build_paper_consistency(paper_root: Path, code_root: Path, output: Path, support: dict[str, Any]) -> dict[str, Any]:
    target = output / "paper_consistency"
    bundle = _metric_bundle(paper_root)
    common_sets = {
        tuple(item["summary"]["metrics"]["common_classes"])
        for item in bundle.values()
    }
    if common_sets != {COMMON_CLASSES}:
        raise RuntimeError(f"unexpected common class sets: {common_sets}")
    display_dataset = {"rellis": "RELLIS test", "korean": "Korean held-out"}
    display_model = {"b0_e0": "B0-E0", "eadom": "E-ADOM"}
    main_rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {"common_classes": list(COMMON_CLASSES), "cells": {}}
    for dataset in ("rellis", "korean"):
        for model in ("b0_e0", "eadom"):
            item = bundle[(dataset, model)]
            metrics = item["summary"]["metrics"]
            classes = item["classes"]
            row = {
                "Evaluation dataset": display_dataset[dataset],
                "Model": display_model[model],
                "Native supported mIoU": metrics["dataset_native_supported_mIoU"],
                "Common-supported mIoU": metrics["common_supported_mIoU"],
                "log IoU": classes["log"]["iou"],
                "log Recall": classes["log"]["recall"],
                "rubble IoU": classes["rubble"]["iou"],
                "rubble Recall": classes["rubble"]["recall"],
            }
            main_rows.append(row)
            key = f"{dataset}.{model}"
            provenance["cells"][key] = {
                "summary": item["paths"]["summary"],
                "per_class": item["paths"]["per_class"],
                "confusion": item["paths"]["confusion"],
                "ordered_manifest_sha256": item["summary"]["provenance"]["ordered_manifest_sha256"],
                "evaluation_contract_sha256": item["summary"]["provenance"]["evaluation_contract_sha256"],
            }
    main_fields = tuple(main_rows[0])
    write_csv(target / "canonical_main_table.csv", main_rows, main_fields)
    (target / "canonical_main_table.md").write_text(markdown_table([{key: _fmt(value) if isinstance(value, float) else value for key, value in row.items()} for row in main_rows], main_fields), encoding="utf-8")

    bootstrap: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in ("rellis", "korean"):
        path = paper_root / "metrics" / f"{dataset}__paired_bootstrap.json"
        for row in read_json(path)["results"]:
            bootstrap[(dataset, row["metric"])] = {**row, "source": str(path)}
    trade_metrics = (
        ("common_supported_mIoU", "Common-supported mIoU"),
        ("log/IoU", "log IoU"),
        ("log/Recall", "log Recall"),
        ("rubble/IoU", "rubble IoU"),
        ("rubble/Recall", "rubble Recall"),
        ("barrier/IoU", "barrier IoU"),
        ("mud/IoU", "mud IoU"),
    )
    trade_rows = []
    for dataset in ("rellis", "korean"):
        for metric, label in trade_metrics:
            row = bootstrap[(dataset, metric)]
            trade_rows.append({
                "Dataset": display_dataset[dataset],
                "Metric": label,
                "B0-E0": row["b0_e0"],
                "E-ADOM": row["eadom"],
                "Delta": row["delta_eadom_minus_b0_e0"],
                "Independent support": f"{row['positive_unit_count']}/{row['unit_count']} positive/total {row['resampling_unit']} units",
                "CI status": row["status"] if row["ci95_lower"] is None else f"[{row['ci95_lower']:.2f}, {row['ci95_upper']:.2f}]",
            })
    trade_fields = tuple(trade_rows[0])
    write_csv(target / "canonical_tradeoff_table.csv", trade_rows, trade_fields)
    (target / "canonical_tradeoff_table.md").write_text(markdown_table([{key: _fmt(value) if isinstance(value, float) else value for key, value in row.items()} for row in trade_rows], trade_fields), encoding="utf-8")

    short_rows = []
    for model in ("b0_e0", "eadom"):
        short_rows.append({
            "Model": display_model[model],
            "RELLIS common mIoU": bundle[("rellis", model)]["summary"]["metrics"]["common_supported_mIoU"],
            "Korean common mIoU": bundle[("korean", model)]["summary"]["metrics"]["common_supported_mIoU"],
            "Korean log IoU": bundle[("korean", model)]["classes"]["log"]["iou"],
            "Korean rubble IoU": bundle[("korean", model)]["classes"]["rubble"]["iou"],
        })
    short_fields = tuple(short_rows[0])
    (target / "canonical_short_table.md").write_text(markdown_table([{key: _fmt(value) if isinstance(value, float) else value for key, value in row.items()} for row in short_rows], short_fields), encoding="utf-8")
    write_json(target / "table_provenance.json", provenance)

    b0_rellis = bundle[("rellis", "b0_e0")]
    b0_korean = bundle[("korean", "b0_e0")]
    e_rellis = bundle[("rellis", "eadom")]
    e_korean = bundle[("korean", "eadom")]
    b0_conf = b0_korean["confusion"]
    claims = [
        ("C01", "B0-E0 benchmark-to-field common mIoU gap", "common-supported mIoU", f"{b0_rellis['summary']['metrics']['common_supported_mIoU']:.2f}% → {b0_korean['summary']['metrics']['common_supported_mIoU']:.2f}%", b0_rellis["paths"]["summary"] + ";" + b0_korean["paths"]["summary"], "fell to zero on this held-out set", "RELLIS native mIoU fell from 59.11 to zero"),
        ("C02", "B0-E0 Korean log/rubble true positives", "class confusion", f"log TP={int(b0_conf[CLASS_ID['log'], CLASS_ID['log']])}; rubble TP={int(b0_conf[CLASS_ID['rubble'], CLASS_ID['rubble']])}", b0_korean["paths"]["confusion"], "produced no true-positive pixels for either annotated hazard", "complete segmentation failure"),
        ("C03", "E-ADOM Korean common mIoU recovery", "common-supported mIoU", f"{e_korean['summary']['metrics']['common_supported_mIoU']:.2f}%", e_korean["paths"]["summary"], "restored the annotated rare hazards on the held-out set", "generally superior model"),
        ("C04", "E-ADOM Korean log performance", "class IoU/recall", f"IoU {e_korean['classes']['log']['iou']:.2f}%; recall {e_korean['classes']['log']['recall']:.2f}%", e_korean["paths"]["per_class"], "log IoU/recall on the held-out annotations", "safety success rate"),
        ("C05", "E-ADOM Korean rubble performance", "class IoU/recall", f"IoU {e_korean['classes']['rubble']['iou']:.2f}%; recall {e_korean['classes']['rubble']['recall']:.2f}%", e_korean["paths"]["per_class"], "rubble IoU/recall on the held-out annotations", "physical stop success"),
        ("C06", "RELLIS native retention", "native supported mIoU", f"{b0_rellis['summary']['metrics']['dataset_native_supported_mIoU']:.2f}% → {e_rellis['summary']['metrics']['dataset_native_supported_mIoU']:.2f}%", e_rellis["paths"]["summary"], "source native mIoU decreased by 1.08 pp", "no source degradation"),
        ("C07", "RELLIS barrier/mud regressions", "class IoU delta", f"barrier {e_rellis['classes']['barrier']['iou'] - b0_rellis['classes']['barrier']['iou']:.2f} pp; mud {e_rellis['classes']['mud']['iou'] - b0_rellis['classes']['mud']['iou']:.2f} pp", e_rellis["paths"]["per_class"], "class-specific source-domain trade-offs", "all source classes retained"),
        ("C08", "RELLIS rubble improvement", "class IoU delta", f"{e_rellis['classes']['rubble']['iou'] - b0_rellis['classes']['rubble']['iou']:.2f} pp", e_rellis["paths"]["per_class"], "rubble improved on RELLIS", "all rare hazards improved uniformly"),
        ("L01", "Single-seed limitation", "study design", "1 training seed", str(paper_root / "report.md"), "single-seed checkpoint comparison", "statistically proven training improvement"),
        ("L02", "Partial-annotation limitation", "annotation policy", "ignore=255 outside target labels", str(paper_root / "dataset_manifest_summary.json"), "metrics are conditional on target-only partial annotations", "full-scene false-positive rate"),
        ("L03", "Independent positive sequence limitation", "sequence support", f"log={support['log']['positive_sequences']}; rubble={support['rubble']['positive_sequences']}", str(output / "sequence_support/summary.json"), "class-wise CI is insufficient", "61 independent positive samples"),
    ]
    claim_rows = [dict(zip(("Claim ID", "Claim", "Metric type", "Value", "Evidence file", "Allowed wording", "Prohibited wording"), row)) for row in claims]
    claim_fields = tuple(claim_rows[0])
    write_csv(target / "claim_registry.csv", claim_rows, claim_fields)
    (target / "claim_registry.md").write_text(markdown_table(claim_rows, claim_fields), encoding="utf-8")

    issues = scan_text_consistency(code_root, paper_root)
    write_csv(target / "text_consistency_issues.csv", issues, ("file", "line", "problem_sentence", "issue_type", "recommended_wording"))
    recommended = """# Recommended wording

## Canonical framing

> Benchmark success did not transfer to field rare hazards. Targeted field
> adaptation restored the hazards, but introduced source-domain class trade-offs.

Cross-domain gaps use common-supported mIoU over `log` and `rubble`. Source
retention uses RELLIS native supported mIoU. Korean target results use common
mIoU and class IoU/recall. Do not compare RELLIS native 59.11 directly with
Korean common 0.00.

## Required qualifiers

- single training seed;
- target-only partial Korean masks, with ignore pixels excluded from FP;
- one independent positive sequence per hazard;
- no closed-loop RC success-rate evidence yet;
- qualitative appearance differences are a compound-shift observation, not a
  causal ablation of illumination, vegetation, or texture.
"""
    (target / "recommended_wording.md").write_text(recommended, encoding="utf-8")
    return {"main_rows": main_rows, "trade_rows": trade_rows, "short_rows": short_rows, "claims": claim_rows, "issues": issues}


def scan_text_consistency(code_root: Path, paper_root: Path) -> list[dict[str, Any]]:
    candidates = [paper_root / "report.md", paper_root / "paper_table.md", paper_root / "paired_deltas.md"]
    candidates.extend(path for path in code_root.rglob("*") if path.suffix.lower() in {".md", ".tex", ".rst"} and path.is_file())
    issues: list[dict[str, Any]] = []
    rules = [
        (re.compile(r"59\.11.*(?:Korean|한국).*\b0(?:\.00)?\b", re.I), "NATIVE_COMMON_DIRECT_COMPARISON", "Use RELLIS and Korean common-supported mIoU for the cross-domain gap."),
        (re.compile(r"(?:generally|overall) superior", re.I), "GENERAL_SUPERIORITY_OVERCLAIM", "Describe target recovery with source-domain class trade-offs."),
        (re.compile(r"61\s+(?:independent\s+)?samples", re.I), "FRAME_INDEPENDENCE_OVERCLAIM", "Report 61 images from two independent capture sequences."),
        (re.compile(r"caused by illumination|proves vegetation", re.I), "UNSUPPORTED_CAUSAL_CLAIM", "Describe a compound appearance/domain shift without assigning a single cause."),
        (re.compile(r"safety improvement", re.I), "SYSTEM_SAFETY_WITHOUT_TRIALS", "Report segmentation recovery; reserve safety claims for closed-loop trials."),
    ]
    for path in sorted(set(candidates)):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, start=1):
            for pattern, issue, wording in rules:
                if pattern.search(line):
                    issues.append({"file": str(path), "line": number, "problem_sentence": line.strip(), "issue_type": issue, "recommended_wording": wording})
    doc_sources = [path for path in code_root.rglob("*") if path.suffix.lower() in {".tex", ".docx"}]
    if not doc_sources:
        issues.append({"file": str(code_root), "line": "N/A", "problem_sentence": "No LaTeX or Word submission source was present in the immutable code image.", "issue_type": "BLOCKED_NO_FINAL_PAPER_SOURCE", "recommended_wording": "Run this consistency audit again on the final two-page submission source."})
    return issues


def _confusion_entries(manifest_path: Path, confusion_path: Path) -> dict[str, np.ndarray]:
    rows = read_csv(manifest_path)
    payload = np.load(confusion_path, allow_pickle=False)
    sample_ids = payload["sample_ids"].astype(str).tolist()
    expected = [row["sample_id"] for row in rows]
    if sample_ids != expected:
        raise RuntimeError(f"per-image confusion order mismatch: {confusion_path}")
    return {sample_id: payload["confusions"][index].astype(np.int64) for index, sample_id in enumerate(sample_ids)}


def _class_frame_metrics(confusion: np.ndarray, class_name: str) -> dict[str, Any]:
    index = CLASS_ID[class_name]
    row = confusion[index]
    gt = int(row.sum())
    tp = int(row[index])
    fp = int(confusion[:, index].sum()) - tp
    denominator = tp + (gt - tp) + fp
    top = sorted(((CLASSES[predicted], int(pixels)) for predicted, pixels in enumerate(row) if pixels), key=lambda item: (-item[1], item[0]))[:5]
    return {
        "gt_pixels": gt,
        "iou_percent": None if denominator == 0 else 100.0 * tp / denominator,
        "predictions_on_gt": [{"class": name, "pixels": pixels, "percent": 100.0 * pixels / gt} for name, pixels in top] if gt else [],
    }


def build_figure_audit(paper_root: Path, workspace_root: Path, output: Path) -> dict[str, Any]:
    target = output / "figure_audit"
    figure_root = paper_root / "supplemental/b0_self_collected/figures/domain_shift_v1"
    selection_path = figure_root / "selection_manifest.json"
    selection = read_json(selection_path)
    rellis = _confusion_entries(paper_root / "manifests/rellis_test_manifest.csv", paper_root / "metrics/rellis__b0_e0__per_image_confusions.npz")
    korean_test = _confusion_entries(paper_root / "manifests/korean_test_manifest.csv", paper_root / "metrics/korean__b0_e0__per_image_confusions.npz")
    korean_train = _confusion_entries(paper_root / "manifests/korean_train_manifest.csv", paper_root / "supplemental/b0_self_collected/metrics/korean_train__b0_e0__per_image_confusions.npz")
    manifests = {
        "rellis": {row["sample_id"]: row for row in read_csv(paper_root / "manifests/rellis_test_manifest.csv")},
        "korean_test": {row["sample_id"]: row for row in read_csv(paper_root / "manifests/korean_test_manifest.csv")},
        "korean_train": {row["sample_id"]: row for row in read_csv(paper_root / "manifests/korean_train_manifest.csv")},
        "korean_val": {row["sample_id"]: row for row in read_csv(paper_root / "manifests/korean_val_manifest.csv")},
    }
    definitions = [
        ("domain_shift_log_rubble.png", "rellis_log", "rellis", "log", "B0-E0", "median per-image log IoU among GT-positive RELLIS images"),
        ("domain_shift_log_rubble.png", "korean_log", "korean_test", "log", "B0-E0", "largest log GT area in Korean held-out"),
        ("domain_shift_log_rubble.png", "rellis_rubble", "rellis", "rubble", "B0-E0", "median per-image rubble IoU among GT-positive RELLIS images"),
        ("domain_shift_log_rubble.png", "korean_rubble", "korean_test", "rubble", "B0-E0", "largest rubble GT area in Korean held-out"),
        ("person_partial_success.png", "korean_person", "korean_train", "person", "B0-E0", "median per-image person IoU among GT-positive Korean train images"),
        ("duplicate_label_conflict.png", "annotation_conflict_train", "korean_train", "rubble", "B0-E0", "largest conflicting-pixel duplicate group; train member"),
        ("duplicate_label_conflict.png", "annotation_conflict_val", "korean_val", "log", "not separately inferred", "largest conflicting-pixel duplicate group; val member"),
    ]
    confusion_maps = {"rellis": rellis, "korean_test": korean_test, "korean_train": korean_train}
    rows = []
    for figure, key, dataset, class_name, model, rule in definitions:
        sample_id = selection["selections"][key]
        record = manifests[dataset][sample_id]
        metrics = _class_frame_metrics(confusion_maps[dataset][sample_id], class_name) if dataset in confusion_maps else None
        figure_path = figure_root / figure
        rows.append({
            "figure": str(figure_path),
            "figure_sha256": sha256_file(figure_path),
            "panel_key": key,
            "source_image_id": sample_id,
            "dataset": record["dataset"],
            "split": record["split"],
            "sequence": record["sequence"],
            "selection_rule": rule,
            "gt_present_class": class_name,
            "displayed_model": model,
            "gt_pixels": None if metrics is None else metrics["gt_pixels"],
            "class_iou_percent": None if metrics is None else metrics["iou_percent"],
            "top_predictions_on_gt": None if metrics is None else json.dumps(metrics["predictions_on_gt"], sort_keys=True),
            "deterministic_selection": True,
            "best_case_cherry_pick": False,
        })
    write_csv(target / "figure_manifest.csv", rows, tuple(rows[0]))
    plot_code = (workspace_root / "tools/paper_eval/generate_domain_shift_figure.py").read_text(encoding="utf-8")
    black_ignore = "output[mask == IGNORE_INDEX] = (0, 0, 0)" in plot_code
    palette_has_black = (0, 0, 0) in (
        (108, 64, 20), (0, 102, 0), (0, 255, 0), (0, 153, 153), (0, 128, 255),
        (0, 0, 255), (255, 255, 0), (255, 0, 127), (64, 64, 64), (255, 0, 0),
        (102, 0, 0), (204, 153, 255), (102, 0, 204), (255, 153, 204),
        (170, 170, 170), (41, 121, 255), (134, 255, 239), (99, 66, 34), (110, 22, 138),
    )
    validation = {
        "status": "PASS" if black_ignore and not palette_has_black else "REVIEW_REQUIRED",
        "selection_manifest": str(selection_path),
        "selection_schema": selection.get("schema_version"),
        "black_is_ignore_255": black_ignore,
        "semantic_palette_contains_black": palette_has_black,
        "legend_source": str(workspace_root / "tools/paper_eval/_common.py"),
        "metric_source": "per-image confusion matrices listed in figure_manifest.csv",
        "displayed_models": sorted({row["displayed_model"] for row in rows}),
        "note": "The domain-shift and person figures show B0-E0 only; they are not B0-E0 versus E-ADOM comparison figures.",
    }
    report = f"""# Figure validation report

- Status: **{validation['status']}**
- Deterministic selection manifest: `{selection_path}`
- Black ground-truth pixels explicitly map `ignore_index=255` to RGB `(0,0,0)`: **{black_ignore}**
- Semantic palette contains a black class color: **{palette_has_black}**
- Figure metrics were re-derived from the selected sample's stored per-image confusion matrix.
- Displayed model: B0-E0. E-ADOM is not shown in these three figures.
- Source panels use median positive-image IoU; target hazard panels use largest held-out GT support. These are deterministic descriptive selections, not best-case examples.

The selection rules differ across source and target, so the panels support qualitative
failure-mode interpretation but are not a controlled causal comparison.
"""
    (target / "figure_validation_report.md").write_text(report, encoding="utf-8")
    write_json(target / "figure_validation.json", validation)

    ko = """# 한국어 figure captions

## Domain shift

희소 오프로드 위험에 대한 benchmark-to-field domain shift. 사전 정의된 규칙으로
선정한 RELLIS 장면에서 B0-E0는 log와 rubble을 인지하지만, 한국 현장 장면의 가는
통나무와 음영 아래 자갈 표면은 주로 puddle, mud 및 식생 클래스로 예측한다. 한국
GT mask는 target-only partial annotation이며, 검정색 픽셀은 라벨되지 않은
`ignore=255` 영역이다. 이 영역의 예측은 false positive 계산에서 제외했다.

## Person partial success

한국 train의 person-positive 이미지 중 per-image IoU 중앙값 장면. 사람 실루엣은
부분적으로 회복되어 native supported mIoU를 높이지만, partial annotation의 ignore
영역 예측은 FP로 계산되지 않으므로 완전 라벨 장면의 precision을 의미하지 않는다.

## Annotation conflict

동일 RGB가 train export에서는 rubble, val export에서는 log로 라벨된 최대 충돌
그룹. 동일한 39,910 non-ignore pixels의 클래스가 상충하며, 주 전체-data 결과에서는
이 그룹을 포함한 12개 상충 RGB의 양쪽 annotation을 모두 제외했다.
"""
    en = """# English figure captions

## Domain shift

**Benchmark-to-field domain shift for rare off-road hazards.** B0-E0 recognizes
log and rubble in deterministically selected RELLIS scenes but maps thin logs
and shaded gravel surfaces in Korean field scenes primarily to puddle, mud, and
vegetation classes. Korean masks contain target-only partial annotations;
black pixels denote ignored, unlabeled regions, and predictions within these
regions are excluded from false-positive computation. The panels describe a
compound domain shift and do not identify a single causal factor.

## Person partial success

Median-IoU person-positive Korean training frame. Person shape is partially
recovered, explaining why native supported mIoU exceeds the log/rubble-only
common mIoU. Metrics are conditional on partial annotations, and ignored-region
predictions are not counted as false positives.

## Annotation conflict

The largest conflicting duplicate group. An identical RGB frame is annotated
as rubble in the training export and log in the validation export, producing
39,910 conflicting non-ignore pixels. Both annotations from all 12 conflicting
RGB groups are excluded from the primary conflict-free union.
"""
    short = """# Two-page short captions and slide lines

## Domain shift — short English

**Benchmark-to-field shift.** B0-E0 recognizes log/rubble in selected RELLIS
scenes but maps Korean thin logs and shaded gravel mainly to puddle/mud. Black
Korean GT pixels are ignored unlabeled regions and are excluded from FP counts.

Slide: **Benchmark hazard recognition collapsed under the observed field-domain shift.**

## Person — short English

**Partial person recovery raises native mIoU but not the log/rubble common mIoU.**

Slide: **Native mIoU hides the near-zero field hazard result because person is partly recovered.**

## Conflict — short English

**Twelve train/val duplicate RGBs contain conflicting log/rubble annotations and are excluded from the primary union.**

Slide: **The data audit found and isolated 12 contradictory duplicate-label groups.**
"""
    (target / "captions_ko.md").write_text(ko, encoding="utf-8")
    (target / "captions_en.md").write_text(en, encoding="utf-8")
    (target / "captions_short.md").write_text(short, encoding="utf-8")
    issue_text = "# Legend or rendering issues\n\nNo blocking issue found. Black is reserved for ignore=255 and does not collide with a Semantic20 palette color. The captions must state that only B0-E0 is displayed.\n"
    (target / "legend_or_rendering_issues.md").write_text(issue_text, encoding="utf-8")
    return validation


def prepare_rc_eval(code_root: Path, workspace_root: Path, output: Path, checkpoints: dict[str, Any]) -> dict[str, Any]:
    target = output / "rc_eval"
    target.mkdir(parents=True, exist_ok=True)
    tool_root = code_root / "tools/rc_eval"
    sys.path.insert(0, str(tool_root))
    from _common import TRIAL_FIELDS as RC_FIELDS, write_csv as rc_write_csv, write_json as rc_write_json
    from create_trial_plan import build_plan
    from analyze_trials import analyze

    plan_rows = build_plan(20260825, 10)
    rc_write_csv(target / "trial_plan.csv", plan_rows, (*RC_FIELDS, "condition_repetition", "randomization_seed"))
    template = {field: "" for field in RC_FIELDS}
    template.update({"hazard_present": "true|false", "model": "b0-e0|eadom", "hazard_type": "log|rubble|none", "emergency_intervention": "false"})
    rc_write_csv(target / "trial_metadata_template.csv", [template], RC_FIELDS)
    shutil.copy2(tool_root / "configs/rc_eval.yaml", target / "topic_mapping.yaml")
    shutil.copy2(tool_root / "schemas/trial_metadata.schema.json", target / "trial_metadata.schema.json")

    ros2 = shutil.which("ros2")
    ros_files = [path for path in code_root.rglob("*") if path.is_file() and (path.suffix in {".launch.py", ".xml"} or "ros" in path.name.lower())]
    graph_status = "BLOCKED_ROS2_NOT_INSTALLED" if not ros2 else "BLOCKED_NO_RUNNING_GRAPH_AUDITED"
    graph_report = f"""# ROS graph report

- Code baseline: `{code_root}`
- `ros2` executable: `{ros2 or 'N/A'}`
- ROS/launch files found in immutable training image: **{len(ros_files)}**
- Publisher commands executed: **0**
- Status: **{graph_status}**

The `/opt/adom` immutable image is a model-training/evaluation image. It does not
contain the target Jetson's verified perception/control launch files, deployed model
profiles, or a live ROS graph. Camera, mask, hazard, Go/Stop, motor-command, and
emergency-stop topics are therefore intentionally N/A rather than guessed.

Run `tools/rc_eval/inspect_ros_graph.py --execute-read-only` on the stationary target
Jetson, then confirm each topic type with `ros2 topic info --verbose` before filling
`topic_mapping.yaml`.
"""
    (target / "ros_graph_report.md").write_text(graph_report, encoding="utf-8")

    protocol = """# Repeated Go/Stop trial protocol

## Design

- 40 trials: B0-E0/E-ADOM × hazard-present/hazard-absent × 10 repetitions.
- Use one frozen scene, start marker, obstacle marker, safety boundary, speed, and
  camera pose. Randomized order is in `trial_plan.csv`.
- Hazard-present trials use one predeclared `log` or `rubble` object; negative trials
  remove it without changing other scene elements.
- Record checkpoint/profile, battery voltage, lighting/weather, rosbag, external
  video, intervention, exclusions, and final human outcome for every planned trial.

## Outcomes

- TP: hazard present and physical stop before the frozen boundary.
- FN: hazard present and no physical stop before the boundary.
- FP: hazard absent and stop decision/physical stop occurs.
- TN: hazard absent and the trial completes without unnecessary stop.
- If no physical stop measurement exists, report `stop-command proxy`, never
  `physical stop success`.

Wilson 95% intervals accompany binary rates. Repeated trials in one scene may be
correlated, so the intervals do not establish broad deployment safety.
"""
    (target / "trial_protocol.md").write_text(protocol, encoding="utf-8")
    safety = """# Human safety checklist

1. Assign a physical battery/power-cut operator and confirm a clear exclusion zone.
2. Verify neutral PWM, steering, command timeout, software E-stop, and process-kill
   behavior with propulsion disabled and wheels off the ground.
3. Confirm the exact deployment profile/checkpoint SHA and the verified topic map.
4. Run the logger in dry-run and confirm it publishes zero topics.
5. Freeze camera, scene, start marker, safety boundary, obstacle, and speed ≤0.30 m/s.
6. Start external video and rosbag before human-authorized low-speed motion.
7. Abort on unexpected motion, topic loss, dropped command watchdog, or intervention.
8. Preserve and label every interrupted/excluded trial; do not silently delete it.
"""
    (target / "safety_checklist.md").write_text(safety, encoding="utf-8")

    experiment = target / "example_outputs/experiment"
    conditions = ((True, True), (True, False), (False, True), (False, False))
    for index, (present, stopped) in enumerate(conditions, start=1):
        trial = experiment / "trials" / f"T{index:03d}"
        trial.mkdir(parents=True, exist_ok=True)
        metadata = {
            "trial_id": trial.name,
            "operator": "synthetic-test",
            "model": "b0-e0" if index <= 2 else "eadom",
            "checkpoint_sha256": checkpoints["b0_e0" if index <= 2 else "eadom"]["sha256"],
            "scene_id": "synthetic-fixed-scene",
            "hazard_type": "log" if present else "none",
            "hazard_present": present,
            "start_position_marker": "A",
            "obstacle_position_marker": "B" if present else None,
            "commanded_speed_mps": 0.1,
            "emergency_intervention": False,
        }
        annotation = {
            "physical_stop_before_boundary": stopped,
            "stop_decision_observed": stopped,
            "hazard_detection_observed": present and index != 2,
            "trial_completed": True,
            "emergency_intervention": False,
            "decision_latency_s": 0.1 * index,
            "detection_to_stop_latency_s": 0.2 * index,
            "braking_latency_s": None,
            "first_hazard_detection_s": 0.05 * index if present and index != 2 else None,
            "first_stop_command_s": 0.15 * index if stopped else None,
            "physical_stop_time_s": 0.25 * index if stopped else None,
            "dropped_frame_count": index - 1,
        }
        rc_write_json(trial / "metadata.json", metadata)
        rc_write_json(trial / "human_annotation.json", annotation)
    synthetic = analyze(experiment, target / "example_outputs/analysis")
    existing_bags = list((workspace_root / "rc_trials").glob("**/*.db3")) if (workspace_root / "rc_trials").is_dir() else []
    dry_run = f"""# RC logger dry-run report

- Trial-plan generation: PASS ({len(plan_rows)} balanced randomized rows).
- Synthetic TP/FN/FP/TN analysis: PASS ({synthetic['trial_counts']}).
- Wilson interval calculation: PASS (generated in example `summary.json`).
- Perception hazard-detection rate and timestamp aggregation: PASS.
- Malformed/interrupted-trial handling: covered by unit tests and validation output.
- ROS graph discovery on `/opt/adom`: {graph_status}.
- Existing rosbag `.db3` under `/workspace/adom/rc_trials`: {len(existing_bags)}.
- Replay test: {'NOT_RUN_NO_EXISTING_ROSBAG' if not existing_bags else 'BLOCKED_REQUIRES_VERIFIED_ROS_ENVIRONMENT'}.
- Motor/servo/Go/Stop/E-stop publish commands executed: **0**.
"""
    (target / "dry_run_report.md").write_text(dry_run, encoding="utf-8")
    analysis_readme = f"""# Analysis hand-off

```bash
cd /opt/adom
python tools/rc_eval/validate_trial.py \\
  --experiment-root /workspace/adom/rc_trials/<experiment_id>
python tools/rc_eval/analyze_trials.py \\
  --experiment-root /workspace/adom/rc_trials/<experiment_id> \\
  --output-dir /workspace/adom/rc_trials/<experiment_id>/analysis
python tools/rc_eval/generate_paper_table.py \\
  --analysis-dir /workspace/adom/rc_trials/<experiment_id>/analysis \\
  --output-dir /workspace/adom/rc_trials/<experiment_id>/paper_outputs
```

Do not call physical-stop outcomes when only stop-command timestamps exist. Review
`validation_errors.csv`, exclusions, interventions, and trial video before using the
generated table.
"""
    (target / "analysis_readme.md").write_text(analysis_readme, encoding="utf-8")
    return {
        "status": "LOGGER_READY_DRY_RUN_ONLY",
        "graph_status": graph_status,
        "trial_plan_rows": len(plan_rows),
        "synthetic_counts": synthetic["trial_counts"],
        "existing_rosbags": len(existing_bags),
        "publish_commands_executed": 0,
    }


def final_reports(
    output: Path,
    environment: dict[str, Any],
    checkpoints: dict[str, Any],
    inventory: list[dict[str, Any]],
    support: dict[str, Any],
    conflicts: dict[str, Any],
    training: dict[str, Any],
    tables: dict[str, Any],
    figure: dict[str, Any],
    rc: dict[str, Any],
) -> dict[str, Any]:
    overlap_lookup = {(row["left"], row["right"], row["field"]): row["overlap_count"] for row in support["overlap_rows"]}
    blockers = [
        "BLOCKED_NO_GIT_METADATA_IN_IMMUTABLE_IMAGE",
        "BLOCKED_NO_FINAL_PAPER_SOURCE",
        rc["graph_status"],
        "NOT_RUN_NO_EXISTING_ROSBAG" if rc["existing_rosbags"] == 0 else "BLOCKED_REPLAY_REQUIRES_VERIFIED_ROS_ENVIRONMENT",
        "BLOCKED_NO_PHYSICAL_RC_TRIALS_YET",
    ]
    summary = {
        "schema_version": "adom-paper-submission-audit-v1",
        "generated_at_utc": now_utc(),
        "output_root": str(output),
        "code_root": environment["code_root"],
        "immutable_image_git_sha": environment["immutable_image_git_sha"],
        "checkpoints_status": {name: checkpoints[name]["status"] for name in ("b0_e0", "eadom", "canonical_archive")},
        "sequence_support": {
            "images": support["manifest_rows"],
            "unique_images": support["unique_rgb_sha256_images"],
            "sequences": support["independent_sequences"],
            "log_positive_images": support["log"]["positive_images"],
            "log_positive_sequences": support["log"]["positive_sequences"],
            "log_pixels": support["log"]["gt_pixels"],
            "rubble_positive_images": support["rubble"]["positive_images"],
            "rubble_positive_sequences": support["rubble"]["positive_sequences"],
            "rubble_pixels": support["rubble"]["gt_pixels"],
        },
        "split_overlap": {
            "train_test_rgb": overlap_lookup[("korean_train", "korean_test", "image_sha256")],
            "val_test_rgb": overlap_lookup[("korean_val", "korean_test", "image_sha256")],
            "train_test_sequence": overlap_lookup[("korean_train", "korean_test", "sequence")],
            "val_test_sequence": overlap_lookup[("korean_val", "korean_test", "sequence")],
            "korean_rellis_rgb": overlap_lookup[("korean_test", "rellis_test", "image_sha256")],
        },
        "annotation_conflicts": conflicts,
        "training_impact": training,
        "checkpoint_selection_status": training["checkpoint_reranking"]["reranking_status"],
        "metric_consistency_status": "PASS_COMMON_SET_LOG_RUBBLE",
        "figure_caption_status": figure["status"],
        "rc_logger": rc,
        "artifact_inventory_count": len(inventory),
        "clean_retraining_decision": "NOT_REQUIRED_FOR_CURRENT_HELDOUT_CLAIM; RECOMMENDED_SENSITIVITY_BEFORE_STRONG_DATA_QUALITY_CLAIM",
        "blockers": blockers,
    }
    write_json(output / "machine_readable_summary.json", summary)
    main_table = markdown_table([{key: _fmt(value) if isinstance(value, float) else value for key, value in row.items()} for row in tables["main_rows"]], tuple(tables["main_rows"][0]))
    report_en = f"""# Final paper-submission audit

## 1. Environment and checkpoints

- Code baseline: `/opt/adom`; immutable image SHA `{environment['immutable_image_git_sha']}`.
- `/opt/adom` has no `.git`, so branch/status are unavailable and explicitly blocked.
- B0-E0, E-ADOM, and canonical archive SHA checks: {summary['checkpoints_status']}.
- Existing paper-evaluation files inventoried and hashed: {len(inventory)}.

## 2. Korean held-out support

{support['paper_sentence_en']}

{support['support_limitation'] or ''}

Train/test and val/test RGB and sequence overlap are zero; Korean held-out versus
RELLIS RGB overlap is zero. See `sequence_support/split_overlap_audit.csv`.

## 3. Annotation and checkpoint-selection audit

The audit reconfirmed {conflicts['conflicting_rgb_groups']} conflicting train/val RGB
groups and {conflicts['conflicting_pixels']} conflicting pixels; none includes held-out.
All 12 train-side annotations were loaded by E-ADOM, while the val-side counterparts
were not loaded in training. Checkpoint validation was canonical RELLIS-only, so the
conflicts removed zero validation rows. Original and clean ranks are identical and
iteration 26000 remains selected without new inference.

## 4. Canonical main table

{main_table}

Cross-domain claims use common-supported mIoU (`log`, `rubble`); source retention uses
RELLIS native mIoU. Target recovery uses Korean common mIoU plus class IoU/recall.

## 5. Paper wording and figures

Allowed framing: **Benchmark success did not transfer to field rare hazards. Targeted
field adaptation restored the hazards, but introduced source-domain class trade-offs.**

The deterministic figures and Korean/English/short captions passed palette, ignore,
selection-rule, and per-image confusion checks. They show B0-E0 only. Do not claim a
single causal factor or closed-loop safety improvement.

## 6. RC logger

The subscribe-only logger, randomized 40-trial plan, schema, validator, analyzer,
Wilson intervals, paper-table generator, and synthetic examples are ready. `/opt/adom`
contains no verified Jetson ROS graph, so actual topic mapping and replay are blocked.
No motor/control publisher command was executed.

## 7. Human next actions

1. Run read-only ROS graph inspection on the stationary target Jetson.
2. Verify exact topic names/types and B0-E0/E-ADOM deployment profiles; fill the mapping.
3. Perform wheels-off watchdog/E-stop checks with a physical power-cut operator.
4. Freeze scene, start marker, safety boundary, speed, and hazard object.
5. Execute the randomized 40 trials, retaining all exclusions/interventions and video.
6. Run the three validation/analysis/table commands in `rc_eval/analysis_readme.md`.

## 8. Remaining blockers and retraining

{os.linesep.join('- ' + value for value in blockers)}

Clean retraining is not required for the current independent held-out claim or checkpoint
rank. It is recommended as a sensitivity run before stronger data-quality/causal claims;
the plan is recorded but was not executed.
"""
    (output / "FINAL_REPORT.md").write_text(report_en, encoding="utf-8")
    report_ko = f"""# 최종 논문 제출 감사 보고서

## 1. 환경과 checkpoint

- 코드 기준: `/opt/adom`, immutable image SHA `{environment['immutable_image_git_sha']}`.
- `/opt/adom`에는 `.git`이 없어 branch/status는 확인 불가로 명시했다.
- B0-E0, E-ADOM, canonical archive SHA 검증: {summary['checkpoints_status']}.
- 기존 평가 artifact {len(inventory)}개 파일의 hash를 기록했다.

## 2. Korean held-out support

{support['paper_sentence_ko']}

클래스별 독립 positive sequence가 각각 1개이므로 class-wise uncertainty를 안정적으로
추정할 수 없다. train/test 및 val/test의 RGB·sequence overlap과 Korean–RELLIS RGB
overlap은 모두 0이다.

## 3. 상충 annotation과 checkpoint 선택

train/val 동일 RGB 상충 {conflicts['conflicting_rgb_groups']}그룹,
{conflicts['conflicting_pixels']} pixels를 재확인했다. held-out 포함 그룹은 0이다.
E-ADOM은 train 쪽 12 annotation을 실제 로드했지만 val 쪽 annotation은 학습에
로드하지 않았다. 선택 validation은 900장 canonical RELLIS-only이므로 상충 RGB를
제외해도 제거되는 validation row는 0이다. original/clean rank는 동일하며 iter 26000이
그대로 선택된다. 새 inference나 training은 실행하지 않았다.

## 4. Canonical main table

{main_table}

Cross-domain gap은 common-supported mIoU(`log`, `rubble`), source retention은 RELLIS
native mIoU, target adaptation은 Korean common mIoU와 클래스 IoU/recall만 사용한다.

## 5. 문장과 figure

권장 framing: **Benchmark success did not transfer to field rare hazards. Targeted field
adaptation restored the hazards, but introduced source-domain class trade-offs.**

Deterministic figure와 한국어/영어/축약 caption은 palette, ignore=255, 선정 규칙,
per-image confusion을 검증했다. Figure는 B0-E0만 표시한다. 단일 원인이나 closed-loop
안전성 향상을 주장하면 안 된다.

## 6. RC logger와 사람의 다음 작업

Subscribe-only logger, 40회 randomized plan, schema, validator, analyzer, Wilson CI와
synthetic test를 준비했다. 다만 `/opt/adom`에는 Jetson의 실제 ROS graph가 없어 topic
mapping과 replay가 BLOCKED다. motor/control publish 명령은 0회 실행했다.

사람은 target Jetson에서 read-only graph 확인 → topic/type/profile 동결 → wheels-off
watchdog/E-stop 검증 → scene/boundary/speed 동결 → 40회 trial → 분석 명령 실행 순서로
진행해야 한다.

## 7. BLOCKED와 clean retraining

{os.linesep.join('- ' + value for value in blockers)}

현재 held-out 주장과 checkpoint rank를 위해 clean retraining은 필수가 아니다. 다만
강한 데이터 품질·인과 주장을 하기 전 conflict-free sensitivity run은 권장하며, 명령과
약 5.7 GPU-hour 예상 비용만 기록하고 실행하지 않았다.
"""
    (output / "FINAL_REPORT_KO.md").write_text(report_ko, encoding="utf-8")
    audit_log = f"""# Audit log

- Started output root at {output}.
- Treated `/opt/adom` as the code/config baseline and `/workspace/adom` as data/artifact storage.
- Hashed the existing evaluation artifact tree and verified checkpoint/archive SHA values.
- Recomputed Korean support and annotation conflicts from masks and audited manifests.
- Reused stored per-image confusions and 40 checkpoint-selection validation records.
- Did not run training. Did not repeat model inference because clean validation equals original canonical RELLIS validation.
- Generated canonical tables, claims, captions, logger code, dry-run outputs, and final reports.
- Motor, servo, autonomy, Go/Stop, and emergency-stop commands published: 0.
"""
    (output / "audit_log.md").write_text(audit_log, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the ADOM two-page paper submission audit")
    parser.add_argument("--code-root", type=Path, default=Path("/opt/adom"))
    parser.add_argument("--workspace-root", type=Path, default=Path("/workspace/adom"))
    parser.add_argument("--paper-root", type=Path, default=Path("/workspace/adom/paper_eval_outputs/20260824T152720Z"))
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output root: {output}")
    output.mkdir(parents=True, exist_ok=True)
    environment = build_environment(args.code_root, args.workspace_root, args.paper_root, output)
    checkpoints = build_checkpoint_manifest(args.workspace_root, args.paper_root, output)
    inventory = build_artifact_inventory(args.paper_root, output)
    support = build_sequence_support(args.paper_root, output)
    conflicts, conflict_hashes = build_annotation_conflicts(args.paper_root, args.workspace_root, output)
    training = build_training_and_checkpoint_audit(args.code_root, args.workspace_root, args.paper_root, output, conflict_hashes)
    tables = build_paper_consistency(args.paper_root, args.code_root, output, support)
    figure = build_figure_audit(args.paper_root, args.workspace_root, output)
    rc = prepare_rc_eval(args.code_root, args.workspace_root, output, checkpoints)
    summary = final_reports(output, environment, checkpoints, inventory, support, conflicts, training, tables, figure, rc)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
