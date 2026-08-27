from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
import shutil
from pathlib import Path
from typing import Any

from mmengine.hooks import Hook
from mmseg.registry import HOOKS

from adom.evaluation_semantic20 import select_constrained_checkpoint


CANONICAL_TEST_UNLOCK_TOKEN = "final-model-confirmed"
FULL_TRAINING_APPROVAL_TOKEN = "user-approved"


def _unwrap_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def backbone_sha256(model: Any) -> str:
    backbone = _unwrap_model(model).backbone
    digest = hashlib.sha256()
    for name, tensor in sorted(backbone.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _write_audit(runner: Any, filename: str, payload: dict[str, Any]) -> None:
    path = Path(runner.work_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metric_value(metrics: dict[str, Any], suffix: str) -> float:
    matches = [value for key, value in metrics.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one validation metric ending with {suffix!r}, "
            f"found {[key for key in metrics if key.endswith(suffix)]}"
        )
    return float(matches[0])


def _replace_link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _publish_wandb(
    runner: Any,
    *,
    summary_prefix: str,
    metrics: dict[str, Any] | None = None,
    files: tuple[str, ...] = (),
) -> None:
    """Best-effort W&B persistence without making tracking a training gate."""

    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    try:
        if metrics:
            for key, value in metrics.items():
                wandb.run.summary[f"{summary_prefix}/{key}"] = float(value)
        root = Path(runner.work_dir)
        for filename in files:
            path = root / filename
            if path.is_file():
                wandb.save(str(path), base_path=str(root), policy="now")
    except Exception as error:  # pragma: no cover - depends on remote service state
        runner.logger.warning("W&B audit artifact upload failed: %s", error)


@HOOKS.register_module()
class CanonicalTestLockHook(Hook):
    """Reject canonical test execution unless the orchestrator unlocks it."""

    priority = "HIGHEST"

    def before_test(self, runner: Any) -> None:
        if os.getenv("ADOM_CANONICAL_TEST_UNLOCK") != CANONICAL_TEST_UNLOCK_TOKEN:
            raise RuntimeError(
                "Canonical test is locked. Select the final model using validation "
                "and run it through semantic20_cycle --run-test "
                "--final-test-model <b0|b2|b5>."
            )


@HOOKS.register_module()
class TA0AblationContractHook(Hook):
    """Fail closed when a TA0 ablation drifts from its common comparison contract."""

    priority = "HIGHEST"

    def before_train(self, runner: Any) -> None:
        contract = dict(runner.cfg.get("ta0_contract", {}))
        if not contract:
            raise RuntimeError("TA0 config is missing ta0_contract metadata")
        if contract.get("ablation_axis") == "combined_interaction_check" and bool(
            contract.get("provisional", True)
        ):
            raise RuntimeError(
                "TA0-R is a syntax-checked placeholder until independent ablation "
                "winners are committed; provisional combined training is forbidden"
            )
        checkpoint_sha = os.getenv("ADOM_EXPECTED_INITIAL_CHECKPOINT_SHA256", "")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", checkpoint_sha):
            raise RuntimeError(
                "TA0 requires the cycle-validated E0 SHA in "
                "ADOM_EXPECTED_INITIAL_CHECKPOINT_SHA256"
            )
        checkpoint_path_value = os.getenv("ADOM_INITIAL_CHECKPOINT", "")
        checkpoint_path = Path(checkpoint_path_value)
        if not checkpoint_path.is_file():
            raise RuntimeError(
                "TA0 requires the cycle-validated E0 path in ADOM_INITIAL_CHECKPOINT"
            )
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != checkpoint_sha.lower():
            raise RuntimeError("TA0 E0 checkpoint changed after cycle validation")
        train_dataset = runner.cfg["train_dataloader"]["dataset"]
        if train_dataset.get("split") != "splits/ta0_train.txt":
            raise RuntimeError("TA0 is locked to splits/ta0_train.txt")
        sampler = runner.train_dataloader.sampler
        requested_weights = getattr(sampler, "source_weights", None)
        if requested_weights != {"rellis3d": 1.0}:
            raise RuntimeError(f"TA0 source quota changed: {requested_weights}")
        accumulative = int(runner.cfg["optim_wrapper"].get("accumulative_counts", 1))
        micro_batch = int(runner.cfg["train_dataloader"]["batch_size"])
        world_size = int(getattr(runner, "world_size", 1))
        effective_batch = micro_batch * accumulative * world_size
        if effective_batch != int(contract.get("effective_batch", 16)):
            raise RuntimeError(f"TA0 effective batch changed: {effective_batch}")
        configured_seed = int(runner.cfg["randomness"]["seed"])
        if configured_seed != int(contract["seed"]):
            raise RuntimeError(
                f"TA0 seed mismatch: config={configured_seed}, contract={contract['seed']}"
            )
        runner_iterations = int(runner.cfg["train_cfg"]["max_iters"])
        if runner_iterations % accumulative:
            raise RuntimeError("TA0 runner iterations are not divisible by accumulation")
        phase_updates = runner_iterations // accumulative
        full_phase_updates = int(contract["phase_optimizer_updates"])
        full_total_updates = int(contract["total_optimizer_updates"])
        comparison_total_updates = int(
            os.getenv("ADOM_TA_TOTAL_OPTIMIZER_UPDATES", str(full_total_updates))
        )
        expected_phase_updates = round(
            comparison_total_updates * full_phase_updates / full_total_updates
        )
        if full_phase_updates < full_total_updates and full_phase_updates > full_total_updates / 2:
            expected_phase_updates = comparison_total_updates - round(
                comparison_total_updates
                * (full_total_updates - full_phase_updates)
                / full_total_updates
            )
        config_probe = os.getenv("ADOM_TA_CONFIG_PROBE", "").lower() == "true"
        if not config_probe and phase_updates != expected_phase_updates:
            raise RuntimeError(
                f"TA0 phase update drift: actual={phase_updates}, "
                f"expected={expected_phase_updates}, total={comparison_total_updates}"
            )
        if (
            not config_probe
            and phase_updates > 500
            and os.getenv("ADOM_TA0_FULL_TRAINING_APPROVED")
            != FULL_TRAINING_APPROVAL_TOKEN
        ):
            raise RuntimeError(
                "TA0 full training is locked pending explicit user approval; "
                "smoke/mini runs up to 500 optimizer updates remain available"
            )
        _write_audit(
            runner,
            "ta0_contract.json",
            {
                "schema_version": "adom-ta0-ablation-contract-v1",
                **contract,
                "validated_initial_checkpoint_sha256": checkpoint_sha.lower(),
                "validated_initial_checkpoint": str(checkpoint_path.resolve()),
                "actual_optimizer_updates": phase_updates,
                "comparison_total_optimizer_updates": comparison_total_updates,
                "effective_batch": effective_batch,
                "world_size": world_size,
                "config_probe": config_probe,
                "train_split": train_dataset["split"],
                "source_weights": requested_weights,
                "canonical_test_locked": True,
                "full_training_approved": phase_updates <= 500
                or os.getenv("ADOM_TA0_FULL_TRAINING_APPROVED")
                == FULL_TRAINING_APPROVAL_TOKEN,
            },
        )


@HOOKS.register_module()
class MetricArtifactHook(Hook):
    """Persist final validation and test metrics as machine-readable JSON."""

    priority = "LOWEST"

    def after_val_epoch(
        self,
        runner: Any,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if metrics:
            payload = {
                "iteration": runner.iter,
                "metrics": {key: float(value) for key, value in metrics.items()},
            }
            _write_audit(runner, "val_metrics.json", payload)
            history_path = Path(runner.work_dir) / "val_history.jsonl"
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
            _write_audit(
                runner,
                f"val_metrics_iter_{int(runner.iter)}.json",
                payload,
            )
            _publish_wandb(
                runner,
                summary_prefix="clean_v1/val",
                metrics=metrics,
                files=(
                    "val_metrics.json",
                    "val_history.jsonl",
                    "dataset_contract.json",
                    "class_support.json",
                ),
            )

    def after_test_epoch(
        self,
        runner: Any,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if metrics:
            payload = {
                "iteration": runner.iter,
                "metrics": {key: float(value) for key, value in metrics.items()},
            }
            _write_audit(runner, "test_metrics.json", payload)
            _publish_wandb(
                runner,
                summary_prefix="clean_v1/test",
                metrics=metrics,
                files=(
                    "test_metrics.json",
                    "confusion_matrix.json",
                    "semantic20_metrics.json",
                ),
            )


@HOOKS.register_module()
class ConstrainedCheckpointSelectionHook(Hook):
    """Persist the approved overall-noninferior, rare-risk-best checkpoint."""

    priority = "LOWEST"

    def __init__(self, tolerance_pp: float = 1.0) -> None:
        self.tolerance_pp = float(tolerance_pp)
        self.records: list[dict[str, Any]] = []

    def before_train(self, runner: Any) -> None:
        history_path = Path(runner.work_dir) / "checkpoint_selection_history.json"
        if history_path.is_file():
            payload = json.loads(history_path.read_text(encoding="utf-8"))
            self.records = list(payload.get("records", []))

    def after_val_epoch(
        self,
        runner: Any,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if not metrics or getattr(runner, "rank", 0) != 0:
            return
        iteration = int(runner.iter)
        record = {
            "iteration": iteration,
            "overall_miou": _metric_value(metrics, "mIoU/ValSupported13"),
            "rare_risk_miou": _metric_value(metrics, "mIoU/RareRisk4"),
            "metrics": {key: float(value) for key, value in metrics.items()},
        }
        self.records = [
            previous
            for previous in self.records
            if int(previous["iteration"]) != iteration
        ]
        self.records.append(record)
        self.records.sort(key=lambda value: int(value["iteration"]))

        best_overall = max(value["overall_miou"] for value in self.records)
        threshold = best_overall - self.tolerance_pp
        candidate_root = Path(runner.work_dir) / "selection_candidates"
        candidate_root.mkdir(parents=True, exist_ok=True)
        candidate_path = candidate_root / f"iter_{iteration}.pth"
        if record["overall_miou"] >= threshold and not candidate_path.is_file():
            runner.save_checkpoint(
                str(candidate_root),
                candidate_path.name,
                save_optimizer=False,
                save_param_scheduler=False,
                meta={
                    "iter": iteration,
                    "epoch": int(runner.epoch),
                    "iteration": iteration,
                    "selection_metrics": {
                        "ValSupported13_mIoU": record["overall_miou"],
                        "RareRisk4_mIoU": record["rare_risk_miou"],
                    },
                },
                by_epoch=False,
            )

        eligible_iterations = {
            int(value["iteration"])
            for value in self.records
            if value["overall_miou"] >= threshold
        }
        for path in candidate_root.glob("iter_*.pth"):
            try:
                stored_iteration = int(path.stem.removeprefix("iter_"))
            except ValueError:
                continue
            if stored_iteration not in eligible_iterations:
                path.unlink()

        selectable = [
            value
            for value in self.records
            if int(value["iteration"]) in eligible_iterations
            and (candidate_root / f"iter_{int(value['iteration'])}.pth").is_file()
        ]
        selected = select_constrained_checkpoint(
            selectable,
            tolerance_pp=self.tolerance_pp,
        )
        selected_candidate = candidate_root / f"iter_{int(selected['iteration'])}.pth"
        stable_path = Path(runner.work_dir) / (
            f"best_clean_selection_iter_{int(selected['iteration'])}.pth"
        )
        previous = list(Path(runner.work_dir).glob("best_clean_selection_iter_*.pth"))
        if not stable_path.is_file():
            for path in previous:
                path.unlink()
            _replace_link_or_copy(selected_candidate, stable_path)

        payload = {
            "schema_version": "semantic20-clean-v1",
            "rule": (
                "max RareRisk4 mIoU among checkpoints within "
                f"{self.tolerance_pp:.1f}pp of best ValSupported13 mIoU"
            ),
            "best_overall_miou": best_overall,
            "eligibility_threshold": threshold,
            "selected": {
                **selected,
                "checkpoint": str(stable_path.resolve()),
            },
            "records": self.records,
        }
        _write_audit(runner, "checkpoint_selection.json", payload)
        _write_audit(runner, "checkpoint_selection_history.json", payload)
        _publish_wandb(
            runner,
            summary_prefix="clean_v1/selection",
            metrics={
                "iteration": int(selected["iteration"]),
                "ValSupported13_mIoU": selected["overall_miou"],
                "RareRisk4_mIoU": selected["rare_risk_miou"],
            },
            files=("checkpoint_selection.json", "checkpoint_selection_history.json"),
        )


@HOOKS.register_module()
class FiniteLossHook(Hook):
    """Fail immediately when the optimization loss becomes NaN or Inf."""

    priority = "VERY_HIGH"

    def after_train_iter(
        self,
        runner: Any,
        batch_idx: int,
        data_batch: Any = None,
        outputs: dict[str, Any] | None = None,
    ) -> None:
        if not outputs or "loss" not in outputs:
            raise RuntimeError("Training step did not return a loss tensor")
        import torch

        loss = torch.as_tensor(outputs["loss"])
        if not torch.isfinite(loss).all():
            raise FloatingPointError(
                f"Non-finite loss detected at iteration {runner.iter}"
            )


@HOOKS.register_module()
class SourceExposureAuditHook(Hook):
    """Record the source identity of every batch actually consumed by training."""

    priority = "LOWEST"

    def __init__(self, interval: int = 100) -> None:
        if interval < 1:
            raise ValueError("Source exposure audit interval must be positive")
        self.interval = int(interval)
        self.counts: Counter[str] = Counter()
        self.class_image_counts: Counter[tuple[str, int]] = Counter()

    def before_train(self, runner: Any) -> None:
        path = Path(runner.work_dir) / "source_exposure.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.counts.update(
                {key: int(value) for key, value in payload.get("sample_counts", {}).items()}
            )
            for source, values in payload.get("post_transform_class_image_counts", {}).items():
                self.class_image_counts.update(
                    {(source, int(class_id)): int(count) for class_id, count in values.items()}
                )

    def after_train_iter(
        self,
        runner: Any,
        batch_idx: int,
        data_batch: Any = None,
        outputs: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(data_batch, dict):
            raise RuntimeError("Source exposure audit requires a dict data batch")
        samples = data_batch.get("data_samples")
        if not samples:
            raise RuntimeError("Source exposure audit found no data_samples")
        for sample in samples:
            sample_id = str(sample.metainfo.get("sample_id", ""))
            if "/" not in sample_id:
                raise RuntimeError(f"Source exposure audit found invalid sample_id: {sample_id}")
            source = sample_id.split("/", 1)[0]
            self.counts[source] += 1
            ground_truth = getattr(getattr(sample, "gt_sem_seg", None), "data", None)
            if ground_truth is None:
                raise RuntimeError(
                    f"Source exposure audit found no transformed GT: {sample_id}"
                )
            import torch

            for class_id in torch.unique(ground_truth).detach().cpu().tolist():
                class_id = int(class_id)
                if class_id != 255:
                    self.class_image_counts[(source, class_id)] += 1
        if getattr(runner, "rank", 0) == 0 and (runner.iter + 1) % self.interval == 0:
            _write_audit(runner, "source_exposure.json", self._payload(runner))

    def after_train(self, runner: Any) -> None:
        if getattr(runner, "rank", 0) != 0:
            return
        _write_audit(runner, "source_exposure.json", self._payload(runner))

    def _payload(self, runner: Any) -> dict[str, Any]:
        sampler = runner.train_dataloader.sampler
        total = sum(self.counts.values())
        return {
            "schema_version": "adom-source-exposure-v2",
            "sampler": type(sampler).__name__,
            "requested_weights": getattr(sampler, "source_weights", None),
            "seed": getattr(sampler, "seed", None),
            "start_index": getattr(sampler, "start_index", None),
            "rare_class_ids": list(getattr(sampler, "rare_class_ids", ())),
            "rare_probability": getattr(sampler, "rare_probability", None),
            "rare_temperature": getattr(sampler, "temperature", None),
            "rare_minimum_pixels": getattr(sampler, "minimum_pixels", None),
            "rcs_class_probabilities": getattr(
                getattr(sampler, "schedule", None),
                "source_class_probabilities",
                None,
            ),
            "rcs_eligible_image_counts": {
                source: {
                    str(class_id): len(indices)
                    for class_id, indices in sorted(values.items())
                }
                for source, values in sorted(
                    getattr(
                        getattr(sampler, "schedule", None),
                        "source_class_indices",
                        {},
                    ).items()
                )
            },
            "sample_counts": dict(sorted(self.counts.items())),
            "sample_shares": {
                source: count / total if total else 0.0
                for source, count in sorted(self.counts.items())
            },
            "post_transform_class_image_counts": {
                source: {
                    str(class_id): self.class_image_counts[(source, class_id)]
                    for class_id in range(19)
                }
                for source in sorted(self.counts)
            },
            "post_transform_class_image_rates": {
                source: {
                    str(class_id): (
                        self.class_image_counts[(source, class_id)] / self.counts[source]
                        if self.counts[source]
                        else 0.0
                    )
                    for class_id in range(19)
                }
                for source in sorted(self.counts)
            },
            "total_samples": total,
        }


@HOOKS.register_module()
class FreezeBackboneHook(Hook):
    """Freeze the MiT encoder and prove its weights did not change."""

    priority = "VERY_HIGH"

    def before_train(self, runner: Any) -> None:
        backbone = _unwrap_model(runner.model).backbone
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        backbone.eval()
        self.initial_hash = backbone_sha256(runner.model)
        runner.logger.info("ADOM Stage 1 backbone frozen: %s", self.initial_hash)

    def before_train_epoch(self, runner: Any) -> None:
        _unwrap_model(runner.model).backbone.eval()

    def before_train_iter(
        self,
        runner: Any,
        batch_idx: int,
        data_batch: Any = None,
    ) -> None:
        # Keep this invariant even if a loop/model implementation calls
        # model.train() between epochs or iterations.
        _unwrap_model(runner.model).backbone.eval()

    def after_train(self, runner: Any) -> None:
        final_hash = backbone_sha256(runner.model)
        passed = final_hash == self.initial_hash
        payload = {
            "mode": "frozen",
            "initial_sha256": self.initial_hash,
            "final_sha256": final_hash,
            "passed": passed,
        }
        _write_audit(runner, "backbone_freeze_check.json", payload)
        if not passed:
            raise RuntimeError("Backbone changed during frozen Stage 1")


@HOOKS.register_module()
class BackboneAuditHook(Hook):
    """Require the encoder weights to change during end-to-end Stage 2."""

    priority = "VERY_HIGH"

    def before_train(self, runner: Any) -> None:
        backbone = _unwrap_model(runner.model).backbone
        for parameter in backbone.parameters():
            parameter.requires_grad = True
        self.initial_hash = backbone_sha256(runner.model)

    def after_train(self, runner: Any) -> None:
        final_hash = backbone_sha256(runner.model)
        passed = final_hash != self.initial_hash
        payload = {
            "mode": "trainable",
            "initial_sha256": self.initial_hash,
            "final_sha256": final_hash,
            "passed": passed,
        }
        _write_audit(runner, "backbone_update_check.json", payload)
        if not passed:
            raise RuntimeError("Backbone did not change during end-to-end Stage 2")
