from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mmengine.hooks import Hook
from mmseg.registry import HOOKS


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
