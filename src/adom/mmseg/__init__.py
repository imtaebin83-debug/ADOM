"""MMSegmentation registry extensions used by ADOM configs."""

from .dataset import AdomCost4Dataset
from .hooks import (
    BackboneAuditHook,
    FiniteLossHook,
    FreezeBackboneHook,
    MetricArtifactHook,
)
from .metrics import AdomSafetyMetric

__all__ = [
    "AdomCost4Dataset",
    "AdomSafetyMetric",
    "BackboneAuditHook",
    "FiniteLossHook",
    "FreezeBackboneHook",
    "MetricArtifactHook",
]
