"""MMSegmentation registry extensions used by ADOM configs."""

from .dataset import AdomCost4Dataset, AdomSemantic20Dataset
from .hooks import (
    BackboneAuditHook,
    CanonicalTestLockHook,
    ConstrainedCheckpointSelectionHook,
    FiniteLossHook,
    FreezeBackboneHook,
    MetricArtifactHook,
    SourceExposureAuditHook,
    TA0AblationContractHook,
)
from .metrics import AdomSafetyMetric, AdomSemantic20Metric
from .samplers import SourceRareClassInfiniteSampler, SourceWeightedInfiniteSampler

__all__ = [
    "AdomCost4Dataset",
    "AdomSemantic20Dataset",
    "AdomSafetyMetric",
    "AdomSemantic20Metric",
    "BackboneAuditHook",
    "CanonicalTestLockHook",
    "ConstrainedCheckpointSelectionHook",
    "FiniteLossHook",
    "FreezeBackboneHook",
    "MetricArtifactHook",
    "SourceExposureAuditHook",
    "TA0AblationContractHook",
    "SourceRareClassInfiniteSampler",
    "SourceWeightedInfiniteSampler",
]
