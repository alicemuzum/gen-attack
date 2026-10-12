"""Target models wrapped as robustness benchmarks."""

from gen_attack.models.detr import DetectionAttackResult, Detections, DETRBenchmark
from gen_attack.models.resnet import AttackOutcome, QueryCurve, ResNetBenchmark, RobustnessCurve

__all__ = [
    "AttackOutcome",
    "DETRBenchmark",
    "DetectionAttackResult",
    "Detections",
    "QueryCurve",
    "ResNetBenchmark",
    "RobustnessCurve",
]
