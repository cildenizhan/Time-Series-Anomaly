"""
evaluation paketi — metrik, istatistik, gorsellestirme ve raporlama.
"""
from .metrics import compute_metrics, compute_fold_stats, get_roc_curve
from .statistical_tests import wilcoxon_test, mcnemar_test
from .report import ExperimentReport

__all__ = [
    "compute_metrics",
    "compute_fold_stats",
    "get_roc_curve",
    "wilcoxon_test",
    "mcnemar_test",
    "ExperimentReport",
]
