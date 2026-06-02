import numpy as np
from scipy import stats
from typing import Dict, Tuple

def wilcoxon_test(scores_a: list, scores_b: list,
                  alternative: str = "two-sided") -> Dict:
    stat, pval = stats.wilcoxon(scores_a, scores_b,
                                alternative=alternative)
    return {
        "test":        "wilcoxon",
        "statistic":   round(float(stat), 6),
        "p_value":     round(float(pval), 6),
        "significant": bool(pval < 0.05),
        "alternative": alternative,
    }

def mcnemar_test(y_true: np.ndarray,
                 y_pred_a: np.ndarray,
                 y_pred_b: np.ndarray) -> Dict:
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    n00 = int(np.sum(~correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a &  correct_b))
    n10 = int(np.sum( correct_a & ~correct_b))
    n11 = int(np.sum( correct_a &  correct_b))

    if (n01 + n10) == 0:
        chi2, pval = 0.0, 1.0
    else:
        chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        pval = float(1 - stats.chi2.cdf(chi2, df=1))

    return {
        "test":              "mcnemar",
        "statistic":         round(chi2, 6),
        "p_value":           round(pval, 6),
        "significant":       bool(pval < 0.05),
        "contingency_table": [[n00, n01], [n10, n11]],
    }
