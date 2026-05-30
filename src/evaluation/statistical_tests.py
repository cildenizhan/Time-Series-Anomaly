"""
Istatistiksel test modulu.
Wilcoxon ve McNemar testleriyle model farklarinin anlamliligi olculur.
"""
import numpy as np
from scipy import stats
from typing import Dict, Tuple


def wilcoxon_test(scores_a: list, scores_b: list,
                  alternative: str = "two-sided") -> Dict:
    """
    Wilcoxon isaretli-siralama testi uygular.

    Iki modelin performans skorlarini (ornegin F1) karsilastirir.
    Normal dagilim varsayimi gerekmez, kucuk orneklemler icin uygundur.

    Args:
        scores_a:    Model A'nin fold/seed bazli skorlari.
        scores_b:    Model B'nin fold/seed bazli skorlari.
        alternative: 'two-sided', 'greater' veya 'less'.

    Returns:
        {statistic, p_value, significant} sozlugu.
    """
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
    """
    McNemar testi uygular.

    Iki ikili siniflandirici arasindaki hata farkinin istatistiksel
    anlamiligini test eder.

    Args:
        y_true:   Gercek etiketler.
        y_pred_a: Model A tahminleri (0/1).
        y_pred_b: Model B tahminleri (0/1).

    Returns:
        {statistic, p_value, significant, contingency_table} sozlugu.
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Contingency table
    n00 = int(np.sum(~correct_a & ~correct_b))  # her ikisi yanlis
    n01 = int(np.sum(~correct_a &  correct_b))  # sadece B dogru
    n10 = int(np.sum( correct_a & ~correct_b))  # sadece A dogru
    n11 = int(np.sum( correct_a &  correct_b))  # her ikisi dogru

    # McNemar chi-kare (Yates duzeltmesiyle)
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
