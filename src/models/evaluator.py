"""
Model performansini olcen metrik hesaplama modulu.
Accuracy, Precision, Recall ve F1-Score hesaplanir.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)


class ModelEvaluator:
    """
    Siniflandirma modellerinin performansini degerlendiren yardimci sinif.
    SKAB icin fold bazli, BATADAL icin zaman sirali degerlendirme destekler.
    """

    def __init__(self, average: str = "binary"):
        """
        Args:
            average: Metrik ortalamalama yontemi ('binary', 'macro', 'weighted').
        """
        self.average = average

    # ------------------------------------------------------------------
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Temel siniflandirma metriklerini hesaplar.

        Args:
            y_true: Gercek etiketler.
            y_pred: Tahmin edilen etiketler.

        Returns:
            Metrik degerlerini iceren sozluk.
        """
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=self.average,
                               zero_division=0)
        rec  = recall_score(y_true, y_pred, average=self.average,
                            zero_division=0)
        f1   = f1_score(y_true, y_pred, average=self.average,
                        zero_division=0)
        cm   = confusion_matrix(y_true, y_pred)

        return {
            "accuracy":  round(acc, 4),
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1_score":  round(f1, 4),
            "confusion_matrix": cm.tolist(),
        }

    # ------------------------------------------------------------------
    def compute_fold_stats(self, fold_results: list) -> dict:
        """
        Birden fazla fold sonucunun ortalama ve standart sapmasini hesaplar.

        Args:
            fold_results: Her fold icin compute() ciktisi olan liste.

        Returns:
            Her metrik icin mean ve std iceren sozluk.
        """
        keys = ["accuracy", "precision", "recall", "f1_score"]
        stats = {}
        for k in keys:
            vals = [r[k] for r in fold_results if k in r]
            stats[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)), 4),
            }
        return stats
