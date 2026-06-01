import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

class ModelEvaluator:

    def __init__(self, average: str = "binary"):
        self.average = average

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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

    def compute_fold_stats(self, fold_results: list) -> dict:
        keys = ["accuracy", "precision", "recall", "f1_score"]
        stats = {}
        for k in keys:
            vals = [r[k] for r in fold_results if k in r]
            stats[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)), 4),
            }
        return stats
