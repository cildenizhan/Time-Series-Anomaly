import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)
from typing import Dict, List, Tuple, Optional

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: Optional[np.ndarray] = None,
                    average: str = "binary") -> Dict:
    result = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average=average,
                                           zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average=average,
                                        zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred, average=average,
                                    zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_prob is not None:
        try:
            result["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
        except ValueError:
            result["roc_auc"] = None

    return result

def compute_fold_stats(fold_results: List[Dict]) -> Dict:
    keys  = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    stats = {}
    for k in keys:
        vals = [r[k] for r in fold_results if k in r and r[k] is not None]
        if vals:
            stats[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)), 4),
                "n":    len(vals),
            }
    return stats

def get_roc_curve(y_true: np.ndarray,
                  y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return roc_curve(y_true, y_prob)

def get_pr_curve(y_true: np.ndarray,
                 y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return precision_recall_curve(y_true, y_prob)
