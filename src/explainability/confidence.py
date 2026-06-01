from typing import Tuple

def compute_confidence_score(path_probability: float,
                              low_threshold: float = 0.05) -> Tuple[float, str]:
    score    = round(float(path_probability), 6)
    decision = "anomaly" if score < low_threshold else "normal"
    return score, decision

def confidence_label(score: float) -> str:
    if score < 0.1:
        return "Low"
    elif score < 0.5:
        return "Medium"
    else:
        return "High"
