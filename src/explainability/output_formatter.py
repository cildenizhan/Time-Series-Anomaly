import json
from typing import Dict, List, Optional

from .confidence import compute_confidence_score, confidence_label

def format_decision(time_step: int,
                    state: str,
                    pattern: str,
                    status: str,
                    mapped_to: str,
                    path_probability: float,
                    low_threshold: float = 0.05,
                    transitions: Optional[List[Dict]] = None) -> Dict:
    score, decision = compute_confidence_score(path_probability, low_threshold)
    label           = confidence_label(score)

    result = {
        "time_step":   time_step,
        "state":       state,
        "pattern":     pattern,
        "status":      status,
        "mapped_to":   mapped_to,
        "probability": score,
        "confidence":  label,
        "decision":    decision,
    }

    if transitions:
        result["transitions"] = transitions

    return result

def to_json(decision_dict: Dict, indent: int = 2) -> str:
    return json.dumps(decision_dict, ensure_ascii=False, indent=indent)

def to_table_row(decision_dict: Dict) -> str:
    d = decision_dict
    return (
        f"t={d['time_step']:>4} | "
        f"state={d['state']:<8} | "
        f"pattern={d['pattern']:<8} | "
        f"status={d['status']:<6} | "
        f"prob={d['probability']:.4f} | "
        f"conf={d['confidence']:<6} | "
        f"decision={d['decision']}"
    )

def print_report(decisions: List[Dict]) -> None:
    header = f"{'t':>5} | {'state':<8} | {'pattern':<8} | {'status':<6} | {'prob':>6} | {'conf':<6} | decision"
    print(header)
    print("-" * len(header))
    for d in decisions:
        print(to_table_row(d))
