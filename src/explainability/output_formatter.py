"""
JSON ve tablo formatinda karar raporu uretici.

Proje rubriginde belirtilen cikti formatini uygular.
"""
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
    """
    Otomata kararını standart sozluk formatinda uretir.

    Proje cikti formati (rubrik Section X.F):
    {
        "time_step": 5,
        "state": "aab",
        "pattern": "adc",
        "status": "unseen",
        "mapped_to": "abc",
        "probability": 0.108,
        "decision": "anomaly"
    }

    Args:
        time_step:        Zaman adim indeksi.
        state:            Mevcut otomata durumu.
        pattern:          Gozlemlenen SAX kelimesi.
        status:           'seen' veya 'unseen'.
        mapped_to:        Cozumlenen durum (unseen ise en yakin).
        path_probability: Yol olasiligi.
        low_threshold:    Anomali esigi.
        transitions:      [{from, to, prob}] listesi (opsiyonel).

    Returns:
        Karar sozlugu.
    """
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
    """Karar sozlugunu JSON string'e donusturur."""
    return json.dumps(decision_dict, ensure_ascii=False, indent=indent)


def to_table_row(decision_dict: Dict) -> str:
    """Karar sozlugunu tek satirlik tablo formatina donusturur."""
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
    """Tum karar listesini tablo formatinda yazdirir."""
    header = f"{'t':>5} | {'state':<8} | {'pattern':<8} | {'status':<6} | {'prob':>6} | {'conf':<6} | decision"
    print(header)
    print("-" * len(header))
    for d in decisions:
        print(to_table_row(d))
