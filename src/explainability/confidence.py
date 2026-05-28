"""
Guven skoru (confidence score) hesaplama modulu.

Path probability degerini insan-okunabilir bir guven skoru ve
karar etiketi (anomali / normal) ile eslestiren fonksiyonlar icerir.
"""
from typing import Tuple


def compute_confidence_score(path_probability: float,
                              low_threshold: float = 0.05) -> Tuple[float, str]:
    """
    Yol olasiligini guven skoruna ve karar etiketine donusturur.

    Dusuk olasilik -> Anomali adayi (beklenmedik davranis)
    Yuksek olasilik -> Normal davranis

    Args:
        path_probability: AutomataBuilder.path_probability() ciktisi.
        low_threshold:    Bu degerin altindaki olasiliklar anomali sayilir.

    Returns:
        (confidence_score, decision) tuple'i.
        confidence_score: 0.0 - 1.0 arasi normalize edilmis guven degeri.
        decision:         "anomaly" veya "normal".

    Ornek:
        >>> compute_confidence_score(0.108)
        (0.108, 'anomaly')
        >>> compute_confidence_score(0.72)
        (0.72, 'normal')
    """
    score    = round(float(path_probability), 6)
    decision = "anomaly" if score < low_threshold else "normal"
    return score, decision


def confidence_label(score: float) -> str:
    """
    Guven skorunu sozel etikete donusturur.

    Args:
        score: compute_confidence_score() ciktisi.

    Returns:
        'Low', 'Medium' veya 'High'.
    """
    if score < 0.1:
        return "Low"
    elif score < 0.5:
        return "Medium"
    else:
        return "High"
