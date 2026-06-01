import numpy as np

def compute_paa(series: np.ndarray, n_segments: int) -> np.ndarray:
    T = len(series)
    if T < n_segments:
        raise ValueError(
            f"Seri uzunlugu ({T}) segment sayisindan ({n_segments}) kucuk olamaz."
        )

    segment_size = T / n_segments
    paa = np.zeros(n_segments)

    for i in range(n_segments):
        start = int(np.round(i * segment_size))
        end   = int(np.round((i + 1) * segment_size))
        paa[i] = np.mean(series[start:end])

    return paa

def batch_paa(windows: np.ndarray, n_segments: int) -> np.ndarray:
    return np.array([compute_paa(w, n_segments) for w in windows])
