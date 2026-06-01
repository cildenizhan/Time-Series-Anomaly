import numpy as np
from typing import Tuple

def extract_windows(series: np.ndarray, window_size: int,
                    step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    T = len(series)
    if window_size > T:
        raise ValueError(
            f"window_size ({window_size}) seri uzunlugundan ({T}) buyuk olamaz."
        )

    indices = np.arange(0, T - window_size + 1, step)
    windows = np.array([series[i: i + window_size] for i in indices])
    return windows, indices

def extract_windows_2d(data: np.ndarray, window_size: int,
                       step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    T = data.shape[0]
    if window_size > T:
        raise ValueError(
            f"window_size ({window_size}) seri uzunlugundan ({T}) buyuk olamaz."
        )

    indices = np.arange(0, T - window_size + 1, step)
    windows = np.array([data[i: i + window_size, :] for i in indices])
    return windows, indices
