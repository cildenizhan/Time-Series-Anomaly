"""
Kayan pencere (sliding window) ile zaman serisinden oruntu cikarimi.

Her pencere PAA + SAX isleminden gecirilerek sembolik temsile donusturulur.
"""
import numpy as np
from typing import Tuple


def extract_windows(series: np.ndarray, window_size: int,
                    step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zaman serisinden kayan pencereler olusturur.

    Args:
        series:      1-boyutlu zaman serisi. Sekil: (T,)
        window_size: Her pencerenin uzunlugu.
        step:        Pencereler arasi adim buyuklugu (varsayilan: 1).

    Returns:
        windows: Pencere matrisi. Sekil: (N, window_size)
        indices: Her pencerenin baslangic indeksi. Sekil: (N,)

    Raises:
        ValueError: window_size seriden buyukse.

    Ornek:
        >>> s = np.array([1, 2, 3, 4, 5])
        >>> windows, idx = extract_windows(s, window_size=3, step=1)
        >>> windows
        array([[1, 2, 3],
               [2, 3, 4],
               [3, 4, 5]])
    """
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
    """
    Cok degiskenli zaman serisinden kayan pencereler olusturur.

    Her pencere tum ozellikleri icerir; PCA oncesinde kullanilabilir.

    Args:
        data:        2-boyutlu veri matrisi. Sekil: (T, F)
        window_size: Her pencerenin adim sayisi.
        step:        Pencereler arasi adim.

    Returns:
        windows: Sekil: (N, window_size, F)
        indices: Sekil: (N,)
    """
    T = data.shape[0]
    if window_size > T:
        raise ValueError(
            f"window_size ({window_size}) seri uzunlugundan ({T}) buyuk olamaz."
        )

    indices = np.arange(0, T - window_size + 1, step)
    windows = np.array([data[i: i + window_size, :] for i in indices])
    return windows, indices
