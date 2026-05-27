"""
Piecewise Aggregate Approximation (PAA) donusumu.

PAA, zaman serisini esit uzunlukta segmentlere boler ve
her segmentin ortalamasini alarak boyut indirgemesi yapar.
Bu islem SAX kodlamadan once uygulanir.
"""
import numpy as np


def compute_paa(series: np.ndarray, n_segments: int) -> np.ndarray:
    """
    Bir zaman serisine PAA donusumu uygular.

    Her segment icin ortalama deger hesaplanir.
    Segment sayisi (n_segments), pencere boyutuna (window_size) esit olmalidir.

    Args:
        series:     1-boyutlu zaman serisi dizisi. Sekil: (T,)
        n_segments: Bolunecek segment sayisi (== pencere boyutu).

    Returns:
        PAA temsili. Sekil: (n_segments,)

    Raises:
        ValueError: Seri uzunlugu segment sayisindan kucukse.

    Ornek:
        >>> series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        >>> compute_paa(series, 4)
        array([1.5, 3.5, 5.5, 7.5])
    """
    T = len(series)
    if T < n_segments:
        raise ValueError(
            f"Seri uzunlugu ({T}) segment sayisindan ({n_segments}) kucuk olamaz."
        )

    # Her segmentin kac veri noktasini kapsadigini hesapla
    segment_size = T / n_segments
    paa = np.zeros(n_segments)

    for i in range(n_segments):
        start = int(np.round(i * segment_size))
        end   = int(np.round((i + 1) * segment_size))
        paa[i] = np.mean(series[start:end])

    return paa


def batch_paa(windows: np.ndarray, n_segments: int) -> np.ndarray:
    """
    Cok sayida pencereye toplu PAA uygular.

    Args:
        windows:    2-boyutlu pencere matrisi. Sekil: (N, T)
        n_segments: Her pencere icin segment sayisi.

    Returns:
        PAA temsilleri matrisi. Sekil: (N, n_segments)
    """
    return np.array([compute_paa(w, n_segments) for w in windows])
