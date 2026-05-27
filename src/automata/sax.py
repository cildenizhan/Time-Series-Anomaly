"""
Symbolic Aggregate approXimation (SAX) kodlayici.

SAX, PAA ile indirgenmiş zaman serisini sembolik bir diziye donusturur.
Normal dagilim breakpoint'leri kullanarak sürekli degerleri harflere esler.
"""
import numpy as np
from scipy.stats import norm


# Alfabe harfleri (maksimum 26 sembol desteklenir)
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def get_breakpoints(alphabet_size: int) -> np.ndarray:
    """
    Normal dagilim esit-alan dilimlerine gore SAX kirma noktalarini hesaplar.

    Args:
        alphabet_size: Kullanilacak sembol sayisi.

    Returns:
        (alphabet_size - 1) uzunlugunda breakpoint dizisi.

    Ornek (alphabet_size=3):
        [-0.4307, 0.4307]  ->  a | b | c
    """
    quantiles = np.arange(1, alphabet_size) / alphabet_size
    return norm.ppf(quantiles)


class SAXEncoder:
    """
    PAA dizisini SAX sembol dizisine donusturucu.

    Kullanim:
        encoder = SAXEncoder(alphabet_size=3)
        word = encoder.encode(paa_array)  # ornek: 'abc'
    """

    def __init__(self, alphabet_size: int = 3):
        """
        Args:
            alphabet_size: Alfabe buyuklugu (3-6 arasi onerilir).
        """
        if alphabet_size < 2 or alphabet_size > 26:
            raise ValueError("alphabet_size 2 ile 26 arasinda olmalidir.")

        self.alphabet_size = alphabet_size
        self.alphabet      = ALPHABET[:alphabet_size]
        self.breakpoints   = get_breakpoints(alphabet_size)

    # ------------------------------------------------------------------
    def encode(self, paa_values: np.ndarray) -> str:
        """
        PAA degerlerini SAX kelimesine donusturur.

        Args:
            paa_values: PAA temsili. Sekil: (n_segments,)

        Returns:
            SAX kelimesi (ornek: 'bac').
        """
        symbols = []
        for val in paa_values:
            # Hangi bolgeye dustugunu bul
            idx = int(np.searchsorted(self.breakpoints, val, side="right"))
            symbols.append(self.alphabet[idx])
        return "".join(symbols)

    # ------------------------------------------------------------------
    def encode_batch(self, paa_matrix: np.ndarray) -> list:
        """
        Cok sayida PAA satirini toplu SAX kelimelerine donusturur.

        Args:
            paa_matrix: Sekil: (N, n_segments)

        Returns:
            N uzunlugunda SAX kelimesi listesi.
        """
        return [self.encode(row) for row in paa_matrix]

    # ------------------------------------------------------------------
    def decode_info(self) -> dict:
        """Kodlama parametrelerini sozluk olarak dondurur."""
        return {
            "alphabet_size": self.alphabet_size,
            "alphabet":      self.alphabet,
            "breakpoints":   self.breakpoints.tolist(),
        }
