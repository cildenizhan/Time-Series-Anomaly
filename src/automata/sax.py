import numpy as np
from scipy.stats import norm

ALPHABET = "abcdefghijklmnopqrstuvwxyz"

def get_breakpoints(alphabet_size: int) -> np.ndarray:
    quantiles = np.arange(1, alphabet_size) / alphabet_size
    return norm.ppf(quantiles)

class SAXEncoder:

    def __init__(self, alphabet_size: int = 3):
        if alphabet_size < 2 or alphabet_size > 26:
            raise ValueError("alphabet_size 2 ile 26 arasinda olmalidir.")

        self.alphabet_size = alphabet_size
        self.alphabet      = ALPHABET[:alphabet_size]
        self.breakpoints   = get_breakpoints(alphabet_size)

    def encode(self, paa_values: np.ndarray) -> str:
        symbols = []
        for val in paa_values:
            idx = int(np.searchsorted(self.breakpoints, val, side="right"))
            symbols.append(self.alphabet[idx])
        return "".join(symbols)

    def encode_batch(self, paa_matrix: np.ndarray) -> list:
        return [self.encode(row) for row in paa_matrix]

    def decode_info(self) -> dict:
        return {
            "alphabet_size": self.alphabet_size,
            "alphabet":      self.alphabet,
            "breakpoints":   self.breakpoints.tolist(),
        }
