"""
Unseen pattern yonetimi — Levenshtein mesafesi ve en yakin eslesme.

Test asamasinda egitimde gorulmemis SAX kaliplari ile karsilasilabilir.
Bu modul, en yakin bilinen durumu bularak otomatayi devam ettirir.
"""
import numpy as np
from typing import List, Tuple, Optional


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Iki string arasindaki Levenshtein (edit) mesafesini hesaplar.

    Ekleme, silme ve degistirme islemlerinin minimum sayisini dondurur.

    Args:
        s1: Birinci string.
        s2: Ikinci string.

    Returns:
        Tam sayi edit mesafesi.

    Ornek:
        >>> levenshtein_distance('abc', 'adc')
        1
        >>> levenshtein_distance('abc', 'xyz')
        3
    """
    m, n = len(s1), len(s2)
    # DP matrisi
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # silme
                    dp[i][j - 1],      # ekleme
                    dp[i - 1][j - 1]   # degistirme
                )

    return int(dp[m][n])


def find_nearest_pattern(unseen_word: str,
                         known_patterns: List[str]) -> Tuple[str, int]:
    """
    Bilinen desenler arasindan Levenshtein mesafesine gore en yakini bulur.

    Args:
        unseen_word:    Egitimde gorulmemis SAX kelimesi.
        known_patterns: Egitim verisindeki bilinen SAX kelimeleri listesi.

    Returns:
        (en_yakin_kelime, mesafe) tuple'i.

    Raises:
        ValueError: known_patterns bos ise.
    """
    if not known_patterns:
        raise ValueError("known_patterns bos olamaz.")

    best_word = None
    best_dist = float("inf")

    for pattern in known_patterns:
        dist = levenshtein_distance(unseen_word, pattern)
        if dist < best_dist:
            best_dist = dist
            best_word = pattern

    return best_word, best_dist


class UnseenHandler:
    """
    Test sirasinda gorulmemis SAX kaliplari icin yedek mekanizma.

    Unseen bir kelime ile karsilasildiginda:
    1. Kelime egitim sozlugunde var mi kontrol edilir.
    2. Yoksa Levenshtein ile en yakin bilinen desen bulunur.
    3. En yakin desen uzerinden otomata devam ettirilir.
    """

    def __init__(self, known_patterns: List[str]):
        """
        Args:
            known_patterns: build_pattern_dict() anahtarlari (egitim kelimeleri).
        """
        self.known_patterns = known_patterns

    def resolve(self, word: str) -> Tuple[str, bool, Optional[int]]:
        """
        Bir SAX kelimesini cozumler: bilinen mi, yoksa unseen mi?

        Args:
            word: Test sirasinda gelen SAX kelimesi.

        Returns:
            (resolved_word, is_unseen, edit_distance)
            - resolved_word:  Kullanilacak durum (bilinen veya en yakin).
            - is_unseen:      True eger kelime egitimde gorulmemisse.
            - edit_distance:  Unseen ise Levenshtein mesafesi, yoksa None.
        """
        if word in self.known_patterns:
            return word, False, None

        nearest, dist = find_nearest_pattern(word, self.known_patterns)
        return nearest, True, dist
