import numpy as np
from typing import List, Tuple, Optional

def levenshtein_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
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
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1]
                )

    return int(dp[m][n])

def find_nearest_pattern(unseen_word: str,
                         known_patterns: List[str]) -> Tuple[str, int]:
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

    def __init__(self, known_patterns: List[str]):
        self.known_patterns = known_patterns

    def resolve(self, word: str) -> Tuple[str, bool, Optional[int]]:
        if word in self.known_patterns:
            return word, False, None

        nearest, dist = find_nearest_pattern(word, self.known_patterns)
        return nearest, True, dist
