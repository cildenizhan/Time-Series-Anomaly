import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

class AutomataBuilder:

    def __init__(self):
        self.states: List[str]                       = []
        self.state_index: Dict[str, int]             = {}
        self.transition_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.transition_probs: Dict[str, Dict[str, float]] = {}
        self.is_fitted = False

    def fit(self, sax_words: List[str]) -> "AutomataBuilder":
        unique = list(dict.fromkeys(sax_words))
        self.states      = unique
        self.state_index = {s: i for i, s in enumerate(unique)}

        self.transition_counts = defaultdict(lambda: defaultdict(int))
        for i in range(len(sax_words) - 1):
            src = sax_words[i]
            dst = sax_words[i + 1]
            self.transition_counts[src][dst] += 1

        self._compute_probabilities()
        self.is_fitted = True
        return self

    def _compute_probabilities(self) -> None:
        self.transition_probs = {}
        for src, targets in self.transition_counts.items():
            total = sum(targets.values())
            self.transition_probs[src] = {
                dst: count / total
                for dst, count in targets.items()
            }

    def transition_prob(self, src: str, dst: str) -> float:
        if not self.is_fitted:
            raise RuntimeError("Once fit() cagirin.")
        return self.transition_probs.get(src, {}).get(dst, 0.0)

    def path_probability(self, sequence: List[str]) -> float:
        if len(sequence) < 2:
            return 1.0
        prob = 1.0
        for i in range(len(sequence) - 1):
            prob *= self.transition_prob(sequence[i], sequence[i + 1])
            if prob == 0.0:
                return 0.0
        return prob

    def get_transition_matrix(self) -> Tuple[np.ndarray, List[str]]:
        n = len(self.states)
        matrix = np.zeros((n, n))
        for src, targets in self.transition_probs.items():
            if src in self.state_index:
                i = self.state_index[src]
                for dst, prob in targets.items():
                    if dst in self.state_index:
                        j = self.state_index[dst]
                        matrix[i, j] = prob
        return matrix, self.states
