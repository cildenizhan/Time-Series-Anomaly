"""
Probabilistik otomata olusturucu.

SAX kelimelerinden durum uzayi ve gecis matrisi hesaplanir.
Her benzersiz SAX kelimesi bir durum olarak tanimlanir.
"""
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


class AutomataBuilder:
    """
    SAX kelimelerinden probabilistik sonlu otomata olusturur.

    Durumlar: Egitim verisindeki benzersiz SAX kelimeleri.
    Gecisler: Ard arda gelen kelimeler arasi gecis sayilari ve olasiliklari.

    Kullanim:
        builder = AutomataBuilder()
        builder.fit(sax_words)
        prob = builder.transition_prob("abc", "bcd")
    """

    def __init__(self):
        self.states: List[str]                       = []
        self.state_index: Dict[str, int]             = {}
        # transition_counts[kaynak][hedef] = sayi
        self.transition_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.transition_probs: Dict[str, Dict[str, float]] = {}
        self.is_fitted = False

    # ------------------------------------------------------------------
    def fit(self, sax_words: List[str]) -> "AutomataBuilder":
        """
        SAX kelime listesinden otomata olusturur.

        Ard arda gelen (s_i -> s_{i+1}) gecisler sayilir.

        Args:
            sax_words: Egitim verisinden elde edilen SAX kelimeleri.

        Returns:
            self (zincirleme kullanim icin).
        """
        # Benzersiz durumlar
        unique = list(dict.fromkeys(sax_words))  # siraya gore teksizduru
        self.states      = unique
        self.state_index = {s: i for i, s in enumerate(unique)}

        # Gecis sayimlarini hesapla
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        for i in range(len(sax_words) - 1):
            src = sax_words[i]
            dst = sax_words[i + 1]
            self.transition_counts[src][dst] += 1

        # Gecis olasilikları hesapla
        self._compute_probabilities()
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    def _compute_probabilities(self) -> None:
        """
        Ham gecis sayilarindan olasiliklari hesaplar.

        P(S_i -> S_j) = sayi(S_i -> S_j) / toplam_cikis(S_i)
        """
        self.transition_probs = {}
        for src, targets in self.transition_counts.items():
            total = sum(targets.values())
            self.transition_probs[src] = {
                dst: count / total
                for dst, count in targets.items()
            }

    # ------------------------------------------------------------------
    def transition_prob(self, src: str, dst: str) -> float:
        """
        Iki durum arasindaki gecis olasiligini dondurur.

        Args:
            src: Kaynak durum (SAX kelimesi).
            dst: Hedef durum (SAX kelimesi).

        Returns:
            Gecis olasiligi (0.0 eger gecis yoksa).
        """
        if not self.is_fitted:
            raise RuntimeError("Once fit() cagirin.")
        return self.transition_probs.get(src, {}).get(dst, 0.0)

    # ------------------------------------------------------------------
    def path_probability(self, sequence: List[str]) -> float:
        """
        Bir SAX kelimesi dizisinin yol olasiligini hesaplar.

        P(dizi) = CARP(P(s_i -> s_{i+1}))

        Args:
            sequence: SAX kelimelerinden olusan dizi.

        Returns:
            Yol olasiligi (float).
        """
        if len(sequence) < 2:
            return 1.0
        prob = 1.0
        for i in range(len(sequence) - 1):
            prob *= self.transition_prob(sequence[i], sequence[i + 1])
            if prob == 0.0:
                return 0.0
        return prob

    # ------------------------------------------------------------------
    def get_transition_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Tam gecis olasilik matrisini dondurur.

        Returns:
            matrix: Sekil: (N, N) olasilik matrisi.
            labels: Durum etiketleri.
        """
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
