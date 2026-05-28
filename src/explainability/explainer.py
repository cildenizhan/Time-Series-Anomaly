"""
Probabilistik aciklanabilirlik modulu — Explainer sinifi.

Her karar icin asagidakileri uretir:
  - Mevcut durum (state)
  - Gozlemlenen oruntu (pattern)
  - Unseen mi, bilinen mi?
  - Gerceklesen durum gecisleri
  - Her gecisin olasiligi
  - Oruntu dizisinin toplam yol olasiligi (path probability)
"""
from typing import List, Dict, Optional

from src.automata.automata_builder import AutomataBuilder
from src.automata.unseen_handler import UnseenHandler


class Explainer:
    """
    Otomata kararlarini aciklanabilir bicimlere donusturucu.

    Her zaman adimi icin bir karar raporu uretir:
    state, pattern, status (seen/unseen), transitions ve path probability.
    """

    def __init__(self, builder: AutomataBuilder,
                 unseen_handler: UnseenHandler):
        """
        Args:
            builder:        Egitilmis AutomataBuilder nesnesi.
            unseen_handler: Egitilmis UnseenHandler nesnesi.
        """
        self.builder         = builder
        self.unseen_handler  = unseen_handler

    # ------------------------------------------------------------------
    def explain_step(self, time_step: int,
                     prev_word: Optional[str],
                     curr_word: str) -> Dict:
        """
        Tek bir zaman adimi icin acıklama uretir.

        Args:
            time_step:  Zaman adim indeksi.
            prev_word:  Onceki SAX kelimesi (ilk adimda None).
            curr_word:  Gecerli SAX kelimesi.

        Returns:
            Aciklama sozlugu.
        """
        # Unseen kontrol
        resolved, is_unseen, edit_dist = self.unseen_handler.resolve(curr_word)

        # Gecis olasiligi
        if prev_word is not None:
            resolved_prev, _, _ = self.unseen_handler.resolve(prev_word)
            trans_prob = self.builder.transition_prob(resolved_prev, resolved)
        else:
            resolved_prev = None
            trans_prob = None

        return {
            "time_step":      time_step,
            "previous_state": resolved_prev,
            "current_state":  resolved,
            "pattern":        curr_word,
            "status":         "unseen" if is_unseen else "seen",
            "mapped_to":      resolved if is_unseen else curr_word,
            "edit_distance":  edit_dist,
            "transition_prob": round(trans_prob, 6) if trans_prob is not None else None,
        }

    # ------------------------------------------------------------------
    def explain_sequence(self, sax_words: List[str]) -> List[Dict]:
        """
        Tam bir SAX kelimesi dizisi icin adim adim acıklamalar uretir.

        Args:
            sax_words: Zaman sirasinda SAX kelimeleri.

        Returns:
            Her adim icin acıklama sozlugu listesi.
        """
        explanations = []
        for t, word in enumerate(sax_words):
            prev = sax_words[t - 1] if t > 0 else None
            exp  = self.explain_step(t, prev, word)
            explanations.append(exp)
        return explanations

    # ------------------------------------------------------------------
    def compute_path_probability(self, sax_words: List[str]) -> float:
        """
        Bir SAX kelimesi dizisinin yol olasiligini hesaplar.

        Unseen kelimeler once cozumlenir, sonra olasilik hesaplanir.

        Args:
            sax_words: Test sirasindaki SAX kelimeleri.

        Returns:
            Yol olasiligi (float, 0.0 - 1.0 arasi).
        """
        resolved_words = [self.unseen_handler.resolve(w)[0] for w in sax_words]
        return self.builder.path_probability(resolved_words)
