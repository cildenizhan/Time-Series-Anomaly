from typing import List, Dict, Optional

from src.automata.automata_builder import AutomataBuilder
from src.automata.unseen_handler import UnseenHandler

class Explainer:

    def __init__(self, builder: AutomataBuilder,
                 unseen_handler: UnseenHandler):
        self.builder         = builder
        self.unseen_handler  = unseen_handler

    def explain_step(self, time_step: int,
                     prev_word: Optional[str],
                     curr_word: str) -> Dict:
        resolved, is_unseen, edit_dist = self.unseen_handler.resolve(curr_word)

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

    def explain_sequence(self, sax_words: List[str]) -> List[Dict]:
        explanations = []
        for t, word in enumerate(sax_words):
            prev = sax_words[t - 1] if t > 0 else None
            exp  = self.explain_step(t, prev, word)
            explanations.append(exp)
        return explanations

    def compute_path_probability(self, sax_words: List[str]) -> float:
        resolved_words = [self.unseen_handler.resolve(w)[0] for w in sax_words]
        return self.builder.path_probability(resolved_words)
