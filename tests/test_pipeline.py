import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.automata.paa import compute_paa, batch_paa
from src.automata.sax import SAXEncoder
from src.automata.sliding_window import extract_windows
from src.automata.automata_builder import AutomataBuilder
from src.automata.pattern_dict import build_pattern_dict
from src.automata.unseen_handler import UnseenHandler
from src.explainability.explainer import Explainer
from src.explainability.output_formatter import format_decision

def make_sax_sequence(n: int = 30, alphabet_size: int = 3,
                      window_size: int = 4, seed: int = 42) -> list:
    rng    = np.random.default_rng(seed)
    series = rng.standard_normal(n + window_size)
    windows, _ = extract_windows(series, window_size)
    paa        = batch_paa(windows, window_size)
    encoder    = SAXEncoder(alphabet_size)
    return encoder.encode_batch(paa)

class TestPipelineIntegration:

    def setup_method(self):
        self.window_size   = 4
        self.alphabet_size = 3
        self.sax_train     = make_sax_sequence(50, self.alphabet_size,
                                               self.window_size, seed=42)
        self.sax_test      = make_sax_sequence(20, self.alphabet_size,
                                               self.window_size, seed=99)

    def test_automata_fit_and_predict(self):
        builder = AutomataBuilder().fit(self.sax_train)
        assert len(builder.states) > 0
        assert builder.is_fitted

    def test_path_probability_range(self):
        builder      = AutomataBuilder().fit(self.sax_train)
        known        = list(build_pattern_dict(self.sax_train).keys())
        unseen       = UnseenHandler(known)
        explainer    = Explainer(builder, unseen)
        path_prob    = explainer.compute_path_probability(self.sax_test)
        assert 0.0 <= path_prob <= 1.0

    def test_explain_sequence_length(self):
        builder   = AutomataBuilder().fit(self.sax_train)
        known     = list(build_pattern_dict(self.sax_train).keys())
        unseen    = UnseenHandler(known)
        explainer = Explainer(builder, unseen)
        exps      = explainer.explain_sequence(self.sax_test)
        assert len(exps) == len(self.sax_test)

    def test_format_decision_output(self):
        result = format_decision(
            time_step=0, state="abc", pattern="abc",
            status="seen", mapped_to="abc",
            path_probability=0.3
        )
        assert "decision" in result
        assert "probability" in result
        assert result["decision"] in ("normal", "anomaly")

    def test_unseen_rate(self):
        builder   = AutomataBuilder().fit(self.sax_train)
        known     = set(build_pattern_dict(self.sax_train).keys())
        n_unseen  = sum(1 for w in self.sax_test if w not in known)
        rate      = n_unseen / len(self.sax_test)
        assert 0.0 <= rate <= 1.0
