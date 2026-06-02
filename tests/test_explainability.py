import pytest
from unittest.mock import MagicMock

from src.explainability.confidence import compute_confidence_score, confidence_label
from src.explainability.output_formatter import format_decision, to_json, to_table_row
from src.automata.unseen_handler import levenshtein_distance, find_nearest_pattern, UnseenHandler

class TestLevenshtein:
    def test_identical_strings(self):
        assert levenshtein_distance("abc", "abc") == 0

    def test_single_substitution(self):
        assert levenshtein_distance("abc", "adc") == 1

    def test_empty_string(self):
        assert levenshtein_distance("", "abc") == 3
        assert levenshtein_distance("abc", "") == 3

    def test_completely_different(self):
        assert levenshtein_distance("abc", "xyz") == 3

class TestUnseenHandler:
    def setup_method(self):
        self.known = ["abc", "bcd", "cde", "aab"]
        self.handler = UnseenHandler(self.known)

    def test_known_word_not_unseen(self):
        resolved, is_unseen, dist = self.handler.resolve("abc")
        assert resolved == "abc"
        assert is_unseen is False
        assert dist is None

    def test_unseen_word_resolved(self):
        resolved, is_unseen, dist = self.handler.resolve("zzz")
        assert is_unseen is True
        assert resolved in self.known
        assert isinstance(dist, int)

    def test_find_nearest(self):
        nearest, dist = find_nearest_pattern("abd", ["abc", "xyz"])
        assert nearest == "abc"
        assert dist == 1

class TestConfidenceScore:
    def test_low_probability_is_anomaly(self):
        score, decision = compute_confidence_score(0.01)
        assert decision == "anomaly"

    def test_high_probability_is_normal(self):
        score, decision = compute_confidence_score(0.8)
        assert decision == "normal"

    def test_threshold_boundary(self):
        score, decision = compute_confidence_score(0.05)
        assert decision == "normal"

    def test_confidence_label_low(self):
        assert confidence_label(0.01) == "Low"

    def test_confidence_label_medium(self):
        assert confidence_label(0.3) == "Medium"

    def test_confidence_label_high(self):
        assert confidence_label(0.9) == "High"

class TestOutputFormatter:
    def test_format_decision_keys(self):
        result = format_decision(
            time_step=5, state="aab", pattern="adc",
            status="unseen", mapped_to="abc",
            path_probability=0.108
        )
        for key in ["time_step", "state", "pattern", "status",
                    "mapped_to", "probability", "confidence", "decision"]:
            assert key in result

    def test_anomaly_decision(self):
        result = format_decision(
            time_step=1, state="aab", pattern="adc",
            status="unseen", mapped_to="abc",
            path_probability=0.01
        )
        assert result["decision"] == "anomaly"

    def test_to_json_valid(self):
        result = format_decision(
            time_step=1, state="aab", pattern="abc",
            status="seen", mapped_to="abc",
            path_probability=0.72
        )
        json_str = to_json(result)
        import json
        parsed = json.loads(json_str)
        assert parsed["decision"] == "normal"
