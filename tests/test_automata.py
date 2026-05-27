"""
Otomata modulleri icin birim testler.
PAA, SAX ve sliding window fonksiyonlarinin dogrulugunu test eder.
"""
import pytest
import numpy as np

from src.automata.paa import compute_paa, batch_paa
from src.automata.sax import SAXEncoder, get_breakpoints
from src.automata.sliding_window import extract_windows, extract_windows_2d
from src.automata.pattern_dict import build_pattern_dict, is_known_pattern
from src.automata.automata_builder import AutomataBuilder


# ======================================================================
# PAA Testleri
# ======================================================================

class TestPAA:
    def test_basic_paa(self):
        """Basit bir dizi uzerinde PAA segmentlerini dogrular."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        result = compute_paa(series, n_segments=4)
        expected = np.array([1.5, 3.5, 5.5, 7.5])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_paa_output_length(self):
        """PAA ciktisinin dogru segment sayisinda oldugunu dogrular."""
        series = np.arange(20, dtype=float)
        for n in [2, 4, 5, 10]:
            result = compute_paa(series, n_segments=n)
            assert len(result) == n

    def test_paa_raises_on_small_series(self):
        """Seri uzunlugu segment sayisindan kucukse hata firlatilmali."""
        with pytest.raises(ValueError):
            compute_paa(np.array([1.0, 2.0]), n_segments=5)

    def test_batch_paa_shape(self):
        """batch_paa ciktisinin beklenen sekle sahip oldugunu dogrular."""
        windows = np.random.randn(10, 8)
        result = batch_paa(windows, n_segments=4)
        assert result.shape == (10, 4)


# ======================================================================
# SAX Testleri
# ======================================================================

class TestSAX:
    def test_encode_length(self):
        """Kodlanmis kelimenin PAA uzunluguna esit oldugunu dogrular."""
        encoder = SAXEncoder(alphabet_size=3)
        paa = np.array([0.5, -0.5, 0.1])
        word = encoder.encode(paa)
        assert len(word) == 3

    def test_encode_alphabet_chars(self):
        """Kodlanmis kelimenin yalnizca alfabe karakterlerini icerdigini dogrular."""
        encoder = SAXEncoder(alphabet_size=3)
        paa = np.random.randn(4)
        word = encoder.encode(paa)
        for ch in word:
            assert ch in "abc"

    def test_invalid_alphabet_size(self):
        """Gecersiz alfabe boyutu hatasi kontrolu."""
        with pytest.raises(ValueError):
            SAXEncoder(alphabet_size=1)
        with pytest.raises(ValueError):
            SAXEncoder(alphabet_size=27)

    def test_encode_batch(self):
        """Toplu kodlamanin dogru sayida kelime uretigini dogrular."""
        encoder = SAXEncoder(alphabet_size=3)
        paa_matrix = np.random.randn(5, 4)
        words = encoder.encode_batch(paa_matrix)
        assert len(words) == 5


# ======================================================================
# Sliding Window Testleri
# ======================================================================

class TestSlidingWindow:
    def test_window_count(self):
        """Dogru pencere sayisi uretildigini dogrular."""
        series = np.arange(10, dtype=float)
        windows, idx = extract_windows(series, window_size=3, step=1)
        assert len(windows) == 8  # 10 - 3 + 1
        assert len(idx) == 8

    def test_window_content(self):
        """Pencere iceriginin dogru oldugunu dogrular."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        windows, _ = extract_windows(series, window_size=3, step=1)
        np.testing.assert_array_equal(windows[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(windows[-1], [3.0, 4.0, 5.0])


# ======================================================================
# AutomataBuilder Testleri
# ======================================================================

class TestAutomataBuilder:
    def test_fit_states(self):
        """Fit sonrasi durumların dogru sekilde olusturuldugunu dogrular."""
        words = ["abc", "bcd", "abc", "cde", "bcd"]
        builder = AutomataBuilder()
        builder.fit(words)
        assert "abc" in builder.states
        assert "bcd" in builder.states

    def test_transition_prob_sum(self):
        """Bir durumdan cikan gecis olasilikları toplami 1.0 olmali."""
        words = ["abc", "bcd", "abc", "bcd", "abc", "cde"]
        builder = AutomataBuilder().fit(words)
        probs = builder.transition_probs.get("abc", {})
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-9

    def test_path_probability_zero(self):
        """Bilinmeyen gecis icin olasilik 0.0 olmali."""
        words = ["abc", "bcd", "cde"]
        builder = AutomataBuilder().fit(words)
        prob = builder.path_probability(["abc", "zzz"])
        assert prob == 0.0
