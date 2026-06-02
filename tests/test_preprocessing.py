import pytest
import numpy as np
import pandas as pd
from src.automata.paa import compute_paa
from src.automata.sax import SAXEncoder

def test_paa_reduction():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    res = compute_paa(data, n_segments=2)
    assert len(res) == 2
    assert res[0] == pytest.approx(2.5)
    assert res[1] == pytest.approx(6.5)

def test_paa_padding():
    data = np.array([1, 2, 3, 4, 5])
    res = compute_paa(data, n_segments=2)
    assert len(res) == 2

def test_sax_encoding():
    encoder = SAXEncoder(alphabet_size=3)
    data = np.array([[-1.0], [0.0], [1.0]])
    words = encoder.encode_batch(data)
    assert words[0] == 'a'
    assert words[1] == 'b'
    assert words[2] == 'c'

def test_sax_different_alphabet():
    encoder = SAXEncoder(alphabet_size=5)
    data = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    words = encoder.encode_batch(data)
    assert words[0] == 'a'
    assert words[-1] == 'e'

def test_sax_unseen_values():
    encoder = SAXEncoder(alphabet_size=3)
    data = np.array([[-1.0], [0.0], [1.0]])
    unseen = np.array([[-10.0], [10.0]])
    words = encoder.encode_batch(unseen)
    assert words[0] == 'a'
    assert words[1] == 'c'
