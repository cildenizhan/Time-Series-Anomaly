"""
Uctan uce pipeline modulu.

Veri yukleme -> on isleme -> pencere olusturma ->
PAA/SAX donusumu -> otomata egitimi/tahmini -> aciklama
asamalarini tek bir arayuzde birlestiren Pipeline sinifi.
"""
import numpy as np
import logging
from typing import Optional

from src.utils import load_config
from src.data.loader import load_skab, load_batadal
from src.data.preprocessor import TimeSeriesPreprocessor
from src.data.splitter import split_batadal, get_skab_cv
from src.automata.sliding_window import extract_windows
from src.automata.paa import batch_paa
from src.automata.sax import SAXEncoder
from src.automata.automata_builder import AutomataBuilder
from src.automata.pattern_dict import build_pattern_dict
from src.automata.unseen_handler import UnseenHandler
from src.explainability.explainer import Explainer

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Tam anomali tespit pipeline'i.

    config.yaml uzerinden tum parametreleri okur ve
    modulleri sirasi ile calistirir.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg          = load_config(config_path)
        self.window_size  = self.cfg["automata"]["window_size"]
        self.alphabet_size= self.cfg["automata"]["alphabet_size"]

        self.preprocessor: Optional[TimeSeriesPreprocessor] = None
        self.sax_encoder:  Optional[SAXEncoder]             = None
        self.builder:      Optional[AutomataBuilder]        = None
        self.unseen:       Optional[UnseenHandler]          = None
        self.explainer:    Optional[Explainer]              = None

    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray) -> "Pipeline":
        """
        Egitim verisiyle pipeline'i egitir.

        Normalizasyon -> Pencere -> PAA -> SAX -> Otomata adimlari.

        Args:
            X_train: Ham egitim verisi. Sekil: (T, F) veya (T,)

        Returns:
            self
        """
        logger.info("Pipeline egitimi basliyor...")

        # 1. On isleme
        self.preprocessor = TimeSeriesPreprocessor(use_pca=True)
        X_scaled = self.preprocessor.fit_transform(X_train)
        X_1d     = X_scaled.flatten()

        # 2. Pencere olustur
        windows, _ = extract_windows(X_1d, self.window_size)

        # 3. PAA
        paa_matrix = batch_paa(windows, self.window_size)

        # 4. SAX
        self.sax_encoder = SAXEncoder(self.alphabet_size)
        sax_words        = self.sax_encoder.encode_batch(paa_matrix)

        # 5. Otomata
        self.builder = AutomataBuilder().fit(sax_words)

        # 6. Pattern sozlugu & Unseen handler
        pattern_dict = build_pattern_dict(sax_words)
        self.unseen  = UnseenHandler(list(pattern_dict.keys()))

        # 7. Explainer
        self.explainer = Explainer(self.builder, self.unseen)

        logger.info("Pipeline egitimi tamamlandi. Durum sayisi: %d",
                    len(self.builder.states))
        return self

    # ------------------------------------------------------------------
    def predict(self, X_test: np.ndarray) -> dict:
        """
        Test verisi uzerinde anomali tespiti yapar.

        Args:
            X_test: Ham test verisi.

        Returns:
            predictions, path_probabilities, explanations icerik sozlugu.
        """
        if self.preprocessor is None:
            raise RuntimeError("Once fit() cagirin.")

        X_scaled  = self.preprocessor.transform(X_test)
        X_1d      = X_scaled.flatten()
        windows, _= extract_windows(X_1d, self.window_size)
        paa       = batch_paa(windows, self.window_size)
        sax_words = self.sax_encoder.encode_batch(paa)

        explanations   = self.explainer.explain_sequence(sax_words)
        path_prob      = self.explainer.compute_path_probability(sax_words)
        predictions    = [1 if e.get("transition_prob", 1.0) == 0.0 else 0
                          for e in explanations]

        return {
            "sax_words":    sax_words,
            "explanations": explanations,
            "path_probability": path_prob,
            "predictions":  predictions,
        }
