"""
Model egitim ve validasyon dongusu.
Tum modeller (LSTM, GRU, 1D-CNN) bu modul uzerinden egitilir.
"""
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Derin ogrenme modellerinin egitim surecini yoneten sinif.
    Deney parametrelerini kaydeder ve egitim gecmisini raporlar.
    """

    def __init__(self, model, config: dict, seed: int = 42):
        """
        Args:
            model:  BaseModel turevi bir model nesnesi.
            config: Yapilandirma sozlugu.
            seed:   Yeniden uretilebilirlik icin rastgele tohum.
        """
        self.model  = model
        self.config = config
        self.seed   = seed
        self._set_seed(seed)

    # ------------------------------------------------------------------
    @staticmethod
    def _set_seed(seed: int) -> None:
        """TensorFlow ve NumPy rasgeleligini sabitler."""
        import tensorflow as tf
        import random
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    # ------------------------------------------------------------------
    def run(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            input_shape: tuple) -> dict:
        """
        Modeli olusturur, egitir ve sonuclari dondurur.

        Args:
            X_train:     Egitim ozellikleri.
            y_train:     Egitim etiketleri.
            X_val:       Dogrulama ozellikleri.
            y_val:       Dogrulama etiketleri.
            input_shape: (timesteps, features).

        Returns:
            Egitim sonuclari sozlugu.
        """
        logger.info("Model olusturuluyor: %s | seed=%d",
                    self.model.__class__.__name__, self.seed)
        self.model.build(input_shape)

        start = time.time()
        history = self.model.train(X_train, y_train, X_val, y_val)
        elapsed = time.time() - start

        result = {
            "model":    self.model.__class__.__name__,
            "seed":     self.seed,
            "epochs":   len(history.get("loss", [])),
            "val_loss": history.get("val_loss", [None])[-1],
            "val_acc":  history.get("val_accuracy", [None])[-1],
            "train_sec": round(elapsed, 2),
            "history":  history,
        }

        logger.info("Egitim tamamlandi — %s epoch, %.2fs", result["epochs"], elapsed)
        return result

    # ------------------------------------------------------------------
    def run_multi_seed(self, seeds: list, X_train, y_train,
                       X_val, y_val, input_shape) -> list:
        """
        Ayni modeli farkli seed'lerle calistirip tum sonuclari dondurur.

        Args:
            seeds: Kullanilacak rastgele tohum listesi.

        Returns:
            Her seed icin bir sonuc sozlugu iceren liste.
        """
        results = []
        for s in seeds:
            self._set_seed(s)
            self.seed = s
            res = self.run(X_train, y_train, X_val, y_val, input_shape)
            results.append(res)
        return results
