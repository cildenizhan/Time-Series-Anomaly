"""
Tum derin ogrenme modelleri icin soyut temel sinif.
Her model bu sinifte tanimlanan arayuzu uygulamak zorundadir.
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Zaman serisi anomali tespit modelleri icin soyut temel sinif.
    LSTM, GRU ve 1D-CNN bu siniftan turetilir.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: config.yaml dosyasindan yuklenen yapilandirma sozlugu.
        """
        self.config = config
        self.model = None
        self.history = None
        self.is_trained = False

    @abstractmethod
    def build(self, input_shape: tuple) -> None:
        """
        Model mimarisini olusturur.

        Args:
            input_shape: Giris verisinin sekli (timesteps, features).
        """
        pass

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """
        Modeli egitir ve egitim gecmisini dondurur.

        Args:
            X_train: Egitim ozellikleri.
            y_train: Egitim etiketleri.
            X_val:   Dogrulama ozellikleri.
            y_val:   Dogrulama etiketleri.

        Returns:
            Egitim gecmisi (loss, val_loss, vb.)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Tahmin uretir.

        Args:
            X: Tahmin yapilacak girdi.

        Returns:
            Tahmin dizisi (sinif veya olasilik).
        """
        pass

    def summary(self) -> None:
        """Keras model ozetini yazdirir."""
        if self.model is not None:
            self.model.summary()
        else:
            print("Model henuz olusturulmadi. Once build() cagirin.")
