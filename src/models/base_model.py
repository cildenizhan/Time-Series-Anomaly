from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.history = None
        self.is_trained = False

    @abstractmethod
    def build(self, input_shape: tuple) -> None:
        pass

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> dict:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def summary(self) -> None:
        if self.model is not None:
            self.model.summary()
        else:
            print("Model henuz olusturulmadi. Once build() cagirin.")
