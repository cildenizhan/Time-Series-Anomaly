import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from .base_model import BaseModel

class CNN1DModel(BaseModel):

    def __init__(self, config: dict):
        super().__init__(config)
        self.training_cfg = config.get("training", {})
        self.max_epochs  = self.training_cfg.get("max_epochs", 50)
        self.batch_size  = self.training_cfg.get("batch_size", 32)
        self.patience    = self.training_cfg.get("patience", 5)

    def build(self, input_shape: tuple) -> None:
        inp = keras.Input(shape=input_shape, name="input")

        x = layers.Conv1D(64, kernel_size=3, activation="relu",
                          padding="same", name="conv1d_1")(inp)
        x = layers.BatchNormalization(name="bn_1")(x)
        x = layers.Conv1D(128, kernel_size=3, activation="relu",
                          padding="same", name="conv1d_2")(x)
        x = layers.BatchNormalization(name="bn_2")(x)
        x = layers.GlobalAveragePooling1D(name="gap")(x)
        x = layers.Dropout(0.3, name="dropout_1")(x)
        x = layers.Dense(32, activation="relu", name="dense_1")(x)
        out = layers.Dense(1, activation="sigmoid", name="output")(x)

        self.model = keras.Model(inputs=inp, outputs=out, name="CNN1DAnomalyDetector")
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> dict:
        if self.model is None:
            raise RuntimeError("Once build() ile model olusturun.")

        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True
        )

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        self.is_trained = True
        return self.history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model henuz egitilmedi. Once train() cagirin.")
        probs = self.model.predict(X, verbose=0).flatten()
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model henuz egitilmedi.")
        return self.model.predict(X, verbose=0).flatten()
