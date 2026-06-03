import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseModel


class _LSTMNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.fc1   = nn.Linear(32, 16)
        self.relu  = nn.ReLU()
        self.fc2   = nn.Linear(16, 1)
        self.sig   = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out    = self.drop1(out)
        out, _ = self.lstm2(out)
        out    = self.drop2(out[:, -1, :])
        out    = self.relu(self.fc1(out))
        return self.sig(self.fc2(out)).squeeze(1)


class LSTMModel(BaseModel):

    def __init__(self, config: dict):
        super().__init__(config)
        self.training_cfg = config.get("training", {})
        self.max_epochs   = self.training_cfg.get("max_epochs", 50)
        self.batch_size   = self.training_cfg.get("batch_size", 32)
        self.patience     = self.training_cfg.get("patience", 5)
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build(self, input_shape: tuple) -> None:
        input_size   = input_shape[1]
        self.model   = _LSTMNet(input_size).to(self.device)
        self.history = {"loss": [], "val_loss": [], "val_accuracy": []}

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> dict:
        if self.model is None:
            raise RuntimeError("Once build() ile model olusturun.")

        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.BCELoss()

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.float32)
        X_va = torch.tensor(X_val,   dtype=torch.float32)
        y_va = torch.tensor(y_val,   dtype=torch.float32)

        loader = DataLoader(TensorDataset(X_tr, y_tr),
                            batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        wait = 0

        for epoch in range(self.max_epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_out  = self.model(X_va.to(self.device))
                val_loss = criterion(val_out, y_va.to(self.device)).item()
                val_acc  = ((val_out >= 0.5).float() == y_va.to(self.device)).float().mean().item()

            self.history["loss"].append(loss.item())
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        self.model.load_state_dict(best_state)
        self.is_trained = True
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model henuz egitilmedi. Once train() cagirin.")
        self.model.eval()
        with torch.no_grad():
            probs = self.model(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy()
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model henuz egitilmedi.")
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy()
