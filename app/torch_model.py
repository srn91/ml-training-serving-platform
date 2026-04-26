from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from torch import nn

from app.dataset import FEATURE_NAMES

TORCH_HIDDEN_DIM = 16
TORCH_EPOCHS = 160
TORCH_LEARNING_RATE = 0.03
TORCH_WEIGHT_DECAY = 1e-4
TORCH_SEED = 20260426


class CreditRiskTorchNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = TORCH_HIDDEN_DIM) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass(frozen=True)
class TorchModelBundle:
    model_version: str
    feature_mean: list[float]
    feature_std: list[float]
    network: CreditRiskTorchNet

    @property
    def framework(self) -> str:
        return "torch"

    def _normalize(self, features: list[list[float]] | np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("torch model expects a 2D feature matrix")
        if matrix.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"expected {len(FEATURE_NAMES)} features")
        mean = np.asarray(self.feature_mean, dtype=np.float32)
        std = np.asarray(self.feature_std, dtype=np.float32)
        return (matrix - mean) / std

    def predict_proba(self, features: list[list[float]] | np.ndarray) -> np.ndarray:
        normalized = self._normalize(features)
        self.network.eval()
        with torch.no_grad():
            logits = self.network(torch.as_tensor(normalized, dtype=torch.float32))
            probabilities = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, features: list[list[float]] | np.ndarray) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)


def _seed_everything() -> None:
    torch.manual_seed(TORCH_SEED)
    np.random.seed(TORCH_SEED)


def load_torch_bundle(model_file: Path) -> TorchModelBundle:
    payload = torch.load(model_file, map_location="cpu")
    network = CreditRiskTorchNet(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
    )
    network.load_state_dict(payload["state_dict"])
    return TorchModelBundle(
        model_version=str(payload["model_version"]),
        feature_mean=[float(value) for value in payload["feature_mean"]],
        feature_std=[float(value) for value in payload["feature_std"]],
        network=network,
    )


def train_torch_candidate(
    *,
    role: str,
    model_version: str,
    x_train: list[list[float]],
    y_train: list[int],
    x_test: list[list[float]],
    y_test: list[int],
    model_file: Path,
) -> dict[str, object]:
    _seed_everything()
    train_array = np.asarray(x_train, dtype=np.float32)
    test_array = np.asarray(x_test, dtype=np.float32)
    labels = np.asarray(y_train, dtype=np.float32)

    feature_mean = train_array.mean(axis=0)
    feature_std = train_array.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    normalized_train = (train_array - feature_mean) / feature_std
    network = CreditRiskTorchNet(input_dim=len(FEATURE_NAMES))
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=TORCH_LEARNING_RATE,
        weight_decay=TORCH_WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()

    feature_tensor = torch.as_tensor(normalized_train, dtype=torch.float32)
    label_tensor = torch.as_tensor(labels, dtype=torch.float32)
    final_loss = 0.0
    for _ in range(TORCH_EPOCHS):
        network.train()
        optimizer.zero_grad()
        logits = network(feature_tensor)
        loss = criterion(logits, label_tensor)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    payload = {
        "framework": "torch",
        "model_version": model_version,
        "input_dim": len(FEATURE_NAMES),
        "hidden_dim": TORCH_HIDDEN_DIM,
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "state_dict": network.state_dict(),
        "training_epochs": TORCH_EPOCHS,
        "training_loss": round(final_loss, 4),
    }
    model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, model_file)

    bundle = load_torch_bundle(model_file)
    probabilities = bundle.predict_proba(test_array)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "role": role,
        "framework": "torch",
        "model_version": model_version,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(y_test, probabilities)), 4),
        "final_loss": round(final_loss, 4),
        "training_epochs": TORCH_EPOCHS,
        "hidden_dim": TORCH_HIDDEN_DIM,
    }
    return {
        "role": role,
        "framework": "torch",
        "model_version": model_version,
        "model_file": str(model_file),
        "metrics": metrics,
    }
