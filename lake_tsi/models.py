"""Load a trained TSI model (.pkl/.joblib or BNN-MCD .pth) and run inference.

Ported from the notebook's model-loading block in main() (cell 16) plus the
BNN_MCD class + predict_bnn from cell 15. Logic unchanged; just reorganised
into functions callable from Streamlit without CLI prompts.
"""
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gee_utils import FEATURE_NAMES
from .trophic import TROPHIC_CLASSES, tsi_to_class_id


class BNN_MCD(nn.Module):
    """Bayesian NN via Monte-Carlo Dropout - must match the training script."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_rate, training=True)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout_rate, training=True)
        x = self.fc3(x)
        return torch.sigmoid(x) * 100


def predict_bnn(model, X: np.ndarray, n_samples: int = 30, batch_size: int = 2048, device: str = "cpu"):
    """Monte-Carlo dropout inference -> (mean_prediction, uncertainty_std)."""
    model.to(device)
    model.train()  # keep dropout ON for MC sampling
    tensor_X = torch.tensor(X, dtype=torch.float32).to(device)

    all_preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            batch_preds = []
            for start in range(0, len(tensor_X), batch_size):
                batch = tensor_X[start:start + batch_size]
                out = model(batch)
                batch_preds.append(out.cpu().numpy())
            all_preds.append(np.vstack(batch_preds))

    all_preds = np.array(all_preds)
    mean_pred = all_preds.mean(axis=0).flatten()
    uncertainty = all_preds.std(axis=0).flatten()
    model.eval()
    return mean_pred, uncertainty


def load_model(model_path: str):
    """Load a model file, returning (model, kind) where kind is 'sklearn' or 'bnn'."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    ext = os.path.splitext(model_path)[1].lower()

    if ext in (".pkl", ".joblib"):
        return joblib.load(model_path), "sklearn"

    if ext in (".pth", ".pt"):
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model = BNN_MCD(
                checkpoint["input_dim"], checkpoint["hidden_dim"], 1, checkpoint["dropout"]
            )
            model.load_state_dict(checkpoint["model_state"])
        else:
            model = BNN_MCD(input_dim=len(FEATURE_NAMES), hidden_dim=64, output_dim=1, dropout_rate=0.3)
            model.load_state_dict(checkpoint)
        model.eval()
        return model, "bnn"

    raise ValueError(f"Unsupported model format: {ext}")


def predict_tsi(pixel_df: pd.DataFrame, model, kind: str) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Run TSI inference over every pixel row. Returns (result_df, uncertainty|None)."""
    X = pixel_df[FEATURE_NAMES].values

    if kind == "bnn":
        tsi_preds, uncertainty = predict_bnn(model, X)
    else:
        tsi_preds = model.predict(X)
        uncertainty = None

    class_ids = [tsi_to_class_id(float(t)) for t in tsi_preds]
    result_df = pixel_df[["latitude", "longitude"]].copy()
    result_df["TSI_Score"] = np.round(tsi_preds, 2)
    result_df["Class_ID"] = class_ids
    result_df["Trophic_Status"] = [TROPHIC_CLASSES[c]["label"] for c in class_ids]
    result_df["Color"] = [TROPHIC_CLASSES[c]["color"] for c in class_ids]
    for f in FEATURE_NAMES:
        result_df[f] = pixel_df[f].values

    return result_df, uncertainty
