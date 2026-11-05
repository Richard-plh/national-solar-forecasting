"""Shared-GRU multi-scale forecasting and optional training supervision."""

from __future__ import annotations

import copy
import hashlib
import os
import random

import numpy as np
import torch
from torch import nn

from .config import (
    FEATURE_ORDER,
    HORIZON,
    LOOKBACKS,
    L_MAX,
    MODEL_NAMES,
    SUPERVISION_MODELS,
)


def set_deterministic(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def order_seed(country: str, seed: int) -> int:
    offsets = {"Spain": 11, "Germany": 23, "Greece": 37}
    return int(seed) * 1009 + offsets[country]


def component_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class SharedGRUEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=10,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[2] != len(FEATURE_ORDER):
            raise ValueError("Expected [batch, history, 10]")
        _, hidden = self.gru(x)
        return hidden[-1]


class ScaleGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(152, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 6),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, context, horizon_one_hot):
        horizon = horizon_one_hot.unsqueeze(0).expand(context.shape[0], -1, -1)
        context = context.unsqueeze(1).expand(-1, HORIZON, -1)
        return torch.softmax(
            self.network(torch.cat([context, horizon], dim=-1)), dim=-1
        )


class ForecastHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(152, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)
        )

    def forward(self, fused, horizon_one_hot):
        horizon = horizon_one_hot.unsqueeze(0).expand(fused.shape[0], -1, -1)
        return self.network(torch.cat([fused, horizon], dim=-1)).squeeze(-1)


class MultiScaleForecaster(nn.Module):
    def __init__(self, fusion: str = "horizon-fusion"):
        super().__init__()
        if fusion not in ("equal-fusion", "global-fusion", "horizon-fusion"):
            raise ValueError("Unknown fusion strategy")
        self.model_name = fusion
        # Construction order is part of matched random initialization.
        self.encoder = SharedGRUEncoder()
        self.forecast_head = ForecastHead()
        if fusion == "global-fusion":
            self.global_logits = nn.Parameter(torch.zeros(6))
        elif fusion == "horizon-fusion":
            self.gate = ScaleGate()
        self.register_buffer("horizon_one_hot", torch.eye(HORIZON))

    def encode_scales(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (L_MAX, 10):
            raise ValueError("Expected [batch, 336, 10]")
        return torch.stack(
            [self.encoder(x[:, -length:, :]) for length in LOOKBACKS], dim=1
        )

    def scale_weights(self, encoded=None):
        if self.model_name == "equal-fusion":
            weights = self.horizon_one_hot.new_full((HORIZON, 6), 1.0 / 6.0)
        elif self.model_name == "global-fusion":
            weights = (
                torch.softmax(self.global_logits, dim=0).view(1, 6).expand(HORIZON, 6)
            )
        else:
            # Fixed zero context: no observed sample state enters the gate.
            context = self.horizon_one_hot.new_zeros((1, 128))
            weights = self.gate(context, self.horizon_one_hot).squeeze(0)
        return (
            weights
            if encoded is None
            else weights.unsqueeze(0).expand(encoded.shape[0], -1, -1)
        )

    def forward(self, x, *, return_features=False):
        encoded = self.encode_scales(x)
        weights = self.scale_weights(encoded)
        fused = torch.einsum("bhs,bsd->bhd", weights, encoded)
        prediction = self.forecast_head(fused, self.horizon_one_hot)
        if return_features:
            return prediction, fused, weights
        return prediction


class DynamicsForecaster(nn.Module):
    def __init__(
        self, base: MultiScaleForecaster, supervision: str, auxiliary_seed: int
    ):
        super().__init__()
        if supervision not in SUPERVISION_MODELS:
            raise ValueError("Unknown supervision strategy")
        self.base, self.model_name = base, supervision
        before = component_hash(base)
        if supervision in ("state-supervision", "joint-supervision"):
            set_deterministic(auxiliary_seed)
            self.auxiliary_head = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 5)
            )
        if component_hash(base) != before:
            raise RuntimeError(
                "Auxiliary initialization changed the forecasting components"
            )

    def forward(self, x, *, training_outputs=False):
        if not training_outputs:
            return self.base(x)
        prediction, fused, weights = self.base(x, return_features=True)
        logits = self.auxiliary_head(fused) if hasattr(self, "auxiliary_head") else None
        return prediction, logits, fused, weights


def shared_components(model):
    base = model.base if isinstance(model, DynamicsForecaster) else model
    return {
        name: component_hash(getattr(base, name))
        for name in ("encoder", "forecast_head", "gate")
        if hasattr(base, name)
    }


def build_matched(model_name: str, country: str, seed: int):
    """Fresh random weights with matched encoder/head/gate; never loads a checkpoint."""
    if model_name not in MODEL_NAMES:
        raise ValueError("Unknown model")
    derived = order_seed(country, seed)
    base_name = "horizon-fusion" if model_name in SUPERVISION_MODELS else model_name
    set_deterministic(derived)
    master = MultiScaleForecaster("horizon-fusion")
    base = MultiScaleForecaster(base_name)
    base.encoder.load_state_dict(copy.deepcopy(master.encoder.state_dict()))
    base.encoder.gru.flatten_parameters()
    base.forecast_head.load_state_dict(copy.deepcopy(master.forecast_head.state_dict()))
    if base_name == "horizon-fusion":
        base.gate.load_state_dict(copy.deepcopy(master.gate.state_dict()))
    if base_name == "global-fusion":
        nn.init.zeros_(base.global_logits)
    auxiliary_seed = derived + 50_000_033
    model = (
        DynamicsForecaster(base, model_name, auxiliary_seed)
        if model_name in SUPERVISION_MODELS
        else base
    )
    model.initialization_metadata = {
        "country": country,
        "seed": seed,
        "model": model_name,
        "derived_seed": derived,
        "shared_components": shared_components(model),
        "auxiliary_seed": auxiliary_seed if hasattr(model, "auxiliary_head") else None,
        "auxiliary_hash": (
            component_hash(model.auxiliary_head)
            if hasattr(model, "auxiliary_head")
            else None
        ),
        "trained_weights_loaded": False,
    }
    set_deterministic(derived)
    return model


def trajectory_delta(values, anchor):
    if values.ndim != 2 or values.shape[1] != 24 or anchor.shape != (len(values),):
        raise ValueError("Expected [batch, 24] and [batch] anchor")
    if not torch.isfinite(values).all() or not torch.isfinite(anchor).all():
        raise ValueError("Nonfinite trajectory")
    return torch.cat(
        [values[:, :1] - anchor[:, None], values[:, 1:] - values[:, :-1]], dim=1
    )


def state_labels(delta, q30, q80):
    if not np.isfinite([q30, q80]).all() or not 0 <= q30 < q80:
        raise ValueError("Require finite 0 <= Q30 < Q80")
    if not torch.isfinite(delta).all():
        raise ValueError("Nonfinite Solar changes")
    labels = torch.full(delta.shape, 2, dtype=torch.int64, device=delta.device)
    labels[delta < -q80] = 0
    labels[(delta >= -q80) & (delta < -q30)] = 1
    labels[(delta > q30) & (delta <= q80)] = 3
    labels[delta > q80] = 4
    return labels


def loss_components(model_name, prediction, target, x, logits=None, labels=None):
    if (
        model_name not in MODEL_NAMES
        or prediction.shape != target.shape
        or target.shape != (len(x), 24)
    ):
        raise ValueError("Invalid model or loss shapes")
    if x.shape[1:] != (336, 10) or not all(
        torch.isfinite(v).all() for v in (prediction, target, x)
    ):
        raise ValueError("Expected finite [batch, 336, 10] histories")
    parts = {"main_mse": nn.functional.mse_loss(prediction, target)}
    if model_name in ("trajectory-supervision", "joint-supervision"):
        anchor = x[:, -1, 3]
        parts["trajectory_mae"] = torch.mean(
            torch.abs(
                trajectory_delta(prediction, anchor) - trajectory_delta(target, anchor)
            )
        )
    if model_name in ("state-supervision", "joint-supervision"):
        if (
            logits is None
            or logits.shape != (*target.shape, 5)
            or labels is None
            or labels.shape != target.shape
        ):
            raise ValueError("Expected [batch, 24, 5] logits and [batch, 24] labels")
        if (
            labels.dtype != torch.int64
            or not torch.isfinite(logits).all()
            or labels.min() < 0
            or labels.max() > 4
        ):
            raise ValueError("Invalid classification logits or labels")
        parts["classification_ce"] = nn.functional.cross_entropy(
            logits.reshape(-1, 5), labels.reshape(-1)
        )
    parts["total"] = sum(parts.values())
    if not all(torch.isfinite(v) for v in parts.values()):
        raise RuntimeError("Nonfinite loss")
    return parts
