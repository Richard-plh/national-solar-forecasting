"""Scientific settings for national-level hourly Solar forecasting."""

from __future__ import annotations

import json
from pathlib import Path

TASK_ID = "national_solar_forecasting"
PREPROCESSING_VERSION = "complete_hourly_windows_v1"
TARGET_NAME = "solar_generation_actual"
POWER_COLUMNS = ("load_actual", "wind_generation_actual", TARGET_NAME)
FEATURE_ORDER = (
    "NetLoad",
    *POWER_COLUMNS,
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "month_sin",
    "month_cos",
)
LOOKBACKS = (12, 24, 48, 96, 168, 336)
HORIZON, L_MAX = 24, 336
COUNTRIES = {
    "Spain": "ES_60min.csv",
    "Germany": "DE_60min.csv",
    "Greece": "GR_60min.csv",
}
SEEDS = (42, 123, 2026, 3407, 7811)
FUSION_MODELS = ("equal-fusion", "global-fusion", "horizon-fusion")
SUPERVISION_MODELS = (
    "trajectory-supervision",
    "state-supervision",
    "joint-supervision",
)
MODEL_NAMES = FUSION_MODELS + SUPERVISION_MODELS
SOURCE_HASHES = {
    "Spain": "6b6d354113f8b069032d9d1d385e1d3f4b625e5f596776f71cc91dd820bdb49b",
    "Germany": "80ce31c62b4f4f568111b661a11efc63dc7a0deacb381c5b0b2fe4f87f7a3a69",
    "Greece": "c27f5658575e5581ead01cc7b85e236dc190d568130a79fbd0388dab015d0e97",
}
TRAINING = {
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "batch_size": 64,
    "max_epochs": 80,
    "patience": 12,
    "precision": "FP32",
    "selection_metric": "Validation standardized Solar MAE",
    "num_workers": 0,
    "amp": False,
    "scheduler": None,
    "gradient_clipping": None,
    "gradient_accumulation": False,
}
THRESHOLDS = {
    "Spain": {"q30_MW": 18.0, "q80_MW": 591.0, "active_MW": 245.55},
    "Germany": {"q30_MW": 0.0, "q80_MW": 2408.0, "active_MW": 925.0500000000001},
    "Greece": {"q30_MW": 0.0, "q80_MW": 261.0, "active_MW": 79.4},
}
EXPERIMENT = {
    "schema_version": 1,
    "task": TASK_ID,
    "target": TARGET_NAME,
    "unit": "MW",
    "countries": list(COUNTRIES),
    "models": list(MODEL_NAMES),
    "seeds": list(SEEDS),
    "features": list(FEATURE_ORDER),
    "lookbacks_hours": list(LOOKBACKS),
    "horizon_hours": 24,
    "preprocessing_version": PREPROCESSING_VERSION,
    "temporal_split": [0.7, 0.1, 0.2],
    "training": TRAINING,
    "thresholds": THRESHOLDS,
    "loss_weights": {"trajectory": 1.0, "classification": 1.0},
    "curve_seed": 42,
    "curve_dates": {
        "Validation": ["2018-08-21 00:00", "2018-12-21 00:00"],
        "Test": ["2019-06-21 00:00", "2019-12-21 00:00"],
    },
}


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config != EXPERIMENT:
        raise ValueError(
            "The supplied settings differ from the reproducibility configuration"
        )
    return config


def run_matrix() -> list[dict]:
    return [
        {
            "country": country,
            "model": model,
            "seed": seed,
            "group": "fusion" if model in FUSION_MODELS else "supervision",
            "run_path": f"runs/{country}/{model}/seed_{seed}",
        }
        for models in (FUSION_MODELS, SUPERVISION_MODELS)
        for country in COUNTRIES
        for seed in SEEDS
        for model in models
    ]
