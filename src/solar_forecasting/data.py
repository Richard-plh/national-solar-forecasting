"""Hourly Solar inputs, Train-only scaling and complete-window selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    COUNTRIES,
    FEATURE_ORDER,
    HORIZON,
    L_MAX,
    LOOKBACKS,
    POWER_COLUMNS,
    PREPROCESSING_VERSION,
    TARGET_NAME,
    TASK_ID,
)
from .utils import hash_array, hash_json, read_json, sha256_file, write_json

SCHEMA_VERSION = "solar-hourly-v1"
SCALER_POLICY = "Train complete physical rows, unique hours, population std, ddof=0"


def row_ranges(count: int) -> dict[str, tuple[int, int]]:
    """Return the half-open 70/10/20 temporal split, with int flooring."""
    if count < 1:
        raise ValueError("Empty time axis.")
    return {
        "Train": (0, int(count * 0.7)),
        "Validation": (int(count * 0.7), int(count * 0.8)),
        "Test": (int(count * 0.8), count),
    }


def inspect_times(values: Any) -> np.ndarray:
    times = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if times.tz is not None:
        raise ValueError(
            "Timezone-aware input is not the frozen stored time axis; no implicit conversion."
        )
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError("Timestamps must be finite, unique and ordered.")
    result = times.to_numpy(dtype="datetime64[ns]")
    if len(result) > 1 and not np.all(np.diff(result) == np.timedelta64(1, "h")):
        raise ValueError("Time rows must remain strictly continuous hourly records.")
    return result


def build_features(frame: pd.DataFrame) -> np.ndarray:
    """Build 10 raw features without reconstructing any missing power value.

    NetLoad is invalid whenever any required physical measurement is nonfinite.
    CSV Net_load is never read. Stored naive Time is used without timezone shift.
    """
    required = {"Time", *POWER_COLUMNS}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing required CSV fields: {sorted(missing)}")
    times = pd.DatetimeIndex(inspect_times(frame["Time"]))
    power = (
        frame[list(POWER_COLUMNS)]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=np.float64)
    )
    complete = np.isfinite(power).all(axis=1)
    net = np.full(len(frame), np.nan, dtype=np.float64)
    net[complete] = power[complete, 0] - power[complete, 1] - power[complete, 2]
    columns = [net, *[power[:, index] for index in range(3)]]
    for values, period in (
        (times.hour, 24),
        (times.dayofweek, 7),
        (times.month - 1, 12),
    ):
        angle = 2 * np.pi * np.asarray(values, dtype=np.float64) / period
        columns.extend([np.sin(angle), np.cos(angle)])
    return np.column_stack(columns)


def fit_scalers(raw_features: np.ndarray, train_end: int) -> tuple[dict, dict, int]:
    """Fit once per complete Train hour; never weight hours by sliding windows.

    A zero/nonfinite standard deviation raises, under the fixed preprocessing policy.
    It is not silently replaced by one. Zero observations in a varying column
    remain valid, are included in its statistics, and can be inverse-transformed.
    """
    values = np.asarray(raw_features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_ORDER):
        raise ValueError("Expected ten raw features in the locked order.")
    if not 0 < train_end <= len(values):
        raise ValueError("Invalid Train boundary.")
    train = values[:train_end]
    valid = np.isfinite(train[:, 1:4]).all(axis=1)
    fit_values = train[valid, :4]
    if not len(fit_values) or not np.isfinite(fit_values).all():
        raise ValueError("No valid finite complete Train rows for scaler fitting.")
    feature_scalers = {}
    for column, name in enumerate(FEATURE_ORDER[:4]):
        mean = float(fit_values[:, column].mean())
        std = float(fit_values[:, column].std(ddof=0))
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError(
                f"Invalid Train standard deviation for {name}; constant features are rejected."
            )
        feature_scalers[name] = {"name": name, "mean": mean, "std": std, "ddof": 0}
    # Separate objects and explicit semantic target guard: never alias feature 0.
    target_scaler = dict(feature_scalers[TARGET_NAME])
    return feature_scalers, target_scaler, int(valid.sum())


def _check_target_scaler(scaler: dict) -> None:
    if scaler.get("name") != TARGET_NAME:
        raise ValueError(
            "Target scaler must explicitly identify solar_generation_actual."
        )
    if (
        scaler.get("ddof") != 0
        or not np.isfinite(scaler["mean"])
        or not np.isfinite(scaler["std"])
        or scaler["std"] <= 0
    ):
        raise ValueError("Invalid Solar target scaler.")


def encode_features(raw_features: np.ndarray, feature_scalers: dict) -> np.ndarray:
    """Transform using supplied Train statistics; retain nonfinite rows and time features."""
    encoded = np.asarray(raw_features, dtype=np.float64).copy()
    if set(feature_scalers) != set(FEATURE_ORDER[:4]):
        raise ValueError(
            "Feature scaler keys do not match the four physical feature names."
        )
    for column, name in enumerate(FEATURE_ORDER[:4]):
        scaler = feature_scalers[name]
        if (
            scaler.get("name") != name
            or scaler.get("ddof") != 0
            or not np.isfinite(scaler["mean"])
            or not np.isfinite(scaler["std"])
            or scaler["std"] <= 0
        ):
            raise ValueError(f"Invalid feature scaler for {name}.")
        encoded[:, column] = (encoded[:, column] - scaler["mean"]) / scaler["std"]
    return encoded.astype(np.float32)


def standardize_target(values: np.ndarray, target_scaler: dict) -> np.ndarray:
    _check_target_scaler(target_scaler)
    return (np.asarray(values) - target_scaler["mean"]) / target_scaler["std"]


def inverse_target(values: np.ndarray, target_scaler: dict) -> np.ndarray:
    """Return MW without clipping; negative standardized values need not be negative MW."""
    _check_target_scaler(target_scaler)
    return np.asarray(values) * target_scaler["std"] + target_scaler["mean"]


def valid_origins(physical: np.ndarray, solar: np.ndarray) -> tuple[np.ndarray, dict]:
    """Filter first-label indices with cumulative invalid counts, retaining all rows."""
    power, target = np.asarray(physical), np.asarray(solar)
    if power.ndim != 2 or power.shape[1] != 3 or target.shape != (len(power),):
        raise ValueError("Expected [hours,3] physical input and [hours] Solar target.")
    n = len(target)
    origins = np.arange(L_MAX, max(L_MAX, n - HORIZON + 1), dtype=np.int64)
    history_invalid = ~np.isfinite(power).all(axis=1)
    label_invalid = ~np.isfinite(target)
    history_prefix = np.r_[0, np.cumsum(history_invalid, dtype=np.int64)]
    label_prefix = np.r_[0, np.cumsum(label_invalid, dtype=np.int64)]
    bad_history = (history_prefix[origins] - history_prefix[origins - L_MAX]) > 0
    bad_target = (label_prefix[origins + HORIZON] - label_prefix[origins]) > 0
    excluded = bad_history | bad_target
    selected = origins[~excluded]
    statistics = {
        "raw_rows": n,
        "theoretical_windows": len(origins),
        "excluded_history": int(bad_history.sum()),
        "excluded_future_solar": int(bad_target.sum()),
        "excluded_both": int((bad_history & bad_target).sum()),
        "excluded_union": int(excluded.sum()),
        "valid_windows": len(selected),
        "valid_window_ratio": (
            float(len(selected) / len(origins)) if len(origins) else 0.0
        ),
        "history_incomplete_hours": int(history_invalid.sum()),
        "solar_nonfinite_hours": int(label_invalid.sum()),
        "boundary_exclusion": {
            "initial_history_hours": min(n, L_MAX),
            "terminal_incomplete_horizon_origins": min(max(n - L_MAX, 0), HORIZON - 1),
        },
    }
    return selected, statistics


def missing_segments(values: np.ndarray, times: np.ndarray) -> list[dict]:
    """Record nonfinite runs on the unchanged full time axis (zero-based rows).

    The end row is exclusive; end_timestamp is the last missing hourly label.
    Missing NaN and +/-inf are all invalid, independently of finite anomalies.
    """
    values, times = np.asarray(values), np.asarray(times)
    if values.ndim != 1 or times.shape != values.shape:
        raise ValueError(
            "Missing-segment values and times must be equal-length vectors."
        )
    invalid = ~np.isfinite(values)
    edges = np.diff(np.r_[False, invalid, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [
        {
            "start_row": int(start),
            "end_row_exclusive": int(end),
            "start_timestamp": str(times[start]),
            "end_timestamp": str(times[end - 1]),
            "count": int(end - start),
        }
        for start, end in zip(starts, ends)
    ]


def _quality_observations(power: np.ndarray, times: np.ndarray) -> dict:
    """Descriptive finite-value observations only; never used in validity filtering."""
    result = {
        "action": "record only; no smoothing, replacement, removal or threshold selection",
        "columns": {},
    }
    for column, name in enumerate(POWER_COLUMNS):
        values = power[:, column]
        finite = np.isfinite(values)
        differences = np.abs(np.diff(values))
        pair_finite = finite[1:] & finite[:-1]
        indices = np.flatnonzero(pair_finite)
        chosen = indices[np.argsort(differences[indices], kind="stable")[-5:]][::-1]
        result["columns"][name] = {
            "nonfinite_hours": int((~finite).sum()),
            "nan_hours": int(np.isnan(values).sum()),
            "nonfinite_segments": missing_segments(values, times),
            "finite_negative_hours": int((finite & (values < 0)).sum()),
            "finite_zero_hours": int((finite & (values == 0)).sum()),
            "consecutive_equal_finite_pairs": int(
                (pair_finite & (np.diff(values) == 0)).sum()
            ),
            "largest_absolute_adjacent_changes": [
                {
                    "left_timestamp": str(times[j]),
                    "right_timestamp": str(times[j + 1]),
                    "left_MW": float(values[j]),
                    "right_MW": float(values[j + 1]),
                    "absolute_change_MW": float(differences[j]),
                }
                for j in chosen
            ],
        }
    return result


@dataclass
class PreparedSplit:
    name: str
    features: np.ndarray
    raw_features: np.ndarray
    targets: np.ndarray
    raw_targets: np.ndarray
    times: np.ndarray
    origins: np.ndarray
    statistics: dict
    origin_hash: str
    global_start: int
    global_end: int

    @property
    def target_standardized(self) -> np.ndarray:
        return self.targets

    @property
    def target_original(self) -> np.ndarray:
        return self.raw_targets

    @property
    def origin_indices(self) -> np.ndarray:
        return self.origins


@dataclass
class PreparedCountry:
    country: str
    source_sha256: str
    feature_scalers: dict
    target_scaler: dict
    splits: dict[str, PreparedSplit]
    metadata: dict

    @property
    def scaler_hash(self) -> str:
        return self.metadata["identity"]["scaler_hash"]

    @property
    def identity(self) -> dict:
        return self.metadata["identity"]

    def dataset(self, split_name, *, allow_test=False):
        return SolarWindowDataset(self, split_name, allow_test=allow_test)

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        return inverse_target(values, self.target_scaler)


class SolarWindowDataset:
    """Split-local, complete historical windows; Test requires an explicit opt-in."""

    def __init__(self, country, split_name, *, allow_test=False):
        if split_name not in ("Train", "Validation", "Test"):
            raise ValueError("Unknown split")
        if split_name == "Test" and not allow_test:
            raise PermissionError("Test tensors require an explicit evaluation opt-in")
        self.data = country.splits[split_name]
        self.split = split_name
        self.country = country.country
        self.origin_indices = self.data.origins
        self.origin_hash = self.data.origin_hash
        self.identity = country.identity

    def __len__(self):
        return len(self.origin_indices)

    def __getitem__(self, index):
        import torch

        origin = int(self.origin_indices[index])
        x = np.ascontiguousarray(self.data.features[origin - L_MAX : origin])
        y = np.ascontiguousarray(self.data.targets[origin : origin + HORIZON])
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise RuntimeError("Nonfinite data inside a complete window")
        return torch.from_numpy(x), torch.from_numpy(y)


def _origin_hash(
    origins: np.ndarray, times: np.ndarray, bounds: tuple[int, int]
) -> str:
    return hash_json(
        {
            "local_first_label_indices": hash_array(origins),
            "first_target_times_ns": hash_array(times[origins].view(np.int64)),
            "global_row_bounds": list(bounds),
            "history": L_MAX,
            "horizon": HORIZON,
        }
    )


def prepare_frame(
    frame: pd.DataFrame, country: str = "Spain", source_sha256: str = "synthetic"
) -> PreparedCountry:
    """Pure preparation entry for a supplied frame, also used by synthetic tests."""
    if country not in COUNTRIES:
        raise ValueError(f"Country must be a configured full name: {country!r}")
    times = inspect_times(frame["Time"])
    raw = build_features(frame)
    bounds = row_ranges(len(frame))
    feature_scalers, target_scaler, fit_count = fit_scalers(raw, bounds["Train"][1])
    encoded = encode_features(raw, feature_scalers)
    raw_targets = frame[TARGET_NAME].to_numpy(dtype=np.float64)
    targets = standardize_target(raw_targets, target_scaler).astype(np.float32)
    scaler_configuration = {
        "policy": SCALER_POLICY,
        "fit_split": "Train",
        "fit_rows": fit_count,
        "feature_scalers": feature_scalers,
        "target_scaler": target_scaler,
        "target_name": TARGET_NAME,
        "source_csv_sha256": source_sha256,
        "split_bounds": {name: list(pair) for name, pair in bounds.items()},
        "ddof": 0,
        "constant_feature_policy": "raise ValueError; constant power features rejected",
        "time_features_scaled": False,
    }
    scaler_hash = hash_json(scaler_configuration)
    splits, split_metadata = {}, {}
    for name, (start, end) in bounds.items():
        split_raw = raw[start:end].copy()
        split_target = raw_targets[start:end].copy()
        split_times = times[start:end].copy()
        origins, statistics = valid_origins(split_raw[:, 1:4], split_target)
        origin_hash = _origin_hash(origins, split_times, (start, end))
        splits[name] = PreparedSplit(
            name,
            encoded[start:end].copy(),
            split_raw,
            targets[start:end].copy(),
            split_target,
            split_times,
            origins,
            statistics,
            origin_hash,
            start,
            end,
        )
        split_metadata[name] = {
            **statistics,
            "global_start": start,
            "global_end": end,
            "start_timestamp": str(split_times[0]) if len(split_times) else None,
            "end_timestamp": str(split_times[-1]) if len(split_times) else None,
            "first_valid_first_target_timestamp": (
                str(split_times[origins[0]]) if len(origins) else None
            ),
            "last_valid_first_target_timestamp": (
                str(split_times[origins[-1]]) if len(origins) else None
            ),
            "origin_hash": origin_hash,
        }
    identity = {
        "task_id": TASK_ID,
        "country": country,
        "target_name": TARGET_NAME,
        "feature_order": list(FEATURE_ORDER),
        "source_csv_sha256": source_sha256,
        "preprocessing_version": PREPROCESSING_VERSION,
        "split_bounds": {name: list(pair) for name, pair in bounds.items()},
        "scaler_hash": scaler_hash,
        "origin_hash": {name: split.origin_hash for name, split in splits.items()},
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "source_kind": "frozen derived hourly country CSV; not the unprocessed OPSD table",
        "unit": "MW",
        "target_name": TARGET_NAME,
        "feature_order": list(FEATURE_ORDER),
        "raw_rows": len(frame),
        "time_start": str(times[0]),
        "time_end": str(times[-1]),
        "source_columns": list(frame.columns),
        "time_axis": "stored naive Time; no local timezone conversion",
        "time_interval_semantics": "upstream interval-start convention; entire local transformation chain not recovered",
        "missing_policy": "retain every hour; no imputation; filter origins only by nonfinite required physical observations",
        "net_load_policy": "recompute load_actual - wind_generation_actual - solar_generation_actual; NaN if any input nonfinite",
        "scaler_configuration": scaler_configuration,
        "splits": split_metadata,
        "quality_observations": _quality_observations(raw[:, 1:4], times),
        "lookbacks": list(LOOKBACKS),
        "horizon": HORIZON,
        "max_history": L_MAX,
        "formal_test_inference": 0,
    }
    return PreparedCountry(
        country, source_sha256, feature_scalers, target_scaler, splits, metadata
    )


def prepare_country(
    csv_path: str | Path, country: str, expected_sha256: str | None = None
) -> PreparedCountry:
    path = Path(csv_path)
    source_hash = sha256_file(path)
    if expected_sha256 is not None and source_hash.lower() != expected_sha256.lower():
        raise ValueError(f"Frozen CSV SHA-256 mismatch for {country}.")
    prepared = prepare_frame(pd.read_csv(path), country, source_hash)
    if sha256_file(path) != source_hash:
        raise RuntimeError("CSV changed during read.")
    prepared.metadata["source_file_name"] = path.name
    prepared.metadata["source_file_bytes"] = path.stat().st_size
    return prepared


def save_prepared(country: PreparedCountry, output_dir: str | Path) -> Path:
    """Write one versioned full-row NPZ plus small metadata, never overwrite."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    arrays_path, metadata_path = directory / "arrays.npz", directory / "metadata.json"
    if arrays_path.exists() or metadata_path.exists():
        raise FileExistsError(
            "Prepared output exists; use a new version/output directory."
        )
    arrays = {}
    for name, split in country.splits.items():
        for key in (
            "features",
            "raw_features",
            "targets",
            "raw_targets",
            "times",
            "origins",
        ):
            values = getattr(split, key)
            arrays[f"{name}_{key}"] = (
                values.view(np.int64) if key == "times" else values
            )
    # Full-row arrays retain missing/nonfinite entries; no overlapping windows are materialized.
    np.savez_compressed(arrays_path, **arrays)
    metadata = dict(country.metadata)
    metadata["arrays_file"] = arrays_path.name
    metadata["arrays_sha256"] = sha256_file(arrays_path)
    metadata["array_hashes"] = {
        name: hash_array(values) for name, values in arrays.items()
    }
    metadata["metadata_content_hash"] = hash_json(metadata)
    write_json(metadata_path, metadata)
    return metadata_path


def load_prepared(
    metadata_path: str | Path, expected_identity: dict | None = None
) -> PreparedCountry:
    """Read only a correctly bound Solar cache; refuse another task or stale identity."""
    path = Path(metadata_path)
    metadata = read_json(path)
    claimed_hash = metadata.pop("metadata_content_hash", None)
    if claimed_hash != hash_json(metadata):
        raise ValueError("Prepared metadata content hash mismatch.")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported prepared data schema.")
    identity = metadata["identity"]
    required = {
        "task_id": TASK_ID,
        "target_name": TARGET_NAME,
        "feature_order": list(FEATURE_ORDER),
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    for key, expected in required.items():
        if identity.get(key) != expected:
            raise ValueError(f"Prepared cache identity mismatch: {key}")
    for key, expected in (expected_identity or {}).items():
        if identity.get(key) != expected:
            raise ValueError(f"Prepared cache requested identity mismatch: {key}")
    configuration = metadata["scaler_configuration"]
    if identity["scaler_hash"] != hash_json(configuration):
        raise ValueError("Scaler configuration hash mismatch.")
    _check_target_scaler(configuration["target_scaler"])
    if configuration["target_scaler"] != configuration["feature_scalers"][TARGET_NAME]:
        raise ValueError("Solar input and target scaler statistics differ.")
    array_name = metadata["arrays_file"]
    if Path(array_name).name != array_name:
        raise ValueError(
            "Prepared array filename must be local to its version directory."
        )
    array_path = path.parent / array_name
    if sha256_file(array_path) != metadata["arrays_sha256"]:
        raise ValueError("Prepared array file hash mismatch.")
    splits = {}
    with np.load(array_path, allow_pickle=False) as archive:
        if set(archive.files) != set(metadata["array_hashes"]):
            raise ValueError("Prepared array inventory mismatch.")
        for name in ("Train", "Validation", "Test"):
            arrays = {}
            for key in (
                "features",
                "raw_features",
                "targets",
                "raw_targets",
                "times",
                "origins",
            ):
                array_key = f"{name}_{key}"
                values = archive[array_key].copy()
                if hash_array(values) != metadata["array_hashes"][array_key]:
                    raise ValueError(f"Prepared array content mismatch: {array_key}")
                arrays[key] = (
                    values.view("datetime64[ns]") if key == "times" else values
                )
            start, end = identity["split_bounds"][name]
            n = end - start
            for key, shape in (
                ("features", (n, len(FEATURE_ORDER))),
                ("raw_features", (n, len(FEATURE_ORDER))),
                ("targets", (n,)),
                ("raw_targets", (n,)),
                ("times", (n,)),
            ):
                if arrays[key].shape != shape:
                    raise ValueError(f"Prepared {name} {key} shape mismatch.")
            if arrays["origins"].ndim != 1 or arrays["origins"].dtype != np.dtype(
                "int64"
            ):
                raise ValueError("Prepared origins must be an int64 vector.")
            if len(arrays["origins"]) and (
                arrays["origins"].min() < L_MAX or arrays["origins"].max() + HORIZON > n
            ):
                raise ValueError(
                    "Prepared origins exceed split-local history/label bounds."
                )
            inspect_times(arrays["times"])
            if not np.array_equal(
                arrays["raw_targets"], arrays["raw_features"][:, 3], equal_nan=True
            ):
                raise ValueError(
                    "Prepared raw targets do not match the explicit Solar channel."
                )
            expected_features = encode_features(
                arrays["raw_features"], configuration["feature_scalers"]
            )
            expected_targets = standardize_target(
                arrays["raw_targets"], configuration["target_scaler"]
            ).astype(np.float32)
            if not np.array_equal(
                arrays["features"], expected_features, equal_nan=True
            ):
                raise ValueError(
                    "Prepared encoded features do not match the stored Train scalers."
                )
            if not np.array_equal(arrays["targets"], expected_targets, equal_nan=True):
                raise ValueError(
                    "Prepared targets do not match the Solar target scaler."
                )
            origin_hash = _origin_hash(arrays["origins"], arrays["times"], (start, end))
            if origin_hash != identity["origin_hash"][name]:
                raise ValueError("Prepared origin hash mismatch.")
            verified_origins, statistics = valid_origins(
                arrays["raw_features"][:, 1:4], arrays["raw_targets"]
            )
            if not np.array_equal(verified_origins, arrays["origins"]):
                raise ValueError("Stored origins violate complete-window selection.")
            splits[name] = PreparedSplit(
                name=name,
                **arrays,
                statistics=statistics,
                origin_hash=origin_hash,
                global_start=start,
                global_end=end,
            )
    return PreparedCountry(
        identity["country"],
        identity["source_csv_sha256"],
        configuration["feature_scalers"],
        dict(configuration["target_scaler"]),
        splits,
        metadata,
    )
