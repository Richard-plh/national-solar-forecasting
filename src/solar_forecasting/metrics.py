"""Point and dynamics metrics on raw, unclipped Solar predictions."""

from __future__ import annotations

from collections import defaultdict
import numpy as np

from .config import HORIZON, MODEL_NAMES, SEEDS, TARGET_NAME
from .utils import hash_json

POINT_METRICS = ("standardized_MAE", "MAE_MW", "RMSE_MW", "ActiveMAE_MW")
DYNAMIC_METRICS = ("High_RampMAE_MW", "RMRR", "High_Ramp_DirectionAccuracy")


def basic_metrics(actual, prediction):
    actual, prediction = np.asarray(actual, dtype=np.float64), np.asarray(
        prediction, dtype=np.float64
    )
    if actual.shape != prediction.shape or actual.size == 0:
        raise ValueError("Metric arrays must have matching, nonempty shapes")
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require finite arrays")
    mse = float(np.mean((prediction - actual) ** 2))
    return {
        "MAE": float(np.abs(prediction - actual).mean()),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
    }


def inverse_solar(values, scaler):
    if (
        scaler.get("name") != TARGET_NAME
        or not np.isfinite([scaler["mean"], scaler["std"]]).all()
        or scaler["std"] <= 0
    ):
        raise ValueError("A finite, positive-scale Solar target scaler is required")
    return np.asarray(values, dtype=np.float64) * scaler["std"] + scaler["mean"]


def changes_mw(values, anchor):
    values, anchor = np.asarray(values, dtype=np.float64), np.asarray(
        anchor, dtype=np.float64
    )
    if values.ndim != 2 or values.shape[1] != HORIZON or anchor.shape != (len(values),):
        raise ValueError(
            "Expected [origins, 24] MW and one historical Solar anchor per origin"
        )
    if not np.isfinite(values).all() or not np.isfinite(anchor).all():
        raise ValueError("Nonfinite trajectory")
    return np.concatenate(
        [values[:, :1] - anchor[:, None], np.diff(values, axis=1)], axis=1
    )


def active_metrics(actual, prediction, epsilon):
    basic_metrics(actual, prediction)
    if not np.isfinite(epsilon):
        raise ValueError("Nonfinite Active threshold")
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    mask = actual > epsilon
    if not mask.any():
        raise ValueError("Active mask is empty")
    return {
        "ActiveMAE_MW": float(np.abs(prediction - actual)[mask].mean()),
        "active_points": int(mask.sum()),
        "total_points": int(mask.size),
        "active_fraction": float(mask.mean()),
        "epsilon_MW": float(epsilon),
    }


def negative_diagnostics(prediction):
    values = np.asarray(prediction, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Expected finite MW predictions")
    mask = values < 0
    return {
        "negative_prediction_MW_count": int(mask.sum()),
        "negative_prediction_MW_fraction": float(mask.mean()),
        "minimum_prediction_MW": float(values.min()),
        "negative_fraction_by_horizon": mask.mean(axis=0).tolist(),
        "postprocessing": "none",
    }


def dynamic_metrics(actual, prediction, anchor, q80):
    basic_metrics(actual, prediction)
    if not np.isfinite(q80) or q80 <= 0:
        raise ValueError("Require positive finite Q80 in MW")
    true, pred = changes_mw(actual, anchor), changes_mw(prediction, anchor)
    mask = np.abs(true) >= q80
    denominator = float(np.abs(true)[mask].sum())
    if not mask.any() or denominator <= 0:
        raise ValueError("High-Ramp mask is empty or its denominator is nonpositive")
    return {
        "High_RampMAE_MW": float(np.abs(pred - true)[mask].mean()),
        "RMRR": float(np.abs(pred)[mask].sum() / denominator),
        "High_Ramp_DirectionAccuracy": float(
            (np.sign(pred[mask]) == np.sign(true[mask])).mean()
        ),
        "high_points": int(mask.sum()),
        "high_fraction": float(mask.mean()),
        "true_magnitude_denominator_MW": denominator,
        "q80_MW": float(q80),
    }


def evaluate_arrays(
    actual_z,
    prediction_z,
    scaler,
    raw_actual,
    anchor,
    epsilon,
    q80,
    *,
    physical_reference=None,
):
    standardized = basic_metrics(actual_z, prediction_z)
    actual = np.asarray(raw_actual, dtype=np.float64)
    if actual.shape != np.asarray(actual_z).shape or not np.allclose(
        inverse_solar(actual_z, scaler), actual, atol=1e-3, rtol=1e-6
    ):
        raise ValueError("Raw Solar labels disagree with the target scaler")
    prediction = inverse_solar(prediction_z, scaler)
    if physical_reference is not None:
        if not np.allclose(prediction, physical_reference, atol=1e-9, rtol=1e-12):
            raise ValueError("Physical reference is not inverse-consistent")
        prediction = np.asarray(physical_reference, dtype=np.float64)
    physical = basic_metrics(actual, prediction)
    return {
        "standardized_MAE": standardized["MAE"],
        "MAE_MW": physical["MAE"],
        "RMSE_MW": physical["RMSE"],
        **active_metrics(actual, prediction, epsilon),
        **dynamic_metrics(actual, prediction, anchor, q80),
        **negative_diagnostics(prediction),
        "per_horizon_MAE_MW": np.abs(prediction - actual).mean(axis=0).tolist(),
    }


def validate_seed_group(rows):
    if len(rows) != 5 or {r["seed"] for r in rows} != set(SEEDS):
        raise ValueError("Exactly five unique configured seeds are required")
    if len({(r["country"], r["model"], r["split"]) for r in rows}) != 1:
        raise ValueError("Mixed country, model or split")
    if any(
        not np.isfinite(r[k]) or r[k] < 0
        for r in rows
        for k in POINT_METRICS + DYNAMIC_METRICS
    ):
        raise ValueError("Nonfinite or negative metric")
    return sorted(rows, key=lambda r: r["seed"])


def aggregate_five_seed(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["model"] not in MODEL_NAMES:
            raise ValueError("Deterministic persistence is not a five-seed model")
        groups[(row["country"], row["model"], row["split"])].append(row)
    result = []
    for (country, model, split), group in groups.items():
        group = validate_seed_group(group)
        item = {"country": country, "model": model, "split": split, "n": 5, "ddof": 1}
        for key in POINT_METRICS:
            values = np.array([r[key] for r in group])
            item[key + "_mean"] = float(values.mean())
            item[key + "_sample_SD"] = float(values.std(ddof=1))
        for key in DYNAMIC_METRICS:
            item[key + "_median"] = float(np.median([r[key] for r in group]))
        result.append(item)
    return result


def paired_gain(reference, candidate, metric="standardized_MAE"):
    reference, candidate = validate_seed_group(reference), validate_seed_group(
        candidate
    )
    if [(r["country"], r["seed"], r["split"]) for r in reference] != [
        (r["country"], r["seed"], r["split"]) for r in candidate
    ]:
        raise ValueError("Paired groups must match country, seed and split")
    denominators = np.array([r[metric] for r in reference])
    if (denominators <= 0).any():
        raise ValueError("Relative improvement requires positive denominators")
    gains = (
        100 * (denominators - np.array([r[metric] for r in candidate])) / denominators
    )
    return {
        "seed_gains": dict(zip([str(r["seed"]) for r in reference], gains.tolist())),
        "mean_percent": float(gains.mean()),
        "sample_SD_percent": float(gains.std(ddof=1)),
        "positive_seeds": int((gains > 0).sum()),
    }


def horizon_gain(reference_errors, candidate_errors):
    reference, candidate = np.asarray(reference_errors), np.asarray(candidate_errors)
    if reference.shape != (5, 24) or candidate.shape != (5, 24):
        raise ValueError("Expected five-seed by 24-horizon errors")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Nonfinite horizon errors")
    denominator = reference.mean(axis=0)
    if (denominator <= 0).any():
        raise ValueError("Horizon percentage denominator must be positive")
    return 100 * (denominator - candidate.mean(axis=0)) / denominator


def mean_weights(matrices):
    values = np.asarray(matrices, dtype=np.float64)
    if (
        values.shape != (5, 24, 6)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not np.allclose(values.sum(axis=2), 1, atol=1e-6)
    ):
        raise ValueError("Expected five finite 24-by-6 simplex weight matrices")
    return values.mean(axis=0)


def freeze_comparators(rows, config_hash):
    """Select each single mechanism using Validation RampMAE, never Test."""
    if any(r["split"] != "Validation" for r in rows):
        raise ValueError("Comparator selection requires Validation")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["country"], r["model"])].append(r)
    countries = sorted({r["country"] for r in rows})
    selections = []
    for country in countries:
        left = validate_seed_group(groups[(country, "trajectory-supervision")])
        right = validate_seed_group(groups[(country, "state-supervision")])
        for a, b in zip(left, right):
            best = a if a["High_RampMAE_MW"] <= b["High_RampMAE_MW"] else b
            checkpoint = best.get("checkpoint_sha256")
            if not isinstance(checkpoint, str) or len(checkpoint) != 64:
                raise ValueError("Comparator requires an accepted checkpoint hash")
            selections.append(
                {
                    "country": country,
                    "seed": best["seed"],
                    "model": best["model"],
                    "checkpoint_sha256": checkpoint,
                }
            )
    document = {
        "schema": 1,
        "config_hash": config_hash,
        "selection_split": "Validation",
        "test_used_for_selection": False,
        "selections": selections,
    }
    return {**document, "content_hash": hash_json(document)}


def verify_comparators(document, config_hash, countries):
    body = {k: v for k, v in document.items() if k != "content_hash"}
    if (
        document.get("content_hash") != hash_json(body)
        or document.get("config_hash") != config_hash
    ):
        raise ValueError("Comparator identity changed")
    if (
        document.get("selection_split") != "Validation"
        or document.get("test_used_for_selection") is not False
    ):
        raise ValueError("Comparators must be fixed using Validation")
    keys = [(r["country"], r["seed"]) for r in document["selections"]]
    if len(keys) != len(countries) * 5 or set(keys) != {
        (c, s) for c in countries for s in SEEDS
    }:
        raise ValueError("Missing or duplicate comparator")
    if any(
        r["model"] not in ("trajectory-supervision", "state-supervision")
        for r in document["selections"]
    ):
        raise ValueError("Invalid comparator model")
    return document


def matched_comparison(rows, comparators):
    if len({(r["country"], r["model"], r["seed"], r["split"]) for r in rows}) != len(
        rows
    ):
        raise ValueError("Duplicate metric row")
    lookup = {(r["country"], r["model"], r["seed"]): r for r in rows}
    pairs = []
    for choice in comparators["selections"]:
        key = choice["country"], choice["model"], choice["seed"]
        base = lookup[key]
        joint = lookup[(choice["country"], "joint-supervision", choice["seed"])]
        if base["split"] != joint["split"]:
            raise ValueError("Mixed comparison splits")
        if base["checkpoint_sha256"] != choice["checkpoint_sha256"]:
            raise ValueError("Validation-selected comparator checkpoint changed")
        if (
            not np.isfinite(
                [
                    base["High_RampMAE_MW"],
                    base["MAE_MW"],
                    joint["High_RampMAE_MW"],
                    joint["MAE_MW"],
                ]
            ).all()
            or base["High_RampMAE_MW"] <= 0
            or base["MAE_MW"] <= 0
        ):
            raise ValueError("Invalid matched percentage denominator")
        pairs.append(
            {
                "country": choice["country"],
                "seed": choice["seed"],
                "split": base["split"],
                "comparator": choice["model"],
                "RampMAE_gain_percent": 100
                * (base["High_RampMAE_MW"] - joint["High_RampMAE_MW"])
                / base["High_RampMAE_MW"],
                "PointMAE_change_percent": 100
                * (joint["MAE_MW"] - base["MAE_MW"])
                / base["MAE_MW"],
            }
        )
    summaries = []
    for country in sorted({p["country"] for p in pairs}):
        group = [p for p in pairs if p["country"] == country]
        if len(group) != 5 or {p["seed"] for p in group} != set(SEEDS):
            raise ValueError("Matched comparison requires five seeds")
        summaries.append(
            {
                "country": country,
                "RampMAE_gain_median_percent": float(
                    np.median([p["RampMAE_gain_percent"] for p in group])
                ),
                "PointMAE_change_median_percent": float(
                    np.median([p["PointMAE_change_percent"] for p in group])
                ),
                "positive_seeds": sum(p["RampMAE_gain_percent"] > 0 for p in group),
            }
        )
    return {"seed_pairs": pairs, "summaries": summaries}
