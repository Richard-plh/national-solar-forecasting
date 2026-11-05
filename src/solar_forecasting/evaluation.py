"""Serial best-checkpoint inference and reproducible, array-only rechecking."""

from __future__ import annotations

import csv
import gc
import io
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .atomic_io import atomic_write
from .config import COUNTRIES, FUSION_MODELS, MODEL_NAMES, SEEDS, THRESHOLDS, run_matrix
from .metrics import (
    aggregate_five_seed,
    evaluate_arrays,
    freeze_comparators,
    horizon_gain,
    inverse_solar,
    matched_comparison,
    paired_gain,
    verify_comparators,
)
from .utils import hash_json, read_json, sha256_file, write_json
from .workflow import (
    checkpoint_read,
    code_hash,
    inspect_task,
    load_country,
    prepared_path,
)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty result table")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, lambda f: f.write(stream.getvalue().encode("utf-8")))


def seasonal_persistence(split):
    return np.stack([split.raw_targets[i - 24 : i] for i in split.origins])


def accepted_plan(workspace, config, config_dir, split, group, comparators=None):
    if split not in ("Validation", "Test") or group not in (
        "all",
        "fusion",
        "supervision",
    ):
        raise ValueError("Invalid evaluation split/group")
    if split == "Test" and group != "all":
        raise PermissionError("The Test plan must include all 90 models")
    rows = [r for r in run_matrix() if group == "all" or r["group"] == group]
    if group == "supervision":
        rows += [r for r in run_matrix() if r["model"] == "horizon-fusion"]
    accepted = []
    for row in rows:
        state = inspect_task(
            workspace, config, row["country"], row["model"], row["seed"]
        )
        if state["status"] != "COMPLETE":
            raise PermissionError("Every requested model must pass full run acceptance")
        accepted.append(
            {
                **row,
                "run_path": Path(state["directory"]).relative_to(workspace).as_posix(),
                "identity": state["identity"],
                "checkpoint_sha256": state["checkpoint_sha256"],
            }
        )
    if split == "Test":
        verify_comparators(comparators, hash_json(config), COUNTRIES)
        hashes = {
            (r["country"], r["model"], r["seed"]): r["checkpoint_sha256"]
            for r in accepted
        }
        for choice in comparators["selections"]:
            if (
                hashes[(choice["country"], choice["model"], choice["seed"])]
                != choice["checkpoint_sha256"]
            ):
                raise ValueError("Validation-selected comparator checkpoint changed")
    # Metadata-only plan creation: no Test labels or model forward here.
    data = {c: read_json(prepared_path(workspace, c))["identity"] for c in COUNTRIES}
    for row in accepted:
        if row["identity"]["data_identity"] != data[row["country"]]:
            raise ValueError("Accepted checkpoint and prepared data identities differ")
    body = {
        "schema": "solar-evaluation-plan-v1",
        "split": split,
        "group": group,
        "config_hash": hash_json(config),
        "code_hash": code_hash(),
        "models": accepted,
        "data_identity": data,
        "thresholds": THRESHOLDS,
        "comparators": comparators,
        "fixed_dates": config["curve_dates"][split],
        "curve_seed": 42,
        "prediction_postprocessing": "none",
        "persistence": "raw Solar[origin-24:origin]",
        "point_aggregation": "mean and sample SD, ddof=1",
        "dynamics_aggregation": "median",
    }
    return {**body, "plan_hash": hash_json(body)}


@torch.inference_mode()
def predict(model, dataset, *, device, batch_size=64):
    model.to(device).eval()
    values = []
    for x, _ in DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
    ):
        prediction = model(x.to(device))
        if prediction.shape != (len(x), 24) or not torch.isfinite(prediction).all():
            raise RuntimeError("Invalid prediction shape or nonfinite output")
        values.append(prediction.cpu().numpy())
    if not values:
        raise ValueError("No valid prediction windows")
    return np.concatenate(values)


def evaluation_tables(rows, persistence, comparators):
    summary = aggregate_five_seed(rows)
    fusion = []
    ablation = []
    horizon = []
    baseline_comparison = []
    for country in COUNTRIES:
        groups = {
            m: [r for r in rows if r["country"] == country and r["model"] == m]
            for m in MODEL_NAMES
        }
        reference = groups["horizon-fusion"]
        if groups["global-fusion"] and reference:
            fusion.append(
                {"country": country, **paired_gain(groups["global-fusion"], reference)}
            )
            gains = horizon_gain(
                [
                    r["per_horizon_MAE_MW"]
                    for r in sorted(groups["global-fusion"], key=lambda r: r["seed"])
                ],
                [
                    r["per_horizon_MAE_MW"]
                    for r in sorted(reference, key=lambda r: r["seed"])
                ],
            )
            horizon += [
                {"country": country, "horizon": h + 1, "fusion_gain_percent": float(v)}
                for h, v in enumerate(gains)
            ]
        for model in (
            "trajectory-supervision",
            "state-supervision",
            "joint-supervision",
        ):
            if groups[model] and reference:
                ablation.append(
                    {
                        "country": country,
                        "model": model,
                        **paired_gain(reference, groups[model]),
                    }
                )
        base = next(r for r in persistence if r["country"] == country)
        for row in [r for r in summary if r["country"] == country]:
            if base["MAE_MW"] <= 0 or base["ActiveMAE_MW"] <= 0:
                raise ValueError("Persistence percentage denominator must be positive")
            baseline_comparison.append(
                {
                    "country": country,
                    "model": row["model"],
                    "Overall_improvement_percent": 100
                    * (base["MAE_MW"] - row["MAE_MW_mean"])
                    / base["MAE_MW"],
                    "Active_improvement_percent": 100
                    * (base["ActiveMAE_MW"] - row["ActiveMAE_MW_mean"])
                    / base["ActiveMAE_MW"],
                }
            )
    matched = matched_comparison(rows, comparators) if comparators else None
    return {
        "five_seed": summary,
        "fusion_gain": fusion,
        "supervision_gain": ablation,
        "horizon_gain": horizon,
        "persistence_comparison": baseline_comparison,
        "matched": matched,
    }


def evaluate(
    workspace,
    config,
    config_dir,
    *,
    split="Validation",
    group="all",
    device="cuda",
    allow_test=False,
    comparator_file=None,
):
    if split == "Test" and not allow_test:
        raise PermissionError(
            "Test evaluation requires --allow-test and Validation-selected comparators"
        )
    if split == "Test" and comparator_file is None:
        raise PermissionError("A Validation comparator file is required before Test")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; choose --device cpu explicitly if intended"
        )
    workspace = Path(workspace).resolve()
    comparators = read_json(comparator_file) if comparator_file else None
    plan = accepted_plan(workspace, config, config_dir, split, group, comparators)
    destination = (
        workspace
        / "evaluations"
        / ("Test" if split == "Test" else f"Validation_{group}")
    )
    # This is the one evaluation directory for the selected Test batch. No resume/overwrite.
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "plan.json", plan)
    state = {"status": "RUNNING", "plan_hash": plan["plan_hash"], "completed_models": 0}
    write_json(destination / "manifest.json", state)
    rows, persistence = [], []
    try:
        for country in COUNTRIES:
            prepared = load_country(workspace, country, config_dir)
            dataset = prepared.dataset(split, allow_test=allow_test)
            data = prepared.splits[split]
            indices = data.origins[:, None] + np.arange(24)
            actual_z, actual = data.targets[indices], data.raw_targets[indices]
            anchor = data.raw_targets[data.origins - 1]
            threshold = THRESHOLDS[country]
            seasonal = seasonal_persistence(data)
            folder = destination / country
            folder.mkdir()
            observation = folder / "observations.npz"
            atomic_write(
                observation,
                lambda f: np.savez_compressed(
                    f,
                    actual_standardized=actual_z,
                    actual_MW=actual,
                    anchor_MW=anchor,
                    origin_indices=data.origins,
                    first_target_times_ns=data.times[data.origins].view(np.int64),
                    persistence_MW=seasonal,
                ),
            )
            scaler = prepared.target_scaler
            write_json(folder / "scaler.json", scaler)
            curve = {"country": country, "split": split, "seed": 42, "cases": {}}
            for date in config["curve_dates"][split]:
                found = np.flatnonzero(
                    data.times[data.origins] == np.datetime64(date, "ns")
                )
                if len(found) != 1:
                    raise ValueError(
                        "Fixed case unavailable; no automatic replacement: " + date
                    )
                index = int(found[0])
                curve["cases"][date] = {
                    "index": index,
                    "actual_MW": actual[index].tolist(),
                    "predictions_MW": {
                        "seasonal-persistence": seasonal[index].tolist()
                    },
                }
            for item in [r for r in plan["models"] if r["country"] == country]:
                checkpoint = workspace / item["run_path"] / "best.pth"
                if sha256_file(checkpoint) != item["checkpoint_sha256"]:
                    raise ValueError(
                        "Checkpoint changed after evaluation plan creation"
                    )
                model, _ = checkpoint_read(checkpoint, item["identity"])
                prediction_z = predict(
                    model,
                    dataset,
                    device=device,
                    batch_size=config["training"]["batch_size"],
                )
                metrics = evaluate_arrays(
                    actual_z,
                    prediction_z,
                    scaler,
                    actual,
                    anchor,
                    threshold["active_MW"],
                    threshold["q80_MW"],
                )
                row = {
                    "country": country,
                    "model": item["model"],
                    "seed": item["seed"],
                    "split": split,
                    "checkpoint_sha256": item["checkpoint_sha256"],
                    "origin_hash": data.origin_hash,
                    **metrics,
                }
                name = f"{item['model']}_seed_{item['seed']}"
                atomic_write(
                    folder / (name + ".npz"),
                    lambda f: np.savez_compressed(
                        f, prediction_standardized=prediction_z
                    ),
                )
                write_json(folder / (name + ".json"), row)
                base = model.base if hasattr(model, "base") else model
                write_json(
                    folder / (name + "_weights.json"),
                    base.scale_weights().detach().cpu().tolist(),
                )
                rows.append(row)
                if item["seed"] == 42:
                    predicted = inverse_solar(prediction_z, scaler)
                    for case in curve["cases"].values():
                        case["predictions_MW"][item["model"]] = predicted[
                            case["index"]
                        ].tolist()
                state["completed_models"] += 1
                state["current"] = {k: item[k] for k in ("country", "model", "seed")}
                write_json(destination / "manifest.json", state)
                print(
                    f"{split}: {state['completed_models']}/{len(plan['models'])} {country} {name}",
                    flush=True,
                )
                del model, base, prediction_z
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            persistence_z = (seasonal - scaler["mean"]) / scaler["std"]
            persistence.append(
                {
                    "country": country,
                    "model": "seasonal-persistence",
                    "seed": None,
                    "split": split,
                    **evaluate_arrays(
                        actual_z,
                        persistence_z,
                        scaler,
                        actual,
                        anchor,
                        threshold["active_MW"],
                        threshold["q80_MW"],
                        physical_reference=seasonal,
                    ),
                }
            )
            write_json(folder / "fixed_cases.json", curve)
        if split == "Validation" and group != "fusion":
            comparators = freeze_comparators(rows, hash_json(config))
            verify_comparators(comparators, hash_json(config), COUNTRIES)
            write_json(destination / "validation_comparators.json", comparators)
        summary = {
            "split": split,
            "group": group,
            "plan_hash": plan["plan_hash"],
            "seed_rows": rows,
            "persistence": persistence,
            **evaluation_tables(rows, persistence, comparators),
        }
        write_json(destination / "summary.json", summary)
        scalar_rows = [
            {k: v for k, v in r.items() if not isinstance(v, (list, dict))}
            for r in rows
        ]
        write_csv(destination / "model_seed_metrics.csv", scalar_rows)
        write_csv(destination / "five_seed_summary.csv", summary["five_seed"])
        write_csv(
            destination / "seasonal_persistence.csv",
            [
                {k: v for k, v in r.items() if not isinstance(v, (list, dict))}
                for r in persistence
            ],
        )
        write_csv(
            destination / "per_horizon_MAE_MW.csv",
            [
                {
                    "country": r["country"],
                    "model": r["model"],
                    "seed": r["seed"],
                    "split": split,
                    "horizon": h + 1,
                    "MAE_MW": v,
                }
                for r in rows
                for h, v in enumerate(r["per_horizon_MAE_MW"])
            ],
        )
        write_csv(
            destination / "persistence_comparison.csv",
            summary["persistence_comparison"],
        )
        if summary["matched"]:
            write_csv(
                destination / "matched_seedwise.csv", summary["matched"]["seed_pairs"]
            )
        state["file_hashes"] = {
            p.relative_to(destination).as_posix(): sha256_file(p)
            for p in destination.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        }
        state["status"] = "COMPLETE"
        write_json(destination / "manifest.json", state)
        audit_saved(destination)
        return {
            "status": "COMPLETE",
            "models": len(rows),
            "directory": str(destination),
        }
    except BaseException as exc:
        state.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
        write_json(destination / "manifest.json", state)
        raise


def audit_saved(destination):
    """Read saved arrays and independently recompute; never loads or executes models."""
    destination = Path(destination)
    manifest = read_json(destination / "manifest.json")
    if manifest["status"] != "COMPLETE":
        raise ValueError("Evaluation is incomplete")
    for name, expected in manifest["file_hashes"].items():
        if sha256_file(destination / name) != expected:
            raise ValueError("Saved result hash mismatch: " + name)
    plan = read_json(destination / "plan.json")
    body = {k: v for k, v in plan.items() if k != "plan_hash"}
    if hash_json(body) != plan["plan_hash"]:
        raise ValueError("Evaluation plan changed")
    summary = read_json(destination / "summary.json")
    rows, baseline = [], []
    for country in COUNTRIES:
        folder = destination / country
        with np.load(folder / "observations.npz", allow_pickle=False) as z:
            actual_z, actual, anchor, seasonal = [
                z[k]
                for k in (
                    "actual_standardized",
                    "actual_MW",
                    "anchor_MW",
                    "persistence_MW",
                )
            ]
        scaler = read_json(folder / "scaler.json")
        threshold = plan["thresholds"][country]
        for item in [r for r in plan["models"] if r["country"] == country]:
            name = f"{item['model']}_seed_{item['seed']}"
            row = read_json(folder / (name + ".json"))
            with np.load(folder / (name + ".npz"), allow_pickle=False) as z:
                pred = z["prediction_standardized"]
            recomputed = evaluate_arrays(
                actual_z,
                pred,
                scaler,
                actual,
                anchor,
                threshold["active_MW"],
                threshold["q80_MW"],
            )
            if any(
                (
                    not np.allclose(row[k], value, rtol=1e-12, atol=1e-10)
                    if isinstance(value, (float, list))
                    else row[k] != value
                )
                for k, value in recomputed.items()
            ):
                raise ValueError("Saved metrics disagree with saved arrays: " + name)
            rows.append(row)
        baseline.append(
            {
                "country": country,
                "model": "seasonal-persistence",
                "seed": None,
                "split": plan["split"],
                **evaluate_arrays(
                    actual_z,
                    (seasonal - scaler["mean"]) / scaler["std"],
                    scaler,
                    actual,
                    anchor,
                    threshold["active_MW"],
                    threshold["q80_MW"],
                    physical_reference=seasonal,
                ),
            }
        )
    comparators = plan["comparators"]
    if plan["split"] == "Validation" and plan["group"] != "fusion":
        comparators = read_json(destination / "validation_comparators.json")
    if comparators:
        verify_comparators(comparators, plan["config_hash"], COUNTRIES)
    expected = evaluation_tables(rows, baseline, comparators)
    if (
        rows != summary["seed_rows"]
        or baseline != summary["persistence"]
        or any(summary[k] != v for k, v in expected.items())
    ):
        raise ValueError("Seed metrics, persistence or five-seed aggregates disagree")
    return {
        "status": "PASS",
        "model_results": len(rows),
        "persistence_results": len(baseline),
        "new_model_forward": 0,
    }
