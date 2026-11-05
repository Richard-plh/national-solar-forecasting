"""Portable preparation, run identities and explicit execution boundaries."""

from __future__ import annotations

import math
from pathlib import Path
import platform
import numpy as np
import torch

from .config import COUNTRIES, SOURCE_HASHES, THRESHOLDS, run_matrix
from .data import load_prepared, prepare_country, save_prepared
from .models import build_matched, shared_components
from .utils import hash_json, read_json, sha256_file


def code_hash():
    root = Path(__file__).parent
    return hash_json({p.name: sha256_file(p) for p in sorted(root.glob("*.py"))})


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def prepared_path(workspace, country):
    return Path(workspace) / "data" / "processed" / country / "metadata.json"


def check_prepared(prepared, references):
    reference = references[prepared.country]
    if prepared.source_sha256 != reference["sha256"]:
        raise ValueError("Country CSV identity changed")
    if prepared.metadata["raw_rows"] != reference["rows"]:
        raise ValueError("Hourly row count changed")
    if (
        prepared.target_scaler != reference["target_scaler"]
        or prepared.feature_scalers != reference["feature_scalers"]
    ):
        raise ValueError("Train-only scaler statistics changed")
    if (
        prepared.metadata["scaler_configuration"]["fit_rows"]
        != reference["scaler_fit_hours"]
    ):
        raise ValueError("Scaler-fit population changed")
    for split, count in reference["valid_windows"].items():
        if (
            len(prepared.splits[split].origins) != count
            or prepared.splits[split].origin_hash != reference["origin_hashes"][split]
        ):
            raise ValueError("Complete-window origins changed: " + split)
    # Threshold checks use unique raw Train hours and adjacent Solar pairs only.
    train = prepared.splits["Train"].raw_targets
    finite = train[np.isfinite(train)]
    pairs = np.isfinite(train[:-1]) & np.isfinite(train[1:])
    differences = np.abs(np.diff(train)[pairs])
    q30, q80 = np.quantile(differences, [0.3, 0.8], method="linear")
    active = 0.05 * np.quantile(finite, 0.95, method="linear")
    expected = THRESHOLDS[prepared.country]
    if not np.array_equal(
        [q30, q80, active],
        [expected["q30_MW"], expected["q80_MW"], expected["active_MW"]],
    ):
        raise ValueError("Train threshold statistics changed")
    return prepared


def prepare_data(raw_dir, workspace, config_dir):
    references = read_json(Path(config_dir) / "data_reference.json")
    summary = {}
    for country, filename in COUNTRIES.items():
        path = prepared_path(workspace, country)
        if path.exists():
            item = load_prepared(path)
            # A matching processed cache does not replace verification of the supplied CSV.
            if sha256_file(Path(raw_dir) / filename) != SOURCE_HASHES[country]:
                raise ValueError("CSV hash mismatch: " + filename)
        else:
            item = prepare_country(
                Path(raw_dir) / filename, country, SOURCE_HASHES[country]
            )
            check_prepared(item, references)
            save_prepared(item, path.parent)
        check_prepared(item, references)
        summary[country] = {s: len(v.origins) for s, v in item.splits.items()}
    return summary


def load_country(workspace, country, config_dir):
    return check_prepared(
        load_prepared(prepared_path(workspace, country)),
        read_json(Path(config_dir) / "data_reference.json"),
    )


def validate_initialization(model, country, seed, config_dir):
    references = read_json(Path(config_dir) / "initialization_checksums.json")
    reference = [r for r in references if (r["country"], r["seed"]) == (country, seed)]
    if len(reference) != 1:
        raise ValueError("Missing initialization reference")
    for key, value in shared_components(model).items():
        if value != reference[0][key]:
            raise ValueError("Shared random initialization differs: " + key)


def verify_all_initializations(config_dir):
    from .config import MODEL_NAMES, SEEDS

    checked = 0
    for country in COUNTRIES:
        for seed in SEEDS:
            for name in MODEL_NAMES:
                model = build_matched(name, country, seed)
                validate_initialization(model, country, seed, config_dir)
                checked += 1
    return {"verified_initializations": checked, "country_seed_groups": 15}


def task_row(country, model, seed):
    rows = [
        r
        for r in run_matrix()
        if (r["country"], r["model"], r["seed"]) == (country, model, seed)
    ]
    if len(rows) != 1:
        raise ValueError("Task is outside the fixed run matrix")
    return rows[0]


def run_identity(config, prepared, model, seed, attempt):
    task_row(prepared.country, model, seed)
    return {
        "schema": "solar-run-v1",
        "country": prepared.country,
        "model": model,
        "seed": seed,
        "attempt": attempt,
        "config_hash": hash_json(config),
        "data_identity": prepared.identity,
        "sample_counts": {
            name: len(prepared.splits[name].origins) for name in ("Train", "Validation")
        },
        "code_hash": code_hash(),
        "target": config["target"],
        "precision": "FP32",
    }


def run_directory(workspace, country, model, seed, attempt=1):
    task = task_row(country, model, seed)
    if not isinstance(attempt, int) or attempt < 1:
        raise ValueError("Attempt must be a positive integer")
    if attempt == 1:
        return Path(workspace) / task["run_path"]
    return (
        Path(workspace)
        / "attempts"
        / country
        / model
        / f"seed_{seed}"
        / f"attempt_{attempt:02d}"
    )


def checkpoint_read(path, identity):
    from .atomic_io import validate_checkpoint_archive
    from .models import component_hash

    validate_checkpoint_archive(Path(path))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("format") != "solar-checkpoint-v1"
        or payload.get("identity") != identity
        or payload.get("identity_hash") != hash_json(identity)
    ):
        raise ValueError("Checkpoint identity mismatch")
    # Reconstruct only the configured architecture, never run a forward pass.
    model = build_matched(identity["model"], identity["country"], identity["seed"])
    model.load_state_dict(payload["model_state"], strict=True)
    if component_hash(model) != payload["model_state_hash"]:
        raise ValueError("Checkpoint state hash mismatch")
    model.eval()
    return model, payload


def inspect_run(directory, config):
    directory = Path(directory)
    if not directory.exists():
        return {"status": "NOT_STARTED", "directory": str(directory)}
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return {"status": "INCOMPLETE", "directory": str(directory)}
    manifest = read_json(manifest_path)
    if manifest.get("status") != "COMPLETE":
        return {
            "status": manifest.get("status", "INCOMPLETE"),
            "directory": str(directory),
        }
    identity = manifest["identity"]
    if (
        identity["config_hash"] != hash_json(config)
        or identity["code_hash"] != code_hash()
    ):
        raise ValueError("Completed run configuration/code changed")
    task_row(identity["country"], identity["model"], identity["seed"])
    for name, expected in manifest["artifact_hashes"].items():
        if Path(name).name != name or sha256_file(directory / name) != expected:
            raise ValueError("Run artifact hash mismatch: " + name)
    required = {"config.json", "initialization.json", "history.json", "best.pth"}
    if set(manifest["artifact_hashes"]) != required:
        raise ValueError("Run artifact inventory is incomplete")
    if read_json(directory / "config.json") != config:
        raise ValueError("Saved training configuration differs")
    fresh = build_matched(identity["model"], identity["country"], identity["seed"])
    if read_json(directory / "initialization.json") != fresh.initialization_metadata:
        raise ValueError("Initial random component identity differs")
    _, checkpoint = checkpoint_read(directory / "best.pth", identity)
    history = read_json(directory / "history.json")
    settings = config["training"]
    if not history or [r["epoch"] for r in history] != list(range(1, len(history) + 1)):
        raise ValueError("Incomplete epoch history")
    counts = manifest["sample_counts"]
    if counts != identity["sample_counts"]:
        raise ValueError("Run sample counts differ from the prepared identity")
    for row in history:
        if (
            any(
                not math.isfinite(row[k])
                for k in ("main_mse", "total", "validation_MAE", "validation_MSE")
            )
            or not row["finite_gradients"]
        ):
            raise ValueError("Invalid epoch values")
        for split, prefix in [("Train", "train"), ("Validation", "validation")]:
            if (
                row[prefix + "_batches"]
                != math.ceil(counts[split] / settings["batch_size"])
                or row[prefix + "_points"] != counts[split] * 24
            ):
                raise ValueError("Incomplete epoch batches/points")
    best = min(history, key=lambda r: r["validation_MAE"])
    if (
        checkpoint["epoch"] != best["epoch"]
        or checkpoint["validation_MAE"] != best["validation_MAE"]
    ):
        raise ValueError("Checkpoint is not the best Validation epoch")
    if (
        len(history) != settings["max_epochs"]
        and len(history) - best["epoch"] != settings["patience"]
    ):
        raise ValueError("Invalid stopping condition")
    if not manifest["best_restored"] or manifest["best_epoch"] != best["epoch"]:
        raise ValueError("Best checkpoint restoration not recorded")
    if (
        manifest["restored_model_hash"] != checkpoint["model_state_hash"]
        or not manifest["parameters_updated"]
    ):
        raise ValueError("Best restoration or parameter update proof is missing")
    return {
        "status": "COMPLETE",
        "directory": str(directory),
        "identity": identity,
        "checkpoint_sha256": sha256_file(directory / "best.pth"),
        "best_epoch": best["epoch"],
        "validation_MAE": best["validation_MAE"],
    }


def inspect_task(workspace, config, country, model, seed):
    directories = [run_directory(workspace, country, model, seed)]
    retry_root = Path(workspace) / "attempts" / country / model / f"seed_{seed}"
    if retry_root.exists():
        directories += sorted(p for p in retry_root.iterdir() if p.is_dir())
    inspections = [inspect_run(p, config) for p in directories]
    for result in inspections:
        if result["status"] == "COMPLETE":
            identity = result["identity"]
            if (identity["country"], identity["model"], identity["seed"]) != (
                country,
                model,
                seed,
            ):
                raise ValueError("Result directory belongs to a different task")
    complete = [v for v in inspections if v["status"] == "COMPLETE"]
    if len(complete) > 1:
        raise ValueError(
            "Multiple successful attempts require explicit resolution; no score-based selection"
        )
    chosen = complete[0] if complete else inspections[-1]
    return {
        "country": country,
        "model": model,
        "seed": seed,
        **chosen,
        "attempts": inspections,
    }
