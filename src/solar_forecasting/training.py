"""FP32 Adam training, deterministic epoch order and best-checkpoint acceptance."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .atomic_io import atomic_write, validate_checkpoint_archive
from .config import SUPERVISION_MODELS, THRESHOLDS
from .models import (
    DynamicsForecaster,
    build_matched,
    component_hash,
    loss_components,
    order_seed,
    set_deterministic,
    state_labels,
)
from .utils import hash_array, hash_json, sha256_file, write_json
from .workflow import (
    checkpoint_read,
    environment,
    inspect_run,
    inspect_task,
    load_country,
    run_directory,
    run_identity,
    validate_initialization,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def training_lock(workspace):
    """One training process per workspace; stale locks are never silently removed."""
    path = Path(workspace) / ".training.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink()


@dataclass
class EarlyStopping:
    patience: int = 12
    best_mae: float = float("inf")
    best_epoch: int = 0
    bad_epochs: int = 0

    @property
    def should_stop(self):
        return self.bad_epochs >= self.patience

    def observe(self, epoch, mae):
        if epoch < 1 or not math.isfinite(mae) or mae < 0:
            raise ValueError("Invalid Validation metric")
        if mae < self.best_mae:
            self.best_mae, self.best_epoch, self.bad_epochs = mae, epoch, 0
            return True
        self.bad_epochs += 1
        return False


def train_order_indices(sample_count, derived_seed, epoch):
    generator = torch.Generator().manual_seed(derived_seed + epoch * 1_000_003)
    return (
        torch.randperm(sample_count, generator=generator)
        .numpy()
        .astype(np.int64, copy=False)
    )


class FixedOrderSampler(Sampler):
    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=np.int64)

    def __iter__(self):
        return iter(self.indices.tolist())

    def __len__(self):
        return len(self.indices)


class SupervisedWindows:
    def __init__(self, prepared):
        self.windows = prepared.dataset("Train")
        self.data = prepared.splits["Train"]
        self.thresholds = THRESHOLDS[prepared.country]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        x, y = self.windows[index]
        origin = int(self.data.origins[index])
        raw = torch.as_tensor(
            self.data.raw_targets[origin : origin + 24], dtype=torch.float64
        )
        anchor = torch.tensor(self.data.raw_targets[origin - 1], dtype=torch.float64)
        delta = torch.cat([raw[:1] - anchor, raw[1:] - raw[:-1]])
        labels = state_labels(
            delta, self.thresholds["q30_MW"], self.thresholds["q80_MW"]
        )
        return x, y, labels


@torch.no_grad()
def validation_metrics(model, dataset, device, batch_size):
    if dataset.split != "Validation":
        raise PermissionError("Checkpoint selection accepts Validation only")
    model.eval()
    absolute, squared, points, batches = 0.0, 0.0, 0, 0
    for x, y in DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
    ):
        prediction = model(x.to(device))
        target = y.to(device)
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise RuntimeError("Nonfinite Validation values")
        absolute += float((prediction - target).abs().sum())
        squared += float(((prediction - target) ** 2).sum())
        points += target.numel()
        batches += 1
    if points == 0:
        raise ValueError("Empty Validation dataset")
    return {
        "MAE": absolute / points,
        "MSE": squared / points,
        "points": points,
        "batches": batches,
    }


def train(
    workspace,
    config,
    config_dir,
    country,
    model_name,
    seed,
    *,
    execute_training=False,
    device="cuda",
    attempt=1,
):
    if not execute_training:
        raise PermissionError("Training requires --execute-training")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; choose --device cpu explicitly if intended"
        )
    workspace = Path(workspace)
    state = inspect_task(workspace, config, country, model_name, seed)
    if state["status"] == "COMPLETE":
        raise FileExistsError(
            "Accepted task already exists; no overwrite or score-based retraining"
        )
    directory = run_directory(workspace, country, model_name, seed, attempt)
    if directory.exists():
        raise FileExistsError(
            "Attempt directory exists; no overwrite or checkpoint resume"
        )
    if attempt > 1:
        previous = run_directory(workspace, country, model_name, seed, attempt - 1)
        if inspect_run(previous, config)["status"] not in ("FAILED", "INTERRUPTED"):
            raise ValueError(
                "A new attempt requires an explicitly recorded failed/interrupted predecessor"
            )
    with training_lock(workspace):
        prepared = load_country(workspace, country, config_dir)
        model = build_matched(model_name, country, seed)
        validate_initialization(model, country, seed, config_dir)
        identity = run_identity(config, prepared, model_name, seed, attempt)
        manifest = {
            "status": "RUNNING",
            "identity": identity,
            "pid": os.getpid(),
            "started_utc": utc_now(),
            "environment": environment(),
        }
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "manifest.json", manifest)
        try:
            result = _train_loop(directory, model, prepared, config, identity, device)
            manifest.update(result)
            manifest.update(status="COMPLETE", finished_utc=utc_now())
            manifest["artifact_hashes"] = {
                name: sha256_file(directory / name)
                for name in (
                    "config.json",
                    "initialization.json",
                    "history.json",
                    "best.pth",
                )
            }
            write_json(directory / "manifest.json", manifest)
            accepted = inspect_run(directory, config)
            if accepted["status"] != "COMPLETE":
                raise RuntimeError("Run acceptance failed")
            return accepted
        except BaseException as exc:
            manifest.update(
                status=(
                    "INTERRUPTED"
                    if isinstance(exc, (KeyboardInterrupt, SystemExit))
                    else "FAILED"
                ),
                finished_utc=utc_now(),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            write_json(directory / "manifest.json", manifest)
            raise


def _train_loop(directory, model, prepared, config, identity, device):
    settings = config["training"]
    supervised = identity["model"] in SUPERVISION_MODELS
    train_data = (
        SupervisedWindows(prepared) if supervised else prepared.dataset("Train")
    )
    validation = prepared.dataset("Validation")
    if not len(train_data) or not len(validation):
        raise ValueError("Empty Train/Validation windows")
    write_json(directory / "config.json", config)
    write_json(directory / "initialization.json", model.initialization_metadata)
    derived = order_seed(identity["country"], identity["seed"])
    set_deterministic(derived)
    model.to(device)
    initial_model_hash = component_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    stopping = EarlyStopping(settings["patience"])
    history = []
    start = time.perf_counter()
    for epoch in range(1, settings["max_epochs"] + 1):
        if stopping.should_stop:
            break
        indices = train_order_indices(len(train_data), derived, epoch)
        loader = DataLoader(
            train_data,
            batch_size=settings["batch_size"],
            sampler=FixedOrderSampler(indices),
            num_workers=0,
            drop_last=False,
            pin_memory=False,
        )
        model.train()
        sums, points, batches = {}, 0, 0
        for batch in loader:
            x, target = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, DynamicsForecaster):
                prediction, logits, _, _ = model(x, training_outputs=True)
                labels = batch[2].to(device)
            else:
                prediction, logits, labels = model(x), None, None
            parts = loss_components(
                identity["model"], prediction, target, x, logits, labels
            )
            parts["total"].backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                raise RuntimeError("Nonfinite training gradient")
            optimizer.step()
            n = target.numel()
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach()) * n
            points += n
            batches += 1
        if points == 0:
            raise RuntimeError("Empty training epoch")
        val = validation_metrics(model, validation, device, settings["batch_size"])
        row = {
            "epoch": epoch,
            **{k: v / points for k, v in sums.items()},
            "validation_MAE": val["MAE"],
            "validation_MSE": val["MSE"],
            "train_batches": batches,
            "train_points": points,
            "validation_batches": val["batches"],
            "validation_points": val["points"],
            "train_order_hash": hash_array(indices),
            "finite_gradients": True,
        }
        history.append(row)
        if stopping.observe(epoch, val["MAE"]):
            payload = {
                "format": "solar-checkpoint-v1",
                "identity": identity,
                "identity_hash": hash_json(identity),
                "epoch": epoch,
                "validation_MAE": val["MAE"],
                "model_state": model.state_dict(),
                "model_state_hash": component_hash(model),
            }
            atomic_write(
                directory / "best.pth",
                lambda stream: torch.save(payload, stream),
                validator=validate_checkpoint_archive,
            )
        write_json(directory / "history.json", history)
        print(
            f"{identity['country']} {identity['model']} seed={identity['seed']} "
            f"epoch={epoch} main_MSE={row['main_mse']:.6f} "
            f"Validation_MAE={val['MAE']:.6f} best={stopping.best_mae:.6f}",
            flush=True,
        )
    parameters_updated = component_hash(model) != initial_model_hash
    restored, payload = checkpoint_read(directory / "best.pth", identity)
    model.load_state_dict(restored.state_dict())
    return {
        "best_restored": component_hash(model) == payload["model_state_hash"],
        "restored_model_hash": component_hash(model),
        "parameters_updated": parameters_updated,
        "best_epoch": stopping.best_epoch,
        "best_validation_MAE": stopping.best_mae,
        "epochs": len(history),
        "termination": "early_stopping" if stopping.should_stop else "max_epochs",
        "seconds": time.perf_counter() - start,
        "sample_counts": {"Train": len(train_data), "Validation": len(validation)},
        "test_used_for_selection": False,
    }
