import copy
import json
import threading
from pathlib import Path
from unittest import mock

import pytest
import torch

from solar_forecasting import atomic_io
from solar_forecasting.cli import main
from solar_forecasting.config import EXPERIMENT, load_config, run_matrix
from solar_forecasting.evaluation import accepted_plan, evaluate
from solar_forecasting.models import build_matched, component_hash
from solar_forecasting.training import (
    EarlyStopping,
    train,
    training_lock,
    validation_metrics,
)
from solar_forecasting.utils import hash_json, read_json, sha256_file, write_json
from solar_forecasting.workflow import checkpoint_read, code_hash, inspect_run


def test_matrix_complete_and_unique(config_dir):
    config = load_config(config_dir / "experiment.json")
    assert config == EXPERIMENT
    rows = run_matrix()
    assert (
        len(rows) == 90
        and len({(r["country"], r["model"], r["seed"]) for r in rows}) == 90
    )
    assert sum(r["group"] == "fusion" for r in rows) == 45
    assert sum(r["group"] == "supervision" for r in rows) == 45


def test_scientific_config_change_rejected(config_dir, tmp_path):
    config = copy.deepcopy(EXPERIMENT)
    config["training"]["batch_size"] = 8
    path = tmp_path / "changed.json"
    write_json(path, config)
    with pytest.raises(ValueError):
        load_config(path)


def test_training_refused_before_any_model_or_data(tmp_path, config_dir):
    with mock.patch("solar_forecasting.training.build_matched") as model:
        with mock.patch("solar_forecasting.training.load_country") as data:
            with pytest.raises(PermissionError):
                train(tmp_path, EXPERIMENT, config_dir, "Spain", "horizon-fusion", 42)
            model.assert_not_called()
            data.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("allow,comparator", [(False, None), (True, None)])
def test_test_permission_independent_and_before_prediction(
    tmp_path, config_dir, allow, comparator
):
    with mock.patch("solar_forecasting.evaluation.predict") as forward:
        with mock.patch("solar_forecasting.evaluation.load_country") as data:
            with pytest.raises(PermissionError):
                evaluate(
                    tmp_path,
                    EXPERIMENT,
                    config_dir,
                    split="Test",
                    allow_test=allow,
                    comparator_file=comparator,
                )
            forward.assert_not_called()
            data.assert_not_called()
    assert not (tmp_path / "evaluations").exists()


def test_test_plan_requires_all_models_before_labels(tmp_path, config_dir):
    with pytest.raises(PermissionError):
        accepted_plan(tmp_path, EXPERIMENT, config_dir, "Test", "fusion")
    with mock.patch("solar_forecasting.evaluation.load_country") as data:
        with pytest.raises(PermissionError):
            accepted_plan(tmp_path, EXPERIMENT, config_dir, "Validation", "all")
        data.assert_not_called()


def test_training_lock_and_original_file_preserved(tmp_path):
    with training_lock(tmp_path):
        with pytest.raises(FileExistsError):
            with training_lock(tmp_path):
                pass
        assert (tmp_path / ".training.lock").exists()
    assert not (tmp_path / ".training.lock").exists()


def test_early_stopping_uses_minimum_mae_and_strict_ties():
    stopping = EarlyStopping(2)
    assert stopping.observe(1, 0.5)
    assert stopping.observe(2, 0.4)
    assert not stopping.observe(3, 0.4)
    assert not stopping.observe(4, 0.45)
    assert stopping.should_stop and stopping.best_epoch == 2
    with pytest.raises(ValueError):
        stopping.observe(5, float("nan"))


def test_selection_refuses_test_dataset():
    dataset = mock.Mock(split="Test")
    with pytest.raises(PermissionError):
        validation_metrics(mock.Mock(), dataset, "cpu", 64)


def checkpoint_fixture(tmp_path):
    model = build_matched("horizon-fusion", "Spain", 42)
    # Synthetic state perturbation, not training or an optimizer update.
    with torch.no_grad():
        model.forecast_head.network[-1].bias.add_(0.001)
    identity = {
        "schema": "solar-run-v1",
        "country": "Spain",
        "model": "horizon-fusion",
        "seed": 42,
        "attempt": 1,
        "config_hash": hash_json(EXPERIMENT),
        "code_hash": code_hash(),
        "data_identity": {},
        "sample_counts": {"Train": 3, "Validation": 2},
        "target": "solar_generation_actual",
        "precision": "FP32",
    }
    payload = {
        "format": "solar-checkpoint-v1",
        "identity": identity,
        "identity_hash": hash_json(identity),
        "model_state": model.state_dict(),
        "model_state_hash": component_hash(model),
        "epoch": 1,
        "validation_MAE": 0.1,
    }
    path = tmp_path / "best.pth"
    atomic_io.atomic_write(
        path,
        lambda stream: torch.save(payload, stream),
        validator=atomic_io.validate_checkpoint_archive,
    )
    return path, identity


def test_checkpoint_integrity_and_identity(tmp_path):
    path, identity = checkpoint_fixture(tmp_path)
    model, payload = checkpoint_read(path, identity)
    assert component_hash(model) == payload["model_state_hash"]
    with pytest.raises(ValueError):
        checkpoint_read(path, {**identity, "seed": 123})
    path.write_bytes(b"truncated")
    with pytest.raises(Exception):
        checkpoint_read(path, identity)


def test_complete_acceptance_requires_history_and_best_linkage(tmp_path):
    path, identity = checkpoint_fixture(tmp_path)
    write_json(tmp_path / "config.json", EXPERIMENT)
    write_json(
        tmp_path / "initialization.json",
        build_matched("horizon-fusion", "Spain", 42).initialization_metadata,
    )
    history = [
        {
            "epoch": i,
            "main_mse": 0.2,
            "total": 0.2,
            "validation_MAE": 0.1 if i == 1 else 0.2,
            "validation_MSE": 0.2,
            "finite_gradients": True,
            "train_batches": 1,
            "train_points": 72,
            "validation_batches": 1,
            "validation_points": 48,
        }
        for i in range(1, 14)
    ]
    write_json(tmp_path / "history.json", history)
    manifest = {
        "status": "COMPLETE",
        "identity": identity,
        "sample_counts": {"Train": 3, "Validation": 2},
        "best_restored": True,
        "restored_model_hash": checkpoint_read(path, identity)[1]["model_state_hash"],
        "parameters_updated": True,
        "best_epoch": 1,
        "artifact_hashes": {
            name: sha256_file(tmp_path / name)
            for name in [
                "config.json",
                "initialization.json",
                "history.json",
                "best.pth",
            ]
        },
    }
    write_json(tmp_path / "manifest.json", manifest)
    assert inspect_run(tmp_path, EXPERIMENT)["status"] == "COMPLETE"
    history[1]["validation_MAE"] = 0.05
    write_json(tmp_path / "history.json", history)
    manifest["artifact_hashes"]["history.json"] = sha256_file(tmp_path / "history.json")
    write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="best Validation"):
        inspect_run(tmp_path, EXPERIMENT)


def test_checkpoint_file_alone_is_not_complete(tmp_path):
    checkpoint_fixture(tmp_path)
    assert inspect_run(tmp_path, EXPERIMENT)["status"] == "INCOMPLETE"


def denied(code=32):
    error = PermissionError("simulated sharing conflict")
    error.winerror = code
    return error


def test_atomic_save_replace_and_temporary_retry(tmp_path, monkeypatch):
    path = tmp_path / "history.json"
    write_json(path, {"epoch": 1})
    real = atomic_io.os.replace
    calls = []

    def flaky(source, target):
        calls.append(str(source))
        if len(calls) < 3:
            raise denied()
        return real(source, target)

    monkeypatch.setattr(atomic_io.os, "replace", flaky)
    monkeypatch.setattr(atomic_io.time, "sleep", lambda _: None)
    write_json(path, {"epoch": 2})
    assert read_json(path) == {"epoch": 2} and len(calls) == 3


def test_atomic_persistent_denial_keeps_original_and_evidence(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    write_json(path, {"status": "RUNNING"})
    before = path.read_bytes()
    monkeypatch.setattr(atomic_io.os, "replace", mock.Mock(side_effect=denied()))
    monkeypatch.setattr(atomic_io.time, "sleep", lambda _: None)
    with pytest.raises(atomic_io.AtomicWriteError) as got:
        write_json(path, {"status": "COMPLETE"})
    assert path.read_bytes() == before
    assert got.value.temporary_path.exists() and got.value.failure_path.exists()
    assert len(got.value.evidence["replace_attempts"]) == 7


def test_incomplete_candidate_never_published(tmp_path):
    path = tmp_path / "identity.json"
    write_json(path, {"valid": True})

    def reject(_):
        raise ValueError("incomplete")

    with pytest.raises(atomic_io.AtomicWriteError):
        atomic_io.atomic_write(path, lambda f: f.write(b"partial"), validator=reject)
    assert read_json(path) == {"valid": True}


def test_concurrent_json_reads_and_writes(tmp_path):
    path = tmp_path / "history.json"
    write_json(path, {"epoch": 0})
    errors = []

    def reader():
        try:
            for _ in range(40):
                assert isinstance(read_json(path)["epoch"], int)
        except BaseException as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=reader)
    thread.start()
    for epoch in range(1, 15):
        write_json(path, {"epoch": epoch})
    thread.join()
    assert not errors and read_json(path)["epoch"] == 14


def test_cli_plan_has_no_training_side_effect(tmp_path, config_dir, capsys):
    main(
        [
            "plan",
            "--config",
            str(config_dir / "experiment.json"),
            "--workspace",
            str(tmp_path),
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["training_started"] is False and len(result["tasks"]) == 90
    assert list(tmp_path.iterdir()) == []
