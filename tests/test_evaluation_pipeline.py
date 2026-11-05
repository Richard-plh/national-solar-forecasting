"""Complete result-persistence paths with mocked inference, never real data."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from solar_forecasting.config import EXPERIMENT, run_matrix
from solar_forecasting.data import SolarWindowDataset
from solar_forecasting.evaluation import accepted_plan, audit_saved, evaluate
from solar_forecasting.utils import hash_json, read_json, sha256_file


def synthetic_country(country):
    n = 384
    index = np.arange(n)
    raw = 5000 + index * 10 + 4000 * np.sin(index * np.pi / 3)
    features = np.zeros((n, 10), dtype=np.float32)
    features[:, 3] = (raw - 5000) / 1000
    times = np.datetime64("2018-08-07T00:00", "ns") + index * np.timedelta64(1, "h")
    split = SimpleNamespace(
        raw_targets=raw,
        targets=features[:, 3],
        features=features,
        origins=np.array([336, 360]),
        times=times,
        origin_hash="synthetic-origin",
    )
    prepared = SimpleNamespace(
        country=country,
        splits={"Validation": split, "Test": split},
        identity={"synthetic_fixture": True},
        target_scaler={
            "name": "solar_generation_actual",
            "mean": 5000.0,
            "std": 1000.0,
            "ddof": 0,
        },
    )
    prepared.dataset = lambda split_name, allow_test=False: SolarWindowDataset(
        prepared, split_name, allow_test=allow_test
    )
    return prepared


class FakeModel:
    """No neural network and no forward method."""

    def scale_weights(self):
        return torch.ones(24, 6) / 6


def test_90_model_saved_result_pipeline_with_mock_inference(
    tmp_path, config_dir, monkeypatch
):
    config = copy.deepcopy(EXPERIMENT)
    dates = ["2018-08-21 00:00", "2018-08-22 00:00"]
    config["curve_dates"] = {"Validation": dates, "Test": dates}
    tasks = []
    for row in run_matrix():
        directory = tmp_path / row["run_path"]
        directory.mkdir(parents=True)
        cp = directory / "best.pth"
        cp.write_bytes((row["country"] + row["model"] + str(row["seed"])).encode())
        tasks.append(
            {
                **row,
                "identity": {"synthetic_fixture": True},
                "checkpoint_sha256": sha256_file(cp),
            }
        )

    def plan(workspace, cfg, config_dir, split, group, comparators=None):
        body = {
            "schema": "synthetic-plan-test",
            "split": split,
            "group": group,
            "config_hash": hash_json(cfg),
            "models": tasks,
            "thresholds": cfg["thresholds"],
            "comparators": comparators,
        }
        return {**body, "plan_hash": hash_json(body)}

    monkeypatch.setattr("solar_forecasting.evaluation.accepted_plan", plan)
    monkeypatch.setattr(
        "solar_forecasting.evaluation.load_country",
        lambda workspace, country, config_dir: synthetic_country(country),
    )
    monkeypatch.setattr(
        "solar_forecasting.evaluation.checkpoint_read", lambda *a: (FakeModel(), {})
    )
    calls = []

    def fake_predict(model, dataset, **kwargs):
        calls.append(dataset.split)
        data = dataset.data
        return data.targets[data.origins[:, None] + np.arange(24)] + 0.1

    monkeypatch.setattr("solar_forecasting.evaluation.predict", fake_predict)
    validation = evaluate(tmp_path, config, config_dir, device="cpu")
    assert validation["models"] == 90 and len(calls) == 90
    comparator = Path(validation["directory"]) / "validation_comparators.json"
    test = evaluate(
        tmp_path,
        config,
        config_dir,
        split="Test",
        allow_test=True,
        comparator_file=comparator,
        device="cpu",
    )
    assert test["models"] == 90 and len(calls) == 180
    directory = Path(test["directory"])
    assert audit_saved(directory) == {
        "status": "PASS",
        "model_results": 90,
        "persistence_results": 3,
        "new_model_forward": 0,
    }
    assert len(calls) == 180
    assert len(read_json(directory / "summary.json")["five_seed"]) == 18
    with pytest.raises(FileExistsError):
        evaluate(
            tmp_path,
            config,
            config_dir,
            split="Test",
            allow_test=True,
            comparator_file=comparator,
            device="cpu",
        )
    assert len(calls) == 180
    changed = directory / "Spain" / "horizon-fusion_seed_42.npz"
    changed.write_bytes(changed.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_saved(directory)


def test_actual_plan_has_all_90_identities_before_arrays(
    tmp_path, config_dir, monkeypatch
):
    def accepted(workspace, config, country, model, seed):
        path = tmp_path / f"runs/{country}/{model}/seed_{seed}"
        return {
            "status": "COMPLETE",
            "directory": str(path),
            "identity": {"data_identity": {"synthetic_fixture": True}},
            "checkpoint_sha256": "a" * 64,
        }

    monkeypatch.setattr("solar_forecasting.evaluation.inspect_task", accepted)
    monkeypatch.setattr(
        "solar_forecasting.evaluation.read_json",
        lambda p: {"identity": {"synthetic_fixture": True}},
    )
    monkeypatch.setattr(
        "solar_forecasting.evaluation.load_country",
        lambda *a: pytest.fail("No arrays may be loaded during plan creation"),
    )
    plan = accepted_plan(tmp_path, EXPERIMENT, config_dir, "Validation", "all")
    assert len(plan["models"]) == 90
    assert len({(r["country"], r["model"], r["seed"]) for r in plan["models"]}) == 90
