import copy

import numpy as np
import pytest

from solar_forecasting.config import SEEDS
from solar_forecasting.metrics import (
    active_metrics,
    aggregate_five_seed,
    basic_metrics,
    changes_mw,
    dynamic_metrics,
    evaluate_arrays,
    freeze_comparators,
    horizon_gain,
    inverse_solar,
    matched_comparison,
    mean_weights,
    negative_diagnostics,
    paired_gain,
    verify_comparators,
)


def rows(model="state-supervision", country="Spain", split="Validation", ramp=10.0):
    return [
        {
            "country": country,
            "model": model,
            "split": split,
            "seed": s,
            "checkpoint_sha256": str(i) * 64,
            "standardized_MAE": float(i + 1),
            "MAE_MW": float(i + 10),
            "RMSE_MW": float(i + 20),
            "ActiveMAE_MW": float(i + 15),
            "High_RampMAE_MW": ramp + i,
            "RMRR": 0.9,
            "High_Ramp_DirectionAccuracy": 1.0,
        }
        for i, s in enumerate(SEEDS)
    ]


def test_basic_formula():
    metric = basic_metrics([1, 3], [2, 1])
    assert metric["MAE"] == 1.5 and metric["MSE"] == 2.5
    assert metric["RMSE"] == np.sqrt(2.5)


@pytest.mark.parametrize(
    "a,p", [([], []), ([1], [1, 2]), ([np.nan], [0]), ([0], [np.inf])]
)
def test_bad_metric_arrays(a, p):
    with pytest.raises(ValueError):
        basic_metrics(a, p)


def test_active_truth_mask_and_empty():
    metric = active_metrics([0, 5, 10], [100, 6, 7], 5)
    assert metric["ActiveMAE_MW"] == 3 and metric["active_points"] == 1
    assert metric["active_fraction"] == 1 / 3
    with pytest.raises(ValueError):
        active_metrics([0, 5], [1, 1], 5)


def test_physical_negative_not_standardized_and_no_clipping():
    scaler = {"name": "solar_generation_actual", "mean": 100.0, "std": 10.0}
    values = np.zeros((1, 24))
    values[0, :2] = [-1, -11]
    power = inverse_solar(values, scaler)
    assert power[0, 0] == 90 and power[0, 1] == -10
    metric = negative_diagnostics(power)
    assert metric["negative_prediction_MW_count"] == 1
    assert metric["minimum_prediction_MW"] == -10
    np.testing.assert_array_equal(values[0, :2], [-1, -11])


def test_dynamic_anchor_boundary_total_ratio_and_direction():
    truth = np.full((1, 24), 40.0)
    truth[0, 0] = 30
    prediction = np.full((1, 24), 50.0)
    prediction[0, 0] = 20
    metric = dynamic_metrics(truth, prediction, [10], 10)
    # First two true deltas: 20,10. Predicted: 10,30.
    assert changes_mw(truth, [10])[0, :2].tolist() == [20, 10]
    assert metric["high_points"] == 2
    assert metric["High_RampMAE_MW"] == 15
    assert metric["RMRR"] == 40 / 30
    assert metric["RMRR"] != np.mean([10 / 20, 30 / 10])
    assert metric["High_Ramp_DirectionAccuracy"] == 1
    with pytest.raises(ValueError):
        dynamic_metrics(np.ones((1, 24)), np.ones((1, 24)), [1], 1)
    with pytest.raises(ValueError):
        dynamic_metrics(truth, prediction, [10], 0)


def test_shared_anchor_is_independent_of_loss_cancellation():
    truth = np.full((1, 24), 1.0)
    prediction = np.full((1, 24), 2.0)
    correct = dynamic_metrics(truth, prediction, [0], 1)
    wrong = dynamic_metrics(truth, prediction, [10], 1)
    assert correct["RMRR"] == 2
    assert wrong["RMRR"] == 8 / 9


def test_point_mean_sd_and_dynamic_median():
    data = rows()
    result = aggregate_five_seed(data)[0]
    assert result["standardized_MAE_mean"] == 3
    assert result["standardized_MAE_sample_SD"] == np.std([1, 2, 3, 4, 5], ddof=1)
    assert result["High_RampMAE_MW_median"] == 12


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "nonfinite", "mixed_split", "persistence"]
)
def test_bad_aggregation(change):
    data = rows()
    if change == "missing":
        data.pop()
    if change == "duplicate":
        data[-1]["seed"] = data[0]["seed"]
    if change == "nonfinite":
        data[0]["MAE_MW"] = np.inf
    if change == "mixed_split":
        data[0]["split"] = "Test"
    if change == "persistence":
        data[0]["model"] = "seasonal-persistence"
    with pytest.raises(ValueError):
        aggregate_five_seed(data)


def test_pair_mean_differs_from_ratio_of_means():
    ref = rows("global-fusion")
    cand = rows("horizon-fusion")
    for i, r in enumerate(ref):
        r["standardized_MAE"] = float(i + 1)
    for r in cand:
        r["standardized_MAE"] = 1.0
    got = paired_gain(ref, cand)
    expected = np.mean([100 * (x - 1) / x for x in range(1, 6)])
    assert got["mean_percent"] == expected
    assert got["mean_percent"] != 100 * (3 - 1) / 3
    assert got["positive_seeds"] == 4


def test_horizon_ratio_of_seed_means():
    ref = np.tile(np.arange(1, 6)[:, None], (1, 24))
    got = horizon_gain(ref, np.ones((5, 24)))
    np.testing.assert_allclose(got, 100 * (3 - 1) / 3)
    with pytest.raises(ValueError):
        horizon_gain(np.zeros((5, 24)), np.ones((5, 24)))
    with pytest.raises(ValueError):
        horizon_gain(ref[:4], np.ones((5, 24)))


def test_weights_are_five_seed_cell_means():
    matrices = np.ones((5, 24, 6)) / 6
    np.testing.assert_allclose(mean_weights(matrices), matrices[0])
    with pytest.raises(ValueError):
        mean_weights(matrices[:4])


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (8, 10, "trajectory-supervision"),
        (10, 8, "state-supervision"),
        (10, 10, "trajectory-supervision"),
    ],
)
def test_comparator_selection_and_tie(left, right, expected):
    data = rows("trajectory-supervision", ramp=left) + rows(
        "state-supervision", ramp=right
    )
    document = freeze_comparators(data, "hash")
    verify_comparators(document, "hash", ["Spain"])
    assert all(r["model"] == expected for r in document["selections"])


def test_point_change_uses_same_ramp_comparator_and_seedwise_median():
    left = rows("trajectory-supervision", ramp=8)
    right = rows("state-supervision", ramp=10)
    joint = rows("joint-supervision", ramp=7)
    for a, b, j in zip(left, right, joint):
        a["MAE_MW"] = 20
        b["MAE_MW"] = 1
        j["MAE_MW"] = 22
    choices = freeze_comparators(left + right, "hash")
    result = matched_comparison(left + right + joint, choices)
    assert result["summaries"][0]["PointMAE_change_median_percent"] == 10
    expected = np.median(
        [
            100 * (a["High_RampMAE_MW"] - j["High_RampMAE_MW"]) / a["High_RampMAE_MW"]
            for a, j in zip(left, joint)
        ]
    )
    assert result["summaries"][0]["RampMAE_gain_median_percent"] == expected


@pytest.mark.parametrize(
    "change", ["test", "missing", "duplicate", "nonfinite", "hash"]
)
def test_invalid_comparator_document(change):
    data = rows("trajectory-supervision") + rows("state-supervision")
    if change == "test":
        for r in data:
            r["split"] = "Test"
    if change == "missing":
        data.pop()
    if change == "duplicate":
        data[-1]["seed"] = data[-2]["seed"]
    if change == "nonfinite":
        data[-1]["High_RampMAE_MW"] = np.nan
    if change == "hash":
        document = freeze_comparators(data, "hash")
        document["selections"][0]["model"] = "joint-supervision"
        with pytest.raises(ValueError):
            verify_comparators(document, "hash", ["Spain"])
    else:
        with pytest.raises(ValueError):
            freeze_comparators(data, "hash")


def test_persistence_physical_precision_preserved():
    scaler = {"name": "solar_generation_actual", "mean": 100.0, "std": 30.0}
    truth = np.tile(np.arange(24) * 100.0, (1, 1))
    baseline = truth + 7
    result = evaluate_arrays(
        (truth - 100) / 30,
        (baseline - 100) / 30,
        scaler,
        truth,
        [0],
        10,
        100,
        physical_reference=baseline,
    )
    assert result["MAE_MW"] == 7
