import numpy as np
import pytest

from solar_forecasting.config import FEATURE_ORDER
from solar_forecasting.data import (
    build_features,
    fit_scalers,
    inspect_times,
    inverse_target,
    load_prepared,
    prepare_frame,
    row_ranges,
    save_prepared,
    standardize_target,
    valid_origins,
)
from solar_forecasting.evaluation import seasonal_persistence


def test_column_order_and_recomputed_feature(frame):
    frame["Net_load"] = -999999
    values = build_features(frame)
    assert values.shape == (len(frame), 10)
    np.testing.assert_array_equal(
        values[:, 0],
        frame.load_actual
        - frame.wind_generation_actual
        - frame.solar_generation_actual,
    )
    np.testing.assert_array_equal(values[:, 3], frame.solar_generation_actual)
    assert FEATURE_ORDER[3] == "solar_generation_actual"
    assert np.max(abs(values[:, 4:])) <= 1


def test_split_boundaries():
    assert row_ranges(43825) == {
        "Train": (0, 30677),
        "Validation": (30677, 35060),
        "Test": (35060, 43825),
    }


@pytest.mark.parametrize("change", ["missing", "duplicate", "reverse", "timezone"])
def test_timestamp_errors(frame, change):
    times = frame.Time.copy()
    if change == "missing":
        times = times.drop(index=2)
    elif change == "duplicate":
        times.iloc[2] = times.iloc[1]
    elif change == "reverse":
        times = times.iloc[::-1]
    else:
        times = times.dt.tz_localize("UTC")
    with pytest.raises(ValueError):
        inspect_times(times)


@pytest.mark.parametrize(
    "column", ["load_actual", "wind_generation_actual", "solar_generation_actual"]
)
@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_missing_rows_retained_and_windows_filtered(frame, column, missing):
    frame.loc[500, column] = missing
    prepared = prepare_frame(frame, "Spain", "synthetic")
    assert sum(len(x.times) for x in prepared.splits.values()) == len(frame)
    split = prepared.splits["Train"]
    expected = []
    raw = split.raw_features[:, 1:4]
    for origin in range(336, len(raw) - 23):
        if (
            np.isfinite(raw[origin - 336 : origin]).all()
            and np.isfinite(split.raw_targets[origin : origin + 24]).all()
        ):
            expected.append(origin)
    np.testing.assert_array_equal(split.origins, expected)
    assert not any(origin - 336 <= 500 < origin for origin in split.origins)


def test_window_alignment_and_six_suffixes(frame):
    prepared = prepare_frame(frame, "Spain", "synthetic")
    dataset = prepared.dataset("Train")
    origin = int(dataset.origin_indices[0])
    x, y = dataset[0]
    assert x.shape == (336, 10) and y.shape == (24,)
    np.testing.assert_array_equal(
        x, prepared.splits["Train"].features[origin - 336 : origin]
    )
    np.testing.assert_array_equal(
        y, prepared.splits["Train"].targets[origin : origin + 24]
    )
    for length in (12, 24, 48, 96, 168, 336):
        np.testing.assert_array_equal(
            x[-length:], prepared.splits["Train"].features[origin - length : origin]
        )
    assert (
        prepared.splits["Train"].times[origin - 1] + np.timedelta64(1, "h")
        == prepared.splits["Train"].times[origin]
    )


def test_train_only_scaler_and_no_future_feature_leak(frame):
    before = prepare_frame(frame, "Spain", "synthetic")
    future = frame.copy()
    future.loc[
        4500:, ["load_actual", "wind_generation_actual", "solar_generation_actual"]
    ] *= 17
    after = prepare_frame(future, "Spain", "synthetic")
    assert before.feature_scalers == after.feature_scalers
    np.testing.assert_array_equal(
        before.splits["Train"].features, after.splits["Train"].features
    )
    origin = 1200
    changed = frame.copy()
    changed.loc[origin:, "solar_generation_actual"] += 500
    np.testing.assert_array_equal(
        build_features(frame)[origin - 336 : origin],
        build_features(changed)[origin - 336 : origin],
    )


def test_scaler_common_rows_population_std_and_roundtrip(frame):
    frame.loc[10, "wind_generation_actual"] = np.nan
    raw = build_features(frame)
    features, target, count = fit_scalers(raw, 4200)
    assert count == 4199
    expected = raw[:4200][np.isfinite(raw[:4200, 1:4]).all(axis=1), 3]
    assert target["mean"] == expected.mean() and target["std"] == expected.std(ddof=0)
    assert target == features["solar_generation_actual"]
    values = np.array([0.0, 1.0, -1.0, 100.0])
    np.testing.assert_allclose(
        inverse_target(standardize_target(values, target), target), values, atol=1e-12
    )
    with pytest.raises(ValueError):
        inverse_target(values, {**target, "name": "load_actual"})
    raw[:, 3] = 0
    with pytest.raises(ValueError):
        fit_scalers(raw, 4200)


def test_cache_identity_and_no_overwrite(frame, tmp_path):
    prepared = prepare_frame(frame, "Spain", "synthetic")
    path = save_prepared(prepared, tmp_path / "data")
    loaded = load_prepared(path)
    assert loaded.target_scaler == prepared.target_scaler
    np.testing.assert_array_equal(
        loaded.splits["Train"].origins, prepared.splits["Train"].origins
    )
    with pytest.raises(FileExistsError):
        save_prepared(prepared, tmp_path / "data")
    with pytest.raises(ValueError):
        load_prepared(path, {"task_id": "foreign"})
    arrays = path.parent / "arrays.npz"
    arrays.write_bytes(arrays.read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        load_prepared(path)


def test_test_tensor_opt_in_is_independent(frame):
    prepared = prepare_frame(frame, "Spain", "synthetic")
    prepared.dataset("Train")
    with pytest.raises(PermissionError):
        prepared.dataset("Test")
    assert len(prepared.dataset("Test", allow_test=True)) > 0


def test_persistence_uses_only_the_previous_24_hours(frame):
    prepared = prepare_frame(frame, "Spain", "synthetic")
    split = prepared.splits["Train"]
    origin = int(split.origins[0])
    expected = split.raw_targets[origin - 24 : origin].copy()
    split.raw_targets[origin:] = 99999
    actual = seasonal_persistence(split)
    np.testing.assert_array_equal(actual[0], expected)


@pytest.mark.parametrize("n", [1, 335, 336, 359, 360, 361])
def test_origin_edges(n):
    origins, _ = valid_origins(np.ones((n, 3)), np.ones(n))
    assert len(origins) == max(0, n - 359)
