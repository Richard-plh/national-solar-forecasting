import json

import numpy as np
import pytest
import torch

from solar_forecasting.config import COUNTRIES, MODEL_NAMES, SEEDS
from solar_forecasting.models import (
    DynamicsForecaster,
    MultiScaleForecaster,
    build_matched,
    component_hash,
    loss_components,
    order_seed,
    shared_components,
    state_labels,
    trajectory_delta,
)
from solar_forecasting.training import train_order_indices


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_shapes_gradients_and_inference(model_name):
    model = build_matched(model_name, "Spain", 42)
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(2, 336, 10, generator=generator)
    target = torch.randn(2, 24, generator=generator)
    if isinstance(model, DynamicsForecaster):
        prediction, logits, fused, weights = model(x, training_outputs=True)
    else:
        prediction, fused, weights = model(x, return_features=True)
        logits = None
    assert (
        prediction.shape == (2, 24)
        and fused.shape == (2, 24, 128)
        and weights.shape == (2, 24, 6)
    )
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 24))
    labels = torch.randint(0, 5, (2, 24), generator=generator)
    parts = loss_components(model_name, prediction, target, x, logits, labels)
    parts["total"].backward()
    base = model.base if isinstance(model, DynamicsForecaster) else model
    assert base.encoder.gru.weight_ih_l0.grad is not None
    assert base.forecast_head.network[0].weight.grad is not None
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    if hasattr(model, "auxiliary_head"):
        assert model.auxiliary_head[0].weight.grad is not None
        assert [type(layer).__name__ for layer in model.auxiliary_head] == [
            "Linear",
            "ReLU",
            "Linear",
        ]
    else:
        assert "classification_ce" not in parts
    model.eval()
    with torch.no_grad():
        output = model(x)
    assert output.shape == (2, 24)


@pytest.mark.parametrize("country", COUNTRIES)
@pytest.mark.parametrize("seed", SEEDS)
def test_all_public_initializations_match_reference(config_dir, country, seed):
    references = json.loads((config_dir / "initialization_checksums.json").read_text())
    ref = next(r for r in references if r["country"] == country and r["seed"] == seed)
    shared = []
    auxiliary = []
    for name in MODEL_NAMES:
        model = build_matched(name, country, seed)
        for key, value in shared_components(model).items():
            assert value == ref[key]
        shared.append(shared_components(model))
        if hasattr(model, "auxiliary_head"):
            auxiliary.append(component_hash(model.auxiliary_head))
    assert len(set(auxiliary)) == 1
    assert shared[2] == shared[3] == shared[4] == shared[5]


def test_gate_sample_independence_and_sharing():
    model = MultiScaleForecaster("horizon-fusion").eval()
    with torch.no_grad():
        model.gate.network[-1].weight.fill_(0.05)
    first = model.scale_weights(torch.randn(3, 6, 128))
    second = model.scale_weights(torch.randn(3, 6, 128) * 100)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[0], first[2])
    assert len([m for m in model.modules() if isinstance(m, torch.nn.GRU)]) == 1
    assert model.forecast_head.network[0].in_features == 152
    assert model.gate.network[0].in_features == 152
    assert model.horizon_one_hot.shape == (24, 24)


@pytest.mark.parametrize(
    "name,count",
    [
        ("equal-fusion", 63617),
        ("global-fusion", 63623),
        ("horizon-fusion", 75687),
        ("trajectory-supervision", 75687),
        ("state-supervision", 84268),
        ("joint-supervision", 84268),
    ],
)
def test_parameter_counts(name, count):
    model = build_matched(name, "Germany", 123)
    assert sum(p.numel() for p in model.parameters()) == count


@pytest.mark.parametrize("q30,q80", [(18.0, 591.0), (0.0, 2408.0), (0.0, 261.0)])
def test_five_class_boundaries(q30, q80):
    d = torch.tensor([-q80 - 1, -q80, -q30, 0, q30, q80, q80 + 1], dtype=torch.float64)
    assert state_labels(d, q30, q80).tolist() == [0, 1, 2, 2, 2, 3, 4]
    assert state_labels(d, q30, q80).dtype == torch.int64


@pytest.mark.parametrize(
    "q30,q80", [(-1, 2), (0, 0), (2, 2), (3, 2), (np.nan, 2), (0, np.inf)]
)
def test_invalid_thresholds(q30, q80):
    with pytest.raises(ValueError):
        state_labels(torch.zeros(2), q30, q80)


def test_correct_anchor_and_difference_direction():
    x = torch.zeros(2, 336, 10)
    x[:, -1, 0] = 1000
    x[:, -1, 3] = torch.tensor([7.0, 9.0])
    y = torch.arange(1, 25).float().repeat(2, 1)
    delta = trajectory_delta(y, x[:, -1, 3])
    assert delta[:, 0].tolist() == [-6.0, -8.0]
    assert torch.all(delta[:, 1:] == 1)
    assert state_labels(delta, 0, 7)[:, 0].tolist() == [1, 0]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_loss_formula_independent_of_network(name):
    p = torch.arange(1, 49).reshape(2, 24).float().requires_grad_()
    y = torch.zeros(2, 24)
    x = torch.zeros(2, 336, 10)
    x[:, -1, 3] = 3
    logits = torch.zeros(2, 24, 5, requires_grad=True)
    labels = torch.full((2, 24), 2, dtype=torch.int64)
    parts = loss_components(name, p, y, x, logits, labels)
    expected = (p**2).mean()
    if name in ("trajectory-supervision", "joint-supervision"):
        difference = torch.cat([p[:, :1], p[:, 1:] - p[:, :-1]], 1)
        expected = expected + difference.abs().mean()
    if name in ("state-supervision", "joint-supervision"):
        expected = expected + np.log(5)
    torch.testing.assert_close(parts["total"], expected)
    parts["total"].backward()
    assert torch.isfinite(p.grad).all() and torch.all(p.grad != 0)


def test_deterministic_order_and_reset():
    build_matched("horizon-fusion", "Spain", 42)
    one = torch.rand(8)
    build_matched("joint-supervision", "Spain", 42)
    two = torch.rand(8)
    torch.testing.assert_close(one, two)
    seed = order_seed("Spain", 42)
    a = train_order_indices(100, seed, 1)
    b = train_order_indices(100, seed, 1)
    np.testing.assert_array_equal(a, b)
    assert set(a) == set(range(100))
    assert not np.array_equal(a, train_order_indices(100, seed, 2))
