from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def small_cpu_tests():
    torch.set_num_threads(1)


@pytest.fixture
def config_dir():
    return Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def frame():
    n = 6000
    x = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "Time": pd.date_range("2015-01-01", periods=n, freq="h"),
            "load_actual": 1000 + x * 0.05 + 80 * np.sin(x / 13),
            "wind_generation_actual": 100 + 30 * np.cos(x / 19),
            "solar_generation_actual": 250
            * np.maximum(0, np.sin((x % 24 - 6) * np.pi / 12))
            + (x % 7),
        }
    )
