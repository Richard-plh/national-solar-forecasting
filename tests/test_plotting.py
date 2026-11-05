import numpy as np
import pytest

from solar_forecasting.config import COUNTRIES, SEEDS
from solar_forecasting.plotting import plot_saved
from solar_forecasting.utils import write_json


def test_plotting_reads_only_saved_synthetic_results(tmp_path):
    pytest.importorskip("matplotlib")
    write_json(tmp_path / "manifest.json", {"status": "COMPLETE"})
    write_json(
        tmp_path / "summary.json", {"split": "Validation", "supervision_gain": []}
    )
    for country in COUNTRIES:
        folder = tmp_path / country
        folder.mkdir()
        for seed in SEEDS:
            write_json(
                folder / f"horizon-fusion_seed_{seed}_weights.json",
                (np.ones((24, 6)) / 6).tolist(),
            )
        values = (np.arange(24) * 100).tolist()
        write_json(
            folder / "fixed_cases.json",
            {
                "country": country,
                "split": "Validation",
                "seed": 42,
                "cases": {
                    "2018-08-21 00:00": {
                        "index": 0,
                        "actual_MW": values,
                        "predictions_MW": {
                            "seasonal-persistence": values,
                            "horizon-fusion": values,
                            "state-supervision": values,
                            "joint-supervision": values,
                        },
                    }
                },
            },
        )
    result = plot_saved(tmp_path)
    assert result["new_model_forward"] == 0 and len(result["figures"]) == 12
    assert all(
        (tmp_path / "figures" / name).stat().st_size > 1000
        for name in result["figures"]
    )
    with pytest.raises(FileExistsError):
        plot_saved(tmp_path)
