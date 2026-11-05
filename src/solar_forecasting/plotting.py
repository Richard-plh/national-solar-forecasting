"""Optional figures from saved evaluation arrays; no model execution."""

from pathlib import Path

import numpy as np

from .config import COUNTRIES, SEEDS
from .metrics import mean_weights
from .utils import read_json

LABELS = {
    "equal-fusion": "Equal-weight fusion",
    "global-fusion": "Global-weight fusion",
    "horizon-fusion": "Horizon-aware fusion",
    "trajectory-supervision": "Trajectory supervision",
    "state-supervision": "State supervision",
    "joint-supervision": "Joint supervision",
    "seasonal-persistence": "Seasonal persistence",
}
COLORS = {
    "seasonal-persistence": "#A0A5AB",
    "horizon-fusion": "#7CA294",
    "trajectory-supervision": "#DAB49D",
    "state-supervision": "#7CA3BA",
    "joint-supervision": "#A38BB8",
}


def plot_saved(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    summary = read_json(directory / "summary.json")
    if read_json(directory / "manifest.json")["status"] != "COMPLETE":
        raise ValueError("Figures require a complete saved evaluation")
    output = directory / "figures"
    output.mkdir(exist_ok=False)
    split = summary["split"]
    generated = []

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            path = output / (name + "." + extension)
            fig.savefig(path, dpi=220)
            generated.append(path.name)
        plt.close(fig)

    for country in COUNTRIES:
        folder = directory / country
        matrices = [
            read_json(folder / f"horizon-fusion_seed_{seed}_weights.json")
            for seed in SEEDS
        ]
        weights = mean_weights(matrices)
        fig, ax = plt.subplots(figsize=(7, 3))
        image = ax.imshow(weights.T, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(6), [12, 24, 48, 96, 168, 336])
        ax.set_xticks([0, 5, 11, 17, 23], [1, 6, 12, 18, 24])
        ax.set(
            xlabel="Forecast horizon (h)",
            ylabel="Historical lookback (h)",
            title=f"{split}: {country}, five-seed mean weights",
        )
        fig.colorbar(image, ax=ax, label="Scale weight")
        save(fig, f"{country}_scale_weights")
        cases = read_json(folder / "fixed_cases.json")
        for date, case in cases["cases"].items():
            fig, ax = plt.subplots(figsize=(8, 3.4))
            ax.plot(
                range(1, 25),
                case["actual_MW"],
                color="black",
                linewidth=2,
                label="Actual Solar",
            )
            for model, values in case["predictions_MW"].items():
                if model in ("equal-fusion", "global-fusion"):
                    continue
                ax.plot(
                    range(1, 25),
                    values,
                    label=LABELS[model],
                    color=COLORS[model],
                    linestyle="--" if model == "seasonal-persistence" else "-",
                )
            ax.set(
                xlabel="Forecast horizon (h)",
                ylabel="Solar Generation (MW)",
                title=f"{split}: {country}, {date}, seed 42 (stored time)",
            )
            ax.legend(fontsize=8, ncol=2)
            ax.grid(axis="y", alpha=0.2)
            save(fig, f"{country}_{date[:10]}_forecast")
    if summary["supervision_gain"]:
        fig, ax = plt.subplots(figsize=(9, 3.6))
        x = np.arange(3)
        for k, model in enumerate(
            ("trajectory-supervision", "state-supervision", "joint-supervision")
        ):
            values = [
                next(
                    r["mean_percent"]
                    for r in summary["supervision_gain"]
                    if r["country"] == country and r["model"] == model
                )
                for country in COUNTRIES
            ]
            ax.bar(
                x + (k - 1) * 0.24,
                values,
                0.24,
                color=COLORS[model],
                label=LABELS[model],
            )
        ax.axhline(0, color="#64727E", linewidth=1)
        ax.set_xticks(x, list(COUNTRIES))
        ax.set(
            ylabel="Mean paired MAE improvement (%)",
            title=f"{split}: supervision vs horizon-aware fusion",
        )
        ax.legend(fontsize=8)
        save(fig, "supervision_ablation")
    return {"figures": generated, "directory": str(output), "new_model_forward": 0}
