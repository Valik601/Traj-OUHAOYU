"""Evaluate and visualize the original Social LSTM and three modifications."""

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trajnetplusplustools


MODEL_SPECS = {
    "original": {
        "label": "Original Social",
        "checkpoint": ROOT / "grid_triple_experiment/models/social_original_grid.pkl",
        "color": "#0072B2",
        "linestyle": "--",
        "marker": "o",
    },
    "no_neighbours": {
        "label": "No neighbour info",
        "checkpoint": ROOT / "neighbor_ablation/models/social_no_neighbor_information.pkl",
        "color": "#E69F00",
        "linestyle": "-.",
        "marker": "^",
    },
    "tenth_grid": {
        "label": "0.1× grid range",
        "checkpoint": ROOT / "grid_tenth_experiment/models/social_tenth_grid.pkl",
        "color": "#009E73",
        "linestyle": ":",
        "marker": "s",
    },
    "triple_grid": {
        "label": "3× grid range",
        "checkpoint": ROOT / "grid_triple_experiment/models/social_triple_grid.pkl",
        "color": "#D55E00",
        "linestyle": (0, (5, 1, 1, 1)),
        "marker": "D",
    },
}

CV_KEY = "constant_velocity"
CV_LABEL = "Constant velocity"
CV_COLOR = "#CC79A7"


def rollout(model, history, num_tracks):
    with torch.no_grad():
        _, positions = model(
            history,
            torch.zeros(num_tracks, 2),
            torch.tensor([0, num_tracks]),
            n_predict=12,
        )
    return positions[-12:, 0]


def displacement_errors(prediction, truth):
    return torch.linalg.vector_norm(prediction - truth, dim=-1).numpy()


def evaluate_split(models, split):
    reader = trajnetplusplustools.Reader(
        str(ROOT / f"DATA_BLOCK/circle_classroom/{split}/circle.ndjson"),
        scene_type="paths",
    )
    errors = {key: [] for key in models}
    errors[CV_KEY] = []
    per_scene = []
    example = None

    for scene_id, paths in reader.scenes():
        xy = torch.tensor(
            trajnetplusplustools.Reader.paths_to_xy(paths), dtype=torch.float32
        )
        history, truth = xy[:8].clone(), xy[8:, 0]
        predictions = {
            key: rollout(model, history, xy.shape[1])
            for key, model in models.items()
        }
        cv = history[-1, 0] + torch.arange(1, 13)[:, None] * (
            history[-1, 0] - history[-2, 0]
        )
        predictions[CV_KEY] = cv

        for key, prediction in predictions.items():
            value = displacement_errors(prediction, truth)
            errors[key].append(value)
            per_scene.append(
                {
                    "split": split,
                    "scene_id": int(scene_id),
                    "model": key,
                    "ADE": float(value.mean()),
                    "FDE": float(value[-1]),
                }
            )

        if example is None:
            example = {
                "scene_id": int(scene_id),
                "history": history[:, 0].numpy(),
                "truth": truth.numpy(),
                "predictions": {
                    key: value.numpy() for key, value in predictions.items()
                },
            }

    summary = []
    for key, values in errors.items():
        stacked = np.stack(values)
        summary.append(
            {
                "split": split,
                "model": key,
                "ADE": float(stacked.mean()),
                "FDE": float(stacked[:, -1].mean()),
                "scenes": int(stacked.shape[0]),
            }
        )
    return summary, per_scene, example


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(summary, output_dir):
    lookup = {(row["split"], row["model"]): row for row in summary}
    keys = list(MODEL_SPECS) + [CV_KEY]
    labels = [MODEL_SPECS[key]["label"] for key in MODEL_SPECS] + [CV_LABEL]
    colors = [MODEL_SPECS[key]["color"] for key in MODEL_SPECS] + [CV_COLOR]
    x = np.arange(len(keys))
    width = 0.36

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
        for ax, metric in zip(axes, ("ADE", "FDE")):
            val = [lookup[("val", key)][metric] for key in keys]
            test = [lookup[("test", key)][metric] for key in keys]
            bars_val = ax.bar(
                x - width / 2,
                val,
                width,
                color=colors,
                edgecolor="black",
                linewidth=0.7,
                label="Validation",
            )
            bars_test = ax.bar(
                x + width / 2,
                test,
                width,
                color=colors,
                edgecolor="black",
                linewidth=0.7,
                hatch="///",
                alpha=0.78,
                label="Test",
            )
            for bars in (bars_val, bars_test):
                ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7, rotation=90)
            ax.set(
                title=f"{metric} comparison",
                ylabel=f"{metric} (scaled coordinate units)",
                xticks=x,
                xticklabels=labels,
                ylim=(0, max(max(val), max(test)) * 1.22),
            )
            ax.tick_params(axis="x", labelrotation=20)
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.legend()
        fig.suptitle("Trajectory prediction errors (12 scenes per split; one training seed)")
        fig.savefig(output_dir / "metrics_comparison.png", dpi=220)
        fig.savefig(output_dir / "metrics_comparison.svg")
        plt.close(fig)


def plot_trajectories(example, output_dir):
    history = example["history"]
    truth = example["truth"]
    predictions = example["predictions"]

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    ):
        fig, ax = plt.subplots(figsize=(7.2, 6.5), layout="constrained")
        ax.plot(
            history[:, 0], history[:, 1], "o-", color="#666666", label="History"
        )
        truth_line = np.vstack([history[-1], truth])
        ax.plot(
            truth_line[:, 0],
            truth_line[:, 1],
            "o-",
            color="#000000",
            linewidth=2.2,
            label="Truth",
        )
        for key, spec in MODEL_SPECS.items():
            values = np.vstack([history[-1], predictions[key]])
            ax.plot(
                values[:, 0],
                values[:, 1],
                color=spec["color"],
                linestyle=spec["linestyle"],
                marker=spec["marker"],
                linewidth=1.8,
                markersize=4,
                label=spec["label"],
            )
        cv = np.vstack([history[-1], predictions[CV_KEY]])
        ax.plot(
            cv[:, 0],
            cv[:, 1],
            color=CV_COLOR,
            linestyle="--",
            marker="x",
            linewidth=1.8,
            markersize=5,
            label=CV_LABEL,
        )
        ax.set(
            xlabel="x (scaled coordinate units)",
            ylabel="y (scaled coordinate units)",
            title=f"Test split, fixed first scene {example['scene_id']}",
        )
        ax.axis("equal")
        ax.grid(color="#E3E3E3", linewidth=0.7)
        ax.legend(loc="best")
        fig.savefig(output_dir / "trajectory_comparison.png", dpi=220)
        fig.savefig(output_dir / "trajectory_comparison.svg")
        plt.close(fig)


def main():
    torch.set_num_threads(2)
    output_dir = ROOT / "combined_comparison/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    models = {
        key: torch.load(spec["checkpoint"], map_location="cpu").model.eval()
        for key, spec in MODEL_SPECS.items()
    }

    summary = []
    per_scene = []
    test_example = None
    for split in ("val", "test"):
        split_summary, split_scenes, example = evaluate_split(models, split)
        summary.extend(split_summary)
        per_scene.extend(split_scenes)
        if split == "test":
            test_example = example

    write_csv(
        output_dir / "metrics_summary.csv",
        summary,
        ["split", "model", "ADE", "FDE", "scenes"],
    )
    write_csv(
        output_dir / "per_scene_metrics.csv",
        per_scene,
        ["split", "scene_id", "model", "ADE", "FDE"],
    )
    (output_dir / "combined_metrics.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "per_scene": per_scene,
                "parameter_counts": {
                    key: sum(parameter.numel() for parameter in model.parameters())
                    for key, model in models.items()
                },
                "units": "scaled coordinate units; metres only under unverified cm assumption",
                "protocol": "12 primary-pedestrian scenes per split; all 64 tracks recursively predicted from history only",
                "uncertainty": "not estimated: one training seed and overlapping windows",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_metrics(summary, output_dir)
    plot_trajectories(test_example, output_dir)
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(
            {
                "source_checkpoints": {
                    key: str(spec["checkpoint"].relative_to(ROOT))
                    for key, spec in MODEL_SPECS.items()
                },
                "source_data": "DATA_BLOCK/circle_classroom/{val,test}/circle.ndjson",
                "transformations": [
                    "history-only recursive mean rollout for all 64 tracks",
                    "Euclidean displacement per forecast step",
                    "ADE: mean over 12 steps and 12 scenes",
                    "FDE: mean final-step displacement over 12 scenes",
                ],
                "random_seed": 42,
                "figure_note": "General course-report figures; no publisher compliance claimed",
                "matplotlib": matplotlib.__version__,
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
