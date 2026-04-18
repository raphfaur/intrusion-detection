from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (6.6, 3.8),
            "figure.dpi": 160,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "font.size": 10,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _set_epoch_limits(ax: plt.Axes, epochs: list[int]) -> None:
    if len(epochs) <= 1:
        center = epochs[0]
        ax.set_xlim(center - 0.5, center + 0.5)
        return
    ax.set_xlim(min(epochs), max(epochs))


def export_gnn_training_plots(
    experiment_name: str,
    history: list[dict[str, Any]],
    report_dir: Path,
    figure_subdir: str,
) -> dict[str, str]:
    if not history:
        return {}

    _configure_plot_style()

    figures_dir = report_dir / figure_subdir
    figures_dir.mkdir(parents=True, exist_ok=True)

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train/loss"]) for row in history]
    train_f1 = [float(row["train/f1"]) for row in history]
    val_f1 = [float(row["val/f1"]) for row in history]

    paths: dict[str, str] = {}

    loss_path = figures_dir / f"{experiment_name}_loss.pdf"
    fig, ax = plt.subplots()
    ax.plot(epochs, train_loss, color="#1f77b4", linewidth=2.0, marker="o", markersize=3.5)
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    _set_epoch_limits(ax, epochs)
    fig.savefig(loss_path)
    plt.close(fig)
    paths["loss_curve"] = str(loss_path)

    f1_path = figures_dir / f"{experiment_name}_f1.pdf"
    fig, ax = plt.subplots()
    ax.plot(epochs, train_f1, color="#d62728", linewidth=2.0, marker="o", markersize=3.5, label="Train F1")
    ax.plot(epochs, val_f1, color="#2ca02c", linewidth=2.0, marker="s", markersize=3.5, label="Validation F1")
    ax.set_title("F1 Score Across Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    _set_epoch_limits(ax, epochs)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")
    fig.savefig(f1_path)
    plt.close(fig)
    paths["f1_curve"] = str(f1_path)

    return paths


def export_gnn_backbone_comparison_plot(
    entries: list[dict[str, Any]],
    report_dir: Path,
    figure_subdir: str,
    filename: str,
) -> str | None:
    gnn_entries = [
        entry
        for entry in entries
        if entry.get("model") == "gnn" and entry.get("architecture") and entry.get("test")
    ]
    if len(gnn_entries) < 2:
        return None

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in gnn_entries:
        key = (str(entry.get("dataset", "")), str(entry.get("scenario", "all")))
        grouped.setdefault(key, []).append(entry)

    valid_groups = [
        (key, sorted(group, key=lambda item: str(item.get("architecture", ""))))
        for key, group in grouped.items()
        if len(group) >= 2
    ]
    if not valid_groups:
        return None

    _configure_plot_style()
    figures_dir = report_dir / figure_subdir
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / filename

    ncols = len(valid_groups)
    fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 4.0), squeeze=False)
    metrics = ["accuracy", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "F1", "ROC-AUC"]
    palette = ["#4c78a8", "#f58518", "#54a24b"]

    for ax, ((dataset, scenario), group) in zip(axes[0], valid_groups, strict=True):
        architectures = [str(entry["architecture"]).upper() for entry in group]
        x = np.arange(len(architectures))
        width = 0.22
        for idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels, strict=True)):
            values = [float(entry["test"].get(metric, 0.0)) for entry in group]
            ax.bar(
                x + (idx - 1) * width,
                values,
                width=width,
                label=metric_label,
                color=palette[idx],
                alpha=0.85,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(architectures)
        ax.set_ylim(0.0, 1.0)
        title = f"{dataset} / {scenario}" if scenario != "all" else dataset
        ax.set_title(title)
        ax.set_ylabel("Test metric")

    axes[0][0].legend(loc="upper left")
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)


def export_pagerank_score_plot(
    experiment_name: str,
    diagnostics: dict[str, Any],
    threshold: float,
    report_dir: Path,
    figure_subdir: str,
) -> str | None:
    scores = diagnostics.get("test_scores", [])
    labels = diagnostics.get("test_labels", [])
    if not scores or not labels or len(scores) != len(labels):
        return None

    _configure_plot_style()
    figures_dir = report_dir / figure_subdir
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / f"{experiment_name}_scores.pdf"

    normal_scores = [float(score) for score, label in zip(scores, labels, strict=True) if int(label) == 0]
    attack_scores = [float(score) for score, label in zip(scores, labels, strict=True) if int(label) == 1]
    if not normal_scores and not attack_scores:
        return None

    fig, ax = plt.subplots()
    bins = np.linspace(0.0, 1.0, 25)
    if normal_scores:
        ax.hist(normal_scores, bins=bins, alpha=0.65, color="#4c78a8", label="Normal", density=True)
    if attack_scores:
        ax.hist(attack_scores, bins=bins, alpha=0.65, color="#e45756", label="Attack", density=True)
    ax.axvline(float(threshold), color="#222222", linestyle="--", linewidth=1.8, label="Decision threshold")
    ax.set_title("PageRank Anomaly Score Distribution")
    ax.set_xlabel("Anomaly rate")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="best")
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)
