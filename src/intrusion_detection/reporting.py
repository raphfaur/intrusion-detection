from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from intrusion_detection.plotting import (
    export_gnn_training_plots,
    export_pagerank_score_plot,
)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "run"


def _camelize(value: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", value)
    return "".join(part[:1].upper() + part[1:] for part in parts) or "Run"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    escaped = value
    for source, target in replacements.items():
        escaped = escaped.replace(source, target)
    return escaped


def _format_metric(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "--"
        return f"{value:.{decimals}f}"
    return str(value)


def _default_experiment_name(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    dataset = str(summary["dataset"])
    model = str(summary["model"])
    scenario = str(summary.get("scenario", "all"))
    architecture = str(summary.get("architecture", "")).strip()
    if architecture:
        return _slugify(f"{dataset}_{scenario}_{model}_{architecture}")
    return _slugify(f"{dataset}_{scenario}_{model}")


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_registry_entry(
    experiment_name: str,
    payload: dict[str, Any],
    output_dir: Path,
    plot_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    summary = payload["summary"]
    result = payload["result"]
    model_name = str(summary["model"])
    architecture = summary.get("architecture")
    if model_name == "gnn" and not architecture:
        architecture = "wgcn_plus"
    model_label = f"{model_name}:{architecture}" if architecture else model_name
    return {
        "experiment_name": experiment_name,
        "dataset": summary["dataset"],
        "scenario": summary.get("scenario", "all"),
        "loaded_scenarios": summary.get("loaded_scenarios", []),
        "model": model_name,
        "architecture": architecture,
        "model_label": model_label,
        "output_dir": str(output_dir),
        "train_samples": summary["train_samples"],
        "val_samples": summary["val_samples"],
        "test_samples": summary["test_samples"],
        "history": result.get("history", []),
        "validation": result.get("validation", {}),
        "test": result.get("test", {}),
        "plots": plot_paths or {},
    }


def _normalize_registry(registry: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(registry)
    for entry in normalized.values():
        if entry.get("model") == "gnn" and not entry.get("architecture"):
            entry["architecture"] = "wgcn_plus"
            entry["model_label"] = "gnn:wgcn_plus"
    return normalized


def _ordered_entries(registry: dict[str, Any], included_experiments: list[str]) -> list[dict[str, Any]]:
    if included_experiments:
        return [registry[name] for name in included_experiments if name in registry]
    return [registry[key] for key in sorted(registry)]


def _render_result_macros(entries: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for entry in entries:
        prefix = "Result" + _camelize(entry["experiment_name"])
        metrics = entry["test"]
        lines.extend(
            [
                rf"\newcommand{{\{prefix}Accuracy}}{{{_format_metric(metrics.get('accuracy'))}}}",
                rf"\newcommand{{\{prefix}FOne}}{{{_format_metric(metrics.get('f1'))}}}",
                rf"\newcommand{{\{prefix}Precision}}{{{_format_metric(metrics.get('precision'))}}}",
                rf"\newcommand{{\{prefix}Recall}}{{{_format_metric(metrics.get('recall'))}}}",
                rf"\newcommand{{\{prefix}RocAuc}}{{{_format_metric(metrics.get('roc_auc'))}}}",
                rf"\newcommand{{\{prefix}Support}}{{{_format_metric(metrics.get('support'))}}}",
            ]
        )
    return lines


def _render_result_table(entries: list[dict[str, Any]]) -> list[str]:
    lines = [
        r"\subsection{Automated Results}",
        r"\begin{table}[t]",
        r"\centering",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lllrrrrr}",
        r"\hline",
        r"Dataset & Scenario & Model & Acc. & F1 & Prec. & Rec. & ROC-AUC \\",
        r"\hline",
    ]

    for entry in entries:
        metrics = entry["test"]
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(entry["dataset"])),
                    _latex_escape(str(entry["scenario"])),
                    _latex_escape(str(entry.get("model_label", entry["model"]))),
                    _format_metric(metrics.get("accuracy")),
                    _format_metric(metrics.get("f1")),
                    _format_metric(metrics.get("precision")),
                    _format_metric(metrics.get("recall")),
                    _format_metric(metrics.get("roc_auc")),
                ]
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"}",
            r"\caption{Test-set metrics exported automatically from the training pipeline.}",
            r"\label{tab:automated-results}",
            r"\end{table}",
        ]
    )
    return lines


def _relative_posix_path(path: str, base_dir: Path) -> str:
    return Path(path).resolve().relative_to(base_dir.resolve()).as_posix()


def _render_training_plots(entries: list[dict[str, Any]], report_dir: Path) -> list[str]:
    lines: list[str] = []
    for entry in entries:
        plots = entry.get("plots", {})
        loss_curve = plots.get("loss_curve")
        f1_curve = plots.get("f1_curve")
        pagerank_scores = plots.get("pagerank_scores")
        if not loss_curve and not f1_curve and not pagerank_scores:
            continue

        lines.extend(
            [
                rf"\subsection{{Diagnostics: {_latex_escape(entry['experiment_name'])}}}",
                r"\begin{figure}[t]",
                r"\centering",
            ]
        )
        if loss_curve:
            lines.append(
                rf"\includegraphics[width=0.48\linewidth]{{{_relative_posix_path(loss_curve, report_dir)}}}"
            )
        if f1_curve:
            lines.append(
                rf"\includegraphics[width=0.48\linewidth]{{{_relative_posix_path(f1_curve, report_dir)}}}"
            )
        if pagerank_scores:
            lines.append(
                rf"\includegraphics[width=0.7\linewidth]{{{_relative_posix_path(pagerank_scores, report_dir)}}}"
            )
        lines.extend(
            [
                (
                    rf"\caption{{Training diagnostics for \texttt{{{_latex_escape(entry['experiment_name'])}}}: "
                    r"cross-entropy loss and F1 score as a function of epoch.}"
                    if not pagerank_scores
                    else rf"\caption{{Diagnostic plot for \texttt{{{_latex_escape(entry['experiment_name'])}}}: "
                    r"distribution of test anomaly scores with the decision threshold.}"
                ),
                r"\end{figure}",
            ]
        )
    return lines


def _render_tex(entries: list[dict[str, Any]], report_dir: Path) -> str:
    lines = [
        "% Auto-generated by intrusion_detection. Do not edit manually.",
        *(_render_result_macros(entries)),
        "",
        *(_render_result_table(entries)),
        "",
        *(_render_training_plots(entries, report_dir)),
        "",
    ]
    return "\n".join(lines)


def export_results_to_latex(
    cfg: Any,
    payload: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    report_dir = Path(cfg.report.output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    registry_path = report_dir / cfg.report.registry_filename
    tex_path = report_dir / cfg.report.tex_filename

    registry = _normalize_registry(_load_registry(registry_path))
    experiment_name = (
        cfg.report.experiment_name
        if str(cfg.report.experiment_name).strip().lower() != "auto"
        else _default_experiment_name(payload)
    )

    plot_paths: dict[str, str] = {}
    selected_plot_experiments = list(getattr(cfg.report, "diagnostic_plot_experiments", []))
    if cfg.report.plots_enabled and experiment_name in selected_plot_experiments and payload["summary"]["model"] == "gnn":
        plot_paths = export_gnn_training_plots(
            experiment_name=experiment_name,
            history=payload["result"].get("history", []),
            report_dir=report_dir,
            figure_subdir=str(cfg.report.figures_subdir),
        )
    if cfg.report.plots_enabled and experiment_name in selected_plot_experiments and payload["summary"]["model"] == "pagerank":
        pagerank_plot = export_pagerank_score_plot(
            experiment_name=experiment_name,
            diagnostics=payload["result"].get("diagnostics", {}),
            threshold=float(payload["result"].get("thresholds", {}).get("anomaly_rate_threshold", 0.0)),
            report_dir=report_dir,
            figure_subdir=str(cfg.report.figures_subdir),
        )
        if pagerank_plot:
            plot_paths["pagerank_scores"] = pagerank_plot

    registry[experiment_name] = _build_registry_entry(
        experiment_name=experiment_name,
        payload=payload,
        output_dir=output_dir,
        plot_paths=plot_paths,
    )

    ordered_entries = _ordered_entries(
        registry,
        list(getattr(cfg.report, "included_experiments", [])),
    )
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tex_path.write_text(_render_tex(ordered_entries, report_dir), encoding="utf-8")

    return {
        "experiment_name": experiment_name,
        "registry_path": str(registry_path),
        "tex_path": str(tex_path),
    }
