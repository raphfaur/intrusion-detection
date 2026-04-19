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


def _metric_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _format_table_metric(value: Any, best: bool = False, decimals: int = 4) -> str:
    formatted = _format_metric(value, decimals=decimals)
    if best and formatted != "--":
        return rf"\textbf{{{formatted}}}"
    return formatted


def _group_best_indices(
    entries: list[dict[str, Any]],
    group_key: callable,
    metric_extractors: dict[str, callable],
) -> dict[tuple[int, str], bool]:
    highlighted: dict[tuple[int, str], bool] = {}
    grouped_indices: dict[Any, list[int]] = {}
    for index, entry in enumerate(entries):
        grouped_indices.setdefault(group_key(entry), []).append(index)

    for indices in grouped_indices.values():
        for metric_name, extractor in metric_extractors.items():
            scored = [
                (idx, _metric_float(extractor(entries[idx])))
                for idx in indices
            ]
            valid = [(idx, value) for idx, value in scored if value is not None]
            if not valid:
                continue
            best_value = max(value for _, value in valid)
            for idx, value in valid:
                if abs(value - best_value) <= 1e-12:
                    highlighted[(idx, metric_name)] = True
    return highlighted


def _append_group_separator(lines: list[str], entries: list[dict[str, Any]], index: int, group_key: callable) -> None:
    is_last = index == len(entries) - 1
    if not is_last and group_key(entries[index]) != group_key(entries[index + 1]):
        lines.append(r"\hline")


def _default_experiment_name(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    dataset = str(summary["dataset"])
    model = str(summary["model"])
    scenario = str(summary.get("scenario", "all"))
    architecture = str(summary.get("architecture", "")).strip()
    paradigm = str(summary.get("paradigm", "")).strip()
    node_profile = str(summary.get("node_feature_profile", "")).strip()
    edge_mode = str(summary.get("edge_weight_mode", "")).strip()
    parts = [dataset, scenario, model]
    if architecture:
        parts.append(architecture)
    if paradigm and paradigm != "supervised":
        parts.append(paradigm)
    if node_profile and node_profile != "full":
        parts.append(node_profile)
    if edge_mode and edge_mode != "weighted":
        parts.append(edge_mode)
    return _slugify("_".join(parts))


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
        "evaluation_protocol": summary.get("evaluation_protocol", "in_distribution"),
        "source_scenario": summary.get("source_scenario"),
        "target_scenario": summary.get("target_scenario"),
        "loaded_scenarios": summary.get("loaded_scenarios", []),
        "model": model_name,
        "paradigm": result.get("paradigm", summary.get("paradigm", "supervised")),
        "architecture": architecture,
        "model_label": model_label,
        "node_feature_profile": summary.get("node_feature_profile", "full"),
        "edge_weight_mode": summary.get("edge_weight_mode", "weighted"),
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
        if entry.get("model") == "pagerank":
            entry["paradigm"] = "anomaly"
        if not entry.get("paradigm"):
            entry["paradigm"] = "supervised" if entry.get("model") == "gnn" else "anomaly"
        entry.setdefault("node_feature_profile", "full")
        entry.setdefault("edge_weight_mode", "weighted")
        entry.setdefault("evaluation_protocol", "in_distribution")
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
    best = _group_best_indices(
        entries,
        group_key=lambda entry: (entry["dataset"], entry["scenario"]),
        metric_extractors={
            "accuracy": lambda entry: entry["test"].get("accuracy"),
            "f1": lambda entry: entry["test"].get("f1"),
            "precision": lambda entry: entry["test"].get("precision"),
            "recall": lambda entry: entry["test"].get("recall"),
            "roc_auc": lambda entry: entry["test"].get("roc_auc"),
        },
    )
    lines = [
        r"\subsection{Automated Results}",
        r"\begin{table}[H]",
        r"\centering",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lllrrrrr}",
        r"\hline",
        r"Dataset & Scenario & Model & Acc. & F1 & Prec. & Rec. & ROC-AUC \\",
        r"\hline",
    ]

    for index, entry in enumerate(entries):
        metrics = entry["test"]
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(entry["dataset"])),
                    _latex_escape(str(entry["scenario"])),
                    _latex_escape(str(entry.get("model_label", entry["model"]))),
                    _format_table_metric(metrics.get("accuracy"), best.get((index, "accuracy"), False)),
                    _format_table_metric(metrics.get("f1"), best.get((index, "f1"), False)),
                    _format_table_metric(metrics.get("precision"), best.get((index, "precision"), False)),
                    _format_table_metric(metrics.get("recall"), best.get((index, "recall"), False)),
                    _format_table_metric(metrics.get("roc_auc"), best.get((index, "roc_auc"), False)),
                ]
            )
            + r" \\"
        )
        _append_group_separator(lines, entries, index, lambda item: (item["dataset"], item["scenario"]))

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


def _pretty_feature_profile(value: str) -> str:
    mapping = {
        "full": "full",
        "embedding_only": "embedding-only",
        "frequency": "frequency",
        "transition": "transition",
        "structural": "structural",
        "behavioral": "behavioral",
        "temporal": "temporal",
    }
    return mapping.get(value, value)


def _pretty_paradigm(value: str) -> str:
    mapping = {
        "supervised": "supervised",
        "contrastive": "contrastive + probe",
        "anomaly": "anomaly detection",
    }
    return mapping.get(value, value)


def _render_feature_ablation_table(entries: list[dict[str, Any]]) -> list[str]:
    if not entries:
        return []

    best = _group_best_indices(
        entries,
        group_key=lambda entry: (entry["dataset"], entry["scenario"]),
        metric_extractors={
            "accuracy": lambda entry: entry["test"].get("accuracy"),
            "f1": lambda entry: entry["test"].get("f1"),
            "roc_auc": lambda entry: entry["test"].get("roc_auc"),
        },
    )
    lines = [
        r"\subsection{Feature Ablation}",
        r"\begin{table}[H]",
        r"\centering",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llllrrr}",
        r"\hline",
        r"Dataset & Scenario & Node features & Edge mode & Acc. & F1 & ROC-AUC \\",
        r"\hline",
    ]
    for index, entry in enumerate(entries):
        metrics = entry["test"]
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(entry["dataset"])),
                    _latex_escape(str(entry["scenario"])),
                    _latex_escape(_pretty_feature_profile(str(entry.get("node_feature_profile", "full")))),
                    _latex_escape(str(entry.get("edge_weight_mode", "weighted"))),
                    _format_table_metric(metrics.get("accuracy"), best.get((index, "accuracy"), False)),
                    _format_table_metric(metrics.get("f1"), best.get((index, "f1"), False)),
                    _format_table_metric(metrics.get("roc_auc"), best.get((index, "roc_auc"), False)),
                ]
            )
            + r" \\"
        )
        _append_group_separator(lines, entries, index, lambda item: (item["dataset"], item["scenario"]))
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"}",
            r"\caption{Ablation of node and edge features for the selected supervised GNN runs.}",
            r"\label{tab:feature-ablation}",
            r"\end{table}",
        ]
    )
    return lines


def _render_paradigm_table(entries: list[dict[str, Any]]) -> list[str]:
    if not entries:
        return []

    best = _group_best_indices(
        entries,
        group_key=lambda entry: (entry["dataset"], entry["scenario"]),
        metric_extractors={
            "accuracy": lambda entry: entry["test"].get("accuracy"),
            "f1": lambda entry: entry["test"].get("f1"),
            "roc_auc": lambda entry: entry["test"].get("roc_auc"),
        },
    )
    lines = [
        r"\subsection{Learning Paradigms}",
        r"\begin{table}[H]",
        r"\centering",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llllrrr}",
        r"\hline",
        r"Dataset & Scenario & Paradigm & Model & Acc. & F1 & ROC-AUC \\",
        r"\hline",
    ]
    for index, entry in enumerate(entries):
        metrics = entry["test"]
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(entry["dataset"])),
                    _latex_escape(str(entry["scenario"])),
                    _latex_escape(_pretty_paradigm(str(entry.get("paradigm", "supervised")))),
                    _latex_escape(str(entry.get("model_label", entry["model"]))),
                    _format_table_metric(metrics.get("accuracy"), best.get((index, "accuracy"), False)),
                    _format_table_metric(metrics.get("f1"), best.get((index, "f1"), False)),
                    _format_table_metric(metrics.get("roc_auc"), best.get((index, "roc_auc"), False)),
                ]
            )
            + r" \\"
        )
        _append_group_separator(lines, entries, index, lambda item: (item["dataset"], item["scenario"]))
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"}",
            r"\caption{Comparison of supervised and anomaly-detection training settings.}",
            r"\label{tab:learning-paradigms}",
            r"\end{table}",
        ]
    )
    return lines


def _pretty_model_label(entry: dict[str, Any]) -> str:
    model = str(entry.get("model", ""))
    if model == "gnn":
        return str(entry.get("model_label", "gnn"))
    if model == "sequence_logreg":
        return "seq:tfidf-logreg"
    if model == "sequence_gru":
        return "seq:gru"
    return str(entry.get("model_label", model))


def _render_sequence_table(entries: list[dict[str, Any]]) -> list[str]:
    if not entries:
        return []

    best = _group_best_indices(
        entries,
        group_key=lambda entry: (entry["dataset"], entry["scenario"]),
        metric_extractors={
            "accuracy": lambda entry: entry["test"].get("accuracy"),
            "f1": lambda entry: entry["test"].get("f1"),
            "roc_auc": lambda entry: entry["test"].get("roc_auc"),
        },
    )
    lines = [
        r"\subsection{Sequence Baselines}",
        r"\begin{table}[H]",
        r"\centering",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llllrrr}",
        r"\hline",
        r"Dataset & Scenario & Model & Paradigm & Acc. & F1 & ROC-AUC \\",
        r"\hline",
    ]
    for index, entry in enumerate(entries):
        metrics = entry["test"]
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(entry["dataset"])),
                    _latex_escape(str(entry["scenario"])),
                    _latex_escape(_pretty_model_label(entry)),
                    _latex_escape(_pretty_paradigm(str(entry.get("paradigm", "supervised")))),
                    _format_table_metric(metrics.get("accuracy"), best.get((index, "accuracy"), False)),
                    _format_table_metric(metrics.get("f1"), best.get((index, "f1"), False)),
                    _format_table_metric(metrics.get("roc_auc"), best.get((index, "roc_auc"), False)),
                ]
            )
            + r" \\"
        )
        _append_group_separator(lines, entries, index, lambda item: (item["dataset"], item["scenario"]))
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"}",
            r"\caption{Non-graph baselines trained directly on the syscall sequence.}",
            r"\label{tab:sequence-baselines}",
            r"\end{table}",
        ]
    )
    return lines


def _render_transfer_table(entries: list[dict[str, Any]]) -> list[str]:
    if not entries:
        return []

    best_target = _group_best_indices(
        entries,
        group_key=lambda entry: (entry.get("source_scenario"), entry.get("target_scenario")),
        metric_extractors={
            "target_accuracy": lambda entry: entry["test"].get("accuracy"),
            "target_f1": lambda entry: entry["test"].get("f1"),
            "target_roc_auc": lambda entry: entry["test"].get("roc_auc"),
        },
    )
    lines = [
        r"\subsection{Cross-Scenario Generalization}",
        r"\begin{table}[H]",
        r"\centering",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llllrrrrrr}",
        r"\hline",
        r"Train scenario & Test scenario & Model & Paradigm & Src. Acc. & Src. F1 & Src. AUC & Tgt. Acc. & Tgt. F1 & Tgt. AUC \\",
        r"\hline",
    ]
    for index, entry in enumerate(entries):
        source_metrics = entry.get("validation", {})
        metrics = entry["test"]
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(entry.get("source_scenario", "--"))),
                    _latex_escape(str(entry.get("target_scenario", "--"))),
                    _latex_escape(_pretty_model_label(entry)),
                    _latex_escape(_pretty_paradigm(str(entry.get("paradigm", "supervised")))),
                    _format_metric(source_metrics.get("accuracy")),
                    _format_metric(source_metrics.get("f1")),
                    _format_metric(source_metrics.get("roc_auc")),
                    _format_table_metric(metrics.get("accuracy"), best_target.get((index, "target_accuracy"), False)),
                    _format_table_metric(metrics.get("f1"), best_target.get((index, "target_f1"), False)),
                    _format_table_metric(metrics.get("roc_auc"), best_target.get((index, "target_roc_auc"), False)),
                ]
            )
            + r" \\"
        )
        _append_group_separator(
            lines,
            entries,
            index,
            lambda item: (item.get("source_scenario"), item.get("target_scenario")),
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"}",
            r"\caption{Transfer from one LID-DS attack scenario to another. The source columns report validation metrics on the source scenario, while the target columns report test metrics on the target scenario.}",
            r"\label{tab:cross-scenario-generalization}",
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
                r"\begin{figure}[H]",
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
                rf"\label{{fig:{_slugify(entry['experiment_name'])}-diagnostics}}",
                r"\end{figure}",
            ]
        )
    return lines


def _render_tex(
    benchmark_entries: list[dict[str, Any]],
    feature_entries: list[dict[str, Any]],
    paradigm_entries: list[dict[str, Any]],
    sequence_entries: list[dict[str, Any]],
    transfer_entries: list[dict[str, Any]],
    diagnostic_entries: list[dict[str, Any]],
    report_dir: Path,
) -> str:
    lines = [
        "% Auto-generated by intrusion_detection. Do not edit manually.",
        *(_render_result_macros(benchmark_entries + feature_entries + paradigm_entries + sequence_entries + transfer_entries)),
        "",
        *(_render_result_table(benchmark_entries)),
        "",
        *(_render_feature_ablation_table(feature_entries)),
        "",
        *(_render_paradigm_table(paradigm_entries)),
        "",
        *(_render_sequence_table(sequence_entries)),
        "",
        *(_render_transfer_table(transfer_entries)),
        "",
        *(_render_training_plots(diagnostic_entries, report_dir)),
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
    feature_entries = _ordered_entries(
        registry,
        list(getattr(cfg.report, "feature_ablation_experiments", [])),
    )
    paradigm_entries = _ordered_entries(
        registry,
        list(getattr(cfg.report, "paradigm_experiments", [])),
    )
    sequence_entries = _ordered_entries(
        registry,
        list(getattr(cfg.report, "sequence_experiments", [])),
    )
    transfer_entries = _ordered_entries(
        registry,
        list(getattr(cfg.report, "transfer_experiments", [])),
    )
    diagnostic_entries = _ordered_entries(
        registry,
        list(getattr(cfg.report, "diagnostic_plot_experiments", [])),
    )
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tex_path.write_text(
        _render_tex(
            ordered_entries,
            feature_entries,
            paradigm_entries,
            sequence_entries,
            transfer_entries,
            diagnostic_entries,
            report_dir,
        ),
        encoding="utf-8",
    )

    return {
        "experiment_name": experiment_name,
        "registry_path": str(registry_path),
        "tex_path": str(tex_path),
    }
