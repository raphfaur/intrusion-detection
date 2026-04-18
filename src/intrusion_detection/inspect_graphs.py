from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import hydra
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from intrusion_detection.data import build_networkx_graph, load_trace_samples

matplotlib.use("Agg")

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")


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


def _format_metric(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        return "--"
    return f"{value:.2f}"


def _dataset_label(dataset_name: str, scenario: str) -> str:
    if dataset_name == "adfa_ld":
        return "ADFA-LD"
    return f"LID-DS ({scenario})"


def _filter_samples_for_pipeline(spec: Any, samples: list[Any]) -> list[Any]:
    split_strategy = getattr(spec, "split_strategy", "random")
    if split_strategy == "original_test_resplit":
        source_split = getattr(spec, "resplit_source_split", "test")
        return [sample for sample in samples if sample.metadata.get("split") == source_split]
    return samples


def _inspect_sample(sample: Any, dataset_label: str) -> tuple[dict[str, Any], nx.DiGraph]:
    graph = build_networkx_graph(sample)
    num_nodes = int(graph.number_of_nodes())
    num_edges = int(graph.number_of_edges())
    self_loops = int(nx.number_of_selfloops(graph))
    possible_edges = max(num_nodes * num_nodes, 1)
    density = float(num_edges / possible_edges)
    mean_edge_weight = (
        float(np.mean([attrs.get("weight", 1.0) for _, _, attrs in graph.edges(data=True)]))
        if num_edges > 0
        else 0.0
    )

    row = {
        "dataset": sample.dataset_name,
        "dataset_label": dataset_label,
        "scenario": sample.metadata.get("scenario", "all"),
        "label": int(sample.label),
        "label_name": "attack" if int(sample.label) == 1 else "normal",
        "file_path": sample.file_path,
        "sequence_length": int(len(sample.sequence)),
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,
        "self_loops": self_loops,
        "mean_edge_weight": mean_edge_weight,
    }
    return row, graph


def _summarize_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for dataset_label, group in frame.groupby("dataset_label", sort=False):
        attack_ratio = float(group["label"].mean()) if len(group) > 0 else 0.0
        summaries.append(
            {
                "dataset_label": dataset_label,
                "num_graphs": int(len(group)),
                "attack_ratio": attack_ratio,
                "sequence_length_mean": float(group["sequence_length"].mean()),
                "nodes_mean": float(group["num_nodes"].mean()),
                "nodes_median": float(group["num_nodes"].median()),
                "edges_mean": float(group["num_edges"].mean()),
                "edges_median": float(group["num_edges"].median()),
                "density_mean": float(group["density"].mean()),
                "density_std": float(group["density"].std(ddof=0)),
                "mean_edge_weight": float(group["mean_edge_weight"].mean()),
            }
        )
    return summaries


def _configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.frameon": False,
            "font.size": 9,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _plot_nodes_vs_edges(frame: pd.DataFrame, output_path: Path) -> None:
    dataset_labels = frame["dataset_label"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(dataset_labels), figsize=(5.0 * len(dataset_labels), 3.8), squeeze=False)
    color_map = {"normal": "#1f77b4", "attack": "#d62728"}

    for axis, dataset_label in zip(axes[0], dataset_labels, strict=True):
        subset = frame[frame["dataset_label"] == dataset_label]
        for label_name in ["normal", "attack"]:
            label_subset = subset[subset["label_name"] == label_name]
            if label_subset.empty:
                continue
            axis.scatter(
                label_subset["num_nodes"],
                label_subset["num_edges"],
                s=12,
                alpha=0.55,
                c=color_map[label_name],
                label=label_name,
            )
        axis.set_title(dataset_label)
        axis.set_xlabel("Number of nodes")
        axis.set_ylabel("Number of edges")
        axis.legend(loc="best")

    fig.savefig(output_path)
    plt.close(fig)


def _representative_graph_records(records: list[dict[str, Any]], frame: pd.DataFrame) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for dataset_label in frame["dataset_label"].drop_duplicates().tolist():
        dataset_frame = frame[frame["dataset_label"] == dataset_label]
        for label_name in ["normal", "attack"]:
            subset = dataset_frame[dataset_frame["label_name"] == label_name]
            if subset.empty:
                continue
            target_nodes = float(subset["num_nodes"].mean())
            target_edges = float(subset["num_edges"].mean())
            target_density = float(subset["density"].mean())
            ranked = subset.assign(
                representative_score=(
                    ((subset["num_nodes"] - target_nodes) / max(target_nodes, 1.0)) ** 2
                    + ((subset["num_edges"] - target_edges) / max(target_edges, 1.0)) ** 2
                    + ((subset["density"] - target_density) / max(target_density, 1e-6)) ** 2
                )
            ).sort_values("representative_score", kind="stable")
            chosen_path = str(ranked.iloc[0]["file_path"])
            for record in records:
                if record["row"]["file_path"] == chosen_path:
                    selected.append(record)
                    break
    return selected


def _top_weighted_nodes(graph: nx.DiGraph, max_nodes: int = 10) -> list[Any]:
    weighted_degree = {
        node: float(graph.in_degree(node, weight="weight") + graph.out_degree(node, weight="weight"))
        for node in graph.nodes()
    }
    return [
        node for node, _ in sorted(weighted_degree.items(), key=lambda item: item[1], reverse=True)[:max_nodes]
    ]


def _plot_representative_graphs(records: list[dict[str, Any]], output_path: Path) -> list[dict[str, Any]]:
    if not records:
        return []

    ncols = 2
    nrows = math.ceil(len(records) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.6, 3.4 * nrows))
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)
    flat_axes = list(axes_array.flat)
    selected_metadata: list[dict[str, Any]] = []
    for axis in flat_axes[len(records):]:
        axis.axis("off")

    for axis, record in zip(flat_axes, records, strict=False):
        row = record["row"]
        graph = record["graph"]
        if graph.number_of_nodes() == 0:
            axis.axis("off")
            continue

        selected_nodes = _top_weighted_nodes(graph, max_nodes=10)
        matrix = nx.to_numpy_array(graph, nodelist=selected_nodes, weight="weight", dtype=float)
        if matrix.size > 0 and np.max(matrix) > 0:
            matrix = matrix / np.max(matrix)

        image = axis.imshow(matrix, cmap="magma", vmin=0.0, vmax=1.0)
        labels = [str(node) for node in selected_nodes]
        axis.set_xticks(range(len(labels)))
        axis.set_yticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=55, ha="right", fontsize=6)
        axis.set_yticklabels(labels, fontsize=6)
        axis.set_title(f"{row['dataset_label']} / {row['label_name']}", fontsize=10)
        axis.set_xlabel("Destination syscall")
        axis.set_ylabel("Source syscall")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        selected_metadata.append(
            {
                "dataset_label": row["dataset_label"],
                "label_name": row["label_name"],
                "file_path": row["file_path"],
                "num_nodes": row["num_nodes"],
                "num_edges": row["num_edges"],
                "density": row["density"],
                "shown_nodes": labels,
            }
        )

    fig.savefig(output_path)
    plt.close(fig)
    return selected_metadata


def _build_interpretation(summary_rows: list[dict[str, Any]]) -> str:
    if not summary_rows:
        return ""

    fragments: list[str] = []
    for row in summary_rows:
        fragments.append(
            f"{row['dataset_label']} contains {row['num_graphs']} graphs with "
            f"{row['nodes_mean']:.1f} nodes and {row['edges_mean']:.1f} edges on average "
            f"(directed density {row['density_mean']:.3f})"
        )

    return (
        "The generated syscall graphs remain in a small-to-medium regime while staying fairly dense: "
        + "; ".join(fragments)
        + ". This profile makes local message-passing models appropriate. We therefore compare "
        "a plain GCN baseline, GraphSAGE for more robust neighborhood aggregation, GAT to test "
        "whether attention over heterogeneous syscall transitions improves discrimination, and our "
        "WGCN+ variant that combines weighted GCN propagation, learned syscall embeddings, structural "
        "node features, layer normalization, and hybrid graph pooling."
    )


def _render_tex(
    summary_rows: list[dict[str, Any]],
    interpretation: str,
    report_dir: Path,
    figures: dict[str, Path],
    exemplar_metadata: list[dict[str, Any]],
) -> str:
    lines = [
        "% Auto-generated graph inspection block.",
        r"\subsection{Graph Inspection}",
        _latex_escape(interpretation),
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Dataset & Graphs & Attack \% & Mean $|V|$ & Mean $|E|$ & Mean density & Mean seq. len. \\",
        r"\hline",
    ]

    for row in summary_rows:
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["dataset_label"]),
                    str(row["num_graphs"]),
                    _format_metric(100.0 * row["attack_ratio"]),
                    _format_metric(row["nodes_mean"]),
                    _format_metric(row["edges_mean"]),
                    _format_metric(row["density_mean"]),
                    _format_metric(row["sequence_length_mean"]),
                ]
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"}",
            r"\caption{Graph-level inspection of the datasets used by the training pipeline. For LID-DS, the inspection follows the same selection as the experiments and therefore focuses on the original \texttt{test} traces before the local stratified re-split.}",
            r"\label{tab:graph-inspection}",
            r"\end{table}",
            "",
            r"\begin{figure}[t]",
            r"\centering",
            rf"\includegraphics[width=0.72\linewidth]{{{figures['nodes_edges'].resolve().relative_to(report_dir.resolve()).as_posix()}}}",
            r"\caption{Number of nodes versus number of edges for the generated syscall graphs, colored by class.}",
            r"\label{fig:graph-inspection}",
            r"\end{figure}",
            "",
        ]
    )
    if exemplar_metadata:
        lines.extend(
            [
                r"\begin{figure}[t]",
                r"\centering",
                rf"\includegraphics[width=\linewidth]{{{figures['representative_graphs'].resolve().relative_to(report_dir.resolve()).as_posix()}}}",
                r"\caption{Representative transition matrices selected near the mean graph profile for each dataset/scenario and class. Each panel shows the normalized weighted adjacency restricted to the most central syscalls of the selected trace. On ADFA-LD, the axis labels are the raw syscall identifiers provided by the dataset; on LID-DS, they are syscall names parsed from the \texttt{.sc} traces.}",
                r"\label{fig:representative-graphs}",
                r"\end{figure}",
                "",
            ]
        )
    return "\n".join(lines)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="inspection")
def main(cfg: DictConfig) -> None:
    report_dir = Path(cfg.report.output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = report_dir / cfg.report.figures_subdir
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    rows: list[dict[str, Any]] = []
    graph_records: list[dict[str, Any]] = []
    for spec in cfg.datasets:
        samples = load_trace_samples(spec)
        filtered_samples = _filter_samples_for_pipeline(spec, samples)
        dataset_label = _dataset_label(spec.name, getattr(spec, "scenario", "all"))
        print(
            f"[inspect] dataset={spec.name} scenario={getattr(spec, 'scenario', 'all')} "
            f"loaded={len(samples)} kept={len(filtered_samples)}"
        )
        for sample in filtered_samples:
            row, graph = _inspect_sample(sample, dataset_label)
            rows.append(row)
            graph_records.append({"row": row, "graph": graph})

    frame = pd.DataFrame(rows)
    summary_rows = _summarize_frame(frame)
    interpretation = _build_interpretation(summary_rows)

    figures = {
        "nodes_edges": figures_dir / cfg.report.nodes_edges_figure,
        "representative_graphs": figures_dir / cfg.report.representative_graphs_figure,
    }

    _configure_plot_style()
    _plot_nodes_vs_edges(frame, figures["nodes_edges"])
    exemplar_records = _representative_graph_records(graph_records, frame)
    exemplar_metadata = _plot_representative_graphs(
        exemplar_records,
        figures["representative_graphs"],
    )

    json_payload = {
        "summary": summary_rows,
        "interpretation": interpretation,
        "representative_graphs": exemplar_metadata,
        "rows": frame.to_dict(orient="records"),
        "figures": {key: str(path) for key, path in figures.items()},
    }
    json_path = report_dir / cfg.report.json_filename
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    tex_path = report_dir / cfg.report.tex_filename
    tex_path.write_text(
        _render_tex(summary_rows, interpretation, report_dir, figures, exemplar_metadata),
        encoding="utf-8",
    )

    run_summary = {
        "report_json": str(json_path),
        "report_tex": str(tex_path),
        "figure_paths": {key: str(path) for key, path in figures.items()},
        "output_dir": str(output_dir),
    }
    (output_dir / "inspection_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
