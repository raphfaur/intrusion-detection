from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from io import TextIOWrapper
import json
from pathlib import Path
import re
from typing import Any
from zipfile import ZipFile

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

SyscallToken = int | str
UNKNOWN_SYSCALL_TOKEN = "<UNK>"

ADFA_NODE_FEATURE_NAMES = [
    "freq",
    "in_deg",
    "out_deg",
    "in_wdeg",
    "out_wdeg",
    "pagerank",
    "self_loop",
]

LID_NODE_FEATURE_NAMES = [
    "freq",
    "count",
    "unique_proc",
    "error_rate",
    "mean_res",
    "std_res",
    "mean_gap",
    "std_gap",
    "in_deg",
    "out_deg",
    "in_wdeg",
    "out_wdeg",
    "pagerank",
    "self_loop",
]


@dataclass(slots=True)
class TraceSample:
    file_path: str
    label: int
    sequence: list[SyscallToken]
    dataset_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_df: pd.DataFrame | None = None


@dataclass(slots=True)
class GraphDataBundle:
    train_dataset: list[Data]
    val_dataset: list[Data]
    test_dataset: list[Data]
    vocab: dict[SyscallToken, int]
    feature_names: list[str]

    @property
    def num_syscalls(self) -> int:
        return len(self.vocab)

    @property
    def num_node_features(self) -> int:
        return len(self.feature_names)


class ADFALDLoader:
    def __init__(self, root_dir: str | Path) -> None:
        self.root = Path(root_dir)
        self.train_normal_dir = self.root / "Training_Data_Master"
        self.val_normal_dir = self.root / "Validation_Data_Master"
        self.attack_dir = self.root / "Attack_Data_Master"

    @staticmethod
    def read_trace_file(file_path: Path) -> list[int]:
        content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            return []
        return [int(token) for token in content.split()]

    def _iter_normal_samples(self, directory: Path, split_hint: str) -> list[TraceSample]:
        if not directory.exists():
            raise FileNotFoundError(f"Missing directory: {directory}")

        samples: list[TraceSample] = []
        for file_path in sorted(directory.glob("*")):
            if not file_path.is_file():
                continue
            sequence = self.read_trace_file(file_path)
            if len(sequence) < 2:
                continue
            samples.append(
                TraceSample(
                    file_path=str(file_path),
                    label=0,
                    sequence=sequence,
                    dataset_name="adfa_ld",
                    metadata={
                        "attack_type": "normal",
                        "split_hint": split_hint,
                    },
                )
            )
        return samples

    def load_all(self) -> list[TraceSample]:
        samples = []
        samples.extend(self._iter_normal_samples(self.train_normal_dir, "train"))
        samples.extend(self._iter_normal_samples(self.val_normal_dir, "validation"))

        if not self.attack_dir.exists():
            raise FileNotFoundError(f"Missing directory: {self.attack_dir}")

        for attack_subdir in sorted(self.attack_dir.iterdir()):
            if not attack_subdir.is_dir():
                continue
            attack_name = attack_subdir.name
            for file_path in sorted(attack_subdir.glob("*")):
                if not file_path.is_file():
                    continue
                sequence = self.read_trace_file(file_path)
                if len(sequence) < 2:
                    continue
                samples.append(
                    TraceSample(
                        file_path=str(file_path),
                        label=1,
                        sequence=sequence,
                        dataset_name="adfa_ld",
                        metadata={
                            "attack_type": attack_name,
                            "split_hint": "attack",
                        },
                    )
                )
        return samples


class LIDDSLoader:
    sc_line_re = re.compile(
        r"^(?P<timestamp>\d+)\s+"
        r"(?P<field2>\S+)\s+"
        r"(?P<field3>\S+)\s+"
        r"(?P<procname>\S+)\s+"
        r"(?P<field5>\S+)\s+"
        r"(?P<syscall>\S+)\s+"
        r"(?P<direction>[<>])\s*"
        r"(?P<rest>.*)$"
    )
    res_re = re.compile(r"\bres=(-?\d+)\b")

    def __init__(
        self,
        root_dir: str | Path,
        scenario: str = "all",
        direction_filter: str | None = ">",
        keep_only_successful_parsed_lines: bool = True,
    ) -> None:
        self.root = Path(root_dir)
        self.scenario = scenario
        self.direction_filter = direction_filter
        self.keep_only_successful_parsed_lines = keep_only_successful_parsed_lines

    @staticmethod
    def _normalize_syscall_value(value: Any) -> SyscallToken | None:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return int(float(text))
        except Exception:
            return text

    def _resolve_roots(self) -> list[Path]:
        if not self.root.exists():
            raise FileNotFoundError(f"Missing dataset directory: {self.root}")

        if self.scenario != "all":
            scenario_root = self.root / self.scenario
            if scenario_root.exists():
                return [scenario_root]
            if self.root.name == self.scenario:
                return [self.root]
            available_scenarios = sorted(
                path.name for path in self.root.iterdir() if path.is_dir() and path.name != "__MACOSX"
            )
            raise FileNotFoundError(
                f"Requested LID-DS scenario '{self.scenario}' not found under {self.root}. "
                f"Available scenarios: {available_scenarios}"
            )

        child_roots = sorted(
            path for path in self.root.iterdir() if path.is_dir() and path.name != "__MACOSX"
        )
        if child_roots:
            return child_roots
        return [self.root]

    def _collect_zip_files(self) -> list[Path]:
        files: list[Path] = []
        for root in self._resolve_roots():
            files.extend(
                sorted(
                    file_path
                    for file_path in root.rglob("*.zip")
                    if file_path.is_file() and "__MACOSX" not in file_path.parts
                )
            )
        return files

    def _infer_split_and_scenario(self, zip_path: Path) -> tuple[str | None, str]:
        parts_lower = [part.lower() for part in zip_path.parts]
        split_aliases = {
            "train": "train",
            "training": "train",
            "validation": "validation",
            "val": "validation",
            "test": "test",
            "testing": "test",
        }

        split = None
        split_idx = None
        for index, part in enumerate(parts_lower):
            if part in split_aliases:
                split = split_aliases[part]
                split_idx = index
                break

        scenario = "unknown"
        if split_idx is not None and split_idx > 0:
            scenario = zip_path.parts[split_idx - 1]

        return split, scenario

    def _parse_sc_line(self, line: str) -> dict[str, Any] | None:
        line = line.strip()
        if not line:
            return None

        match = self.sc_line_re.match(line)
        if not match:
            return None

        data = match.groupdict()
        rest = data["rest"]
        res_match = self.res_re.search(rest)
        res_value = int(res_match.group(1)) if res_match else np.nan
        syscall = self._normalize_syscall_value(data["syscall"])
        if syscall is None:
            return None

        return {
            "timestamp": int(data["timestamp"]),
            "procname": data["procname"],
            "syscall": syscall,
            "direction": data["direction"],
            "res": res_value,
            "raw_rest": rest,
        }

    def _load_sample_from_zip(self, zip_path: Path) -> tuple[dict[str, Any], pd.DataFrame, int]:
        with ZipFile(zip_path, "r") as archive:
            names = archive.namelist()
            json_files = [name for name in names if name.endswith(".json")]
            sc_files = [name for name in names if name.endswith(".sc")]

            if not json_files:
                raise ValueError(f"No .json found in {zip_path}")
            if not sc_files:
                raise ValueError(f"No .sc found in {zip_path}")

            with archive.open(json_files[0]) as handle:
                meta = json.loads(handle.read().decode("utf-8", errors="ignore"))

            with archive.open(sc_files[0]) as handle:
                lines = TextIOWrapper(handle, encoding="utf-8", errors="ignore").read().splitlines()

        events = [self._parse_sc_line(line) for line in lines]
        if self.keep_only_successful_parsed_lines:
            events = [event for event in events if event is not None]

        events_df = pd.DataFrame(events)
        if len(events_df) == 0:
            raise ValueError(f"No parsed syscall lines in {zip_path}")

        if self.direction_filter is not None:
            events_df = events_df[events_df["direction"] == self.direction_filter].copy()
        if len(events_df) == 0:
            raise ValueError(f"No events left after direction filtering in {zip_path}")

        events_df = events_df.sort_values("timestamp").reset_index(drop=True)
        label = int(bool(meta.get("exploit", False)))
        return meta, events_df, label

    def _parse_sc_lines(self, lines: list[str]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for line in lines:
            parsed = self._parse_sc_line(line)
            if parsed is not None:
                rows.append(parsed)
        return pd.DataFrame(rows)

    def _read_sc_zip(self, file_path: Path) -> pd.DataFrame:
        with ZipFile(file_path) as archive:
            members = [name for name in archive.namelist() if name.endswith(".sc")]
            if not members:
                raise ValueError(f"No .sc trace found in archive: {file_path}")
            with archive.open(members[0], "r") as handle:
                lines = TextIOWrapper(handle, encoding="utf-8", errors="ignore").read().splitlines()
        return self._parse_sc_lines(lines)

    def _read_sc_file(self, file_path: Path) -> pd.DataFrame:
        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return self._parse_sc_lines(lines)

    def load_all(self) -> list[TraceSample]:
        samples: list[TraceSample] = []
        for file_path in self._collect_zip_files():
            split, scenario = self._infer_split_and_scenario(file_path)
            if split is None:
                continue

            try:
                meta, trace_df, label = self._load_sample_from_zip(file_path)
            except Exception as error:
                print(f"[WARN] skipping {file_path}: {error}")
                continue

            if len(trace_df) < 2:
                continue

            samples.append(
                TraceSample(
                    file_path=str(file_path),
                    label=label,
                    sequence=trace_df["syscall"].tolist(),
                    dataset_name="lid_ds",
                    metadata={
                        "scenario": scenario,
                        "split": split,
                    },
                    trace_df=trace_df,
                )
            )
        return samples


def load_trace_samples(dataset_cfg: Any) -> list[TraceSample]:
    if dataset_cfg.name == "adfa_ld":
        return ADFALDLoader(dataset_cfg.root_dir).load_all()
    if dataset_cfg.name == "lid_ds":
        return LIDDSLoader(
            dataset_cfg.root_dir,
            scenario=getattr(dataset_cfg, "scenario", "all"),
            direction_filter=getattr(dataset_cfg, "direction_filter", ">"),
            keep_only_successful_parsed_lines=getattr(
                dataset_cfg,
                "keep_only_successful_parsed_lines",
                True,
            ),
        ).load_all()
    raise ValueError(f"Unsupported dataset: {dataset_cfg.name}")


def build_syscall_vocab(samples: list[TraceSample]) -> dict[SyscallToken, int]:
    tokens = sorted(
        {token for sample in samples for token in sample.sequence},
        key=lambda item: (isinstance(item, str), str(item)),
    )
    vocab: dict[SyscallToken, int] = {UNKNOWN_SYSCALL_TOKEN: 0}
    vocab.update({token: index for index, token in enumerate(tokens, start=1)})
    return vocab


def split_samples(
    samples: list[TraceSample],
    test_size: float,
    val_size: float,
    random_state: int,
    split_strategy: str = "random",
    source_split: str = "test",
) -> tuple[list[TraceSample], list[TraceSample], list[TraceSample]]:
    if split_strategy == "predefined":
        train_samples = [sample for sample in samples if sample.metadata.get("split") == "train"]
        val_samples = [sample for sample in samples if sample.metadata.get("split") == "validation"]
        test_samples = [sample for sample in samples if sample.metadata.get("split") == "test"]
        if not train_samples or not val_samples or not test_samples:
            raise ValueError(
                "Predefined splitting requires non-empty train/validation/test samples."
            )
        return train_samples, val_samples, test_samples

    if split_strategy == "original_test_resplit":
        samples = [sample for sample in samples if sample.metadata.get("split") == source_split]
        if not samples:
            raise ValueError(
                f"No samples found with original split '{source_split}' for re-splitting."
            )

    labels = [sample.label for sample in samples]
    indices = np.arange(len(samples))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    train_val_labels = [labels[index] for index in train_val_idx]
    relative_val_size = val_size / (1.0 - test_size)

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val_labels,
    )

    train_samples = [samples[index] for index in train_idx]
    val_samples = [samples[index] for index in val_idx]
    test_samples = [samples[index] for index in test_idx]
    return train_samples, val_samples, test_samples


def _build_adfa_graph(sequence: list[int]) -> nx.DiGraph:
    graph = nx.DiGraph()
    counts = Counter(sequence)
    total_calls = len(sequence)

    for syscall, count in counts.items():
        graph.add_node(
            syscall,
            count=float(count),
            freq=float(count / total_calls),
        )

    for index in range(len(sequence) - 1):
        source = sequence[index]
        target = sequence[index + 1]
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += 1.0
        else:
            graph.add_edge(source, target, weight=1.0)

    return _add_structural_features(graph)


def _build_lid_graph(trace_df: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    total_events = len(trace_df)

    for syscall, group in trace_df.groupby("syscall"):
        timestamps = group["timestamp"].to_numpy(dtype=float)
        if len(timestamps) > 1:
            inter_arrivals = np.diff(np.sort(timestamps))
            mean_gap = float(np.mean(inter_arrivals))
            std_gap = float(np.std(inter_arrivals))
        else:
            mean_gap = 0.0
            std_gap = 0.0

        graph.add_node(
            syscall,
            count=float(len(group)),
            freq=float(len(group) / total_events),
            unique_proc=float(group["procname"].nunique()),
            error_rate=0.0,
            mean_res=0.0,
            std_res=0.0,
            mean_gap=mean_gap,
            std_gap=std_gap,
        )

        exit_events = group[group["res"].notna()]
        if len(exit_events) > 0:
            graph.nodes[syscall]["error_rate"] = float((exit_events["res"] < 0).mean())
            graph.nodes[syscall]["mean_res"] = float(exit_events["res"].mean())
            graph.nodes[syscall]["std_res"] = float(exit_events["res"].std()) if len(exit_events) > 1 else 0.0

    sequence = trace_df["syscall"].tolist()
    for index in range(len(sequence) - 1):
        source = sequence[index]
        target = sequence[index + 1]
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += 1.0
        else:
            graph.add_edge(source, target, weight=1.0)

    return _add_structural_features(graph)


def _add_structural_features(graph: nx.DiGraph) -> nx.DiGraph:
    if graph.number_of_nodes() == 0:
        return graph

    pagerank = (
        nx.pagerank(graph, weight="weight")
        if graph.number_of_edges() > 0
        else {node: 0.0 for node in graph.nodes()}
    )

    for node in graph.nodes():
        graph.nodes[node]["in_deg"] = float(graph.in_degree(node))
        graph.nodes[node]["out_deg"] = float(graph.out_degree(node))
        graph.nodes[node]["in_wdeg"] = float(graph.in_degree(node, weight="weight"))
        graph.nodes[node]["out_wdeg"] = float(graph.out_degree(node, weight="weight"))
        graph.nodes[node]["pagerank"] = float(pagerank[node])
        graph.nodes[node]["self_loop"] = float(graph.has_edge(node, node))
    return graph


def _feature_names_for_dataset(dataset_name: str) -> list[str]:
    if dataset_name == "adfa_ld":
        return ADFA_NODE_FEATURE_NAMES
    if dataset_name == "lid_ds":
        return LID_NODE_FEATURE_NAMES
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _graph_from_sample(sample: TraceSample) -> nx.DiGraph:
    if sample.dataset_name == "adfa_ld":
        return _build_adfa_graph([int(token) for token in sample.sequence])
    if sample.dataset_name == "lid_ds":
        if sample.trace_df is None:
            raise ValueError("LID-DS samples require trace_df metadata.")
        return _build_lid_graph(sample.trace_df)
    raise ValueError(f"Unsupported dataset: {sample.dataset_name}")


def build_networkx_graph(sample: TraceSample) -> nx.DiGraph:
    return _graph_from_sample(sample)


def _sample_to_pyg_data(
    sample: TraceSample,
    vocab: dict[SyscallToken, int],
    feature_names: list[str],
) -> Data:
    graph = _graph_from_sample(sample)
    nodes = list(graph.nodes())
    node_to_index = {node: index for index, node in enumerate(nodes)}

    x = torch.tensor(
        [
            [float(graph.nodes[node].get(feature_name, 0.0)) for feature_name in feature_names]
            for node in nodes
        ],
        dtype=torch.float,
    )
    node_ids = torch.tensor(
        [vocab.get(node, vocab[UNKNOWN_SYSCALL_TOKEN]) for node in nodes],
        dtype=torch.long,
    )

    edges: list[list[int]] = []
    edge_weights: list[float] = []
    for source, target, attributes in graph.edges(data=True):
        edges.append([node_to_index[source], node_to_index[target]])
        edge_weights.append(float(attributes.get("weight", 1.0)))

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float)

    data = Data(
        x=x,
        node_ids=node_ids,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=torch.tensor([sample.label], dtype=torch.long),
    )
    data.file_path = sample.file_path
    for key, value in sample.metadata.items():
        setattr(data, key, value)
    return data


def _fit_scaler(dataset: list[Data]) -> StandardScaler:
    features = np.concatenate([data.x.cpu().numpy() for data in dataset], axis=0)
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler


def _apply_scaler(dataset: list[Data], scaler: StandardScaler) -> list[Data]:
    transformed: list[Data] = []
    for data in dataset:
        clone = data.clone()
        scaled = scaler.transform(clone.x.cpu().numpy())
        clone.x = torch.tensor(scaled, dtype=torch.float)
        transformed.append(clone)
    return transformed


def build_graph_data_bundle(
    train_samples: list[TraceSample],
    val_samples: list[TraceSample],
    test_samples: list[TraceSample],
    scale_node_features: bool = True,
) -> GraphDataBundle:
    if not train_samples:
        raise ValueError("Training split is empty.")

    dataset_name = train_samples[0].dataset_name
    feature_names = _feature_names_for_dataset(dataset_name)
    vocab = build_syscall_vocab(train_samples)

    train_dataset = [_sample_to_pyg_data(sample, vocab, feature_names) for sample in train_samples]
    val_dataset = [_sample_to_pyg_data(sample, vocab, feature_names) for sample in val_samples]
    test_dataset = [_sample_to_pyg_data(sample, vocab, feature_names) for sample in test_samples]

    if scale_node_features:
        scaler = _fit_scaler(train_dataset)
        train_dataset = _apply_scaler(train_dataset, scaler)
        val_dataset = _apply_scaler(val_dataset, scaler)
        test_dataset = _apply_scaler(test_dataset, scaler)

    return GraphDataBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        vocab=vocab,
        feature_names=feature_names,
    )
