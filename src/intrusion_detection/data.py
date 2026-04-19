from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import TextIOWrapper
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable
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

FEATURE_PROFILES = {
    "adfa_ld": {
        "full": ADFA_NODE_FEATURE_NAMES,
        "embedding_only": [],
        "frequency": ["freq"],
        "transition": ["freq", "in_wdeg", "out_wdeg", "pagerank", "self_loop"],
        "structural": ["freq", "in_deg", "out_deg", "in_wdeg", "out_wdeg", "pagerank", "self_loop"],
    },
    "lid_ds": {
        "full": LID_NODE_FEATURE_NAMES,
        "embedding_only": [],
        "structural": ["freq", "count", "in_deg", "out_deg", "in_wdeg", "out_wdeg", "pagerank", "self_loop"],
        "behavioral": ["freq", "count", "unique_proc", "error_rate", "mean_res", "std_res", "mean_gap", "std_gap"],
        "temporal": ["freq", "count", "mean_gap", "std_gap", "pagerank", "self_loop"],
    },
}


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
    res_re = re.compile(r"\bres=(-?\d+)\b")

    def __init__(
        self,
        root_dir: str | Path,
        scenario: str = "all",
        direction_filter: str | None = ">",
        keep_only_successful_parsed_lines: bool = True,
        loader_workers: int | str | None = 1,
    ) -> None:
        self.root = Path(root_dir)
        self.scenario = scenario
        self.direction_filter = direction_filter
        self.keep_only_successful_parsed_lines = keep_only_successful_parsed_lines
        self.loader_workers = loader_workers

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

        parts = line.split(maxsplit=7)
        if len(parts) < 7:
            return None

        timestamp_text, _, _, procname, _, syscall_text, direction, *rest_parts = parts
        if direction not in {"<", ">"}:
            return None

        try:
            timestamp = int(timestamp_text)
        except ValueError:
            return None

        rest = rest_parts[0] if rest_parts else ""
        res_match = self.res_re.search(rest)
        res_value = int(res_match.group(1)) if res_match else np.nan
        syscall = self._normalize_syscall_value(syscall_text)
        if syscall is None:
            return None

        return {
            "timestamp": timestamp,
            "procname": procname,
            "syscall": syscall,
            "direction": direction,
            "res": res_value,
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
                events_df = self._parse_sc_handle(
                    TextIOWrapper(handle, encoding="utf-8", errors="ignore"),
                    source_label=str(zip_path),
                )
        label = int(bool(meta.get("exploit", False)))
        return meta, events_df, label

    def _finalize_events_df(
        self,
        events_df: pd.DataFrame,
        *,
        source_label: str,
        already_sorted: bool = False,
    ) -> pd.DataFrame:
        if len(events_df) == 0:
            raise ValueError(f"No parsed syscall lines in {source_label}")

        if self.direction_filter is not None:
            events_df = events_df[events_df["direction"] == self.direction_filter].copy()
        if len(events_df) == 0:
            raise ValueError(f"No events left after direction filtering in {source_label}")

        if not already_sorted:
            events_df = events_df.sort_values("timestamp").reset_index(drop=True)
        else:
            events_df = events_df.reset_index(drop=True)
        return events_df

    def _parse_sc_handle(self, handle: Iterable[str], *, source_label: str) -> pd.DataFrame:
        if not self.keep_only_successful_parsed_lines:
            events = [self._parse_sc_line(line) for line in handle]
            return self._finalize_events_df(
                pd.DataFrame(events),
                source_label=source_label,
            )

        timestamps: list[int] = []
        procnames: list[str] = []
        syscalls: list[SyscallToken] = []
        directions: list[str] = []
        res_values: list[int | float] = []
        needs_sort = False
        last_timestamp: int | None = None

        for line in handle:
            parsed = self._parse_sc_line(line)
            if parsed is None:
                continue
            if self.direction_filter is not None and parsed["direction"] != self.direction_filter:
                continue

            timestamp = int(parsed["timestamp"])
            if last_timestamp is not None and timestamp < last_timestamp:
                needs_sort = True
            last_timestamp = timestamp

            timestamps.append(timestamp)
            procnames.append(str(parsed["procname"]))
            syscalls.append(parsed["syscall"])
            directions.append(str(parsed["direction"]))
            res_values.append(parsed["res"])

        events_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "procname": procnames,
                "syscall": syscalls,
                "direction": directions,
                "res": res_values,
            }
        )
        return self._finalize_events_df(
            events_df,
            source_label=source_label,
            already_sorted=not needs_sort,
        )

    def _parse_sc_lines(self, lines: list[str]) -> pd.DataFrame:
        return self._parse_sc_handle(iter(lines), source_label="in-memory trace")

    def _read_sc_zip(self, file_path: Path) -> pd.DataFrame:
        with ZipFile(file_path) as archive:
            members = [name for name in archive.namelist() if name.endswith(".sc")]
            if not members:
                raise ValueError(f"No .sc trace found in archive: {file_path}")
            with archive.open(members[0], "r") as handle:
                return self._parse_sc_handle(
                    TextIOWrapper(handle, encoding="utf-8", errors="ignore"),
                    source_label=str(file_path),
                )

    def _read_sc_file(self, file_path: Path) -> pd.DataFrame:
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            return self._parse_sc_handle(handle, source_label=str(file_path))

    def _resolve_loader_workers(self, num_files: int) -> int:
        if num_files <= 1:
            return 1
        if self.loader_workers in {None, 0, "0", 1, "1"}:
            return 1
        if self.loader_workers == "auto":
            return max(1, min(8, os.cpu_count() or 1, num_files))
        return max(1, min(int(self.loader_workers), num_files))

    def _load_trace_sample(self, file_path: Path) -> tuple[TraceSample | None, str | None]:
        split, scenario = self._infer_split_and_scenario(file_path)
        if split is None:
            return None, None

        try:
            meta, trace_df, label = self._load_sample_from_zip(file_path)
        except Exception as error:
            return None, f"[WARN] skipping {file_path}: {error}"

        if len(trace_df) < 2:
            return None, None

        return (
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
            ),
            None,
        )

    def load_all(self) -> list[TraceSample]:
        file_paths = self._collect_zip_files()
        num_workers = self._resolve_loader_workers(len(file_paths))
        samples: list[TraceSample] = []

        if num_workers == 1:
            results = map(self._load_trace_sample, file_paths)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = executor.map(self._load_trace_sample, file_paths)
                for sample, warning in results:
                    if warning is not None:
                        print(warning)
                    if sample is not None:
                        samples.append(sample)
            return samples

        for sample, warning in results:
            if warning is not None:
                print(warning)
            if sample is not None:
                samples.append(sample)
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
            loader_workers=getattr(dataset_cfg, "loader_workers", 1),
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


def split_lid_transfer_samples(
    samples: list[TraceSample],
    source_scenario: str,
    target_scenario: str,
    source_strategy: str = "resplit",
    source_train_split: str = "train",
    source_val_split: str = "validation",
    source_resplit_split: str = "test",
    target_test_split: str = "test",
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[TraceSample], list[TraceSample], list[TraceSample]]:
    if source_strategy == "predefined":
        train_samples = [
            sample
            for sample in samples
            if sample.dataset_name == "lid_ds"
            and sample.metadata.get("scenario") == source_scenario
            and sample.metadata.get("split") == source_train_split
        ]
        val_samples = [
            sample
            for sample in samples
            if sample.dataset_name == "lid_ds"
            and sample.metadata.get("scenario") == source_scenario
            and sample.metadata.get("split") == source_val_split
        ]
    elif source_strategy == "resplit":
        source_samples = [
            sample
            for sample in samples
            if sample.dataset_name == "lid_ds"
            and sample.metadata.get("scenario") == source_scenario
            and sample.metadata.get("split") == source_resplit_split
        ]
        if not source_samples:
            raise ValueError(
                f"No source samples found for scenario={source_scenario} and split={source_resplit_split}."
            )
        labels = [sample.label for sample in source_samples]
        indices = np.arange(len(source_samples))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_size,
            random_state=random_state,
            stratify=labels,
        )
        train_samples = [source_samples[index] for index in train_idx]
        val_samples = [source_samples[index] for index in val_idx]
    else:
        raise ValueError(f"Unsupported transfer source strategy: {source_strategy}")

    test_samples = [
        sample
        for sample in samples
        if sample.dataset_name == "lid_ds"
        and sample.metadata.get("scenario") == target_scenario
        and sample.metadata.get("split") == target_test_split
    ]

    if not train_samples or not val_samples or not test_samples:
        raise ValueError(
            "Cross-scenario evaluation requires non-empty train, validation, and test selections. "
            f"Got train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)} "
            f"for source={source_scenario}, target={target_scenario}."
        )
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


def _feature_names_for_dataset(dataset_name: str, node_profile: str = "full") -> list[str]:
    if dataset_name not in FEATURE_PROFILES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    profiles = FEATURE_PROFILES[dataset_name]
    if node_profile not in profiles:
        raise ValueError(
            f"Unsupported feature profile '{node_profile}' for dataset '{dataset_name}'. "
            f"Available profiles: {sorted(profiles)}"
        )
    return list(profiles[node_profile])


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
    edge_weight_mode: str,
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
        if edge_weight_mode == "weighted":
            edge_weights.append(float(attributes.get("weight", 1.0)))
        elif edge_weight_mode == "binary":
            edge_weights.append(1.0)
        else:
            raise ValueError(f"Unsupported edge_weight_mode: {edge_weight_mode}")

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
    if not dataset or dataset[0].x.shape[1] == 0:
        raise ValueError("Scaler requested on an empty feature set.")
    features = np.concatenate([data.x.cpu().numpy() for data in dataset], axis=0)
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler


def _apply_scaler(dataset: list[Data], scaler: StandardScaler) -> list[Data]:
    if not dataset or dataset[0].x.shape[1] == 0:
        return [data.clone() for data in dataset]
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
    node_feature_profile: str = "full",
    edge_weight_mode: str = "weighted",
) -> GraphDataBundle:
    if not train_samples:
        raise ValueError("Training split is empty.")

    dataset_name = train_samples[0].dataset_name
    feature_names = _feature_names_for_dataset(dataset_name, node_feature_profile)
    vocab = build_syscall_vocab(train_samples)

    train_dataset = [
        _sample_to_pyg_data(sample, vocab, feature_names, edge_weight_mode) for sample in train_samples
    ]
    val_dataset = [
        _sample_to_pyg_data(sample, vocab, feature_names, edge_weight_mode) for sample in val_samples
    ]
    test_dataset = [
        _sample_to_pyg_data(sample, vocab, feature_names, edge_weight_mode) for sample in test_samples
    ]

    if scale_node_features and feature_names:
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
