from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np

from intrusion_detection.metrics import compute_binary_classification_metrics

SyscallToken = int | str


def _aggregate_graph_from_sequences(sequences: Iterable[list[SyscallToken]]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for sequence in sequences:
        for index in range(len(sequence) - 1):
            source = sequence[index]
            target = sequence[index + 1]
            if graph.has_edge(source, target):
                graph[source][target]["weight"] += 1.0
            else:
                graph.add_edge(source, target, weight=1.0)
    return graph


def _windows_from_sequence(sequence: list[SyscallToken], window_size: int) -> list[tuple[SyscallToken, ...]]:
    windows: list[tuple[SyscallToken, ...]] = []
    current = deque(maxlen=window_size)
    for token in sequence:
        current.append(token)
        if len(current) == window_size:
            windows.append(tuple(current))
    return windows


def _k_map_from_graph(
    graph: nx.DiGraph,
    pagerank_scores: dict[SyscallToken, float],
) -> dict[tuple[SyscallToken, SyscallToken], float]:
    weights: dict[tuple[SyscallToken, SyscallToken], float] = {}
    for source, target, attributes in graph.edges(data=True):
        out_degree = graph.out_degree(source, weight="weight")
        target_rank = pagerank_scores.get(target, 0.0)
        if out_degree == 0 or target_rank == 0.0:
            score = 1.0
        else:
            contribution = (pagerank_scores[source] * attributes["weight"] / out_degree) / target_rank
            score = 1.0 - contribution
        weights[(source, target)] = max(0.0, min(1.0, score))
    return weights


def _distance_to_pattern(
    observed_sequence: tuple[SyscallToken, ...],
    pattern_sequence: tuple[SyscallToken, ...],
    weight_map: dict[tuple[SyscallToken, SyscallToken], float],
    default_edge_weight: float,
    prev_token: SyscallToken | None,
) -> float:
    distance = 0.0
    for index, token in enumerate(pattern_sequence):
        if observed_sequence[index] == token:
            continue
        if index == 0:
            edge = (prev_token, observed_sequence[index])
        else:
            edge = (observed_sequence[index - 1], observed_sequence[index])
        distance += weight_map.get(edge, default_edge_weight)
    return distance


@dataclass(slots=True)
class PageRankPrediction:
    anomaly_rate: float
    distances: list[float]


class PageRankAnomalyDetector:
    def __init__(
        self,
        window_size: int,
        distance_threshold: float,
        anomaly_rate_threshold: float,
        default_edge_weight: float,
        max_patterns: int,
        random_state: int,
    ) -> None:
        self.window_size = window_size
        self.distance_threshold = distance_threshold
        self.anomaly_rate_threshold = anomaly_rate_threshold
        self.default_edge_weight = default_edge_weight
        self.max_patterns = max_patterns
        self.random_state = random_state
        self.patterns: list[tuple[SyscallToken, ...]] = []
        self.pattern_set: set[tuple[SyscallToken, ...]] = set()
        self.patterns_by_first: dict[SyscallToken, list[tuple[SyscallToken, ...]]] = {}
        self.patterns_by_prefix: dict[tuple[SyscallToken, ...], list[tuple[SyscallToken, ...]]] = {}
        self.weight_map: dict[tuple[SyscallToken, SyscallToken], float] = {}
        self.distance_cache: dict[tuple[SyscallToken | None, tuple[SyscallToken, ...]], float] = {}

    def fit(self, normal_sequences: list[list[SyscallToken]]) -> None:
        if not normal_sequences:
            raise ValueError("PageRank model requires at least one normal training sequence.")

        graph = _aggregate_graph_from_sequences(normal_sequences)
        pagerank_scores = (
            nx.pagerank(graph, weight="weight")
            if graph.number_of_edges() > 0
            else {node: 0.0 for node in graph.nodes()}
        )
        self.weight_map = _k_map_from_graph(graph, pagerank_scores)

        all_unique_patterns = list(
            dict.fromkeys(
                window
                for sequence in normal_sequences
                for window in _windows_from_sequence(sequence, self.window_size)
            )
        )
        candidate_patterns = list(all_unique_patterns)
        if self.max_patterns > 0 and len(candidate_patterns) > self.max_patterns:
            rng = np.random.default_rng(self.random_state)
            selected_indices = rng.choice(len(candidate_patterns), size=self.max_patterns, replace=False)
            candidate_patterns = [candidate_patterns[index] for index in sorted(selected_indices)]

        self.patterns = candidate_patterns
        self.pattern_set = set(all_unique_patterns)
        self.patterns_by_first = {}
        self.patterns_by_prefix = {}
        for pattern in self.patterns:
            self.patterns_by_first.setdefault(pattern[0], []).append(pattern)
            prefix = pattern[:2]
            self.patterns_by_prefix.setdefault(prefix, []).append(pattern)
        self.distance_cache = {}

    def _min_distance(self, observed_window: tuple[SyscallToken, ...], prev_token: SyscallToken | None) -> float:
        if observed_window in self.pattern_set:
            return 0.0

        cache_key = (prev_token, observed_window)
        cached = self.distance_cache.get(cache_key)
        if cached is not None:
            return cached

        candidates = self.patterns_by_prefix.get(observed_window[:2])
        if not candidates:
            candidates = self.patterns_by_first.get(observed_window[0], self.patterns)

        min_distance = float("inf")
        for pattern in candidates:
            distance = _distance_to_pattern(
                observed_window,
                pattern,
                self.weight_map,
                self.default_edge_weight,
                prev_token,
            )
            if distance < min_distance:
                min_distance = distance
            if min_distance == 0.0:
                break

        if min_distance == float("inf"):
            min_distance = 0.0

        self.distance_cache[cache_key] = min_distance
        return min_distance

    def score_sequence(self, sequence: list[SyscallToken]) -> PageRankPrediction:
        observed_windows = _windows_from_sequence(sequence, self.window_size)
        if not observed_windows:
            return PageRankPrediction(anomaly_rate=0.0, distances=[])

        distances: list[float] = []
        for index, window in enumerate(observed_windows):
            prev_token = sequence[index - 1] if index > 0 else None
            distances.append(self._min_distance(window, prev_token))

        anomaly_rate = float(np.mean(np.asarray(distances) > self.distance_threshold))
        return PageRankPrediction(anomaly_rate=anomaly_rate, distances=distances)

    def predict(self, sequence: list[SyscallToken]) -> tuple[int, PageRankPrediction]:
        prediction = self.score_sequence(sequence)
        label = int(prediction.anomaly_rate > self.anomaly_rate_threshold)
        return label, prediction


def select_pagerank_thresholds(
    predictions: list[PageRankPrediction],
    labels: list[int],
    distance_thresholds: list[float],
    anomaly_rate_thresholds: list[float],
) -> dict[str, float]:
    if len(set(labels)) < 2:
        raise ValueError("Threshold selection requires both classes in validation data.")

    best_result: dict[str, float] | None = None
    best_score = (-1.0, -1.0)

    distance_arrays = [np.asarray(prediction.distances, dtype=float) for prediction in predictions]
    for distance_threshold in distance_thresholds:
        anomaly_rates = [
            float(np.mean(distances > distance_threshold)) if len(distances) else 0.0
            for distances in distance_arrays
        ]
        for anomaly_rate_threshold in anomaly_rate_thresholds:
            y_pred = [int(rate > anomaly_rate_threshold) for rate in anomaly_rates]
            metrics = compute_binary_classification_metrics(labels, y_pred, anomaly_rates)
            score = (metrics["f1"], metrics["accuracy"])
            if score > best_score:
                best_score = score
                best_result = {
                    "distance_threshold": float(distance_threshold),
                    "anomaly_rate_threshold": float(anomaly_rate_threshold),
                    "f1": float(metrics["f1"]),
                    "accuracy": float(metrics["accuracy"]),
                }

    if best_result is None:
        raise RuntimeError("Could not select pagerank thresholds.")
    return best_result
