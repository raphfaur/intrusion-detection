from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from intrusion_detection.data import (
    UNKNOWN_SYSCALL_TOKEN,
    SyscallToken,
    TraceSample,
    build_syscall_vocab,
)

# Number of per-node static features computed from the trace graph.
# [rel_freq, out_weight, in_weight, out_degree_norm, in_degree_norm, self_loop, pagerank]
NUM_STATIC_FEATURES = 7


@dataclass(slots=True)
class TemporalTraceData:
    """A single execution trace as a sequence of temporal syscall transition events."""

    src_ids: Tensor       # (E,) vocab-indexed source syscall ids
    dst_ids: Tensor       # (E,) vocab-indexed destination syscall ids
    src_dt: Tensor        # (E,) log-microsecond time delta since src last seen
    dst_dt: Tensor        # (E,) log-microsecond time delta since dst last seen
    node_features: Tensor # (V, NUM_STATIC_FEATURES) static features for unique nodes
    label: int
    file_path: str


@dataclass(slots=True)
class TemporalDataBundle:
    train_dataset: list[TemporalTraceData]
    val_dataset: list[TemporalTraceData]
    test_dataset: list[TemporalTraceData]
    vocab: dict[SyscallToken, int]

    @property
    def num_syscalls(self) -> int:
        return len(self.vocab)

    @property
    def static_feature_dim(self) -> int:
        return NUM_STATIC_FEATURES


def _pagerank_iter(
    trans: dict[tuple[int, int], int],
    out_cnt: np.ndarray,
    V: int,
    damping: float = 0.85,
    iterations: int = 20,
) -> np.ndarray:
    pr = np.full(V, 1.0 / V, dtype=np.float64)
    for _ in range(iterations):
        new_pr = np.zeros(V, dtype=np.float64)
        for (u, v), cnt in trans.items():
            if out_cnt[u] > 0:
                new_pr[v] += damping * pr[u] * cnt / out_cnt[u]
        new_pr += (1.0 - damping) / V
        pr = new_pr
    return pr.astype(np.float32)


def _compute_node_features(
    src_id_list: list[int],
    dst_id_list: list[int],
    vocab_to_local: dict[int, int],
    V: int,
) -> np.ndarray:
    """Compute per-node static features from the event sequence.

    Features (NUM_STATIC_FEATURES = 7):
        rel_freq, out_weight, in_weight, out_degree_norm, in_degree_norm,
        self_loop, pagerank
    """
    E = len(src_id_list)
    if E == 0:
        return np.zeros((V, NUM_STATIC_FEATURES), dtype=np.float32)

    out_cnt = np.zeros(V, dtype=np.float32)
    in_cnt = np.zeros(V, dtype=np.float32)
    out_nbrs: list[set[int]] = [set() for _ in range(V)]
    in_nbrs: list[set[int]] = [set() for _ in range(V)]
    trans: dict[tuple[int, int], int] = {}

    for s, d in zip(src_id_list, dst_id_list):
        u = vocab_to_local[s]
        v = vocab_to_local[d]
        out_cnt[u] += 1
        in_cnt[v] += 1
        out_nbrs[u].add(v)
        in_nbrs[v].add(u)
        trans[(u, v)] = trans.get((u, v), 0) + 1

    rel_freq = (out_cnt + in_cnt) / (2 * E)
    out_w = out_cnt / E
    in_w = in_cnt / E
    out_deg = np.array([len(out_nbrs[i]) for i in range(V)], dtype=np.float32) / max(V, 1)
    in_deg = np.array([len(in_nbrs[i]) for i in range(V)], dtype=np.float32) / max(V, 1)
    self_lp = np.array([float((i, i) in trans) for i in range(V)], dtype=np.float32)
    pr = _pagerank_iter(trans, out_cnt, V)

    return np.column_stack([rel_freq, out_w, in_w, out_deg, in_deg, self_lp, pr]).astype(np.float32)


def _sample_to_temporal(
    sample: TraceSample,
    vocab: dict[SyscallToken, int],
    max_events: int,
) -> TemporalTraceData | None:
    if sample.trace_df is None or len(sample.trace_df) < 2:
        return None

    sequence = sample.sequence
    raw_ts = sample.trace_df["timestamp"].to_numpy(dtype=np.float64)

    # Build temporal edges: consecutive syscall pairs (seq[i] -> seq[i+1])
    # with the timestamp of the source event
    srcs = sequence[:-1]
    dsts = sequence[1:]
    ts = raw_ts[:-1]

    # Normalize to relative seconds from the first event in this trace
    # LID-DS timestamps are nanosecond unix timestamps
    ts_rel = (ts - ts[0]) / 1e9

    if len(srcs) > max_events:
        # Stride-based subsampling: cover the full trace uniformly rather than
        # truncating the head, so attack events (which may occur anywhere in the
        # trace) are represented in the sample
        indices = np.linspace(0, len(srcs) - 1, max_events, dtype=int)
        srcs = [srcs[i] for i in indices]
        dsts = [dsts[i] for i in indices]
        ts_rel = ts_rel[indices]

    # Precompute per-event time deltas: seconds since the last time each node
    # appeared (either as src or dst). Zero on first occurrence.
    last_seen: dict[Any, float] = {}
    src_dt_list: list[float] = []
    dst_dt_list: list[float] = []
    for src, dst, t in zip(srcs, dsts, ts_rel):
        src_dt_list.append(t - last_seen.get(src, t))
        dst_dt_list.append(t - last_seen.get(dst, t))
        last_seen[src] = t
        last_seen[dst] = t

    # Log-microsecond scale so that freq~N(0,1) produces useful angles.
    # Raw seconds give dt~1e-6, making cos(dt*freq)≈1 for all events.
    src_dt_arr = np.log1p(np.array(src_dt_list, dtype=np.float32) * 1e6)
    dst_dt_arr = np.log1p(np.array(dst_dt_list, dtype=np.float32) * 1e6)

    unk = vocab[UNKNOWN_SYSCALL_TOKEN]
    src_id_list = [vocab.get(s, unk) for s in srcs]
    dst_id_list = [vocab.get(d, unk) for d in dsts]

    # Build sorted unique vocab ids — must match torch.unique order in forward()
    unique_vocab_ids = sorted(set(src_id_list) | set(dst_id_list))
    vocab_to_local = {vid: i for i, vid in enumerate(unique_vocab_ids)}
    V = len(unique_vocab_ids)
    node_features = _compute_node_features(src_id_list, dst_id_list, vocab_to_local, V)

    return TemporalTraceData(
        src_ids=torch.tensor(src_id_list, dtype=torch.long),
        dst_ids=torch.tensor(dst_id_list, dtype=torch.long),
        src_dt=torch.from_numpy(src_dt_arr),
        dst_dt=torch.from_numpy(dst_dt_arr),
        node_features=torch.from_numpy(node_features),
        label=sample.label,
        file_path=sample.file_path,
    )


def build_temporal_data_bundle(
    train_samples: list[TraceSample],
    val_samples: list[TraceSample],
    test_samples: list[TraceSample],
    max_events: int = 2000,
) -> TemporalDataBundle:
    """Convert LID-DS TraceSamples into temporal event sequences for TGN.

    Requires LID-DS samples: ADFA-LD does not provide per-event timestamps.
    Timestamps are normalized to relative seconds and per-node time deltas
    are precomputed to avoid repeated dictionary lookups during training.
    """
    if not train_samples:
        raise ValueError("Training split is empty.")
    if train_samples[0].dataset_name != "lid_ds":
        raise ValueError(
            "TGN temporal pipeline requires LID-DS samples (per-event timestamps). "
            "ADFA-LD provides only syscall sequences without timestamps."
        )

    vocab = build_syscall_vocab(train_samples)

    def convert(samples: list[TraceSample]) -> list[TemporalTraceData]:
        result = []
        for sample in samples:
            item = _sample_to_temporal(sample, vocab, max_events)
            if item is not None:
                result.append(item)
        return result

    return TemporalDataBundle(
        train_dataset=convert(train_samples),
        val_dataset=convert(val_samples),
        test_dataset=convert(test_samples),
        vocab=vocab,
    )
