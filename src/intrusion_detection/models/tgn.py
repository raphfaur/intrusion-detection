from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TimeEncoder(nn.Module):
    """Encodes scalar time deltas into d-dimensional feature vectors.

    Uses learnable frequencies and phases following the time2vec formulation
    in Rossi et al., Temporal Graph Networks for Deep Learning on Dynamic
    Graphs, 2020.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.freq = nn.Parameter(torch.randn(dim))
        self.phase = nn.Parameter(torch.zeros(dim))

    def forward(self, dt: Tensor) -> Tensor:
        """
        Args:
            dt: (...) log-microsecond time delta(s)
        Returns:
            (..., dim) encoded time features
        """
        return torch.cos(dt.unsqueeze(-1) * self.freq + self.phase)


class TGNClassifier(nn.Module):
    """Temporal Graph Network for per-trace intrusion detection.

    Each execution trace is processed as an ordered sequence of syscall
    transition events (src, dst, Δt_src, Δt_dst). Per-node memory vectors
    evolve via GRU updates as events arrive, capturing the temporal dynamics
    of execution that static aggregated graphs cannot represent.

    Each trace is classified independently: memory is reset to learned
    syscall-type embeddings at the start of each forward pass, then updated
    sequentially through the event stream. The final graph-level embedding
    is the concatenation of mean and max pooling over active node memories.

    Optionally, per-trace static node features (frequency, degree, PageRank)
    can seed the initial memory via a learned projection, giving the model
    the same structural prior as a static GCN before temporal updates begin.

    Reference: Rossi et al., Temporal Graph Networks for Deep Learning on
    Dynamic Graphs, NeurIPS Workshop 2020.
    """

    def __init__(
        self,
        num_syscalls: int,
        memory_dim: int = 32,
        time_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        chunk_size: int = 50,
        static_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_syscalls = num_syscalls
        self.memory_dim = memory_dim
        # Events per batch in the sequential memory update loop.
        # chunk_size=1 is exact TGN; larger values trade temporal fidelity
        # for speed (each chunk sees the memory state from its start).
        self.chunk_size = chunk_size

        # Initial memory per syscall type: gives the model a static prior
        # about each syscall's identity before any events are processed
        self.initial_memory = nn.Embedding(num_syscalls, memory_dim)
        nn.init.xavier_uniform_(self.initial_memory.weight)

        # Optional projection of per-trace static features (freq, degree,
        # PageRank) into memory space, added to the initial embedding.
        # This seeds each node's memory with the same structural information
        # that static GCN models receive as pre-computed node features.
        if static_feature_dim > 0:
            self.feature_proj: nn.Linear | None = nn.Linear(static_feature_dim, memory_dim)
            # Zero-init so the hybrid starts identically to vanilla TGN and
            # learns to use static features incrementally during training.
            nn.init.zeros_(self.feature_proj.weight)
            nn.init.zeros_(self.feature_proj.bias)
        else:
            self.feature_proj = None

        self.time_encoder = TimeEncoder(time_dim)

        # Message function: (mem_self, mem_other, time_enc) -> message vector
        self.msg_fn = nn.Linear(2 * memory_dim + time_dim, memory_dim)

        # Memory updater
        self.gru = nn.GRUCell(memory_dim, memory_dim)

        # Concat(mean, max) pooling: 2*memory_dim -> hidden_dim -> 2
        # Max preserves the strongest attack-specific memory signal that
        # plain mean pooling washes out when averaged over many syscalls.
        self.classifier = nn.Sequential(
            nn.Linear(2 * memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        src_ids: Tensor,
        dst_ids: Tensor,
        src_dt: Tensor,
        dst_dt: Tensor,
        node_features: Tensor | None = None,
    ) -> Tensor:
        """Classify a single execution trace.

        Args:
            src_ids:       (E,) vocab indices of source syscalls
            dst_ids:       (E,) vocab indices of destination syscalls
            src_dt:        (E,) log-μs since last occurrence of each source node
            dst_dt:        (E,) log-μs since last occurrence of each destination node
            node_features: (V, F) optional static per-node features in sorted
                           vocab-id order, matching torch.unique(cat([src,dst]))

        Returns:
            logits: (1, 2)
        """
        device = src_ids.device
        unique_ids = torch.unique(torch.cat([src_ids, dst_ids]))
        V = unique_ids.size(0)

        # Map global vocab ids to compact local indices [0, V)
        local_map = torch.zeros(self.num_syscalls, dtype=torch.long, device=device)
        local_map[unique_ids] = torch.arange(V, device=device)
        local_src = local_map[src_ids]  # (E,)
        local_dst = local_map[dst_ids]  # (E,)

        # All time encodings vectorized upfront
        te_src = self.time_encoder(src_dt)  # (E, time_dim)
        te_dst = self.time_encoder(dst_dt)  # (E, time_dim)

        memory = self.initial_memory(unique_ids)  # (V, memory_dim)
        if self.feature_proj is not None and node_features is not None:
            memory = memory + self.feature_proj(node_features.to(device))

        # Chunked memory updates: process chunk_size events per iteration.
        # Within a chunk all events see the same memory snapshot (synchronous
        # update). chunk_size=1 is exact sequential TGN; larger values trade
        # strict temporal fidelity for speed via batched BLAS calls.
        E = src_ids.size(0)
        expand = lambda idx: idx.unsqueeze(1).expand(-1, self.memory_dim)

        for start in range(0, E, self.chunk_size):
            end = min(start + self.chunk_size, E)
            c_src = local_src[start:end]   # (C,)
            c_dst = local_dst[start:end]   # (C,)

            mem_s = memory[c_src]          # (C, memory_dim)
            mem_d = memory[c_dst]          # (C, memory_dim)

            m_s = F.relu(self.msg_fn(torch.cat([mem_s, mem_d, te_src[start:end]], dim=1)))
            m_d = F.relu(self.msg_fn(torch.cat([mem_d, mem_s, te_dst[start:end]], dim=1)))

            new_s = self.gru(m_s, mem_s)  # (C, memory_dim)
            new_d = self.gru(m_d, mem_d)  # (C, memory_dim)

            # Scatter updates back; last write wins on same-chunk conflicts
            memory = memory.scatter(0, expand(c_dst), new_d)
            memory = memory.scatter(0, expand(c_src), new_s)

        graph_embed = torch.cat([memory.mean(dim=0), memory.max(dim=0).values])
        return self.classifier(graph_embed.unsqueeze(0))  # (1, 2)
