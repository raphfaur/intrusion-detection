from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    SAGEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


def _normalize_architecture_name(name: str) -> str:
    normalized = name.strip().lower()
    aliases = {
        "sage": "graphsage",
        "graph_sage": "graphsage",
        "weighted_gcn_plus": "wgcn_plus",
        "wgcna": "wgcn_plus",
    }
    return aliases.get(normalized, normalized)


class SyscallGraphClassifier(nn.Module):
    def __init__(
        self,
        num_syscalls: int,
        num_node_features: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pooling: str = "mean",
        architecture: str = "gcn",
        gat_heads: int = 4,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.architecture = _normalize_architecture_name(architecture)
        self.embedding = nn.Embedding(num_syscalls, embedding_dim)
        self.dropout = dropout
        self.pooling = pooling
        self.use_layer_norm = self.architecture in {"wgcn_plus", "graphsage", "gat"}
        self.use_hybrid_pooling = self.architecture == "wgcn_plus"

        input_dim = embedding_dim + num_node_features
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(self._build_conv(input_dim, hidden_dim, gat_heads))
        self.norms.append(nn.LayerNorm(hidden_dim) if self.use_layer_norm else nn.Identity())
        for _ in range(num_layers - 1):
            self.convs.append(self._build_conv(hidden_dim, hidden_dim, gat_heads))
            self.norms.append(nn.LayerNorm(hidden_dim) if self.use_layer_norm else nn.Identity())

        classifier_input_dim = hidden_dim * 2 if self.use_hybrid_pooling else hidden_dim
        self.graph_embedding_dim = classifier_input_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def _build_conv(self, in_dim: int, out_dim: int, gat_heads: int) -> nn.Module:
        if self.architecture in {"gcn", "wgcn_plus"}:
            return GCNConv(in_dim, out_dim)
        if self.architecture == "graphsage":
            return SAGEConv(in_dim, out_dim, aggr="mean")
        if self.architecture == "gat":
            return GATConv(
                in_dim,
                out_dim,
                heads=gat_heads,
                concat=False,
                dropout=self.dropout,
                edge_dim=1,
            )
        raise ValueError(
            f"Unsupported GNN architecture: {self.architecture}. "
            "Expected one of: gcn, wgcn_plus, graphsage, gat."
        )

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.use_hybrid_pooling:
            return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        if self.pooling == "sum":
            return global_add_pool(x, batch)
        if self.pooling == "max":
            return global_max_pool(x, batch)
        raise ValueError(f"Unsupported pooling mode: {self.pooling}")

    def _apply_conv(self, conv: nn.Module, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        if self.architecture in {"gcn", "wgcn_plus"}:
            return conv(x, edge_index, edge_weight)
        if self.architecture == "gat":
            edge_attr = edge_weight.view(-1, 1) if edge_weight.numel() > 0 else None
            return conv(x, edge_index, edge_attr=edge_attr)
        return conv(x, edge_index)

    def encode(self, data) -> torch.Tensor:
        embedded = self.embedding(data.node_ids)
        x = torch.cat([embedded, data.x], dim=1)

        for index, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            x = self._apply_conv(conv, x, data.edge_index, data.edge_weight)
            x = norm(x)
            x = F.relu(x)
            if index < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return self._pool(x, data.batch)

    def classify_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)

    def forward(self, data) -> torch.Tensor:
        return self.classify_embeddings(self.encode(data))
