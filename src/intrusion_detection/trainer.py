from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from intrusion_detection.data import TraceSample, build_graph_data_bundle
from intrusion_detection.metrics import compute_binary_classification_metrics
from intrusion_detection.models import (
    PageRankAnomalyDetector,
    GRUSequenceClassifier,
    SyscallGraphClassifier,
    select_pagerank_thresholds,
)


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"classification_report", "y_true", "y_pred", "y_score"}
    }


def _build_bundle(cfg: Any, train_samples: list[TraceSample], val_samples: list[TraceSample], test_samples: list[TraceSample]):
    return build_graph_data_bundle(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        scale_node_features=cfg.train.scale_node_features,
        node_feature_profile=cfg.features.node_profile,
        edge_weight_mode=cfg.features.edge_weight_mode,
    )


def _build_loaders(cfg: Any, bundle: Any) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(bundle.train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(bundle.val_dataset, batch_size=cfg.train.eval_batch_size, shuffle=False)
    test_loader = DataLoader(bundle.test_dataset, batch_size=cfg.train.eval_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def _build_model(cfg: Any, bundle: Any, device: torch.device) -> SyscallGraphClassifier:
    return SyscallGraphClassifier(
        num_syscalls=bundle.num_syscalls,
        num_node_features=bundle.num_node_features,
        embedding_dim=cfg.model.embedding_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        pooling=cfg.model.pooling,
        architecture=cfg.model.architecture,
        gat_heads=cfg.model.gat_heads,
    ).to(device)


class TokenSequenceDataset(Dataset):
    def __init__(
        self,
        samples: list[TraceSample],
        vocab: dict[Any, int],
        max_length: int,
        truncate: str = "head",
    ) -> None:
        self.samples = samples
        self.vocab = vocab
        self.max_length = max_length
        self.truncate = truncate

    def __len__(self) -> int:
        return len(self.samples)

    def _encode(self, sequence: list[Any]) -> list[int]:
        token_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in sequence]
        if len(token_ids) <= self.max_length:
            return token_ids
        if self.truncate == "tail":
            return token_ids[-self.max_length :]
        return token_ids[: self.max_length]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        tokens = self._encode(sample.sequence)
        return {
            "tokens": tokens,
            "length": len(tokens),
            "label": int(sample.label),
            "file_path": sample.file_path,
        }


def _build_sequence_vocab(samples: list[TraceSample]) -> dict[Any, int]:
    vocab: dict[Any, int] = {"<PAD>": 0, "<UNK>": 1}
    tokens = sorted(
        {token for sample in samples for token in sample.sequence},
        key=lambda item: (isinstance(item, str), str(item)),
    )
    for token in tokens:
        vocab[token] = len(vocab)
    return vocab


def _sequence_collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    max_length = max(item["length"] for item in batch)
    padded_tokens = []
    lengths = []
    labels = []
    for item in batch:
        tokens = item["tokens"]
        padded_tokens.append(tokens + [0] * (max_length - len(tokens)))
        lengths.append(item["length"])
        labels.append(item["label"])
    return {
        "tokens": torch.tensor(padded_tokens, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _build_sequence_loaders(cfg: Any, train_samples: list[TraceSample], val_samples: list[TraceSample], test_samples: list[TraceSample]):
    vocab = _build_sequence_vocab(train_samples)
    train_dataset = TokenSequenceDataset(
        train_samples,
        vocab=vocab,
        max_length=int(cfg.model.max_sequence_length),
        truncate=str(cfg.model.truncate_strategy),
    )
    val_dataset = TokenSequenceDataset(
        val_samples,
        vocab=vocab,
        max_length=int(cfg.model.max_sequence_length),
        truncate=str(cfg.model.truncate_strategy),
    )
    test_dataset = TokenSequenceDataset(
        test_samples,
        vocab=vocab,
        max_length=int(cfg.model.max_sequence_length),
        truncate=str(cfg.model.truncate_strategy),
    )
    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=_sequence_collate_fn,
    )
    val_loader = TorchDataLoader(
        val_dataset,
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
        collate_fn=_sequence_collate_fn,
    )
    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
        collate_fn=_sequence_collate_fn,
    )
    return vocab, train_loader, val_loader, test_loader


def _make_optimizer(parameters: Any, cfg: Any) -> torch.optim.Optimizer:
    return torch.optim.Adam(parameters, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)


def _make_probe(input_dim: int, cfg: Any, device: torch.device) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, cfg.model.hidden_dim),
        nn.ReLU(),
        nn.Dropout(cfg.model.dropout),
        nn.Linear(cfg.model.hidden_dim, 2),
    ).to(device)


def _train_one_epoch(
    model: SyscallGraphClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.y.view(-1))
        loss.backward()
        optimizer.step()

        batch_size = batch.num_graphs
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def _evaluate_classifier(
    model: SyscallGraphClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        scores = torch.softmax(logits, dim=1)[:, 1]
        predictions = logits.argmax(dim=1)

        y_true.extend(batch.y.view(-1).cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())
        y_score.extend(scores.cpu().tolist())

    metrics = compute_binary_classification_metrics(y_true, y_pred, y_score)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_score"] = y_score
    return metrics


def _augment_batch(batch: Any, edge_dropout: float, feature_mask: float):
    augmented = batch.clone()
    if edge_dropout > 0.0 and augmented.edge_index.numel() > 0:
        keep_mask = torch.rand(augmented.edge_index.size(1), device=augmented.edge_index.device) > edge_dropout
        if int(keep_mask.sum()) == 0:
            keep_mask[torch.randint(augmented.edge_index.size(1), (1,), device=keep_mask.device)] = True
        augmented.edge_index = augmented.edge_index[:, keep_mask]
        augmented.edge_weight = augmented.edge_weight[keep_mask]

    if feature_mask > 0.0 and augmented.x.numel() > 0:
        mask = torch.rand_like(augmented.x) < feature_mask
        augmented.x = augmented.x.clone()
        augmented.x[mask] = 0.0
    return augmented


def _nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    if z1.size(0) != z2.size(0):
        raise ValueError("Contrastive views must contain the same number of graphs.")

    embeddings = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    logits = embeddings @ embeddings.T / temperature
    logits.fill_diagonal_(-1e9)

    batch_size = z1.size(0)
    positives = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size, device=z1.device),
            torch.arange(0, batch_size, device=z1.device),
        ]
    )
    return F.cross_entropy(logits, positives)


def _train_contrastive_epoch(
    model: SyscallGraphClassifier,
    projector: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    edge_dropout: float,
    feature_mask: float,
    temperature: float,
) -> float:
    model.train()
    projector.train()
    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        view_one = _augment_batch(batch, edge_dropout=edge_dropout, feature_mask=feature_mask)
        view_two = _augment_batch(batch, edge_dropout=edge_dropout, feature_mask=feature_mask)

        optimizer.zero_grad()
        z1 = projector(model.encode(view_one))
        z2 = projector(model.encode(view_two))
        loss = _nt_xent_loss(z1, z2, temperature)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

    return total_loss / max(total_graphs, 1)


def _train_probe_epoch(
    encoder: SyscallGraphClassifier,
    probe: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    encoder.eval()
    probe.train()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            embeddings = encoder.encode(batch)
        logits = probe(embeddings)
        loss = F.cross_entropy(logits, batch.y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_examples += batch.num_graphs

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def _evaluate_probe(
    encoder: SyscallGraphClassifier,
    probe: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    encoder.eval()
    probe.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []

    for batch in loader:
        batch = batch.to(device)
        embeddings = encoder.encode(batch)
        logits = probe(embeddings)
        scores = torch.softmax(logits, dim=1)[:, 1]
        predictions = logits.argmax(dim=1)
        y_true.extend(batch.y.view(-1).cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())
        y_score.extend(scores.cpu().tolist())

    metrics = compute_binary_classification_metrics(y_true, y_pred, y_score)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_score"] = y_score
    return metrics


@torch.no_grad()
def _compute_center(model: SyscallGraphClassifier, loader: DataLoader, device: torch.device) -> torch.Tensor:
    model.eval()
    total = None
    count = 0
    for batch in loader:
        batch = batch.to(device)
        embeddings = model.encode(batch)
        total = embeddings.sum(dim=0) if total is None else total + embeddings.sum(dim=0)
        count += embeddings.size(0)
    if total is None or count == 0:
        raise ValueError("Anomaly detection requires at least one normal training graph.")
    return total / float(count)


def _train_anomaly_epoch(
    model: SyscallGraphClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    center: torch.Tensor,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        embeddings = model.encode(batch)
        distances = torch.sum((embeddings - center) ** 2, dim=1)
        loss = distances.mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_examples += batch.num_graphs

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def _score_anomaly_loader(
    model: SyscallGraphClassifier,
    loader: DataLoader,
    center: torch.Tensor,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    for batch in loader:
        batch = batch.to(device)
        embeddings = model.encode(batch)
        distances = torch.sum((embeddings - center) ** 2, dim=1)
        labels.extend(batch.y.view(-1).cpu().tolist())
        scores.extend(distances.cpu().tolist())
    return labels, scores


def _select_score_threshold(scores: list[float], labels: list[int], grid_size: int) -> tuple[float, dict[str, Any]]:
    if not scores:
        return 0.0, compute_binary_classification_metrics([], [], [])
    if len(set(labels)) < 2:
        threshold = float(np.median(scores))
        predictions = [int(score > threshold) for score in scores]
        metrics = compute_binary_classification_metrics(labels, predictions, scores)
        metrics["y_true"] = labels
        metrics["y_pred"] = predictions
        metrics["y_score"] = scores
        return threshold, metrics

    low = float(min(scores))
    high = float(max(scores))
    thresholds = np.linspace(low, high, num=max(grid_size, 2))
    best_threshold = float(thresholds[0])
    best_metrics: dict[str, Any] | None = None
    best_key = (-1.0, -1.0)
    for threshold in thresholds:
        predictions = [int(score > float(threshold)) for score in scores]
        metrics = compute_binary_classification_metrics(labels, predictions, scores)
        metrics["y_true"] = labels
        metrics["y_pred"] = predictions
        metrics["y_score"] = scores
        key = (float(metrics["f1"]), float(metrics["roc_auc"]) if not np.isnan(metrics["roc_auc"]) else -1.0)
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics

    assert best_metrics is not None
    return best_threshold, best_metrics


def _run_supervised_experiment(
    cfg: Any,
    bundle: Any,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    train_loader, val_loader, test_loader = _build_loaders(cfg, bundle)
    model = _build_model(cfg, bundle, device)
    optimizer = _make_optimizer(model.parameters(), cfg)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, device)
        train_metrics = _evaluate_classifier(model, train_loader, device)
        val_metrics = _evaluate_classifier(model, val_loader, device)

        epoch_metrics = {
            "epoch": float(epoch),
            "train/loss": float(train_loss),
            "train/accuracy": float(train_metrics["accuracy"]),
            "train/f1": float(train_metrics["f1"]),
            "val/accuracy": float(val_metrics["accuracy"]),
            "val/f1": float(val_metrics["f1"]),
            "val/roc_auc": float(val_metrics["roc_auc"]),
        }
        history.append(epoch_metrics)
        print(
            f"[GNN:supervised] epoch={epoch:03d}/{cfg.train.epochs:03d} "
            f"train_loss={train_loss:.4f} train_f1={train_metrics['f1']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_roc_auc={val_metrics['roc_auc']:.4f}",
            flush=True,
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_state = {name: parameter.detach().cpu().clone() for name, parameter in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = output_dir / "best_gnn.pt"
    torch.save(model.state_dict(), checkpoint_path)

    val_metrics = _evaluate_classifier(model, val_loader, device)
    test_metrics = _evaluate_classifier(model, test_loader, device)
    print(
        f"[GNN:supervised] final validation_f1={val_metrics['f1']:.4f} "
        f"test_accuracy={test_metrics['accuracy']:.4f} "
        f"test_f1={test_metrics['f1']:.4f} test_roc_auc={test_metrics['roc_auc']:.4f}",
        flush=True,
    )
    return {
        "model_name": "gnn",
        "paradigm": "supervised",
        "architecture": cfg.model.architecture,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "history": history,
        "validation": val_metrics,
        "test": test_metrics,
        "test_scores": {"y_true": test_metrics.get("y_true", []), "y_score": test_metrics.get("y_score", [])},
    }


def _run_contrastive_experiment(
    cfg: Any,
    bundle: Any,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    train_loader, val_loader, test_loader = _build_loaders(cfg, bundle)
    encoder = _build_model(cfg, bundle, device)
    projector = nn.Sequential(
        nn.Linear(encoder.graph_embedding_dim, cfg.model.hidden_dim),
        nn.ReLU(),
        nn.Linear(cfg.model.hidden_dim, cfg.model.hidden_dim),
    ).to(device)
    optimizer = _make_optimizer(list(encoder.parameters()) + list(projector.parameters()), cfg)

    pretrain_history: list[dict[str, float]] = []
    for epoch in range(1, cfg.train.contrastive_pretrain_epochs + 1):
        contrastive_loss = _train_contrastive_epoch(
            encoder,
            projector,
            train_loader,
            optimizer,
            device,
            edge_dropout=float(cfg.train.contrastive_edge_dropout),
            feature_mask=float(cfg.train.contrastive_feature_mask),
            temperature=float(cfg.train.contrastive_temperature),
        )
        pretrain_history.append({"epoch": float(epoch), "contrastive/loss": float(contrastive_loss)})
        print(
            f"[GNN:contrastive] pretrain_epoch={epoch:03d}/{cfg.train.contrastive_pretrain_epochs:03d} "
            f"loss={contrastive_loss:.4f}",
            flush=True,
        )

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    probe = _make_probe(encoder.graph_embedding_dim, cfg, device)
    probe_optimizer = _make_optimizer(probe.parameters(), cfg)
    best_probe_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.train.linear_probe_epochs + 1):
        train_loss = _train_probe_epoch(encoder, probe, train_loader, probe_optimizer, device)
        train_metrics = _evaluate_probe(encoder, probe, train_loader, device)
        val_metrics = _evaluate_probe(encoder, probe, val_loader, device)
        history.append(
            {
                "epoch": float(epoch),
                "train/loss": float(train_loss),
                "train/accuracy": float(train_metrics["accuracy"]),
                "train/f1": float(train_metrics["f1"]),
                "val/accuracy": float(val_metrics["accuracy"]),
                "val/f1": float(val_metrics["f1"]),
                "val/roc_auc": float(val_metrics["roc_auc"]),
            }
        )
        print(
            f"[GNN:contrastive] probe_epoch={epoch:03d}/{cfg.train.linear_probe_epochs:03d} "
            f"train_loss={train_loss:.4f} train_f1={train_metrics['f1']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_roc_auc={val_metrics['roc_auc']:.4f}",
            flush=True,
        )
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_probe_state = {name: parameter.detach().cpu().clone() for name, parameter in probe.state_dict().items()}

    if best_probe_state is not None:
        probe.load_state_dict(best_probe_state)

    checkpoint_path = output_dir / "best_gnn.pt"
    torch.save({"encoder": encoder.state_dict(), "probe": probe.state_dict()}, checkpoint_path)

    val_metrics = _evaluate_probe(encoder, probe, val_loader, device)
    test_metrics = _evaluate_probe(encoder, probe, test_loader, device)
    print(
        f"[GNN:contrastive] final validation_f1={val_metrics['f1']:.4f} "
        f"test_accuracy={test_metrics['accuracy']:.4f} "
        f"test_f1={test_metrics['f1']:.4f} test_roc_auc={test_metrics['roc_auc']:.4f}",
        flush=True,
    )
    return {
        "model_name": "gnn",
        "paradigm": "contrastive",
        "architecture": cfg.model.architecture,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "history": history,
        "pretrain_history": pretrain_history,
        "validation": val_metrics,
        "test": test_metrics,
    }


def _run_anomaly_experiment(
    cfg: Any,
    bundle: Any,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    _, val_loader, test_loader = _build_loaders(cfg, bundle)
    normal_train_dataset = [data for data in bundle.train_dataset if int(data.y.view(-1)[0].item()) == 0]
    normal_loader = DataLoader(normal_train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    model = _build_model(cfg, bundle, device)
    optimizer = _make_optimizer(model.parameters(), cfg)
    center = _compute_center(model, normal_loader, device).detach()

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_threshold = 0.0
    best_val_metrics: dict[str, Any] | None = None
    best_val_f1 = -1.0

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = _train_anomaly_epoch(model, normal_loader, optimizer, center, device)
        val_labels, val_scores = _score_anomaly_loader(model, val_loader, center, device)
        threshold, val_metrics = _select_score_threshold(
            val_scores,
            val_labels,
            grid_size=int(cfg.train.anomaly_threshold_grid_size),
        )
        history.append(
            {
                "epoch": float(epoch),
                "train/loss": float(train_loss),
                "val/accuracy": float(val_metrics["accuracy"]),
                "val/f1": float(val_metrics["f1"]),
                "val/roc_auc": float(val_metrics["roc_auc"]),
            }
        )
        print(
            f"[GNN:anomaly] epoch={epoch:03d}/{cfg.train.epochs:03d} "
            f"train_loss={train_loss:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_roc_auc={val_metrics['roc_auc']:.4f} threshold={threshold:.4f}",
            flush=True,
        )
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_threshold = float(threshold)
            best_val_metrics = val_metrics
            best_state = {name: parameter.detach().cpu().clone() for name, parameter in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = output_dir / "best_gnn.pt"
    torch.save(
        {
            "encoder": model.state_dict(),
            "center": center.cpu(),
            "threshold": best_threshold,
        },
        checkpoint_path,
    )

    test_labels, test_scores = _score_anomaly_loader(model, test_loader, center, device)
    test_pred = [int(score > best_threshold) for score in test_scores]
    test_metrics = compute_binary_classification_metrics(test_labels, test_pred, test_scores)
    test_metrics["y_true"] = test_labels
    test_metrics["y_pred"] = test_pred
    test_metrics["y_score"] = test_scores

    assert best_val_metrics is not None
    print(
        f"[GNN:anomaly] final validation_f1={best_val_metrics['f1']:.4f} "
        f"test_accuracy={test_metrics['accuracy']:.4f} "
        f"test_f1={test_metrics['f1']:.4f} test_roc_auc={test_metrics['roc_auc']:.4f}",
        flush=True,
    )
    return {
        "model_name": "gnn",
        "paradigm": "anomaly",
        "architecture": cfg.model.architecture,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "history": history,
        "validation": best_val_metrics,
        "test": test_metrics,
        "thresholds": {"score_threshold": float(best_threshold)},
    }


def run_gnn_experiment(
    cfg: Any,
    train_samples: list[TraceSample],
    val_samples: list[TraceSample],
    test_samples: list[TraceSample],
    output_dir: Path,
) -> dict[str, Any]:
    bundle = _build_bundle(cfg, train_samples, val_samples, test_samples)
    device = resolve_device(cfg.device)
    paradigm = str(getattr(cfg.train, "paradigm", "supervised")).strip().lower()

    if paradigm == "supervised":
        result = _run_supervised_experiment(cfg, bundle, output_dir, device)
    elif paradigm == "contrastive":
        result = _run_contrastive_experiment(cfg, bundle, output_dir, device)
    elif paradigm == "anomaly":
        result = _run_anomaly_experiment(cfg, bundle, output_dir, device)
    else:
        raise ValueError(f"Unsupported GNN learning paradigm: {paradigm}")

    _save_json(
        output_dir / "metrics_gnn.json",
        {
            "architecture": cfg.model.architecture,
            "paradigm": paradigm,
            "node_feature_profile": cfg.features.node_profile,
            "edge_weight_mode": cfg.features.edge_weight_mode,
            "feature_names": bundle.feature_names,
            "num_syscalls": bundle.num_syscalls,
            "num_node_features": bundle.num_node_features,
            "history": result.get("history", []),
            "pretrain_history": result.get("pretrain_history", []),
            "validation": result.get("validation", {}),
            "test": result.get("test", {}),
            "test_scores": result.get("test_scores", {}),
            "checkpoint": result.get("checkpoint"),
            "thresholds": result.get("thresholds"),
        },
    )
    classification_report = result["test"].get("classification_report")
    if classification_report:
        (output_dir / "classification_report_gnn.txt").write_text(
            classification_report,
            encoding="utf-8",
        )
    result["validation"] = _compact_metrics(result["validation"])
    result["test"] = _compact_metrics(result["test"])
    return result


def _stringify_sequence(sample: TraceSample) -> str:
    return " ".join(str(token) for token in sample.sequence)


def _run_sequence_logreg_experiment(
    cfg: Any,
    train_samples: list[TraceSample],
    val_samples: list[TraceSample],
    test_samples: list[TraceSample],
    output_dir: Path,
) -> dict[str, Any]:
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(int(cfg.model.ngram_min), int(cfg.model.ngram_max)),
        min_df=int(cfg.model.min_df),
    )
    classifier = LogisticRegression(
        max_iter=int(cfg.model.max_iter),
        class_weight="balanced",
        C=float(cfg.model.c),
    )

    train_texts = [_stringify_sequence(sample) for sample in train_samples]
    val_texts = [_stringify_sequence(sample) for sample in val_samples]
    test_texts = [_stringify_sequence(sample) for sample in test_samples]
    y_train = [int(sample.label) for sample in train_samples]
    y_val = [int(sample.label) for sample in val_samples]
    y_test = [int(sample.label) for sample in test_samples]

    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    x_test = vectorizer.transform(test_texts)
    classifier.fit(x_train, y_train)

    val_scores = classifier.predict_proba(x_val)[:, 1].tolist()
    val_pred = classifier.predict(x_val).tolist()
    val_metrics = compute_binary_classification_metrics(y_val, val_pred, val_scores)
    val_metrics["y_true"] = y_val
    val_metrics["y_pred"] = val_pred
    val_metrics["y_score"] = val_scores

    test_scores = classifier.predict_proba(x_test)[:, 1].tolist()
    test_pred = classifier.predict(x_test).tolist()
    test_metrics = compute_binary_classification_metrics(y_test, test_pred, test_scores)
    test_metrics["y_true"] = y_test
    test_metrics["y_pred"] = test_pred
    test_metrics["y_score"] = test_scores

    checkpoint_path = output_dir / "sequence_logreg.joblib"
    try:
        import joblib

        joblib.dump({"vectorizer": vectorizer, "classifier": classifier}, checkpoint_path)
    except Exception:
        checkpoint_path = output_dir / "sequence_logreg.pkl"

    print(
        f"[SEQ:logreg] val_f1={val_metrics['f1']:.4f} "
        f"test_accuracy={test_metrics['accuracy']:.4f} "
        f"test_f1={test_metrics['f1']:.4f} "
        f"test_roc_auc={test_metrics['roc_auc']:.4f}",
        flush=True,
    )
    return {
        "model_name": "sequence_logreg",
        "paradigm": "supervised",
        "checkpoint": str(checkpoint_path),
        "history": [],
        "validation": val_metrics,
        "test": test_metrics,
        "vocab_size": int(len(vectorizer.vocabulary_)),
    }


def _train_sequence_epoch(
    model: GRUSequenceClassifier,
    loader: TorchDataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(tokens, lengths)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_examples += labels.size(0)
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def _evaluate_sequence_classifier(
    model: GRUSequenceClassifier,
    loader: TorchDataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    for batch in loader:
        tokens = batch["tokens"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        logits = model(tokens, lengths)
        scores = torch.softmax(logits, dim=1)[:, 1]
        predictions = logits.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())
        y_score.extend(scores.cpu().tolist())
    metrics = compute_binary_classification_metrics(y_true, y_pred, y_score)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_score"] = y_score
    return metrics


def _run_sequence_gru_experiment(
    cfg: Any,
    train_samples: list[TraceSample],
    val_samples: list[TraceSample],
    test_samples: list[TraceSample],
    output_dir: Path,
) -> dict[str, Any]:
    device = resolve_device(cfg.device)
    vocab, train_loader, val_loader, test_loader = _build_sequence_loaders(
        cfg, train_samples, val_samples, test_samples
    )
    model = GRUSequenceClassifier(
        vocab_size=len(vocab),
        embedding_dim=int(cfg.model.embedding_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        num_layers=int(cfg.model.num_layers),
        dropout=float(cfg.model.dropout),
        pad_index=0,
        bidirectional=bool(cfg.model.bidirectional),
    ).to(device)
    optimizer = _make_optimizer(model.parameters(), cfg)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -1.0
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = _train_sequence_epoch(model, train_loader, optimizer, device)
        train_metrics = _evaluate_sequence_classifier(model, train_loader, device)
        val_metrics = _evaluate_sequence_classifier(model, val_loader, device)
        history.append(
            {
                "epoch": float(epoch),
                "train/loss": float(train_loss),
                "train/accuracy": float(train_metrics["accuracy"]),
                "train/f1": float(train_metrics["f1"]),
                "val/accuracy": float(val_metrics["accuracy"]),
                "val/f1": float(val_metrics["f1"]),
                "val/roc_auc": float(val_metrics["roc_auc"]),
            }
        )
        print(
            f"[SEQ:gru] epoch={epoch:03d}/{cfg.train.epochs:03d} "
            f"train_loss={train_loss:.4f} train_f1={train_metrics['f1']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_roc_auc={val_metrics['roc_auc']:.4f}",
            flush=True,
        )
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_state = {name: parameter.detach().cpu().clone() for name, parameter in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = output_dir / "best_sequence_gru.pt"
    torch.save(model.state_dict(), checkpoint_path)
    val_metrics = _evaluate_sequence_classifier(model, val_loader, device)
    test_metrics = _evaluate_sequence_classifier(model, test_loader, device)
    print(
        f"[SEQ:gru] final validation_f1={val_metrics['f1']:.4f} "
        f"test_accuracy={test_metrics['accuracy']:.4f} "
        f"test_f1={test_metrics['f1']:.4f} "
        f"test_roc_auc={test_metrics['roc_auc']:.4f}",
        flush=True,
    )
    return {
        "model_name": "sequence_gru",
        "paradigm": "supervised",
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "history": history,
        "validation": val_metrics,
        "test": test_metrics,
        "vocab_size": len(vocab),
    }


def run_sequence_experiment(
    cfg: Any,
    train_samples: list[TraceSample],
    val_samples: list[TraceSample],
    test_samples: list[TraceSample],
    output_dir: Path,
) -> dict[str, Any]:
    if cfg.model.name == "sequence_logreg":
        result = _run_sequence_logreg_experiment(cfg, train_samples, val_samples, test_samples, output_dir)
    elif cfg.model.name == "sequence_gru":
        result = _run_sequence_gru_experiment(cfg, train_samples, val_samples, test_samples, output_dir)
    else:
        raise ValueError(f"Unsupported sequence model: {cfg.model.name}")

    _save_json(
        output_dir / f"metrics_{cfg.model.name}.json",
        {
            "model_name": cfg.model.name,
            "paradigm": "supervised",
            "history": result.get("history", []),
            "validation": _compact_metrics(result["validation"]),
            "test": _compact_metrics(result["test"]),
            "checkpoint": result.get("checkpoint"),
            "vocab_size": result.get("vocab_size"),
        },
    )
    classification_report = result["test"].get("classification_report")
    if classification_report:
        (output_dir / f"classification_report_{cfg.model.name}.txt").write_text(
            classification_report,
            encoding="utf-8",
        )
    result["validation"] = _compact_metrics(result["validation"])
    result["test"] = _compact_metrics(result["test"])
    return result


def run_pagerank_experiment(
    cfg: Any,
    train_samples: list[TraceSample],
    val_samples: list[TraceSample],
    test_samples: list[TraceSample],
    output_dir: Path,
) -> dict[str, Any]:
    normal_train_sequences = [sample.sequence for sample in train_samples if sample.label == 0]
    if not normal_train_sequences:
        raise ValueError("PageRank training requires at least one normal sample in the train split.")

    detector = PageRankAnomalyDetector(
        window_size=cfg.model.window_size,
        distance_threshold=cfg.model.distance_threshold,
        anomaly_rate_threshold=cfg.model.anomaly_rate_threshold,
        default_edge_weight=cfg.model.default_edge_weight,
        max_patterns=cfg.model.max_patterns,
        random_state=cfg.seed,
    )
    detector.fit(normal_train_sequences)

    val_predictions = [detector.score_sequence(sample.sequence) for sample in val_samples]
    val_labels = [sample.label for sample in val_samples]
    threshold_selection: dict[str, float] | None = None

    if cfg.model.threshold_search.enabled and len(set(val_labels)) > 1:
        threshold_selection = select_pagerank_thresholds(
            predictions=val_predictions,
            labels=val_labels,
            distance_thresholds=list(cfg.model.threshold_search.distance_thresholds),
            anomaly_rate_thresholds=list(cfg.model.threshold_search.anomaly_rate_thresholds),
        )
        detector.distance_threshold = threshold_selection["distance_threshold"]
        detector.anomaly_rate_threshold = threshold_selection["anomaly_rate_threshold"]

    val_scores = [
        float(sum(distance > detector.distance_threshold for distance in prediction.distances) / len(prediction.distances))
        if prediction.distances
        else 0.0
        for prediction in val_predictions
    ]
    val_pred = [int(score > detector.anomaly_rate_threshold) for score in val_scores]
    val_metrics = compute_binary_classification_metrics(val_labels, val_pred, val_scores)
    val_metrics["y_true"] = val_labels
    val_metrics["y_pred"] = val_pred
    val_metrics["y_score"] = val_scores

    test_predictions = [detector.score_sequence(sample.sequence) for sample in test_samples]
    test_labels = [sample.label for sample in test_samples]
    test_scores = [prediction.anomaly_rate for prediction in test_predictions]
    test_pred = [int(score > detector.anomaly_rate_threshold) for score in test_scores]
    test_metrics = compute_binary_classification_metrics(test_labels, test_pred, test_scores)
    test_metrics["y_true"] = test_labels
    test_metrics["y_pred"] = test_pred
    test_metrics["y_score"] = test_scores

    metrics_payload = {
        "distance_threshold": float(detector.distance_threshold),
        "anomaly_rate_threshold": float(detector.anomaly_rate_threshold),
        "num_patterns": len(detector.patterns),
        "threshold_selection": threshold_selection,
        "validation": _compact_metrics(val_metrics),
        "test": _compact_metrics(test_metrics),
    }
    _save_json(output_dir / "metrics_pagerank.json", metrics_payload)
    (output_dir / "classification_report_pagerank.txt").write_text(
        test_metrics["classification_report"],
        encoding="utf-8",
    )
    print(
        f"[PageRank] distance_threshold={detector.distance_threshold:.4f} "
        f"anomaly_rate_threshold={detector.anomaly_rate_threshold:.4f} "
        f"validation_f1={val_metrics['f1']:.4f} "
        f"test_accuracy={test_metrics['accuracy']:.4f} "
        f"test_f1={test_metrics['f1']:.4f} "
        f"test_roc_auc={test_metrics['roc_auc']:.4f}",
        flush=True,
    )

    return {
        "model_name": "pagerank",
        "paradigm": "anomaly",
        "validation": _compact_metrics(val_metrics),
        "test": _compact_metrics(test_metrics),
        "diagnostics": {
            "validation_scores": val_scores,
            "validation_labels": val_labels,
            "test_scores": test_scores,
            "test_labels": test_labels,
        },
        "thresholds": {
            "distance_threshold": float(detector.distance_threshold),
            "anomaly_rate_threshold": float(detector.anomaly_rate_threshold),
        },
        "num_patterns": len(detector.patterns),
    }
