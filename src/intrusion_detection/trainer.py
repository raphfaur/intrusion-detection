from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from intrusion_detection.data import TraceSample, build_graph_data_bundle
from intrusion_detection.metrics import compute_binary_classification_metrics
from intrusion_detection.models import (
    PageRankAnomalyDetector,
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


def run_gnn_experiment(
    cfg: Any,
    train_samples: list[TraceSample],
    val_samples: list[TraceSample],
    test_samples: list[TraceSample],
    output_dir: Path,
) -> dict[str, Any]:
    bundle = build_graph_data_bundle(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        scale_node_features=cfg.train.scale_node_features,
    )

    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        bundle.val_dataset,
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        bundle.test_dataset,
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
    )

    device = resolve_device(cfg.device)
    model = SyscallGraphClassifier(
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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

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
            f"[GNN] epoch={epoch:03d}/{cfg.train.epochs:03d} "
            f"train_loss={train_loss:.4f} "
            f"train_f1={train_metrics['f1']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_roc_auc={val_metrics['roc_auc']:.4f}",
            flush=True,
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = output_dir / "best_gnn.pt"
    torch.save(model.state_dict(), checkpoint_path)

    val_metrics = _evaluate_classifier(model, val_loader, device)
    test_metrics = _evaluate_classifier(model, test_loader, device)
    _save_json(
        output_dir / "metrics_gnn.json",
        {
            "architecture": cfg.model.architecture,
            "feature_names": bundle.feature_names,
            "num_syscalls": bundle.num_syscalls,
            "num_node_features": bundle.num_node_features,
            "history": history,
            "validation": _compact_metrics(val_metrics),
            "test": _compact_metrics(test_metrics),
            "checkpoint": str(checkpoint_path),
        },
    )
    (output_dir / "classification_report_gnn.txt").write_text(
        test_metrics["classification_report"],
        encoding="utf-8",
    )
    print(
        f"[GNN] final validation_f1={val_metrics['f1']:.4f} "
        f"test_accuracy={test_metrics['accuracy']:.4f} "
        f"test_f1={test_metrics['f1']:.4f} "
        f"test_roc_auc={test_metrics['roc_auc']:.4f}",
        flush=True,
    )

    return {
        "model_name": "gnn",
        "architecture": cfg.model.architecture,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "history": history,
        "validation": _compact_metrics(val_metrics),
        "test": _compact_metrics(test_metrics),
    }


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
