from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from intrusion_detection.data import TraceSample
from intrusion_detection.data_temporal import (
    TemporalDataBundle,
    TemporalTraceData,
    build_temporal_data_bundle,
)
from intrusion_detection.metrics import compute_binary_classification_metrics
from intrusion_detection.models.tgn import TGNClassifier
from intrusion_detection.trainer import resolve_device


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"classification_report", "y_true", "y_pred", "y_score"}
    }


def _class_weights(
    dataset: list[TemporalTraceData],
    device: torch.device,
    mode: str,
    cap: float,
) -> torch.Tensor:
    n_normal = sum(s.label == 0 for s in dataset)
    n_attack = sum(s.label == 1 for s in dataset)
    if n_normal == 0 or n_attack == 0:
        return torch.ones(2, dtype=torch.float, device=device)

    imbalance_ratio = n_normal / n_attack
    normalized_mode = mode.strip().lower()
    if normalized_mode in {"none", "off", "disabled"}:
        positive_weight = 1.0
    elif normalized_mode == "balanced":
        positive_weight = imbalance_ratio
    elif normalized_mode == "sqrt_balanced":
        positive_weight = imbalance_ratio ** 0.5
    elif normalized_mode == "capped_balanced":
        positive_weight = min(imbalance_ratio, cap)
    else:
        raise ValueError(f"Unsupported TGN class_weight_mode: {mode}")

    return torch.tensor([1.0, positive_weight], dtype=torch.float, device=device)


def _node_features(sample: TemporalTraceData, device: torch.device) -> torch.Tensor | None:
    feat = getattr(sample, "node_features", None)
    return feat.to(device) if feat is not None else None


@torch.no_grad()
def _collect_tgn_outputs(
    model: TGNClassifier,
    dataset: list[TemporalTraceData],
    device: torch.device,
) -> tuple[list[int], list[float]]:
    model.eval()
    y_true: list[int] = []
    y_score: list[float] = []

    for sample in dataset:
        logits = model(
            sample.src_ids.to(device),
            sample.dst_ids.to(device),
            sample.src_dt.to(device),
            sample.dst_dt.to(device),
            node_features=_node_features(sample, device),
        )
        score = torch.softmax(logits, dim=1)[0, 1].item()
        y_true.append(sample.label)
        y_score.append(score)

    return y_true, y_score


def _metrics_from_scores(
    y_true: list[int],
    y_score: list[float],
    threshold: float,
) -> dict[str, Any]:
    y_pred = [int(score >= threshold) for score in y_score]
    metrics = compute_binary_classification_metrics(y_true, y_pred, y_score)
    metrics["threshold"] = float(threshold)
    metrics["positive_rate"] = float(sum(y_pred) / max(len(y_pred), 1))
    metrics["positive_predictions"] = int(sum(y_pred))
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_score"] = y_score
    return metrics


def _resolve_threshold_candidates(scores: list[float]) -> list[float]:
    if not scores:
        return [0.5]

    unique_scores = sorted(set(float(score) for score in scores))
    candidates = [0.0]
    candidates.extend(
        (left + right) / 2.0
        for left, right in zip(unique_scores, unique_scores[1:], strict=False)
    )
    candidates.append(unique_scores[-1] + 1e-12)
    return candidates


def _select_decision_threshold(
    y_true: list[int],
    y_score: list[float],
    threshold_cfg: Any,
) -> float:
    if isinstance(threshold_cfg, str) and threshold_cfg.strip().lower() == "auto":
        best_threshold = 0.5
        best_key = (-1.0, -1.0, -1.0, -1.0)
        for threshold in _resolve_threshold_candidates(y_score):
            metrics = _metrics_from_scores(y_true, y_score, threshold)
            key = (
                float(metrics["f1"]),
                float(metrics["accuracy"]),
                float(metrics["precision"]),
                float(threshold),
            )
            if key > best_key:
                best_key = key
                best_threshold = float(threshold)
        return best_threshold

    return float(threshold_cfg)


def _selection_score(metrics: dict[str, Any], metric_name: str) -> float:
    if metric_name == "val_f1":
        return float(metrics["f1"])
    if metric_name == "val_roc_auc":
        return float(metrics["roc_auc"])
    raise ValueError(f"Unsupported TGN selection_metric: {metric_name}")


def run_tgn_experiment(
    cfg: Any,
    train_samples: list[TraceSample],
    val_samples: list[TraceSample],
    test_samples: list[TraceSample],
    output_dir: Path,
) -> dict[str, Any]:
    """Train and evaluate a Temporal Graph Network on LID-DS traces.

    Uses gradient accumulation to simulate mini-batching over individual
    traces (TGN processes one trace per forward pass due to sequential
    memory updates).
    """
    bundle = build_temporal_data_bundle(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        max_events=cfg.model.max_events,
    )

    device = resolve_device(cfg.device)
    use_static = bool(getattr(cfg.model, "use_static_features", False))
    model = TGNClassifier(
        num_syscalls=bundle.num_syscalls,
        memory_dim=cfg.model.memory_dim,
        time_dim=cfg.model.time_dim,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
        chunk_size=int(getattr(cfg.model, "chunk_size", 50)),
        static_feature_dim=bundle.static_feature_dim if use_static else 0,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    # Halve lr when val_roc_auc stops improving for 3 epochs; prevents the
    # model from overshooting after finding an early optimum.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5,
    )

    class_weight = _class_weights(
        bundle.train_dataset,
        device,
        mode=str(getattr(cfg.model, "class_weight_mode", "none")),
        cap=float(getattr(cfg.model, "class_weight_cap", 4.0)),
    )
    grad_accum = int(getattr(cfg.model, "grad_accum_steps", 32))
    threshold_cfg = getattr(cfg.model, "decision_threshold", "auto")
    selection_metric = str(getattr(cfg.model, "selection_metric", "val_f1"))

    best_selection_value = float("-inf")
    best_threshold = 0.5
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        train_data = list(bundle.train_dataset)
        random.shuffle(train_data)

        total_loss = 0.0
        optimizer.zero_grad()
        n_train = len(train_data)

        for step, sample in enumerate(train_data):
            logits = model(
                sample.src_ids.to(device),
                sample.dst_ids.to(device),
                sample.src_dt.to(device),
                sample.dst_dt.to(device),
                node_features=_node_features(sample, device),
            )
            label = torch.tensor([sample.label], dtype=torch.long, device=device)
            # Apply sample weighting manually: cross_entropy(weight=...) cancels
            # itself out at batch_size=1 because reduction='mean'.
            loss = F.cross_entropy(logits, label) * class_weight[sample.label] / grad_accum
            loss.backward()
            total_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0 or step == n_train - 1:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % max(1, n_train // 5) == 0 or step == n_train - 1:
                print(
                    f"[TGN] epoch={epoch:03d} step={step + 1}/{n_train} "
                    f"loss={total_loss / (step + 1):.4f}",
                    flush=True,
                )

        avg_loss = total_loss / max(len(train_data), 1)
        train_y, train_score = _collect_tgn_outputs(model, bundle.train_dataset, device)
        val_y, val_score = _collect_tgn_outputs(model, bundle.val_dataset, device)
        decision_threshold = _select_decision_threshold(val_y, val_score, threshold_cfg)
        train_metrics = _metrics_from_scores(train_y, train_score, decision_threshold)
        val_metrics = _metrics_from_scores(val_y, val_score, decision_threshold)

        history.append({
            "epoch": float(epoch),
            "train/loss": float(avg_loss),
            "train/f1": float(train_metrics["f1"]),
            "val/f1": float(val_metrics["f1"]),
            "val/roc_auc": float(val_metrics["roc_auc"]),
            "decision_threshold": float(decision_threshold),
            "val/positive_rate": float(val_metrics["positive_rate"]),
        })
        scheduler.step(float(val_metrics["roc_auc"]))
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[TGN] epoch={epoch:03d}/{cfg.train.epochs:03d} "
            f"loss={avg_loss:.4f} "
            f"train_f1={train_metrics['f1']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_roc_auc={val_metrics['roc_auc']:.4f} "
            f"thr={decision_threshold:.4f} "
            f"val_pos_rate={val_metrics['positive_rate']:.4f} "
            f"lr={current_lr:.2e}",
            flush=True,
        )

        selection_value = _selection_score(val_metrics, selection_metric)
        if selection_value > best_selection_value:
            best_selection_value = selection_value
            best_threshold = float(decision_threshold)
            best_state = {
                name: param.detach().cpu().clone()
                for name, param in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = output_dir / "best_tgn.pt"
    torch.save(model.state_dict(), checkpoint_path)

    val_y, val_score = _collect_tgn_outputs(model, bundle.val_dataset, device)
    test_y, test_score = _collect_tgn_outputs(model, bundle.test_dataset, device)
    val_metrics = _metrics_from_scores(val_y, val_score, best_threshold)
    test_metrics = _metrics_from_scores(test_y, test_score, best_threshold)

    _save_json(
        output_dir / "metrics_tgn.json",
        {
            "num_syscalls": bundle.num_syscalls,
            "max_events": cfg.model.max_events,
            "history": history,
            "decision_threshold": best_threshold,
            "selection_metric": selection_metric,
            "validation": _compact_metrics(val_metrics),
            "test": _compact_metrics(test_metrics),
            "test_scores": {"y_true": test_y, "y_score": test_score},
            "checkpoint": str(checkpoint_path),
        },
    )
    (output_dir / "classification_report_tgn.txt").write_text(
        test_metrics["classification_report"],
        encoding="utf-8",
    )
    print(
        f"[TGN] final val_f1={val_metrics['f1']:.4f} "
        f"test_accuracy={test_metrics['accuracy']:.4f} "
        f"test_f1={test_metrics['f1']:.4f} "
        f"test_roc_auc={test_metrics['roc_auc']:.4f} "
        f"thr={best_threshold:.4f}",
        flush=True,
    )

    return {
        "model_name": "tgn",
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "decision_threshold": best_threshold,
        "selection_metric": selection_metric,
        "history": history,
        "validation": _compact_metrics(val_metrics),
        "test": _compact_metrics(test_metrics),
    }
