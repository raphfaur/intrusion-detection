from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from intrusion_detection.data import load_trace_samples, split_samples
from intrusion_detection.reporting import export_results_to_latex
from intrusion_detection.trainer import run_gnn_experiment, run_pagerank_experiment

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_ready(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _json_ready(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_json_ready(value) for value in payload]
    if isinstance(payload, tuple):
        return [_json_ready(value) for value in payload]
    if isinstance(payload, np.generic):
        return payload.item()
    return payload


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    _set_seed(cfg.seed)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    samples = load_trace_samples(cfg.dataset)
    train_samples, val_samples, test_samples = split_samples(
        samples=samples,
        test_size=cfg.train.test_size,
        val_size=cfg.train.val_size,
        random_state=cfg.seed,
        split_strategy=getattr(cfg.dataset, "split_strategy", "random"),
        source_split=getattr(cfg.dataset, "resplit_source_split", "test"),
    )

    summary = {
        "dataset": cfg.dataset.name,
        "model": cfg.model.name,
        "num_samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "train_normals": sum(sample.label == 0 for sample in train_samples),
        "train_attacks": sum(sample.label == 1 for sample in train_samples),
    }
    if cfg.model.name == "gnn":
        summary["architecture"] = cfg.model.architecture
    if cfg.dataset.name == "lid_ds":
        summary["scenario"] = getattr(cfg.dataset, "scenario", "all")
        summary["loaded_scenarios"] = sorted({sample.metadata.get("scenario", "unknown") for sample in samples})
        summary["split_strategy"] = getattr(cfg.dataset, "split_strategy", "random")

    if cfg.model.name == "gnn":
        result = run_gnn_experiment(
            cfg=cfg,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            output_dir=output_dir,
        )
    elif cfg.model.name == "pagerank":
        result = run_pagerank_experiment(
            cfg=cfg,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            output_dir=output_dir,
        )
    else:
        raise ValueError(f"Unsupported model: {cfg.model.name}")

    payload = {"summary": summary, "result": _json_ready(result)}
    (output_dir / "run_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if cfg.report.enabled:
        payload["report_export"] = export_results_to_latex(
            cfg=cfg,
            payload=payload,
            output_dir=output_dir,
        )
        (output_dir / "run_summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print(f"Output directory: {output_dir}")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
