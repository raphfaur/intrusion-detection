## Intrusion Detection

The repo contains notebook, that are up to this date :
- pagerank on sendmail 
- gnn on lid_DS
    - on scenario Bruteforce_CWE-307
    - on scenario CVE-2014-0160
`src/intrusion_detection`.

### Available pipelines

- `model=gnn`: graph classification with PyTorch Geometric.
- `model=pagerank`: PageRank-based anomaly detector built from normal syscall patterns.

### GNN backbones

- `model=gnn_gcn`: weighted GCN baseline.
- `model=gnn_graphsage`: GraphSAGE with mean aggregation.
- `model=gnn_gat`: GAT with attention over directed weighted syscall transitions.
- `model=gnn` with `model.architecture=...`: direct override path if you want to tweak a preset.

### Datasets

- `dataset=adfa_ld`: expects `data/ADFA-LD/`
- `dataset=lid_ds`: expects `data/LIS-DS/` in the current repo layout

### Run
Here are some useful command to get our proposed results :

```bash
uv run python inspect_graphs.py
uv run python main.py model=gnn_gcn
uv run python main.py model=gnn_graphsage
uv run python main.py model=gnn_gat
uv run python main.py model=pagerank
uv run python main.py model=gnn_gcn report.enabled=true report.experiment_name=adfa_ld_gcn
uv run python main.py model=gnn_graphsage report.enabled=true report.experiment_name=adfa_ld_graphsage
uv run python main.py model=gnn_gat report.enabled=true report.experiment_name=adfa_ld_gat
uv run python main.py dataset=lid_ds model=gnn
uv run python main.py dataset=lid_ds dataset.scenario=CWE-89-SQL-injection model=gnn
uv run python main.py model=gnn model.architecture=gat
uv run python main.py report.enabled=true report.experiment_name=adfa_ld_gnn
uv run python main.py model=pagerank report.enabled=true report.experiment_name=adfa_ld_pagerank
uv run python main.py train.epochs=5
```

### Hydra

Configs live in `configs/`:

- `configs/dataset/*.yaml`
- `configs/model/*.yaml`
- `configs/config.yaml`

Hydra run outputs are written under `outputs/`.
The pipeline is intentionally local-only: metrics, reports, and checkpoints are stored in Hydra run directories without any external experiment tracker.

### Notes

- The `lid_ds` loader supports both extracted CSV-like traces and the `.sc` traces stored inside the per-sample `.zip` archives currently present under `data/LIS-DS/`.
- For `dataset=lid_ds`, choose a scenario at train time with `dataset.scenario=Bruteforce_CWE-307`, `dataset.scenario=CVE-2014-0160`, or `dataset.scenario=CWE-89-SQL-injection`. The default `dataset.scenario=all` loads all available scenarios.
- The default `lid_ds` split policy follows the notebook: it keeps the original `test` traces and re-splits them stratified into train/validation/test. Override with `dataset.split_strategy=predefined` if you want to use the dataset folders as-is.
- The current graph statistics suggest using message-passing models that work well on small-to-medium directed attributed graphs: `GCN` as a weighted baseline, `GraphSAGE` for more robust neighborhood aggregation, and `GAT` when transition importance may be heterogeneous across edges.
- `uv run python inspect_graphs.py` generates `report /generated_graph_inspection.tex`, `report /generated_graph_inspection.json`, and companion PDF figures for the report.
- GNN training logs are printed epoch by epoch in the console.
- With `report.enabled=true`, each run updates `report /generated_results.tex` and `report /generated_results.json`, and GNN runs also export paper-ready PDF curves under `report /figures/`.
