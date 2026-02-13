# TODO 03: Implement Baselines (MLP + 1 Standard GNN)

Goal: build strong baselines that share the same training/evaluation protocol as the specialized methods.

Work:
- Implement an MLP baseline:
  - Input: node features only.
  - Output: fraud probability for each node.
  - Train on `train_mask`, early stop on `val_mask`, evaluate on `test_mask`.
- Implement one standard GNN baseline:
  - Choose GraphSAGE or GCN (2-layer is enough).
  - Use the same masks and the same metrics as the MLP baseline.
- Implement consistent evaluation:
  - ROC-AUC and AP computed directly from probabilities.
  - F1-macro computed using a threshold selected on validation (do not tune on test).
  - Report mean and std across seeds.
- Integrate baselines into `results.csv` output with a stable model name (example: `mlp`, `sage`).

Done when:
- For clean graphs (severity 0.0), the baselines train successfully and produce non-trivial AUC/AP (better than random).
- For stressed graphs, the baselines produce monotonic-ish degradation trends (not required, but usually expected).

Notes:
- Keep baseline hyperparameters simple; the benchmark is about robustness trends, not SOTA tuning.

