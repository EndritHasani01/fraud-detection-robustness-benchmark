# TODO 08 (Optional): Extensions If Time Allows

Only do these after the core benchmark (MLP + baseline GNN + PMP + SEC-GFD) is complete.

High-value extensions:
- Add CARE-GNN as an additional method focused on camouflage:
  - Fix the data path mismatch in `Repos/CARE-GNN-master/` if needed.
  - Evaluate mainly on the camouflage stress test and compare to PMP/SEC-GFD.
- Add GAGA as an additional method for low-homophily:
  - Use `Repos/GAGA-master/pytorch_gaga/` preprocessing and run scripts.
  - Integrate results into the same `results.csv` schema.
- Add a "train clean, test perturbed" setting:
  - Train once on severity 0.0 and evaluate on increasing severities.
  - This approximates test-time attacks and is often more realistic.
- Add robustness summary metrics:
  - Worst-case performance over severities.
  - Area under the robustness curve.

Done when:
- Any added method appears in `results.csv` and the report contains at least one new insight enabled by the extension.

