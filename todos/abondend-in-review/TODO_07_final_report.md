# TODO 07: Write the Final Report (Based on Benchmark Results)

Goal: produce a clear, defensible final report that explains the benchmark design, evaluation protocol, results, and implications.

Work:
- Use `Paper_Summary.md` for motivation and related work:
  - camouflage (CARE-GNN), heterophily (SEC-GFD, GAGA), noisy neighborhoods (PMP), evaluation variance (Pitfalls).
- Write Methods with exact benchmark definitions:
  - dataset(s), split ratios, seed count, metrics, thresholding rule
  - stress-test algorithms and severity levels
  - implementation details (DGL, reproducibility, caching)
- Present results with:
  - curves (metric vs severity) and mean +/- std
  - a compact summary table (clean vs max-stress and drops)
- Discuss findings:
  - which scenario hurts which method most and why (connect to paper intuitions)
  - practical takeaways for real fraud graphs
- Document limitations:
  - oracle perturbations (if used)
  - static graphs, dataset versioning, compute limits
  - any repo patches and why they were necessary

Done when:
- The report can be read independently and the experiments are reproducible from the repo + config.
- Every figure/table in the report is traceable to `results.csv`.

