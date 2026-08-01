# Project Explanation Guide

This folder is a learning-oriented explanation of the repository. It is not a replacement for the source code, the frozen configs, or the report guide. Its purpose is to help you learn how to explain the project clearly to different audiences, starting from a plain-language version and moving toward code-level implementation details.

The repository implements a robustness benchmark for graph-based fraud detection. In simple terms, it starts from the YelpChi fraud graph, creates controlled stressed versions of that graph, trains or evaluates several fraud detection models, and writes unified results and report artifacts. The important thing to explain is not only what the scripts do, but why the project is structured as a benchmark and how the code keeps experiments reproducible.

The current checked-in repository contains historical v1 and v2 configs, plus a newer v3 implementation surface. The `AGENTS.md` working agreement describes the v2 expectations, while the local README and configs show that v3 has been added on top. The explanations in this folder focus on the current implementation, especially the v3 behavior, and mention v1/v2 where that history helps explain why the code is shaped the way it is.

## Suggested Reading Order

Start with [01_plain_english_overview.md](01_plain_english_overview.md) if you need a non-technical explanation. This file explains the project as a controlled testing lab for fraud models and avoids code-level details.

Read [02_methodology_and_research_design.md](02_methodology_and_research_design.md) when you want to explain the academic idea behind the benchmark. It covers the research question, dataset, metrics, protocols, seeds, and the reason each design choice exists.

Read [03_repository_and_runtime_pipeline.md](03_repository_and_runtime_pipeline.md) to understand how the repository runs end to end. It explains the CLI stages, how `graphs`, `matrix`, `shift`, and `plots` fit together, and why the benchmark separates graph generation from model training.

Read [04_data_configs_and_reproducibility.md](04_data_configs_and_reproducibility.md) to understand the config files, data loading, split masks, graph caching, and run keys. This is the best file for explaining reproducibility.

Read [05_stress_scenarios.md](05_stress_scenarios.md) for the most important benchmark logic. It explains heterophily, feature camouflage, relation camouflage, and random edge noise in terms of what they mean, why they were used, and how the code applies them.

Read [06_models_and_training.md](06_models_and_training.md) to explain the four integrated models and how training is wrapped consistently. It covers `mlp`, `sage`, `pmp`, and `secgfd`, plus validation thresholds and early stopping.

Read [07_results_reporting_and_plots.md](07_results_reporting_and_plots.md) to explain the output files. It covers `graph_variants.csv`, `variant_audit.csv`, `results.csv`, summary CSVs, plot CSVs, completeness diagnostics, and how the reporting layer joins performance with perturbation evidence.

Read [08_code_level_walkthrough.md](08_code_level_walkthrough.md) when you need a detailed technical walkthrough of the implementation. This file follows important functions and modules in the order the program executes.

Read [09_tests_and_quality_controls.md](09_tests_and_quality_controls.md) to understand what the test suite protects. It explains why the tests use small fake graphs, what behavior they verify, and what they do not prove.

Read [10_how_to_explain_the_project.md](10_how_to_explain_the_project.md) when preparing for a presentation, defense, or report discussion. It gives explanation scripts, common questions, and careful language for limitations.

Read [11_what_why_how.md](11_what_why_how.md) when you want a direct concept-by-concept explanation using the pattern "what is this, why did we use it, and how did we implement it."

Read [12_professor_questions_and_interesting_implementation.md](12_professor_questions_and_interesting_implementation.md) when preparing for difficult questions. It covers the most interesting implementation choices, likely professor questions, and defensible answer framing.

Read [GLOSSARY.md](GLOSSARY.md) when a term is unclear. It defines the project-specific words that appear throughout the benchmark.

## The Core Story In One Paragraph

The project is a reproducible robustness benchmark for graph-based fraud detection. It uses DGL's YelpChi dataset, fixes train/validation/test splits, creates deterministic graph variants under several stress scenarios, evaluates feature-only and graph-based models under two protocols, and exports both performance results and perturbation audits. The benchmark is designed to answer comparative questions such as which model degrades most under feature camouflage or strong heterophily. It is not a realistic fraudster simulator, so the reporting must keep oracle status, protocol meaning, and realized perturbation strength visible.
