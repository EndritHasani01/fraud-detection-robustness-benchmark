# Glossary

Average precision is a ranking metric that is useful for imbalanced binary classification. In fraud detection, the fraud class is often rare, so average precision helps show whether the model places fraud nodes near the top of its ranked predictions.

Canonical graph view means the graph representation used by downstream model evaluation. In the current configs, the canonical view is homogeneous, meaning relation types are merged into one graph view for comparability across models.

Camouflage means a stress scenario where fraud nodes are made to look more normal. Feature camouflage changes node features. Relation camouflage changes neighborhood structure by adding benign-looking neighbors and optionally removing suspicious fraud-to-fraud edges.

Clean graph means the unperturbed benchmark graph after loading, type normalization, split-mask creation, and conversion to the canonical evaluation view.

Config means a frozen JSON experiment definition under `configs/`. It defines datasets, splits, models, scenarios, severities, seeds, metrics, and disclosure notes.

Graph seed means the seed used for graph perturbation randomness. It controls choices such as which edges are rewired or which fraud nodes are camouflaged.

Graph variant means one materialized clean or stressed graph state. Variants are recorded in `graph_variants.csv` and usually stored as DGL graph binaries under `runs/<experiment_name>/graphs/`.

Heterophily means edges often connect nodes with different labels. In graph fraud detection, heterophily can hurt message passing because neighbors may send misleading class information.

Homogeneous graph means a graph with one merged node and edge type view. The current benchmark often converts YelpChi from a source heterograph into a homogeneous evaluation graph.

Heterograph-aware generation means a v3 graph view mode where a perturbation is applied to the source heterograph first, using selected relations, and then converted back to the canonical evaluation view.

Macro-F1 is the average of F1 for the normal class and F1 for the fraud class. It is useful when classes are imbalanced because it gives both classes equal weight.

Oracle scenario means a scenario that uses true labels while constructing the perturbation. Oracle scenarios are useful controlled stress tests, but they are not realistic attacker simulations and must be disclosed.

Protocol means the evaluation mode. `train_on_variant` trains and evaluates on each stressed variant. `train_clean_eval_all` trains on the clean graph and evaluates the trained model on all variants.

Realized change means what a perturbation actually changed after sampling constraints. It can differ from requested change if candidate edges are rejected because of self-loops, existing edges, duplicates, or exhausted attempts.

ROC-AUC is a threshold-free ranking metric that measures how well the model ranks positive fraud nodes above negative normal nodes.

Run key means the tuple that uniquely identifies a model evaluation row: dataset, split, scenario, severity, graph seed, training seed, model, and protocol.

Severity means the scenario-specific stress parameter. It is not calibrated across scenario families. A severity of `0.3` can mean different things for heterophily, camouflage, and noise.

Training seed means the seed used for model training randomness, including initialization and stochastic optimization behavior.

Variant audit means the companion evidence table `variant_audit.csv`, which records requested and realized perturbation behavior for every graph variant.

