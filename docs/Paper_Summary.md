# Robustness to camouflage, heterophily, and noisy neighborhoods in graph-based fraud detection.

## Abstract
Graph-based fraud and anomaly detection becomes unstable when the graph is messy. Fraudsters can hide by copying normal features and by linking to benign nodes. CARE-GNN names these as feature camouflage and relation camouflage ([3, Fig. 1]). Many fraud graphs also have low homophily, so neighbors often have different labels. SEC-GFD defines heterophily edges in this way ([2, Eq. 4]). Neighborhoods are also noisy because most nodes are normal or unlabeled, and some graphs are very dense ([1, Appendix D.5]). This review compares PMP, SEC-GFD, CARE-GNN, and GAGA, and uses Pitfalls of GNN Evaluation to discuss evaluation risks ([5, Fig. 1]).

## Introduction
In these papers, graph fraud detection is mainly node-level binary classification with a small labeled set. Class imbalance is strong. PMP reports anomaly rates of 14.53% on YelpChi and 6.87% on Amazon in its benchmark statistics ([1, Appendix D.1, Table 4]). SEC-GFD lists the same two datasets with the same anomaly rates ([2, Table 1]). For T-Social, both PMP and SEC-GFD list around 3.01% anomalies ([1, Appendix D.1, Table 4] and [2, Table 1]). With such imbalance, the neighborhood of a fraud node is usually dominated by normal nodes.

This review focuses on three robustness problems. First is camouflage. CARE-GNN explains that fraudsters can change attributes to look normal, and can connect to benign nodes to blend into normal neighborhoods ([3, Fig. 1]). Second is heterophily. SEC-GFD defines heterophily using label disagreement on edges and around a node ([2, Eq. 4]). GAGA argues low homophily makes common GNN smoothing less effective for fraud graphs ([4, Sec. 1, Fig. 1]). Third is noisy neighborhoods and density. PMP reports that average degrees can be very high and that two-hop neighborhoods can explode in size, which can lead to over-smoothing and “information flooding” ([1, Appendix D.5]).

## Methodology of this review
For each of the five papers I extracted their goal, core method idea, robustness target, datasets and task setup, metrics, and high-level result trends. I cite the papers as [1] PMP, [2] SEC-GFD, [3] CARE-GNN, [4] GAGA, [5] Pitfalls of GNN Evaluation. I use [5] as a lens to judge how stable the fraud-paper conclusions may be.

## Paper summaries

### CARE-GNN: Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters. CIKM 2020.
**Goal.** CARE-GNN aims to make graph fraud detection robust against camouflage, both in features and relations ([3, Sec. 1 and Fig. 1]).

**Main idea.** CARE-GNN learns a label-aware similarity score between a node and its neighbors, then selects neighbors based on this score, and aggregates information across relations ([3, Fig. 2]). The neighbor selection uses reinforcement learning to adapt selection thresholds during training ([3, Sec. 3.3 and Algorithm 1]). The aggregation is relation-aware, so it can combine multiple relations instead of merging all edges into one graph ([3, Sec. 3.4 and Fig. 2]).

**Robustness target.** It targets camouflage directly. Filtering also helps with noisy neighborhoods, because it tries to drop misleading or weakly related neighbors ([3, Sec. 3.3]).

**Datasets and setup.** CARE-GNN evaluates on Yelp and Amazon review fraud datasets with multi-relation graphs ([3, Sec. 4.1.2 and Table 2]). For Yelp, Table 2 lists relations like R-U-R, R-S-R, R-T-R, and also a merged “ALL” relation graph ([3, Table 2]). For Amazon, the text lists U-P-U, U-S-V, and U-V-U ([3, Sec. 4.1.2]), but Table 2 shows a relation named U-S-U, so one relation name is inconsistent in the paper ([3, Table 2]). CARE-GNN varies the labeled training rate (5%, 10%, 20%, 40%) and uses mini-batch training with under-sampling ([3, Sec. 4.1.4]).

**Metrics and results.** CARE-GNN reports ROC-AUC and Recall ([3, Sec. 4.1.6]). It improves over general GNN baselines and several fraud baselines on Yelp and Amazon across label rates, with larger gains when labels are fewer ([3, Table 3]). The paper’s analysis attributes part of the gain to the neighbor selection component ([3, Sec. 6]).

**Figure 2.** CARE-GNN: reinforcement-learning neighbor selection before aggregation. Given a center node v and its neighbors, CARE-GNN computes a label-aware similarity score and uses an RL agent to adapt the selection threshold during training. Neighbors with scores above the threshold are kept for relation-aware aggregation (per relation, then merged), while low-scoring neighbors are dropped as likely camouflage or noise, reducing misleading messages.

### GAGA: Label Information Enhanced Fraud Detection against Low Homophily in Graphs. WWW 2023.
**Goal.** GAGA targets fraud detection under low homophily and argues typical GNN aggregation mixes different-label neighbors and harms detection ([4, Sec. 1, Fig. 1]).

**Main idea.** GAGA builds a Transformer-based model, but first converts a node’s neighborhood into a grouped sequence. Its Group Aggregation uses label information to create group-level neighborhood representations, then adds hop, relation, and group encodings for attention ([4, Fig. 2 and Sec. 4.3]). It uses cross-relation aggregation to combine relation-specific outputs ([4, Fig. 2]).

**Robustness target.** Its main target is heterophily. Grouping reduces harmful early averaging under low homophily ([4, Sec. 1]).

**Datasets and setup.** GAGA evaluates on YelpChi, Amazon, and an industrial dataset BF10M (Baidu) ([4, Table 1]). It reports per-relation homophily ratios ϕr, showing several relations have very low homophily ([4, Table 1]). It uses a 0.4/0.1/0.5 train/validation/test split and early stopping with patience 100 ([4, Sec. 5.4]).

**Metrics and results.** It reports AUC, AP, and F1 (including F1-macro) ([4, Sec. 5.3]). GAGA is the best method in its comparisons on YelpChi and Amazon across these metrics, and it is also best on BF10M in the reported trends ([4, Table 2 and Table 3]). The paper suggests extending to multi-class fraud categories as future work ([4, Sec. 6]).

### PMP: Partitioning Message Passing for Graph Fraud Detection. International Conference on Learning Representations. ICLR 2024.
**Goal.** PMP studies fraud detection with label imbalance and mixed homophily and heterophily. It argues shared-parameter message passing lets majority benign neighbors dominate learning, especially when heterophily exists ([1, Sec. 1 and Fig. 1]).

**Main idea.** PMP partitions message passing by neighbor label status. It uses different transformations for fraud-labeled neighbors, benign-labeled neighbors, and unlabeled neighbors ([1, Sec. 3, Eq. 3]). Unlabeled neighbors are treated as a mixture of fraud and benign transformations, controlled by a node-specific scalar αi ([1, Eq. 4]). PMP also generates root-specific weights, so different center nodes can have different transformation weights ([1, Eq. 5]).

**Robustness target.** PMP targets heterophily and noisy neighborhoods caused by imbalance. It does not prune neighbors, but separates their influence to reduce harmful mixing ([1, Sec. 3 and Sec. 6]).

**Datasets and setup.** PMP evaluates on YelpChi, Amazon, T-Finance, T-Social, and an industry dataset from Grab ([1, Sec. 5.1 and Appendix D.1, Table 4]). It uses 40%/20%/40% splits for supervised and 1%/10%/89% for semi-supervised settings ([1, Sec. 5.1]). It reports 10 trials with varied random seeds and averages results ([1, Sec. 5.1]). It also analyzes graph density and two-hop neighborhood size in Appendix D.5 ([1, Appendix D.5]).

**Metrics and results.** PMP reports AUC, F1-macro, and G-Mean ([1, Sec. 5.1]). It performs best overall in its main comparisons, and ablations show partitioning and root-specific weights both contribute to the gains ([1, Table 1–3]). PMP also observes that deeper GNNs can hurt performance on dense fraud graphs, linked to neighborhood explosion and over-smoothing ([1, Appendix D.5]).

### SEC-GFD: Revisiting Graph-Based Fraud Detection in Sight of Heterophily and Spectrum. AAAI 2024.
**Goal.** SEC-GFD proposes a spectrum-enhanced framework for graph fraud detection, motivated by heterophily and weak label utilization ([2, Abstract and Fig. 2]).

**Main idea.** SEC-GFD has a hybrid-pass spectral filtering module and a local environmental constraint module ([2, Fig. 2]). The spectral module is built from a Laplacian spectral view of graph convolution ([2, Eq. 2]) and uses multiple frequency bands, including band-pass and low-order high-pass components ([2, Eq. 5–9]). The constraint module builds a masked multi-hop neighborhood view and a KNN feature-similarity view, then applies a label-dependent constraint that encourages normal nodes to match their environment more than anomaly nodes ([2, Eq. 11–15]).

**Robustness target.** The spectral module targets heterophily by keeping useful high-frequency information instead of only smoothing ([2, Eq. 4–9]). The constraint module adds a stabilizing signal that can reduce sensitivity to noisy neighborhoods and limited labels ([2, Eq. 14–15]).

**Datasets and setup.** SEC-GFD evaluates on Amazon, YelpChi, T-Finance, and T-Social ([2, Table 1]). It uses a 0.4/0.2/0.4 train/validation/test split and runs 100 epochs for all compared methods ([2, Experiments paragraph near Table 2]).

**Metrics and results.** It reports AUC and F1-macro ([2, Table 2]). SEC-GFD is best or near-best across the four datasets in these metrics, and its ablation suggests both modules contribute ([2, Table 2–3]). It also provides analysis figures related to heterophily and frequency components ([2, Fig. 3–4]).

**Figure 5.** SEC-GFD: hybrid spectral filtering plus environmental constraints. SEC-GFD couples a spectral module (graph Laplacian → frequency analysis → hybrid-pass filter) that retains both low- and high-frequency components to address heterophily, with a constraint module that builds local-context and KNN feature-similarity views to form an environmental constraint loss. The combined objective encourages normal nodes to be context-consistent while allowing fraud nodes to remain outliers, improving stability under noisy neighborhoods and limited labels.

### Pitfalls of Graph Neural Network Evaluation. 2018.
**Goal.** This paper shows that GNN results can be unstable across splits, initializations, and tuning choices, so single-split reporting can be misleading ([5, Sec. 1 and Sec. 4]).

**Method and results.** The authors evaluate models with 100 random data splits and 20 random initializations per split, then report mean and variability in test accuracy ([5, Table 1 caption and Fig. 1 caption]). They show that model ranking can change and that fair tuning is important, including comparisons to MLP baselines ([5, Table 1]).

**Relevance here.** It is not about fraud, but it warns that robustness claims should be supported by repeated evaluation and variance reporting ([5, Fig. 1]).

## Synthesis
All four fraud papers address “bad neighbors”, but they intervene differently.

CARE-GNN filters neighbors before aggregation. This fits camouflage, because camouflage is about creating misleading neighbors ([3, Fig. 1]). The risk is that filtering depends on learned similarity, so it may remove useful neighbors when labels are very limited, and this is not fully stress-tested in the paper ([3, Sec. 4 and Sec. 6]).

PMP keeps neighbors but separates their roles. Partitioning by label status reduces majority-class dominance, and αi gives a controlled way to use unlabeled neighbors without treating them as fully trusted ([1, Eq. 3–4]). PMP’s appendix suggests that depth and neighborhood growth are practical robustness factors in dense fraud graphs ([1, Appendix D.5]).

SEC-GFD changes what information passes through the graph. The hybrid-pass filter keeps multiple frequency views instead of only low-pass smoothing, motivated by heterophily ([2, Eq. 4–9]). The environmental constraint adds another training signal tied to local context, which may reduce sensitivity to neighborhood noise ([2, Eq. 11–15]).

GAGA changes the representation format. Grouping plus encodings reduces early mixing, and attention can combine group tokens more flexibly than simple averaging ([4, Fig. 2]). It also supports multi-relation graphs through relation encoding and cross-relation aggregation ([4, Sec. 4.3 and Fig. 2]).

Across methods, labels are used to control neighbor influence, not only for a final loss. CARE-GNN uses labels for similarity learning and selection reward signals ([3, Sec. 3.2–3.3]). PMP uses labels to define partitions and unlabeled mixing ([1, Sec. 3]). SEC-GFD uses labels in a weighted classification loss and a normal-vs-anomaly constraint loss ([2, Eq. 10 and Eq. 14]). GAGA uses label information to form groups and group encoding ([4, Fig. 2]). A shared pattern is to separate or re-weight neighborhood information using label-related signals.

Evaluation is still a weak point. Pitfalls shows that repeated splits can change model ranking ([5, Fig. 1]). PMP reports 10 trials with different seeds ([1, Sec. 5.1]). The other fraud papers state fixed split ratios, but repeated splits are not clearly reported in their main text ([4, Sec. 5.4], [2, Experiments near Table 2], and [3, Sec. 4.1.4]).

**Figure 6.** Comparative synthesis: four strategies for handling “bad neighbors.” The shared failure driver—neighbors that mislead the classifier due to camouflage and heterophily—is addressed via four distinct strategies: Filtering (CARE-GNN, hard neighbor dropping), Separation (PMP, label-partitioned transformations), Representation change (GAGA, graph-to-grouped-sequence + attention), and Spectral modeling (SEC-GFD, frequency filtering). The diagram also highlights an evaluation caution from Pitfalls of GNN Evaluation: reported rankings can shift with random seeds and data splits.

## Limitations
The papers do not share one robustness benchmark. CARE-GNN focuses on camouflage ([3, Fig. 1]). PMP focuses on imbalance with mixed homophily and heterophily ([1, Sec. 1]). SEC-GFD focuses on heterophily through a spectral view ([2, Eq. 4–9]). GAGA focuses on low homophily and label utilization through grouping ([4, Sec. 1]). Because the targets are different, we cannot directly rank them as “most robust”.

Dataset versions can differ even when names match. CARE-GNN’s Amazon uses a helpful-vote labeling rule and reports 9.5% fraud in its dataset statistics ([3, Sec. 4.1.1 and Table 2]). PMP and SEC-GFD report Amazon anomaly rate 6.87% in their benchmark tables ([1, Appendix D.1, Table 4] and [2, Table 1]). So cross-paper comparisons on “Amazon” can be unfair.

Metrics are inconsistent. CARE-GNN reports AUC and Recall ([3, Sec. 4.1.6]). SEC-GFD reports AUC and F1-macro ([2, Table 2]). GAGA reports AUC, AP, and F1 variants ([4, Sec. 5.3]). PMP reports AUC, F1-macro, and G-Mean ([1, Sec. 5.1]). These metrics react differently under imbalance.

Finally, robustness is mostly evaluated on static graphs. CARE-GNN motivates active camouflage, but tests on fixed datasets ([3, Sec. 1 and Sec. 4]). The other fraud papers also do not test an adaptive attacker. So robustness here mainly means robustness to heterophily, density, and label scarcity in the given datasets.

## Conclusion
The reviewed methods improve robustness by controlling neighbor influence. CARE-GNN filters neighbors to fight camouflage ([3, Fig. 2]). PMP partitions message passing by label status to reduce harmful mixing under imbalance and heterophily ([1, Eq. 3–5]). SEC-GFD uses hybrid-pass spectral filtering plus a local environment constraint to keep useful frequency signals and stabilize training ([2, Eq. 5–15]). GAGA groups neighborhoods and uses attention with hop, relation, and group encodings to better handle low homophily ([4, Fig. 2]).

For a final project extension, I would build a shared robustness test suite and run it across these ideas. One part can increase heterophily using the definition in [2, Eq. 4]. Another part can add camouflage-like changes following CARE-GNN’s two camouflage types ([3, Fig. 1]). A third part can add neighbor noise or degree stress, motivated by PMP’s dense neighborhood analysis ([1, Appendix D.5]). I would evaluate with repeated seeds and, if possible, repeated splits, following the warning in [5, Fig. 1].

## References
[1]	Zhuo, W., Liu, Z., Hooi, B., He, B., Tan, G., Fathony, R., & Chen, J. (2024). Partitioning Message Passing for Graph Fraud Detection. International Conference on Learning Representations (ICLR 2024). https://proceedings.iclr.cc/paper_files/paper/2024/file/9251fa7fcc356bbfc16c040f4c030d83-Paper-Conference.pdf

[2]	Xu, F., Wang, N., Wu, H., Wen, X., Zhao, X., & Wan, H. (2024). Revisiting Graph-Based Fraud Detection in Sight of Heterophily and Spectrum. AAAI 2024. arXiv preprint arXiv:2312.06441v3. https://arxiv.org/abs/2312.06441v3

[3]	Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H., & Yu, P. S. (2020). Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters. CIKM 2020. arXiv preprint arXiv:2008.08692v1. https://arxiv.org/abs/2008.08692v1

[4]	Wang, Y., Zhang, J., Huang, Z., Li, W., Feng, S., Ma, Z., Sun, Y., Yu, D., Dong, F., Jin, J., Wang, B., & Luo, J. (2023). Label Information Enhanced Fraud Detection against Low Homophily in Graphs. The ACM Web Conference 2023 (WWW 2023). arXiv preprint arXiv:2302.10407v1. https://arxiv.org/abs/2302.10407v1

[5]	Shchur, O., Mumme, M., Bojchevski, A., & Günnemann, S. (2018). Pitfalls of Graph Neural Network Evaluation. arXiv preprint arXiv:1811.05868v2. https://arxiv.org/abs/1811.05868v2
