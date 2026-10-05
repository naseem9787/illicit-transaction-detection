# VIVA_QA.md

Likely examiner questions with honest answers. Numbers from `docs/RESULTS.md`.

## The hard ones

**Why does Random Forest beat your model?**
Because on Elliptic a tree ensemble on the released features is simply a
better detector: Random Forest F1 0.790 and Static HGB 0.787, against 0.566
for Adaptive GraphSAGE. We did not beat it and do not claim to. Our
diagnostics (MLP, graph features, graph smoothing) did not close the gap. The
most plausible explanation is that features f94–f165 already contain
one-hop neighbour aggregates, so message passing adds little, but that is a
hypothesis: the 94/72 split is an external convention we could not verify.
GNNs also degrade more under temporal drift (GraphSAGE validation PR-AUC 0.864
falls to 0.430 on test; Static HGB holds 0.795).

**Then what is the point of your study?**
A careful answer to a narrower question: under a leakage-safe chronological
protocol, does online adaptation help a graph model, and is that effect
generic? Result: it helps GraphSAGE modestly (+0.072 F1, 5/5 seeds), it is
not generic (HGB got worse), and graph models still lose to trees. Negative
and boundary results are part of the finding.

**What is novel?**
Nothing we would call a new method. Online fine-tuning, GraphSAGE and
delayed-feedback evaluation are all existing ideas. What the study adds is a
documented, leakage-audited combination on this dataset, a multi-seed
analysis, and controls that test whether the adaptive gain is generic. We do
not claim novelty or state of the art.

**What exactly is "adaptive" here?**
After predicting time step t, its true labels are revealed (simulated delay
of one step, k = 1). The model takes one gradient step (lr 0.001) on those
labelled nodes, with a fixed class weight computed from the training split,
then predicts t + 1. No replay, no reinforcement learning, no
meta-learning. It is plain online fine-tuning.

**Is the one-step feedback delay realistic?**
We do not know. The dataset has no label-arrival timestamps. The delay is an
explicit assumption to make the experiment well-defined. k = 3 gave F1 0.534
(vs 0.562 for k = 1 on the seed-42 reference), so a longer delay reduces the
gain. We make no real-world latency claim.

**Why did adaptive HGB collapse? Is it a bug?**
We checked the obvious things: at t = 35 its F1 is identical to static HGB (the first prediction precedes any adaptation),
the unit tests confirm adaptation only uses past labels and that the ensemble
genuinely grows, and the static HGB number (0.787) is consistent with Random
Forest. We believe it is a real effect of a crude mechanism: 10 boosting
rounds at learning rate 0.1 fitted to one small snapshot is a large
perturbation compared with the GNN's single step at 0.001. Over a 15-step walk
it accumulates, and precision falls from 0.878 to 0.348. We did not tune it
afterwards, because the protocol forbids changes after seeing test. It does
not prove trees cannot adapt, only that this mechanism does not.

**Did you tune on the test set?**
No. Hyperparameters and thresholds were selected on validation (or
rolling-origin folds below t = 35). But honesty requires one more statement:
the earlier baselines and GraphSAGE models had their test results looked at in
Phases 3–7. The final table re-scores those frozen predictions. Only HGB was a
fresh test evaluation.

**Is there leakage?**
Our pipeline has none that we found: chronological split, thresholds from
validation, unknown labels excluded, and a prediction is frozen before its
labels are used. One upstream caveat (L1): the released features are globally
standardized and we cannot verify whether the statistics came from all 49
time steps.

**Is the improvement statistically significant?**
We report direction (5/5 seeds) and spread (F1 0.566 ± 0.025 against
0.494 ± 0.042). With five seeds and temporally ordered, non-independent
steps we do not claim significance. A Wilcoxon test on per-step differences
(p ≈ 0.017, seed-42 only) is exploratory.

**Why does the gain disappear after t = 43?**
Observed pattern only. Illicit prevalence in t = 43–49 is very low, so
per-step F1 is often exactly 0 for every model, including Random Forest.
We do not claim a regime boundary.

## Methodology questions

**Why GraphSAGE and not GAT, GIN, transformers?**
GraphSAGE beat GCN in a controlled comparison; further architecture search
was out of scope and the project deliberately stopped there.

**Why exclude unknown labels?**
They have no ground truth. They stay in the graph for message passing but
never enter loss or metrics.

**Why F1 on the illicit class and PR-AUC?**
Illicit nodes are about 6.5% of labelled test nodes (1,083 of 16,670). Accuracy
is misleading. PR-AUC and illicit-class F1 reflect the minority class.

**How was the threshold chosen?**
Maximum F1 on validation, then frozen. GraphSAGE static and adaptive share
one threshold (0.898) so the comparison is fair.

**Why five seeds?**
To show variability. Five is a small number, so we report mean, std, and
range, not significance. A known gap (L4): the seed-42 reference checkpoint
cannot be regenerated from a bare seed; a clean replicate gave F1 0.440 / 0.537.

**What is direction-aware GraphSAGE and why isn't it in the final table?**
It uses separate in- and out-edge branches. It gave +0.042 validation PR-AUC
in 3/3 folds, but its adaptive version was never evaluated, and the protocol
excluded untested models from the finalists. It is listed as future work.

## Limitations to state unprompted

1. Single dataset and one chronological split.
2. Simulated feedback delay (L2).
3. Feature-standardization provenance (L1).
4. Five seeds; seed-42 reproducibility gap.
5. Adaptive HGB is one crude tree mechanism.
6. Direction-aware GraphSAGE untested on test.
7. Test-set exposure in earlier phases.

## Future work

Direction-aware adaptive GraphSAGE; tree adaptation by retraining on train
plus revealed steps; replay buffers; datasets with real label timestamps;
verifying the origin of f94–f165; a temporal GNN comparison.
