# Framework–code correspondence audit

Audit date: 2026-10-22. Result: **PASS for the original PGCAN architecture**.
Current Figure 2 is the concise seven-panel atlas, reconstructed as an editable PPT and published from its actual PowerPoint export. Panels c/d describe the original model, e the buffered spatial evaluation, and f the separate multi-relation candidate.

| Framework component | Executed implementation | Verification |
|---|---|---|
| Six prior weights, five operational multipliers | `prepare()` in original core and released `pgcan.core` | Exact prepared arrays on all 1,980 real cells |
| Target 0.70 prior + 0.30 activity | `prepare()` | Equality checked; no observed grid emissions |
| Coordinate branch 2→48→48→10, Tanh | `Model.coord` | Nine coefficients + intercept; exact forward outputs |
| Spatial context AX | `graph_for()` and sparse matrix multiply | Exact public/original matrix; symmetric 8NN, self-loops, symmetric degree normalization |
| Context branch 23→48→24→1, ReLU | `Model.context` | 9 predictors + 5 functional indicators + 9 spatial features |
| Residual sum and prior softmax | `Model.forward()` / `train()` | Exact original/public residuals; allocation conservation tested |
| Fit/prior/graph/coefficient loss | `train()` | Weights 1, 0.025, 0.015, 0.0005 preserved |
| Original optimization | `train()` | Adam lr 0.008, weight decay 0.0002, gradient clip 5, 900 epochs, five seeds |
| Accounting output | `carbon()` | Nonnegative; adopted 15 Mt total conserved |
| Evaluation-only cohorts | Original evaluation and later benchmark code | 146 regulatory, 69 energy-user, 59 shared sites; membership excluded from fitting |

## Distinguish the experimental variants

Original PGCAN and its spatial refit both register **5,339 parameters**, but their
validation scopes and graph normalization differ. Panel c of current Figure 2 describes
five full-domain 900-epoch fits. Buffered spatial evaluation instead fits scalers
on training nodes, permits messages only from training sources, row-normalizes
query weights and selects up to 450 epochs using inner spatial validation.
Outer folds use 10 km blocks and a 1 km centroid buffer, with three stochastic seeds.

The **6,638-parameter multi-relation candidate** is specified in manuscript Table 4
and audited in Figure 12. It retains one grid-node type and has three relations,
three 9→16 encoders, a 16→3 gate from [X, coordinates, class indicators], weighted
relation fusion and a 39→48→24→1 context branch. It is not a heterogeneous
building–street graph and is shown separately in panel f of the expanded Figure 2.

The diagram label “8-neighbour” means distance-based 8NN, not a fixed eight-cell
Moore neighbourhood. Tensor/neuron thumbnails and inset result charts are schematic.
GNNWR is a neural-network-weighted spatial regression method, not a graph neural network;
PGCAN is a custom implementation rather than a faithful GNNWR reproduction.

## Evidence limits

The target is a deterministic proxy derived from the declared inputs. Spatial
proxy reconstruction does not establish measured emissions accuracy. Ridge and
graph-removal results are retained even when better than PGCAN. Location screening
uses partially overlapping, POI-geolocated lists and is a distinct diagnostic.
No model was retrained to alter the recorded rankings during this figure revision.
