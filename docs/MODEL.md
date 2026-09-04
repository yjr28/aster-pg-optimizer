# Model

## Baselines first

Aster's first learned model is a random-forest runtime regressor. The graph model must earn its complexity against strong simple baselines.

Planned comparison ladder:

1. PostgreSQL root estimated cost.
2. Linear/ridge runtime regression.
3. Tree ensemble over flat plan summaries (implemented first).
4. MLP over the same summaries.
5. Graph/tree model over explicit plan structure.
6. Pairwise ranking objective if it improves plan selection.

## Current features

The baseline includes log-scaled estimated cost/cardinality/width summaries, plan depth, node count, operator counts, and join-type counts. Relation names are omitted from the first baseline to reduce easy identity memorization.

It excludes `Actual Rows`, actual/execution time, measured buffers, temp I/O from `ANALYZE`, and any label-derived feature.

## Objective

The current model learns `log1p(runtime_ms)`. This is only a baseline. Aster's system objective is **candidate ranking by actual query runtime**, not runtime RMSE.

Future experiments compare absolute runtime regression, pairwise ranking, and query-normalized relative-cost objectives under the same splits.

## Uncertainty / fallback

The current policy uses tree-prediction disagreement, a root-cost training-domain check, and a minimum predicted gain over native PostgreSQL. This is a first risk layer, not a calibrated uncertainty guarantee. The next step is a held-out Pareto curve of speedup vs fallback rate vs regression severity.

## Graph representation

`aster.features.graph.PlanGraph` contains explicit plan nodes and directed parent-to-child edges. Each node carries operator/relation/index/join identities and leak-free numeric estimates. The representation is ready for message passing, but **no GNN result is claimed yet**.
