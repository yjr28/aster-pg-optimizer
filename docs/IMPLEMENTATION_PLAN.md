# Living implementation plan

## Phase 1 — plan evidence foundation ✅

Typed PostgreSQL EXPLAIN parser, structural canonicalization/fingerprints, explicit plan graph representation, and provenance schema.

## Phase 2 — candidate collection ✅

Bounded planner-GUC search, deduplication before measured execution, repeated EXPLAIN ANALYZE collection, and plan-drift guard.

## Phase 3 — learned baseline + fallback ✅

Leak-free flat features, random-forest runtime ensemble, ensemble-disagreement uncertainty, training cost-domain check, and minimum predicted-gain fallback.

## Phase 4 — leakage-safe evaluation ✅

Median repeated-run aggregation, query-template holdout, and measured-runtime metrics versus native PostgreSQL.

## Phase 5 — end-to-end execution + CI ✅

`collect`, `train`, `optimize` CLI; model metadata persistence; PostgreSQL 17 Docker environment; live PostgreSQL GitHub Actions smoke test.

## Phase 6 — recognized workload corpus ⏳

- Import JOB/IMDB workload under compatible licensing.
- Add TPC-H-style generated dataset/query templates.
- Version workload manifests and scale factors.
- Capture PostgreSQL config, indexes, statistics state, and hardware.
- Build first nontrivial measured dataset.

## Phase 7 — baseline study

PostgreSQL estimated-cost ranking, ridge/linear model, tree baselines, MLP, runtime regression vs pairwise ranking, and parameter/template holdouts.

## Phase 8 — graph model

Operator vocabulary/embeddings, relation/statistics features without leakage, TreeLSTM or small message-passing network, and ablations against flat baselines.

## Phase 9 — uncertainty / robustness

Calibrated uncertainty, stronger OOD features, conformal experiments, stale-statistics/data/scale/index/predicate shifts, and speedup/fallback/worst-regression Pareto curves.

## Phase 10 — stronger PostgreSQL integration

Evaluate `pg_hint_plan` exact candidate control and prototype a supported C planner hook if the learned ranker is already valuable. Measure ranking overhead separately from query execution.

## Resume readiness gate

Not complete until recognized workloads, a large unique measured-plan corpus, graph/baseline comparisons, real execution integration, calibrated fallback, full latency distributions, robustness tests, and reproducible benchmark artifacts exist. Target resume numbers are never copied into results.
