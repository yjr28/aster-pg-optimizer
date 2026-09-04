# Living implementation plan

## Phase 1 — plan evidence foundation ✅

Typed PostgreSQL EXPLAIN parser, structural canonicalization/fingerprints, explicit plan graph representation, and provenance schema.

## Phase 2 — candidate collection ✅

Bounded planner-GUC search, deduplication before measured execution, repeated EXPLAIN ANALYZE collection, and plan-drift guard.

## Phase 3 — learned baseline + fallback ✅

Leak-free flat features, random-forest runtime ensemble, training-domain checks, minimum predicted-gain fallback, and calibrated conservative bounds.

## Phase 4 — leakage-safe evaluation ✅

Median repeated-run aggregation and measured-runtime ranking metrics versus native PostgreSQL. Implemented holdout units now include query template, parameterization, workload, dataset snapshot/scale, unseen relation, and benchmark environment.

## Phase 5 — end-to-end execution + CI ✅

`collect`, `train`, `optimize`, and paired `benchmark` CLI paths; model metadata persistence; PostgreSQL 17 Docker environment; live PostgreSQL GitHub Actions smoke coverage; randomized paired execution with plan-drift checks.

## Phase 6 — recognized workload corpus infrastructure ✅ / measured corpus ⏳

Implemented:

- External JOB/IMDB support without vendoring benchmark SQL/data.
- Strict 113-query / 33-family JOB workload fingerprints and 21-table IMDB byte fingerprints.
- External TPC-H support with 22-query / eight-table fingerprints, specification version, and scale factor.
- Resumable atomic per-query collection and integrity-gated finalization for both workloads.
- Candidate-yield auditing before expensive execution.
- Cross-workload corpus merge with duplicate-observation protection.
- PostgreSQL settings, relations, indexes, statistics state, database size, host hardware, and OS fingerprints.
- Environment identity carried through observations, finalization, merge, training, and evaluation.

Still required:

- Collect the first complete nontrivial JOB and TPC-H measured corpora on external benchmark databases.
- Report actual unique-plan yield and corpus size from those artifacts.

## Phase 7 — baseline study ✅ / real-workload results ⏳

Implemented on one shared experiment protocol:

- PostgreSQL estimated-cost ranking.
- Absolute log-runtime Ridge regression.
- Query-normalized Ridge objective.
- Pairwise logistic plan ranking.
- Flat MLP runtime baseline.
- Random-forest runtime ensemble.
- Template, parameter, workload, dataset-version, relation, and environment holdouts.

Still required:

- Run and report the complete baseline matrix on finalized real JOB/TPC-H corpora.
- Promote no learned objective based on synthetic/unit fixtures.

## Phase 8 — graph model ⏳

Operator vocabulary/embeddings, relation/statistics features without leakage, TreeLSTM or a small message-passing network, and ablations against the flat Ridge/MLP/RF baselines.

**Gate:** do not implement or promote the graph model merely for complexity. It must be evaluated on the same measured corpora and split protocol and must beat simpler baselines on database-relevant ranking metrics.

## Phase 9 — uncertainty / robustness infrastructure ✅ / perturbation evidence ⏳

Implemented:

- Ensemble disagreement and multidimensional feature-domain/OOD checks.
- Split-conformal log-runtime calibration using a query-group calibration slice carved only from the training side.
- Held-out interval coverage diagnostics.
- Conservative fallback using calibrated candidate/native bounds.
- Fallback speedup/risk Pareto sweeps.
- Offline robustness matrix for template, parameter, workload, dataset, relation, and environment shift.
- Exact benchmark-environment fingerprints so stale-statistics, index, planner-config, and hardware states can be collected as separate evidence rather than averaged together.

Still required:

- Collect controlled stale-statistics/index/data-scale/predicate/config perturbation corpora on disposable benchmark databases.
- Run live paired benchmarks under those states and compare no-fallback versus calibrated fallback tail risk.

## Phase 10 — stronger PostgreSQL integration ⏳

Evaluate `pg_hint_plan` exact candidate control and prototype a supported C planner hook only if the learned ranker already demonstrates value. Measure candidate discovery, ranking-only overhead, and full selection overhead separately from PostgreSQL execution.

## Resume readiness gate

Not complete until recognized workloads have real measured corpora, baseline/graph comparisons are evidence-backed, full live latency distributions exist, robustness perturbations are measured, calibrated fallback has a defensible gain/risk curve, and every published number is traceable to a versioned benchmark artifact. Target resume numbers are never copied into results.
