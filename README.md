# Aster — Learned Query Optimization for PostgreSQL

Aster is an experimental learned query-plan ranking system for PostgreSQL. It asks PostgreSQL for a bounded set of alternative physical plans, deduplicates them structurally, predicts candidate runtime from planner-visible plan features, applies calibrated uncertainty-aware fallback, and executes the selected candidate back inside PostgreSQL.

> **Research status:** the end-to-end research infrastructure is implemented, but Aster does **not** yet have a published large benchmark corpus or a defensible speedup claim. Target numbers from the project brief are goals, not results.

## Research question

Can a learned model rank alternative PostgreSQL execution plans well enough to outperform the native planner on a meaningful subset of analytical queries **without unacceptable tail-risk regressions or ranking overhead**?

## Current end-to-end path

```mermaid
flowchart LR
    Q[SQL query] --> PG[PostgreSQL planner]
    PG --> C[Bounded physical + GEQO candidates]
    C --> D[Structural canonicalization + dedup]
    D --> F[Leak-free plan features]
    F --> M[Learned runtime ensemble]
    M --> U[Conformal uncertainty + domain + gain gate]
    U -->|confident| A[Aster candidate]
    U -->|fallback| N[Native candidate]
    A --> E[EXPLAIN ANALYZE under selected GUCs]
    N --> E
    E --> O[Measured observation + provenance]
```

Aster currently integrates through **session-local PostgreSQL planner settings**, not a planner hook and not `pg_hint_plan`. The selected candidate is therefore a reproducible planner configuration; `EXPLAIN ANALYZE` executes the query under that configuration. See `docs/POSTGRES_INTEGRATION.md`.

## What is implemented

- PostgreSQL `EXPLAIN (FORMAT JSON)` tree parser and structural fingerprints.
- Explicit graph representation (nodes + parent/child edges).
- Strictly validated planner-setting candidate generation, including deterministic GEQO join-search variants.
- Structural deduplication before measured execution plus candidate-yield auditing.
- Repeated `EXPLAIN ANALYZE (BUFFERS, TIMING OFF)` collection with provenance and plan-drift detection.
- Atomic, resumable per-query collection and integrity-gated finalization.
- External Join Order Benchmark support: 113-query/33-family workload fingerprint plus exact 21-table IMDB snapshot identity.
- External TPC-H support: 22-query workload fingerprint, eight-table snapshot identity, scale/specification provenance, collection, and finalization.
- Audited multi-workload corpus merging for workload-shift experiments.
- Leak-free plan-summary features and a random-forest runtime ensemble.
- PostgreSQL-cost, absolute-runtime Ridge, query-normalized Ridge, and pairwise logistic ranking baselines.
- Split-conformal runtime intervals fitted only from a training-side query holdout.
- Fallback gates for unseen structure, feature-domain shift, ensemble disagreement, and conservative calibrated gain.
- Template, parameter, and whole-workload holdout regimes.
- Offline robustness matrix over measured candidate runtimes with unsupported regimes reported explicitly.
- Randomized paired live benchmarking with raw samples and Aster selection overhead charged.
- Full-JOB benchmark aggregation: no-fallback vs uncertainty-fallback, full distributions, tail regressions, and fallback frequency.
- Benchmark identity bound to exact model bytes, workload/data bytes, host fingerprint, PostgreSQL catalog/settings state, and candidate set.
- CLI: `collect`, `train`, `optimize`, `benchmark`, plus workload/preflight/finalization scripts.
- PostgreSQL 17 Docker environment and live GitHub Actions smoke test.

## Quick start

```bash
docker compose -f docker/compose.yml up -d
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

```bash
export ASTER_DSN='postgresql://aster:aster@localhost:5432/aster'
aster collect --sql-file demo/query.sql --workload demo --query-id demo-west-1 --query-template join-aggregate --parameter-key west-123 --dataset-version demo-v1 --out artifacts/demo.jsonl
```

The single demo query is a functional fixture, not a training benchmark.

For the standardized JOB/TPC-H collection, cross-workload merge, robustness, calibration, and paired-benchmark workflow, see `docs/REPRODUCIBILITY.md`.

## Current model

The primary executable model is deliberately simple: a random-forest ensemble over plan-summary features such as operator counts, estimated cardinalities, widths, depth, and estimated costs. Runtime-only fields are excluded from inference features. Aster constructs plan graphs, but a graph neural model is **not implemented yet**: structural model complexity must beat the simpler measured baselines before becoming primary.

## Benchmark status

| Result | Status |
|---|---|
| Unique measured research plans | not yet published |
| JOB end-to-end benchmark | infrastructure ready; no published corpus/result |
| TPC-H workload-shift benchmark | infrastructure ready; no published corpus/result |
| Geometric-mean speedup vs PostgreSQL | **not yet measured for publication** |
| p95 end-to-end latency change | **not yet measured for publication** |
| Median ranking/selection overhead | **not yet measured for publication** |
| Severe-regression rate with fallback | **not yet measured for publication** |
| Calibrated interval test coverage | protocol implemented; no published corpus result |

No number will be added until traceable to reproducible experiment artifacts. See `docs/BENCHMARKS.md`.

## Important limitations

- Candidate search is bounded; GEQO seeds expand join-order diversity but Aster still does not enumerate the full physical-plan space.
- There is no `pg_hint_plan` adapter or PostgreSQL planner-hook extension yet.
- The demo schema is synthetic and **cannot support resume performance claims**.
- The learned production model is still a flat-feature baseline; plan-graph models must earn their complexity empirically.
- Conformal intervals and fallback policies are implemented, but no calibrated gain/risk curve is a published result until measured on finalized held-out corpora.
- TPC-H query text/data and JOB/IMDB inputs remain external; Aster fingerprints them instead of redistributing benchmark material.
- `EXPLAIN ANALYZE` executes the query; use disposable benchmark databases.

The living plan is `docs/IMPLEMENTATION_PLAN.md`.
