# Aster — Learned Query Optimization for PostgreSQL

Aster is an experimental learned query-plan ranking system for PostgreSQL. It asks PostgreSQL for a bounded set of alternative physical plans, deduplicates them structurally, predicts candidate runtime from planner-visible plan features, applies uncertainty-aware fallback, and executes the selected candidate back inside PostgreSQL.

> **Research status:** the end-to-end foundation is implemented, but Aster does **not** yet have a large benchmark corpus or a defensible speedup claim. The target numbers in the project brief are goals, not results.

## Research question

Can a learned model rank alternative PostgreSQL execution plans well enough to outperform the native planner on a meaningful subset of analytical queries **without unacceptable tail-risk regressions or ranking overhead**?

## Current end-to-end path

```mermaid
flowchart LR
    Q[SQL query] --> PG[PostgreSQL planner]
    PG --> C[Bounded planner-GUC candidates]
    C --> D[Structural canonicalization + dedup]
    D --> F[Leak-free plan features]
    F --> M[Learned runtime ensemble]
    M --> U[Uncertainty / domain / gain gate]
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
- Bounded planner-GUC candidate generation with deduplication before execution.
- Repeated `EXPLAIN ANALYZE (BUFFERS, TIMING OFF)` collection with provenance.
- Leak-free baseline features and a random-forest runtime ensemble.
- Ensemble disagreement, training-domain, and minimum-gain fallback gates.
- Query-template holdout with zero template overlap.
- Evaluation using measured selected-plan runtime vs native runtime.
- CLI: `collect`, `train`, `optimize`.
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

## Current model

The first model is deliberately simple: a random-forest ensemble over plan-summary features such as operator counts, estimated cardinalities, widths, depth, and estimated costs. Runtime-only fields are excluded from inference features. Aster already constructs plan graphs, but the graph neural model is **not implemented yet** and must beat simpler baselines before becoming primary.

## Benchmark status

| Result | Status |
|---|---|
| Unique measured plans | not yet reported |
| Held-out workload | not yet established |
| Geometric-mean speedup vs PostgreSQL | **not yet measured** |
| p95 end-to-end latency change | **not yet measured** |
| Median ranking overhead | **not yet measured** |
| Severe-regression rate with fallback | **not yet measured** |

No number will be added until traceable to reproducible experiment artifacts. See `docs/BENCHMARKS.md`.

## Important limitations

- Candidate search is currently limited to planner GUC combinations; it does not enumerate arbitrary join orders.
- There is no `pg_hint_plan` adapter or PostgreSQL planner-hook extension yet.
- The demo schema is synthetic and **cannot support resume performance claims**.
- The learned model is a flat-feature baseline, not the final graph model.
- Fallback calibration has not yet produced a gain/risk Pareto curve.
- `EXPLAIN ANALYZE` executes the query; use disposable benchmark databases.

The living plan is `docs/IMPLEMENTATION_PLAN.md`.
