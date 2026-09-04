# Architecture

Aster separates **evidence**, **learning**, **policy**, and **execution** so a model prediction can never be confused with a measured database result.

## Components

1. **PostgreSQL integration** — `PsqlExplainRunner` opens a read-only transaction, applies a validated allow-list of `SET LOCAL enable_*` planner switches, and asks PostgreSQL for JSON plans.
2. **Candidate discovery** — `CandidateCollector.discover` requests alternatives without `ANALYZE`, fingerprints physical structure, and removes duplicates.
3. **Measurement** — unique candidates are executed repeatedly with `EXPLAIN ANALYZE, BUFFERS, TIMING OFF`; warmups are never persisted as samples.
4. **Plan representation** — the parser builds typed tree nodes; `build_plan_graph` exports explicit nodes/edges for graph models.
5. **Baseline features** — inference features are derived only from estimated/planner-visible fields.
6. **Learned model** — the current random-forest ensemble predicts log runtime and exposes tree disagreement.
7. **Risk policy** — ranking falls back to native PostgreSQL for high uncertainty, extrapolation outside the observed cost domain, or insufficient predicted gain.
8. **Evaluation** — test metrics use the already measured runtime of the selected candidate, never predicted runtime.

## Critical invariants

### No duplicate-plan inflation

Candidate IDs are search interventions, not examples. Several interventions can produce the same physical plan. Aster fingerprints the operator/relation/index/join topology before execution and preserves only the first plan for each fingerprint.

### No runtime leakage

Actual row counts, execution time, buffer activity, and other `ANALYZE` evidence may be labels/evaluation metadata but are unavailable to the ranking model at decision time. `baseline_feature_dict` consumes only planner-visible estimates and structure.

### Native plan is always present

The candidate list starts with `native`. Fallback returns that exact candidate rather than synthesizing an approximation of PostgreSQL defaults.

### Discovery/measurement stability

A candidate's measured fingerprint must match its discovery fingerprint. If schema/statistics changes cause plan drift, collection fails instead of silently averaging two physical plans under one label.

## Integration boundary

```text
client / benchmark harness
        |
        v
Aster candidate generator
        |
        +---- SET LOCAL enable_hashjoin = off; ...
        v
PostgreSQL planner
        |
        v
EXPLAIN JSON candidate
        |
        v
Aster learned ranker + fallback
        |
        +---- selected SET LOCAL configuration
        v
PostgreSQL EXPLAIN ANALYZE (real execution)
```

This genuinely influences PostgreSQL planning/execution, but it is less expressive than join-order hints or a planner hook. That distinction remains explicit in public claims.
