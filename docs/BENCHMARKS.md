# Benchmarks

Aster's benchmark is query execution, not ML prediction accuracy.

## Required headline statistics

For native PostgreSQL, learned ranking without fallback, and learned ranking with fallback, report geometric-mean execution time/speedup, median, p95, p99 when justified, fractions improved/regressed, worst regression, maximum speedup, decision overhead, end-to-end latency, and fallback frequency.

Full per-query distributions must be retained. Winning subsets are not sufficient.

## Current measured results

**None yet.** Unit-test runtimes, CI timings, synthetic demo timings, and candidate-yield counts are not end-to-end optimizer benchmark results.

| Metric | Native | Aster no fallback | Aster fallback |
|---|---:|---:|---:|
| Geometric-mean execution time | — | — | — |
| Geometric-mean speedup | 1.00x reference | — | — |
| p95 end-to-end latency | — | — | — |
| Queries improved | — | — | — |
| Queries regressed | — | — | — |
| Worst regression | — | — | — |
| Median decision overhead | 0 | — | — |
| Fallback rate | 0% | 0% | — |

## Candidate-generation yield

Candidate generation is measured separately from execution performance. Aster reports:

- planner interventions attempted;
- structurally unique plans produced;
- duplicate interventions;
- unique-plan ratio;
- minimum/median/maximum unique plans per query;
- the exact candidate IDs that collapsed to each physical-plan fingerprint.

These statistics determine dataset efficiency but are **not speedup metrics**.

## Paired single-query benchmark

`aster benchmark` performs paired live execution between PostgreSQL's native plan and Aster's final selected plan.

When the physical plans differ:

1. warm both variants;
2. randomize native/Aster order independently for each repetition;
3. execute both with `EXPLAIN ANALYZE`;
4. verify each execution retained the discovery-time physical fingerprint;
5. retain every execution/planning sample.

When fallback selects the native physical plan, the plan executes once per repetition and that one sample is shared across native/Aster execution statistics. Aster does not double-run the same plan and pretend the samples are independent.

The artifact separates:

- PostgreSQL execution time;
- PostgreSQL planning + execution time;
- ranking-only model/fallback overhead;
- full selection overhead including candidate discovery;
- end-to-end Aster latency with selection overhead charged to Aster.

## Workload-level benchmark protocol

Every published workload run must record PostgreSQL version/settings, machine CPU/RAM/OS/storage, workload/version/scale, schema/index fingerprint, statistics state, query/parameter manifest, warmups/repetitions, cache policy, code revision, model artifact/config/seed, split definition, and raw per-query measurements.

For JOB, the query checkout and 21-table IMDB snapshot are bound by `benchmark_input_sha256` before collection. A benchmark cannot combine observations from different preflight identities.

The demo PostgreSQL schema under `docker/` is a functional CI fixture only and is excluded from resume metrics.

## Tail risk

Aster publishes gain/risk curves over fallback thresholds. The important question is not only how often Aster wins, but the worst loss admitted at a given fallback rate.

A future workload benchmark runner will aggregate the paired per-query artifacts into the headline table above. Until that exists and is run on a documented benchmark host, Aster makes no end-to-end speedup claim.
