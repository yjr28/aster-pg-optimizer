# Benchmarks

Aster's benchmark is query execution, not ML prediction accuracy.

## Required headline statistics

For native PostgreSQL, learned ranking without fallback, and learned ranking with fallback, report geometric mean execution time/speedup, median, p95, p99 when justified, fractions improved/regressed, worst regression, maximum speedup, decision overhead, end-to-end latency, and fallback frequency.

Full per-query distributions must be retained. Winning subsets are not sufficient.

## Current measured results

**None yet.** Unit-test runtimes and synthetic demo timings are not benchmark results.

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

## Measurement protocol

Every published run records PostgreSQL version/settings, machine CPU/RAM/OS/storage, workload/version/scale, schema/index fingerprint, statistics state, query/parameter manifest, warmups/repetitions, cache policy, code revision, model artifact/config/seed, split definition, and raw per-query measurements.

The demo PostgreSQL schema under `docker/` is a functional CI fixture only and is excluded from resume metrics.

## Tail risk

Aster will publish gain/risk curves over fallback thresholds. The important question is not only how often Aster wins, but the worst loss admitted at a given fallback rate.
