# Aster — PostgreSQL Query-Plan Ranking Prototype

**Live demo:** https://yjr28.github.io/yjr28profile/projects/aster/

Aster is an experimental PostgreSQL query-plan ranking project. The current repository contains a **transparent executable baseline**: candidate plans are scored from PostgreSQL cost/row estimates, operator-shape signals, and an uncertainty value, then an uncertainty-aware confidence gate decides whether to accept the ranker's choice or fall back to PostgreSQL's default plan.

The important boundary is explicit: **this repository does not yet claim a trained 180k-plan ML model, a 2.34× speedup, or production latency improvements.** Those become publishable only after a real plan corpus, training pipeline, reproducible evaluation harness, and measured benchmark artifacts exist.

## Why this version exists

A learned optimizer is easy to overclaim and hard to validate. This baseline establishes the contracts that a later model has to beat:

- a concrete candidate-plan schema;
- deterministic ranking;
- explicit uncertainty;
- a fallback policy;
- tests for clear wins, uncertainty fallback, and low-margin fallback;
- a recruiter-facing browser demo that loads this repository's `aster_ranker.py` into Pyodide and calls `rank_plans` directly.

## Run it

Requires Python 3.11+ and no third-party packages.

```bash
python -m unittest -v
```

The core implementation is in [`aster_ranker.py`](aster_ranker.py). Tests are in [`test_aster_ranker.py`](test_aster_ranker.py).

## Current ranking model

Lower score is better:

```text
0.46 * log1p(postgres_cost)
+ 0.14 * log1p(estimated_rows)
+ 0.13 * seq_scans
+ 0.11 * nested_loops
+ 0.16 * uncertainty
```

The top-ranked plan is rejected in favor of PostgreSQL's default when either:

1. the best candidate's uncertainty exceeds the configured ceiling; or
2. the score margin over the runner-up is too small.

This is intentionally simple and auditable. The future learned ranker should plug into the same decision/fallback contract.

## Roadmap

1. ingest real `EXPLAIN (ANALYZE, FORMAT JSON)` plan trees;
2. build workload-safe train/validation/test splits;
3. add plan-tree featurization and learned ranking;
4. calibrate uncertainty and fallback thresholds;
5. measure selection overhead separately from execution latency;
6. publish benchmark artifacts only when reproducible.
