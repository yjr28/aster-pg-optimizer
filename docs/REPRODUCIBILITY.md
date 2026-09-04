# Reproducibility

## Functional environment

```bash
docker compose -f docker/compose.yml up -d
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

The Docker fixture runs PostgreSQL 17 and initializes a small join workload used for functional integration only. It is not a performance benchmark.

## Small functional observation collection

```bash
export ASTER_DSN='postgresql://aster:aster@localhost:5432/aster'
aster collect \
  --sql-file demo/query.sql \
  --workload demo \
  --query-id demo-1 \
  --query-template join-aggregate \
  --parameter-key west-123 \
  --dataset-version demo-v1 \
  --warmups 1 \
  --repetitions 5 \
  --out artifacts/demo.jsonl
```

## Join Order Benchmark pipeline

Aster intentionally does **not** vendor JOB SQL or IMDB data. Point it at your external benchmark query checkout and the corresponding 21-table CSV snapshot.

### 1. Freeze benchmark inputs

```bash
python scripts/job_preflight.py \
  --query-dir /path/to/job/queries \
  --csv-dir /path/to/imdb/csv \
  --out artifacts/job/preflight.json
```

Strict mode requires 113 query files across 33 JOB families and all 21 expected IMDB CSV files. The preflight hashes every query and every CSV and derives one `benchmark_input_sha256`. Changing either query text or dataset bytes changes that identity.

### 2. Audit candidate-generation yield without executing the workload

```bash
python scripts/job_discovery.py \
  --query-dir /path/to/job/queries \
  --preflight artifacts/job/preflight.json \
  --candidate-set research \
  --out artifacts/job/candidate-yield.json
```

The research candidate set combines physical join-method interventions with deterministic GEQO seed/effort variants. The report records attempted interventions, duplicate interventions, physical-plan fingerprints, and unique-plan yield per query. Duplicate planner settings that resolve to the same structural plan are never counted as additional plans.

### 3. Collect measured plans with resumable per-query shards

```bash
python scripts/job_collect.py \
  --query-dir /path/to/job/queries \
  --preflight artifacts/job/preflight.json \
  --output-dir artifacts/job/run-001 \
  --candidate-set research \
  --warmups 1 \
  --repetitions 3 \
  --experiment-id job-run-001
```

Each query is written atomically to `queries/<query-id>.jsonl` only after all unique candidates for that query complete. Failed queries get a separate failure record and do not leave a completed shard. Re-running the same experiment resumes from validated shards.

### 4. Finalize only a complete, failure-free corpus

```bash
python scripts/job_finalize.py \
  --collection-dir artifacts/job/run-001 \
  --out artifacts/job/job-run-001.jsonl
```

Finalization refuses unresolved failures, missing query shards, mixed experiment IDs, mixed dataset versions, non-JOB records, or any dataset-integrity error. The merged JSONL is published atomically only after the full integrity audit succeeds.

## Train and compare objectives

Training requires multiple query templates so template holdout is meaningful:

```bash
aster train \
  --dataset artifacts/job/job-run-001.jsonl \
  --model-out artifacts/models/runtime-baseline.joblib \
  --test-fraction 0.2 \
  --seed 7
```

The adjacent metadata JSON records dataset integrity, held-out template groups, PostgreSQL-cost ranking, absolute Ridge runtime, query-normalized Ridge, pairwise logistic ranking, random-forest ranking, uncertainty-fallback metrics, fallback Pareto points, code revision, Python version, and platform.

## Optimize and execute

```bash
aster optimize \
  --model artifacts/models/runtime-baseline.joblib \
  --sql-file demo/query.sql \
  --workload demo \
  --query-id demo-live \
  --query-template join-aggregate \
  --parameter-key west-123 \
  --dataset-version demo-v1
```

Output includes selected planner settings, prediction/uncertainty for each unique candidate, fallback reason, ranking overhead, and measured selected-plan executions.

## Paired live benchmark

```bash
aster benchmark \
  --model artifacts/models/runtime-baseline.joblib \
  --sql-file demo/query.sql \
  --workload demo \
  --query-id demo-live \
  --query-template join-aggregate \
  --dataset-version demo-v1 \
  --warmups 2 \
  --repetitions 15 \
  --out artifacts/benchmarks/demo-live.json
```

When Aster selects a different physical plan, each repetition executes native and Aster plans in randomized order and validates that neither plan fingerprint drifted. If fallback selects the native physical plan, the query executes once per repetition and the shared execution sample is used for both sides rather than pretending duplicate executions are independent.

The artifact separates PostgreSQL execution/planning latency from Aster's full selection overhead and retains every raw paired sample.

## Performance-claim gate

There are still no published end-to-end Aster speedup claims. A resume or README performance number must come from a versioned, complete workload benchmark artifact with documented hardware/cache/statistics policy—not from CI, unit fixtures, model RMSE, or a cherry-picked query.
