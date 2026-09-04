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

`aster collect` captures the current benchmark environment once, stamps its `environment_sha256` on every observation, and writes `artifacts/demo.jsonl.environment.json`. The environment fingerprint includes PostgreSQL catalog/settings/statistics/index state plus host properties. The demo remains a functional fixture, not performance evidence.

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

At run start, Aster captures `artifacts/job/run-001/environment.json`. The output directory is bound to that environment SHA: attempting to resume it against a different PostgreSQL/index/statistics/host state fails before completed shards are reused.

Each query is written atomically to `queries/<query-id>.jsonl` only after all unique candidates for that query complete. Every observation carries the same environment SHA. Failed queries get a separate failure record and do not leave a completed shard.

### 4. Finalize only a complete, failure-free corpus

```bash
python scripts/job_finalize.py \
  --collection-dir artifacts/job/run-001 \
  --out artifacts/job/job-run-001.jsonl
```

Finalization refuses unresolved failures, missing query shards, mixed experiment IDs, mixed dataset versions, mixed environment fingerprints, non-JOB records, or any dataset-integrity error. The merged JSONL is published atomically only after the full integrity audit succeeds.

## TPC-H workload-shift pipeline

TPC-H inputs are also external. Aster fingerprints executable SQL and the exact eight-table data snapshot rather than redistributing benchmark material. Record the exact specification version used by your external query/data generator.

### 1. Freeze the TPC-H query/data inputs

```bash
python scripts/tpch_preflight.py \
  --query-dir /path/to/tpch/queries \
  --data-dir /path/to/tpch/data \
  --scale-factor 1 \
  --specification-version 3.0.1 \
  --out artifacts/tpch/preflight.json
```

Strict mode requires 22 executable query files and all eight base tables. `.tbl` and `.csv` data files are supported, but ambiguous duplicate formats for one table are rejected. The preflight identity binds query bytes, data bytes, declared scale factor, and specification version.

### 2. Collect and finalize with the same integrity semantics as JOB

```bash
python scripts/tpch_collect.py \
  --query-dir /path/to/tpch/queries \
  --preflight artifacts/tpch/preflight.json \
  --output-dir artifacts/tpch/run-001 \
  --candidate-set research \
  --warmups 1 \
  --repetitions 3 \
  --experiment-id tpch-run-001

python scripts/tpch_finalize.py \
  --collection-dir artifacts/tpch/run-001 \
  --out artifacts/tpch/tpch-run-001.jsonl
```

TPC-H uses the same environment binding, atomic per-query publication, resume validation, and final integrity audit as JOB. A partial or mixed-provenance run cannot be finalized.

## Combine audited corpora for shift experiments

```bash
python scripts/combine_datasets.py \
  --dataset artifacts/job/job-run-001.jsonl \
  --dataset artifacts/tpch/tpch-run-001.jsonl \
  --out artifacts/combined/job-tpch.jsonl \
  --require-multiple-workloads
```

Every input must independently pass `audit_dataset`. The combiner records each input SHA, workload, dataset version, and environment fingerprint. Duplicate observation identity includes environment, so independently measured states are preserved instead of rejected or averaged. The combined file is written atomically and re-audited.

The same path can combine separately collected baseline/stale-statistics/index-state/configuration runs, provided each run has its own environment fingerprint.

## Train and compare objectives

The training split is explicit. The default is full query-template holdout:

```bash
aster train \
  --dataset artifacts/job/job-run-001.jsonl \
  --model-out artifacts/models/runtime-baseline.joblib \
  --split-regime template \
  --test-fraction 0.2 \
  --calibration-fraction 0.15 \
  --conformal-alpha 0.10 \
  --seed 7
```

`--split-regime` supports:

- `template`: entire query templates are unseen at test time;
- `parameter`: parameterizations are held out within each known template;
- `workload`: complete workloads such as JOB or TPC-H are held out;
- `dataset`: complete dataset snapshots/scales are held out;
- `relation`: queries involving one relation are held out while that relation is absent from all training plans;
- `environment`: complete benchmark-environment fingerprints are held out, intended for separately collected statistics/index/config/hardware shifts.

All candidate plans for a query/parameter/environment identity stay together.

The primary test split is created first. If conformal calibration is enabled, a second **query-group holdout is carved only from the training side**; the final test set is never used to fit interval calibration. Query grouping includes environment, dataset, workload, and query ID.

The saved metadata includes calibration target coverage, calibration-set coverage, held-out test coverage, interval width, split groups/notes, and ranking/fallback metrics.

The same fit subset is used for PostgreSQL estimated cost, absolute Ridge runtime, query-normalized Ridge, pairwise logistic ranking, flat MLP runtime, and random-forest runtime baselines. A future graph model must use this same protocol to make its added complexity comparable.

## Run the robustness matrix

```bash
python scripts/robustness_matrix.py \
  --dataset artifacts/combined/job-tpch.jsonl \
  --out artifacts/experiments/robustness.json
```

By default the matrix attempts `template`, `parameter`, `workload`, `dataset`, `relation`, and `environment` holdouts with one shared training protocol. A regime that the dataset cannot support is recorded as `unsupported_for_dataset` with the exact split-construction reason; it is not silently omitted. Model/training failures propagate as failures rather than being reclassified as unsupported data.

These matrix results are labeled `offline_measured_plan_replay`: the candidate runtimes are real measured labels from collection, but this is not a fresh randomized paired database benchmark. Live publication claims still require the benchmark path below.

## Controlled database-state shift protocol

Aster does not automatically mutate production or benchmark databases. To study stale statistics, index changes, planner configuration changes, or scale shifts:

1. prepare each desired state on a disposable benchmark database;
2. run the same JOB/TPC-H preflight for the unchanged workload/data identity as appropriate;
3. collect into a **new output directory** so Aster captures a new `environment.json`;
4. finalize each state independently;
5. combine the finalized corpora;
6. run `--split-regime environment` or the full robustness matrix;
7. for publishable performance evidence, run randomized paired live benchmarks in each state.

This separates state preparation from Aster's read-oriented measurement path and prevents an automated robustness script from silently dropping indexes or corrupting statistics.

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

Output includes selected planner settings, point runtime predictions, calibrated intervals when available, domain diagnostics, fallback reason, ranking overhead, current environment SHA, and measured selected-plan executions.

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

The single-query benchmark artifact embeds the full current benchmark-environment snapshot. Full JOB runs additionally bind resume to exact model SHA, benchmark-input SHA, environment SHA, and candidate-set identity.

The benchmark artifact separates PostgreSQL execution/planning latency from Aster's full selection overhead and retains every raw paired sample. Calibrated prediction bounds and the exact fallback reason are retained per candidate for post-hoc risk auditing.

## Performance-claim gate

There are still no published end-to-end Aster speedup claims. A resume or README performance number must come from a versioned, complete workload benchmark artifact with documented hardware/cache/statistics policy—not from CI, unit fixtures, offline replay, model RMSE, or a cherry-picked query.
