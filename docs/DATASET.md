# Dataset

Aster stores raw measured observations as JSON Lines. Each line corresponds to one repeated execution of one unique physical plan.

## Observation schema

Each observation records workload, query ID/template/parameter key, candidate intervention ID, PostgreSQL version, dataset version, seed, code revision, UTC capture time, structural fingerprint, planner settings, planning time, execution time, repetition index, and the complete JSON EXPLAIN document.

Repeated executions are aggregated to the median runtime for training. If a supposedly identical candidate changes structural fingerprint, collection fails instead of averaging across plan drift.

## Physical-plan identity and deduplication

A plan fingerprint is computed from physical structure: operator type, relation, index, join type, parent relationship, sort key, and recursive child topology. Estimated costs/cardinalities and actual runtime values do not define physical identity.

Candidate discovery records the full intervention-to-fingerprint mapping. If 16 planner interventions produce 6 distinct physical plans, the dataset contains **6 plans**, not 16. `scripts/job_discovery.py` reports this yield before expensive runtime collection.

## JOB input identity

Aster does not vendor the JOB query files or IMDB data. The external inputs are frozen by `scripts/job_preflight.py`:

- every JOB query is SHA-256 hashed;
- all 21 expected IMDB CSV files are byte-hashed and sized;
- the query hashes produce a workload fingerprint;
- the CSV hashes produce a dataset fingerprint and dataset version;
- both are combined into one `benchmark_input_sha256`.

This prevents observations from different query checkouts or IMDB snapshots from being silently mixed under the generic name "JOB".

## TPC-H input identity

TPC-H is handled with the same external-input rule. `scripts/tpch_preflight.py` fingerprints:

- all 22 executable SQL query files;
- the eight base-table `.tbl` or `.csv` files;
- the declared scale factor;
- the declared TPC-H specification version.

The exact SQL and data bytes remain external. The preflight-derived `benchmark_input_sha256` makes a scale/specification/query/data change visible in provenance instead of treating all TPC-H runs as interchangeable.

## Resumable collection and finalization

JOB and TPC-H both use atomic per-query shards. A shard is published only after discovery and all repeated measurements for all of that query's unique physical plans complete. Failures are recorded separately, so an interrupted sweep can resume without treating half-written query data as complete.

Finalization publishes one training JSONL only when:

- the expected number of query shards is present;
- no unresolved failure records remain;
- every record belongs to one experiment and dataset version;
- record workload and shard/query IDs agree;
- the full raw-dataset integrity audit passes.

JOB and TPC-H share the same finalization/audit implementation rather than maintaining workload-specific definitions of "complete".

## Combining corpora

`combine_datasets` / `scripts/combine_datasets.py` creates cross-workload corpora only from individually audited JSONL inputs. It:

- records every input SHA, workload, and dataset version;
- can require at least two distinct workloads;
- rejects duplicate observation identities across inputs;
- writes the combined JSONL atomically;
- re-runs `audit_dataset` on the combined file;
- writes an adjacent combined-corpus manifest.

This is the intended path for workload-shift experiments such as JOB + TPC-H.

## Split regimes

Implemented evaluation regimes are:

1. **Query-template holdout** — entire templates, including all candidates and parameterizations, stay on one side of the split.
2. **Parameter holdout** — the unit is `(query_template, parameter_key)`; all candidate plans for one parameterization stay together and every template must remain represented in training.
3. **Whole-workload holdout** — complete named workloads such as JOB or TPC-H are held out.
4. **Query-group holdout for calibration** — used only inside the primary training side to fit uncertainty calibration; the final test set is never used to calibrate intervals.

A random plan split is intentionally not a headline regime because it leaks near-identical query context across train/test. Relation/table holdout remains a possible future robustness extension where workload structure supports a defensible definition.

## Learning objectives

Aster compares these baselines on the same fit subset and the same held-out measured candidates:

- PostgreSQL estimated cost;
- absolute log-runtime Ridge regression;
- query-normalized relative-runtime Ridge regression;
- pairwise logistic plan ranking;
- random-forest runtime regression.

Model prediction error is secondary. The ranking evaluator chooses a candidate and scores it using its **measured** runtime against the same query's native PostgreSQL plan.

## Uncertainty calibration

The random-forest ensemble exposes raw tree disagreement, but Aster does not treat that raw number as a calibrated probability statement. A split-conformal layer is fitted from a query-group calibration slice drawn only from the training side.

Saved experiment metadata records target coverage, calibration coverage, final held-out test coverage, interval width, and the conformal quantile. Fallback can then require a minimum gain even under the conservative comparison `candidate upper bound` versus `native lower bound`.

These intervals are still empirical, finite-sample statements tied to the calibration distribution; they are not a guarantee under arbitrary distribution shift.

## Offline robustness matrix versus live benchmark

`scripts/robustness_matrix.py` runs the implemented split regimes over finalized measured-plan corpora. Results are explicitly labeled `offline_measured_plan_replay`: plan labels came from real PostgreSQL measurements, but the experiment is not a fresh randomized paired database execution.

A regime the corpus cannot support is recorded as `unsupported_for_dataset` with its reason. It is never silently removed from the matrix.

Live performance claims require the separate paired benchmark pipeline and its environment/model/workload fingerprints.

## Dataset-size claims

Aster counts **unique structural query plans**, not planner interventions and not repeated executions, when reporting dataset size. The aspirational 180k-plan target is not a current result.
