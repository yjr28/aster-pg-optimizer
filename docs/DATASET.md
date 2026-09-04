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

## Resumable collection

A full JOB run is sharded by query. A query shard is atomically published only after discovery and all repeated measurements for all of that query's unique physical plans complete. Failures are recorded separately, so an interrupted sweep can resume without treating half-written query data as complete.

`job_finalize.py` publishes one training JSONL only when:

- the expected number of query shards is present;
- no unresolved failure records remain;
- every record belongs to one experiment and dataset version;
- shard/query IDs agree;
- the full raw-dataset integrity audit passes.

## Split regimes

Evaluation regimes, from weakest to strongest:

1. Random plan split — diagnostic only; never the primary headline.
2. Query-template holdout — implemented and currently the primary split.
3. Parameter holdout within known templates.
4. Relation/table holdout where workload structure permits.
5. Workload shift.

`template_holdout` moves **entire templates**, including all candidate plans and parameterizations, to one side of the split.

## Learning objectives

Aster currently compares these baselines on the same held-out measured candidates:

- PostgreSQL estimated cost;
- absolute log-runtime Ridge regression;
- query-normalized relative-runtime Ridge regression;
- pairwise logistic plan ranking;
- random-forest runtime regression.

Model prediction error is secondary. The primary evaluator executes the lower-ranked candidate virtually against its measured runtime and reports speedup/regression versus the same query's native PostgreSQL plan.

## Dataset-size claims

Aster counts **unique structural query plans**, not planner interventions and not repeated executions, when reporting dataset size. The aspirational 180k-plan target is not a current result.
