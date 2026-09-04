# Dataset

Aster stores raw measured observations as JSON Lines. Each line corresponds to one repeated execution of one unique physical plan.

## Observation schema

Each observation records workload, query ID/template/parameter key, candidate intervention ID, PostgreSQL version, dataset version, seed, code revision, UTC capture time, structural fingerprint, planner settings, planning time, execution time, repetition index, and the complete JSON EXPLAIN document.

Repeated executions are aggregated to the median runtime for training. If a supposedly identical candidate changes structural fingerprint, the loader keeps the plans separate rather than averaging across drift.

## Deduplication

A plan fingerprint is computed from physical structure: operator type, relation, index, join type, parent relationship, sort key, and recursive child topology. Estimated costs/cardinalities and actual runtime values do not define physical identity.

## Split regimes

Planned evaluation regimes, from weakest to strongest:

1. Random plan split — diagnostic only; never the primary headline.
2. Query-template holdout — implemented and intended as the first primary split.
3. Parameter holdout within known templates.
4. Relation/table holdout where workload structure permits.
5. Workload shift.

`template_holdout` moves **entire templates**, including all candidate plans and parameterizations, to one side of the split.

## Dataset-size claims

Aster counts **unique structural plans**, not planner interventions and not repeated executions, when reporting dataset size. The aspirational 180k-plan target is not a current result.
