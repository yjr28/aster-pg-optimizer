# PostgreSQL integration

## Current route: external planner-configuration ranking

Aster changes PostgreSQL planning through an allow-listed set of session-local planner GUCs such as `enable_hashjoin`, `enable_mergejoin`, `enable_nestloop`, and `enable_seqscan`.

For each intervention it begins a read-only transaction, applies `SET LOCAL` settings, requests `EXPLAIN (FORMAT JSON, SETTINGS)`, canonicalizes/deduplicates the physical plan, ranks unique plans, reapplies the selected intervention, and executes it through `EXPLAIN ANALYZE` for measurement.

This means Aster **does influence PostgreSQL execution**. It does **not** yet inject an arbitrary preconstructed plan, force exact join order, or hook planner internals.

## Why start here

The external route is portable, testable against stock PostgreSQL, and sufficient to prove the research loop before building a C extension. It also makes the intervention responsible for a plan explicit in every observation.

## Stronger routes under investigation

- `pg_hint_plan` for join-order/operator hints on supported environments;
- a PostgreSQL C extension using supported planner hooks;
- GEQO/cost-parameter exploration where scientifically useful.

A planner-hook extension is stronger technically, but should be implemented only after the learned ranker demonstrates value with the simpler route.

## Safety

`EXPLAIN ANALYZE` executes the statement. A read-only transaction blocks ordinary DML but does not make arbitrary SQL harmless: reads can be expensive and functions can have external side effects. Benchmarks belong on disposable databases or isolated replicas.
