# Reproducibility

## Functional environment

```bash
docker compose -f docker/compose.yml up -d
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

The Docker fixture runs PostgreSQL 17 and initializes a small join workload used for functional integration only.

## Collect observations

```bash
export ASTER_DSN='postgresql://aster:aster@localhost:5432/aster'
aster collect --sql-file demo/query.sql --workload demo --query-id demo-1 --query-template join-aggregate --parameter-key west-123 --dataset-version demo-v1 --warmups 1 --repetitions 5 --out artifacts/demo.jsonl
```

## Train

Training requires multiple templates so query-template holdout is meaningful:

```bash
aster train --dataset artifacts/workload.jsonl --model-out artifacts/runtime-baseline.joblib --test-fraction 0.2 --seed 7
```

The adjacent metadata JSON records split groups, evaluation metrics, code revision, Python version, and platform.

## Optimize and execute

```bash
aster optimize --model artifacts/runtime-baseline.joblib --sql-file demo/query.sql --workload demo --query-id demo-live --query-template join-aggregate --parameter-key west-123 --dataset-version demo-v1
```

Output includes selected planner settings, prediction/uncertainty for each unique candidate, fallback reason, decision overhead, and measured selected-plan executions.

A future benchmark release will add a versioned workload manifest and raw results under an experiment ID. Until then, the repository makes no end-to-end performance claim.
