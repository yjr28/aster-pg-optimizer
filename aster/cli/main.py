from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from aster.artifacts import load_model, save_model
from aster.candidates import CandidateCollector, default_candidates
from aster.data import audit_dataset
from aster.data.jsonl import append_observations
from aster.data.load import load_training_examples
from aster.experiments import evaluate_fallback_policy, evaluate_ranking, fallback_pareto_sweep, template_holdout
from aster.integration.psql import PsqlExplainRunner, read_query
from aster.models import PostgresCostRanker, RidgeRuntimeModel, RuntimeEnsemble
from aster.ranking import rank_with_fallback


def _revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _runner(args) -> PsqlExplainRunner:
    dsn = args.dsn or os.environ.get("ASTER_DSN")
    if not dsn:
        raise SystemExit("database DSN required via --dsn or ASTER_DSN")
    return PsqlExplainRunner(dsn, timeout_s=args.timeout)


def cmd_collect(args) -> int:
    runner = _runner(args); query = read_query(args.sql_file); collector = CandidateCollector(runner)
    discovered = collector.discover(query, default_candidates()); experiment_id = args.experiment_id or str(uuid4()); total = 0
    for candidate in discovered:
        total += append_observations(args.out, collector.measure(
            query, candidate, workload=args.workload, query_id=args.query_id,
            dataset_version=args.dataset_version, run_seed=args.seed,
            code_revision=args.code_revision or _revision(), experiment_id=experiment_id,
            query_template=args.query_template, parameter_key=args.parameter_key,
            warmups=args.warmups, repetitions=args.repetitions))
    print(json.dumps({"experiment_id": experiment_id, "query_id": args.query_id,
                      "candidate_specs": len(default_candidates()), "unique_plans": len(discovered),
                      "observations_written": total, "out": str(args.out)}, sort_keys=True))
    return 0


def cmd_train(args) -> int:
    integrity = audit_dataset(args.dataset)
    if not integrity.ok:
        raise SystemExit("dataset integrity audit failed:\n- " + "\n- ".join(integrity.errors))
    examples = load_training_examples(args.dataset)
    split = template_holdout(examples, test_fraction=args.test_fraction, seed=args.seed)
    train_examples, test_examples = list(split.train), list(split.test)
    postgres_cost = PostgresCostRanker(); ridge = RidgeRuntimeModel(alpha=args.ridge_alpha).fit(train_examples)
    model = RuntimeEnsemble(trees=args.trees, seed=args.seed, min_samples_leaf=args.min_samples_leaf).fit(train_examples)
    baseline_metrics = {
        "postgres_estimated_cost": asdict(evaluate_ranking(postgres_cost, test_examples)),
        "ridge_log_runtime": asdict(evaluate_ranking(ridge, test_examples)),
        "random_forest_runtime": asdict(evaluate_ranking(model, test_examples)),
    }
    metadata = {
        "model": "random_forest_runtime_baseline", "seed": args.seed, "dataset": str(args.dataset),
        "dataset_integrity": asdict(integrity), "train_examples": len(split.train), "test_examples": len(split.test),
        "train_templates": sorted(split.train_groups), "test_templates": sorted(split.test_groups),
        "baseline_metrics": baseline_metrics, "fallback_metrics": asdict(evaluate_fallback_policy(model, test_examples)),
        "fallback_pareto": [{"max_log_std": p.max_log_std, "min_predicted_gain": p.min_predicted_gain,
                             "metrics": asdict(p.metrics)} for p in fallback_pareto_sweep(model, test_examples)],
        "code_revision": _revision(), "python": sys.version, "platform": platform.platform(),
    }
    save_model(args.model_out, model, metadata); print(json.dumps(metadata, indent=2, sort_keys=True)); return 0


def cmd_optimize(args) -> int:
    runner = _runner(args); query = read_query(args.sql_file); model = load_model(args.model); collector = CandidateCollector(runner)
    discovered = collector.discover(query, default_candidates())
    decision = rank_with_fallback(model, discovered, max_log_std=args.max_log_std,
                                  min_predicted_gain=args.min_predicted_gain, domain_margin=args.domain_margin)
    measured = collector.measure(query, decision.selected, workload=args.workload, query_id=args.query_id,
                                 dataset_version=args.dataset_version, run_seed=args.seed, code_revision=_revision(),
                                 experiment_id=args.experiment_id or str(uuid4()), query_template=args.query_template,
                                 parameter_key=args.parameter_key, warmups=args.warmups, repetitions=args.repetitions)
    print(json.dumps({"selected_candidate": decision.selected.spec.candidate_id,
                      "selected_settings": decision.selected.spec.settings, "fallback": decision.fallback,
                      "reason": decision.reason, "decision_overhead_ms": decision.decision_overhead_ms,
                      "measured_execution_ms": [o.execution_time_ms for o in measured], "unique_candidates": len(discovered),
                      "predictions": [{"candidate": x.candidate.spec.candidate_id, "runtime_ms": x.prediction.runtime_ms,
                                       "log_std": x.prediction.log_std} for x in decision.ranked]}, indent=2, sort_keys=True))
    return 0


def cmd_audit_dataset(args) -> int:
    report = audit_dataset(args.dataset); print(json.dumps(asdict(report), indent=2, sort_keys=True)); return 0 if report.ok else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aster"); sub = parser.add_subparsers(dest="command", required=True)
    def db_args(p):
        p.add_argument("--dsn"); p.add_argument("--timeout", type=int, default=120); p.add_argument("--sql-file", type=Path, required=True)
        p.add_argument("--workload", required=True); p.add_argument("--query-id", required=True); p.add_argument("--query-template")
        p.add_argument("--parameter-key"); p.add_argument("--dataset-version", required=True); p.add_argument("--seed", type=int, default=7)
        p.add_argument("--warmups", type=int, default=1); p.add_argument("--repetitions", type=int, default=3); p.add_argument("--experiment-id")
    collect = sub.add_parser("collect"); db_args(collect); collect.add_argument("--out", type=Path, required=True); collect.add_argument("--code-revision"); collect.set_defaults(func=cmd_collect)
    train = sub.add_parser("train"); train.add_argument("--dataset", type=Path, required=True); train.add_argument("--model-out", type=Path, required=True)
    train.add_argument("--test-fraction", type=float, default=0.2); train.add_argument("--seed", type=int, default=7); train.add_argument("--trees", type=int, default=128)
    train.add_argument("--min-samples-leaf", type=int, default=2); train.add_argument("--ridge-alpha", type=float, default=1.0); train.set_defaults(func=cmd_train)
    audit = sub.add_parser("audit-dataset"); audit.add_argument("--dataset", type=Path, required=True); audit.set_defaults(func=cmd_audit_dataset)
    optimize = sub.add_parser("optimize"); db_args(optimize); optimize.add_argument("--model", type=Path, required=True)
    optimize.add_argument("--max-log-std", type=float, default=0.45); optimize.add_argument("--min-predicted-gain", type=float, default=0.10)
    optimize.add_argument("--domain-margin", type=float, default=0.15); optimize.set_defaults(func=cmd_optimize)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv); return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
