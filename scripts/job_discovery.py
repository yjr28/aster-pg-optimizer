from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

from aster.candidates import CandidateCollector, default_candidates, research_candidates
from aster.integration import PsqlExplainRunner
from aster.workloads import build_job_manifest, load_job_queries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Measure candidate-to-physical-plan yield across JOB without executing queries")
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--dsn")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--candidate-set", choices=("fast", "research"), default="research")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-partial-workload", action="store_true")
    args = parser.parse_args(argv)

    dsn = args.dsn or os.environ.get("ASTER_DSN")
    if not dsn:
        raise SystemExit("database DSN required via --dsn or ASTER_DSN")
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    strict = not args.allow_partial_workload
    workload = build_job_manifest(args.query_dir, strict=strict)
    if workload.workload_sha256 != preflight.get("workload_sha256"):
        raise SystemExit("JOB query checkout does not match preflight workload hash")

    specs = research_candidates() if args.candidate_set == "research" else default_candidates()
    collector = CandidateCollector(PsqlExplainRunner(dsn, timeout_s=args.timeout))
    per_query = []
    for query in load_job_queries(args.query_dir, strict=strict):
        report = collector.discover_report(query.sql, specs)
        per_query.append({
            "query_id": query.query_id,
            "family": query.family,
            "attempted_interventions": report.attempted_interventions,
            "unique_plan_count": report.unique_plan_count,
            "duplicate_interventions": report.duplicate_interventions,
            "uniqueness_ratio": report.uniqueness_ratio,
            "plan_groups": [
                {
                    "fingerprint": group.fingerprint,
                    "representative_candidate_id": group.representative_candidate_id,
                    "candidate_ids": list(group.candidate_ids),
                }
                for group in report.plan_groups
            ],
        })

    unique_counts = [row["unique_plan_count"] for row in per_query]
    attempted = sum(row["attempted_interventions"] for row in per_query)
    unique = sum(unique_counts)
    payload = {
        "candidate_set": args.candidate_set,
        "candidate_specs": len(specs),
        "workload_sha256": workload.workload_sha256,
        "query_count": len(per_query),
        "attempted_interventions": attempted,
        "unique_query_plans": unique,
        "duplicate_interventions": attempted - unique,
        "overall_uniqueness_ratio": unique / attempted if attempted else 0.0,
        "unique_plans_per_query": {
            "minimum": min(unique_counts),
            "median": statistics.median(unique_counts),
            "maximum": max(unique_counts),
        },
        "queries": per_query,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
