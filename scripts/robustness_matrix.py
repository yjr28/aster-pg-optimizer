from __future__ import annotations

import argparse
import json
from pathlib import Path

from aster.data import audit_dataset
from aster.data.load import load_training_examples
from aster.experiments import TrainingProtocol, run_robustness_matrix


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description="Evaluate Aster across leakage-resistant dataset splits")
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--regime",action="append",choices=("template","parameter","workload"))
    parser.add_argument("--test-fraction",type=float,default=.2)
    parser.add_argument("--calibration-fraction",type=float,default=.15)
    parser.add_argument("--conformal-alpha",type=float,default=.10)
    parser.add_argument("--min-log-scale",type=float,default=.05)
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--trees",type=int,default=128)
    parser.add_argument("--min-samples-leaf",type=int,default=2)
    parser.add_argument("--ridge-alpha",type=float,default=1.0)
    parser.add_argument("--pairwise-c",type=float,default=1.0)
    args=parser.parse_args(argv)

    integrity=audit_dataset(args.dataset)
    if not integrity.ok:
        raise SystemExit("dataset integrity audit failed:\n- "+"\n- ".join(integrity.errors))
    protocol=TrainingProtocol(
        split_regime="template",
        test_fraction=args.test_fraction,
        calibration_fraction=args.calibration_fraction,
        conformal_alpha=args.conformal_alpha,
        min_log_scale=args.min_log_scale,
        seed=args.seed,
        trees=args.trees,
        min_samples_leaf=args.min_samples_leaf,
        ridge_alpha=args.ridge_alpha,
        pairwise_c=args.pairwise_c,
    )
    regimes=tuple(args.regime) if args.regime else ("template","parameter","workload")
    matrix=run_robustness_matrix(load_training_examples(args.dataset),protocol,regimes=regimes)
    payload={
        "dataset":str(args.dataset),
        "dataset_sha256":integrity.sha256,
        "dataset_integrity":{
            "observations":integrity.observations,
            "queries":integrity.queries,
            "query_templates":integrity.query_templates,
            "unique_query_plans":integrity.unique_query_plans,
        },
        "matrix":matrix.to_jsonable(),
    }
    encoded=json.dumps(payload,indent=2,sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(encoded+"\n",encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
