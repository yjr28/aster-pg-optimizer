from __future__ import annotations

import argparse
import json
from pathlib import Path

from aster.benchmarks import (
    ENVIRONMENT_DIFF_SECTIONS,
    compare_benchmark_environments,
    validate_perturbation,
)


def _load(path: Path) -> dict:
    try:
        payload=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read benchmark environment {path}: {exc}") from exc
    if not isinstance(payload,dict):
        raise SystemExit(f"benchmark environment must be a JSON object: {path}")
    return payload


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(
        description="Compare two Aster benchmark-environment snapshots by semantic state changes"
    )
    parser.add_argument("--before",type=Path,required=True)
    parser.add_argument("--after",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    parser.add_argument(
        "--require-change",
        action="store_true",
        help="exit nonzero when the two environment fingerprints are identical",
    )
    parser.add_argument(
        "--allow-section",
        action="append",
        choices=sorted(ENVIRONMENT_DIFF_SECTIONS),
        help="declare a change section that is allowed; repeat for multiple sections",
    )
    parser.add_argument(
        "--require-section",
        action="append",
        choices=sorted(ENVIRONMENT_DIFF_SECTIONS),
        help="declare an allowed section that must actually change",
    )
    args=parser.parse_args(argv)
    diff=compare_benchmark_environments(_load(args.before),_load(args.after))
    payload=diff.to_jsonable()

    validation=None
    if args.allow_section is not None or args.require_section is not None:
        allowed=tuple(args.allow_section or ())
        required=tuple(args.require_section or ())
        try:
            validation=validate_perturbation(
                diff,
                allowed_sections=allowed,
                required_sections=required,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        payload["perturbation_validation"]=validation.to_jsonable()

    encoded=json.dumps(payload,indent=2,sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(encoded+"\n",encoding="utf-8")
    print(encoded)
    if args.require_change and diff.identical_fingerprint:
        return 2
    if validation is not None and not validation.valid:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
