from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from aster.data import combine_datasets


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Combine audited Aster JSONL corpora atomically")
    parser.add_argument("--dataset", type=Path, action="append", required=True,
                        help="Input finalized JSONL; repeat for multiple corpora")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-multiple-workloads", action="store_true")
    args = parser.parse_args(argv)
    result = combine_datasets(
        args.dataset,
        args.out,
        overwrite=args.overwrite,
        require_multiple_workloads=args.require_multiple_workloads,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
