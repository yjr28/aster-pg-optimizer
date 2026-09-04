from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from aster.workloads import finalize_job_collection


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Finalize atomic JOB query shards into one audited training JSONL")
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = finalize_job_collection(args.collection_dir, args.out, overwrite=args.overwrite)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
