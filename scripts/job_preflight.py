from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from aster.workloads import build_job_preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fingerprint external JOB queries and IMDB CSV inputs")
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_job_preflight(args.query_dir, args.csv_dir, strict=not args.allow_partial)
    encoded = json.dumps(asdict(manifest), indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
