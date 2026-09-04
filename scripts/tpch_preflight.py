from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from aster.workloads import build_tpch_preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fingerprint external TPC-H queries and data inputs")
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--scale-factor", type=float, required=True)
    parser.add_argument("--specification-version", default="3.0.1")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_tpch_preflight(
        args.query_dir,
        args.data_dir,
        scale_factor=args.scale_factor,
        specification_version=args.specification_version,
        strict=not args.allow_partial,
    )
    encoded = json.dumps(asdict(manifest), indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
