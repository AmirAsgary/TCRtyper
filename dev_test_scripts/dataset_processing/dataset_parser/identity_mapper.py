# src/dataset_parser/identitiy_parser.py
#!/usr/bin/env python3
"""
identity donor mapping:
  donor == 'donor_' + sample_name
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List
import sys

# Ensure this script's directory is on sys.path so sibling imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from merge_donor_samples_from_mapping import merge_donor_samples_from_mapping  # noqa: E402
from tcrtyper.dataset_processing.path_utils import (  # noqa: E402
    processed_dataset_root,
    sample_overview_path,
)


def build_identity_map(root: Path, *, encoding: str = "utf-8") -> Dict[str, List[str]]:
    """
    Build a donor->samples mapping where each sample is its own donor,
    with donor IDs prefixed as 'donor_<sample_name>':

      donors[f"donor_{sample_name}"] = [sample_name]
    """
    sv = sample_overview_path(root)
    if not sv.exists():
        raise SystemExit(f"sample_overview.tsv not found: {sv}")

    donors: Dict[str, List[str]] = {}
    with open(sv, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "sample_name" not in (reader.fieldnames or []):
            raise SystemExit(f"{sv}: missing required column 'sample_name'")

        for row in reader:
            s = (row.get("sample_name") or "").strip()
            if not s:
                continue
            donor_id = f"donor_{s}"
            donors.setdefault(donor_id, []).append(s)

    # normalize: unique + sort
    for k in list(donors.keys()):
        donors[k] = sorted(set(donors[k]))

    return donors


def main():
    ap = argparse.ArgumentParser(
        description="identity parser: build donor=donor_<sample> map, write JSON, and merge donor samples."
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Dataset root (inputs; outputs go to ../processed/<dataset>)",
    )
    ap.add_argument(
        "--map-out",
        default="donor_samples_map.json",
        help="Output mapping JSON filename (relative to processed/<dataset>)",
    )
    ap.add_argument("--keep-sources", action="store_true", help="Keep original per-sample TSVs after merge")
    ap.add_argument("--encoding", default="utf-8")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_root = processed_dataset_root(root)
    out_root.mkdir(parents=True, exist_ok=True)
    donors = build_identity_map(root, encoding=args.encoding)

    # Write standardized mapping JSON
    out_path = out_root / args.map_out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"donors": donors}, f, indent=2, ensure_ascii=False)

    # Quick mapping summary
    n_donors = len(donors)
    n_samples = sum(len(v) for v in donors.values())
    singletons = sum(1 for v in donors.values() if len(v) == 1)
    multis = n_donors - singletons
    print(
        f"Wrote {out_path}  | donors={n_donors}, samples={n_samples}, "
        f"singletons={singletons}, multi={multis}"
    )

    # Run merge + HLA composition
    merge_donor_samples_from_mapping(
        out_root,
        donors,
        encoding=args.encoding,
        keep_sources=args.keep_sources,
    )


if __name__ == "__main__":
    main()
