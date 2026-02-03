# src/dataset_parser/musvosvi2022.py
#!/usr/bin/env python3
"""
musvosvi2022 donor mapping parser:

Donor definition:
  - sample_name first token before the first underscore ('_') is of the form:
        <subject_id>-D<timepoint>
    e.g.  "04-1104-D0", "04-1104-D180", "07-0630-D540".
  - The biological donor is the subject_id without the timepoint suffix, so:
        "04-1104-D0"   -> donor "04-1104"
        "04-1104-D180" -> donor "04-1104"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List
import sys

# FIXME Ensure this script's directory is on sys.path so sibling imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from merge_donor_samples_from_mapping import merge_donor_samples_from_mapping  # noqa: E402
from tcrtyper.dataset_processing.path_utils import (  # noqa: E402
    processed_dataset_root,
    sample_overview_path,
)


# Matches IDs like "04-1104-D0", "04-1104-D180", "07-0630-D540", etc.
# Captures the part before the final "-D<digits>" as group "id".
TIMEPOINT_SUFFIX_RE = re.compile(r"^(?P<id>.+)-D[0-9]+$")


def donor_from_sample_name(sample: str) -> str:
    """
    Infer biological donor from a musvosvi2022 sample name.

    Expected sample_name pattern (simplified):
        <subject_id>-D<timepoint>_<rest>

    Examples:
        "04-1104-D0_TCRB"    -> "04-1104"
        "04-1104-D180_TCRB"  -> "04-1104"
        "07-0630-D0_TCRB"    -> "07-0630"
        "07-0630-D540_TCRB"  -> "07-0630"

    If the pattern does not match, falls back to the first token before '_'.
    """
    sample = sample.strip()
    if not sample:
        return sample

    # First token before the first underscore (if any)
    if "_" in sample:
        base = sample.split("_", 1)[0]
    else:
        base = sample

    m = TIMEPOINT_SUFFIX_RE.match(base)
    if m:
        return m.group("id")
    # Fallback: use the whole base token unchanged
    return base


def build_map(root: Path) -> Dict[str, List[str]]:
    sv = sample_overview_path(root)
    if not sv.exists():
        raise SystemExit(f"sample_overview.tsv not found: {sv}")
    donors: Dict[str, List[str]] = {}
    with open(sv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "sample_name" not in (reader.fieldnames or []):
            raise SystemExit(f"{sv}: missing required column 'sample_name'")
        for row in reader:
            s = (row.get("sample_name") or "").strip()
            if not s:
                continue
            d = donor_from_sample_name(s)
            donors.setdefault(d, []).append(s)
    # normalize: unique + sort
    for k in donors:
        donors[k] = sorted(set(donors[k]))
    return donors


def main():
    ap = argparse.ArgumentParser(
        description="musvosvi2022: build donor map, write JSON, and merge donor samples."
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
    donors = build_map(root)

    # Write standardized mapping JSON
    out_path = out_root / args.map_out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"donors": donors}, f, indent=2, ensure_ascii=False)

    # Quick mapping summary
    n_donors = len(donors)
    n_samples = sum(len(v) for v in donors.values())
    singletons = sum(1 for v in donors.values() if len(v) == 1)
    multis = n_donors - singletons
    print(f"Wrote {out_path}  | donors={n_donors}, samples={n_samples}, singletons={singletons}, multi={multis}")

    # Run merge + HLA composition
    merge_donor_samples_from_mapping(
        out_root,
        donors,
        encoding=args.encoding,
        keep_sources=args.keep_sources,
    )


if __name__ == "__main__":
    main()
