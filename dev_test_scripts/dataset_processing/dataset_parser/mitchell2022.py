# src/dataset_parser/mitchell2022.py
#!/usr/bin/env python3
"""
mitchell2022 donor mapping parser:
  Donor comes from sample_tags:
    - token exactly "Control %03d"  -> donor "Control_###"
    - token exactly "Subject %03d"  -> donor "Subject_###"
    - otherwise                     -> donor = sample_name (singleton)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

# Ensure sibling import works when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_donor_samples_from_mapping import merge_donor_samples_from_mapping  # noqa: E402
from tcrtyper.dataset_processing.path_utils import (  # noqa: E402
    processed_dataset_root,
    sample_overview_path,
)

DONOR_TOKEN_RE = re.compile(r"^(Control|Subject)\s+(\d{3})$")


def donor_from_sample_tags(sample_name: str, sample_tags: str) -> str:
    """
    Return donor ID from tags; fallback to sample_name if no Control/Subject token is present.
    """
    if sample_tags:
        for tok in (t.strip() for t in sample_tags.split(",") if t.strip()):
            m = DONOR_TOKEN_RE.match(tok)
            if m:
                return f"{m.group(1)}_{m.group(2)}"  # e.g., "Control_016" or "Subject_015"
    return sample_name  # singleton fallback


def build_map(root: Path) -> Dict[str, List[str]]:
    sv = sample_overview_path(root)
    if not sv.exists():
        raise SystemExit(f"sample_overview.tsv not found: {sv}")

    donors: Dict[str, List[str]] = {}
    with open(sv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"sample_name", "sample_tags"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{sv}: missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            sname = (row.get("sample_name") or "").strip()
            stags = (row.get("sample_tags") or "").strip()
            if not sname:
                continue
            donor = donor_from_sample_tags(sname, stags)
            donors.setdefault(donor, []).append(sname)

    # normalize: unique + sort
    for k in donors:
        donors[k] = sorted(set(donors[k]))
    return donors


def main():
    ap = argparse.ArgumentParser(description="mitchell2022: build donor map, write JSON, and merge donor samples.")
    ap.add_argument("--root", required=True, help="Dataset root (inputs; outputs go to ../processed/<dataset>)")
    ap.add_argument("--map-out", default="donor_samples_map.json", help="Output mapping JSON filename (relative to processed/<dataset>)")
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
    print(f"Wrote {out_path} | donors={n_donors}, samples={n_samples}, singletons={singletons}, multi={multis}")

    # Merge + HLA composition using shared util (unchanged)
    merge_donor_samples_from_mapping(
        out_root,
        donors,
        encoding=args.encoding,
        keep_sources=args.keep_sources,
    )


if __name__ == "__main__":
    main()
