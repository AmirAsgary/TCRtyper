#!/usr/bin/env python3
"""
AIRR donor mapping parser:
  - donor_id is taken from subject.subject_id
  - sample_name is repertoire_id (matches rearrangement TSV repertoire_id field)
  - merges per-donor TSVs and composes hla_donor_assignments.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Ensure sibling import works when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_donor_samples_from_mapping import merge_donor_samples_from_mapping  # noqa: E402

from tcrtyper.dataset_processing.airr_utils import (  # noqa: E402
    find_airr_metadata,
    get_repertoire_id,
    get_subject_id,
    load_airr_metadata,
)
from tcrtyper.dataset_processing.path_utils import processed_dataset_root  # noqa: E402


def build_map(dataset_root: Path, metadata_name: str | None) -> Dict[str, List[str]]:
    metadata_path = (
        dataset_root / metadata_name if metadata_name else find_airr_metadata(dataset_root)
    )
    reps = load_airr_metadata(metadata_path)

    donors: Dict[str, List[str]] = {}
    for rep in reps:
        rep_id = get_repertoire_id(rep)
        if not rep_id:
            continue
        donor_id = get_subject_id(rep) or rep_id
        donors.setdefault(donor_id, []).append(rep_id)

    for donor in donors:
        donors[donor] = sorted(set(donors[donor]))
    return donors


def main() -> None:
    ap = argparse.ArgumentParser(
        description="AIRR: build donor map from metadata and merge donor samples.",
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Dataset root (inputs; outputs go to ../processed/<dataset>).",
    )
    ap.add_argument(
        "--metadata",
        default=None,
        help="Metadata JSON filename under dataset root (default: auto-detect *metadata.json).",
    )
    ap.add_argument(
        "--map-out",
        default="donor_samples_map.json",
        help="Output mapping JSON filename (relative to processed/<dataset>).",
    )
    ap.add_argument(
        "--keep-sources",
        action="store_true",
        help="Keep original per-sample TSVs after merge.",
    )
    ap.add_argument(
        "--encoding",
        default="utf-8",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_root = processed_dataset_root(root)
    out_root.mkdir(parents=True, exist_ok=True)

    donors = build_map(root, args.metadata)

    out_path = out_root / args.map_out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"donors": donors}, f, indent=2, ensure_ascii=False)

    n_donors = len(donors)
    n_samples = sum(len(v) for v in donors.values())
    singletons = sum(1 for v in donors.values() if len(v) == 1)
    multis = n_donors - singletons
    print(
        f"Wrote {out_path} | donors={n_donors}, samples={n_samples}, "
        f"singletons={singletons}, multi={multis}"
    )

    merge_donor_samples_from_mapping(
        out_root,
        donors,
        encoding=args.encoding,
        keep_sources=args.keep_sources,
    )


if __name__ == "__main__":
    main()
