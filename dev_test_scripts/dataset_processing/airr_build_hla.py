#!/usr/bin/env python3
"""
Build hla_assignments.json from AIRR Data Commons metadata JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tcrtyper.config import config
from tcrtyper.dataset_processing.airr_utils import (
    extract_hla_from_repertoire,
    find_airr_metadata,
    get_repertoire_id,
    load_airr_metadata,
)
from tcrtyper.dataset_processing.hla_utils import build_hla_entry
from tcrtyper.dataset_processing.path_utils import processed_dataset_root

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Extract HLA assignments from AIRR metadata JSON.",
    )
    ap.add_argument(
        "--dataset-dir",
        default=None,
        help="AIRR dataset directory (contains metadata JSON).",
    )
    ap.add_argument(
        "--root",
        default=None,
        help="Alias for --dataset-dir.",
    )
    ap.add_argument(
        "--metadata",
        default=None,
        help="Metadata JSON filename under dataset dir (default: auto-detect *metadata.json).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output path for hla_assignments.json (default: processed/<dataset>/hla_assignments.json).",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging.",
    )
    return ap.parse_args()


def _configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)

    dataset_arg = args.dataset_dir or args.root
    if not dataset_arg:
        raise SystemExit("Must provide --dataset-dir (or --root).")
    dataset_dir = Path(dataset_arg).resolve()
    metadata_path = (
        dataset_dir / args.metadata
        if args.metadata
        else find_airr_metadata(dataset_dir)
    )
    reps = load_airr_metadata(metadata_path)

    entries = []
    missing_id = 0
    for rep in reps:
        rep_id = get_repertoire_id(rep)
        if not rep_id:
            missing_id += 1
            continue
        hla_i, hla_ii = extract_hla_from_repertoire(rep)
        rel_path = f"{config.data.export_subdir_name}/{rep_id}.tsv"
        entries.append(
            build_hla_entry(
                sample_name=rep_id,
                rel_path=rel_path,
                hla_i=hla_i,
                hla_ii=hla_ii,
                has_tcr=True,
            )
        )

    out_root = processed_dataset_root(dataset_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_root / config.data.hla_assignments_filename
    out_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    if missing_id:
        logger.warning("Skipped %d repertoire entries missing repertoire_id.", missing_id)
    logger.info("Wrote %s with %d entries.", out_path, len(entries))


if __name__ == "__main__":
    main()
