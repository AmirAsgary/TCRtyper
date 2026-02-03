#!/usr/bin/env python3
"""
Probe AIRR rearrangement TSV structure:
  - Detect whether repertoire_id appears in contiguous blocks or is interleaved.
  - Report basic row counts and TRB filtering stats.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Optional

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcrtyper.dataset_processing.airr_split_rearrangements import find_single_tsv

logger = logging.getLogger(__name__)


def _fmt_gb(n_bytes: int) -> str:
    return f"{n_bytes / (1024 ** 3):.2f}GB"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Probe AIRR rearrangement TSV for repertoire_id ordering.",
    )
    ap.add_argument(
        "--dataset-dir",
        required=True,
        help="AIRR dataset directory containing the rearrangement TSV.",
    )
    ap.add_argument(
        "--rearrangement-tsv",
        default=None,
        help="Rearrangement TSV filename under dataset dir (default: auto-detect single *.tsv).",
    )
    ap.add_argument(
        "--log-every-gb",
        type=float,
        default=0.5,
        help="Emit progress every N GB processed (default: 0.5).",
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


def probe_tsv(src: Path, log_every_gb: float) -> dict:
    total_bytes = src.stat().st_size
    last_progress_bytes = 0

    total_rows = 0
    trb_rows = 0
    skipped_non_trb = 0
    skipped_missing = 0

    current_rep: Optional[str] = None
    current_block_len = 0
    block_lengths = []
    seen = set()
    closed = set()
    reps_with_reentry = set()
    reentry_blocks = 0
    reentry_rows = 0

    with src.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{src} is missing header row.")
        if "repertoire_id" not in reader.fieldnames or "locus" not in reader.fieldnames:
            raise ValueError(
                f"{src} missing required columns: repertoire_id, locus"
            )
        logger.info(
            "Probing %s (%s, columns=%d).",
            src,
            _fmt_gb(total_bytes),
            len(reader.fieldnames),
        )

        for row in reader:
            total_rows += 1
            locus = (row.get("locus") or "").strip()
            if locus and locus.upper() != "TRB":
                skipped_non_trb += 1
                continue
            if not locus:
                skipped_missing += 1
                continue

            rep_id = (row.get("repertoire_id") or "").strip()
            if not rep_id:
                skipped_missing += 1
                continue

            trb_rows += 1

            if current_rep is None:
                current_rep = rep_id
                current_block_len = 1
                seen.add(rep_id)
            elif rep_id == current_rep:
                current_block_len += 1
            else:
                block_lengths.append(current_block_len)
                closed.add(current_rep)
                current_block_len = 1
                current_rep = rep_id
                if rep_id in closed:
                    reentry_blocks += 1
                    reps_with_reentry.add(rep_id)
                    reentry_rows += 1
                seen.add(rep_id)

            if rep_id in closed and rep_id != current_rep:
                reentry_rows += 1

            if log_every_gb and total_bytes:
                processed = fh.buffer.tell()
                if processed - last_progress_bytes >= log_every_gb * 1024 ** 3:
                    remaining = max(total_bytes - processed, 0)
                    pct = (processed / total_bytes * 100.0) if total_bytes else 0.0
                    logger.info(
                        "Probe progress: %s/%s (%.1f%%), remaining %s",
                        _fmt_gb(processed),
                        _fmt_gb(total_bytes),
                        pct,
                        _fmt_gb(remaining),
                    )
                    last_progress_bytes = processed

    if current_rep is not None:
        block_lengths.append(current_block_len)
        closed.add(current_rep)

    blocks = len(block_lengths)
    unique_reps = len(seen)
    avg_block = (sum(block_lengths) / blocks) if blocks else 0.0
    max_block = max(block_lengths) if block_lengths else 0

    return {
        "total_rows": total_rows,
        "trb_rows": trb_rows,
        "skipped_non_trb": skipped_non_trb,
        "skipped_missing": skipped_missing,
        "unique_repertoires": unique_reps,
        "blocks": blocks,
        "avg_block_len": avg_block,
        "max_block_len": max_block,
        "reentry_blocks": reentry_blocks,
        "reentry_rows": reentry_rows,
        "repertoires_with_reentry": len(reps_with_reentry),
        "contiguous_blocks": reentry_blocks == 0,
    }


def main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)

    dataset_dir = Path(args.dataset_dir).resolve()
    src = (
        dataset_dir / args.rearrangement_tsv
        if args.rearrangement_tsv
        else find_single_tsv(dataset_dir)
    )

    report = probe_tsv(src, args.log_every_gb)
    logger.info("Probe summary: %s", report)


if __name__ == "__main__":
    main()
