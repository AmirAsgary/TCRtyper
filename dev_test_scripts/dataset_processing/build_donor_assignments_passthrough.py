#!/usr/bin/env python3
"""
Create hla_donor_assignments.json by treating each sample as its own donor.

This is useful for datasets where samples are already donor-unique.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

from tcrtyper.config import config
from tcrtyper.dataset_processing.path_utils import processed_dataset_root
from tcrtyper.dataset_processing.hla_utils import (
    donor_hla_typing_flags,
    normalize_hla_entry,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build donor assignments by copying per-sample HLA entries.",
    )
    ap.add_argument(
        "--dataset-dir",
        required=True,
        help="Dataset directory containing hla_assignments.json.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help=(
            "Output path for hla_donor_assignments.json "
            "(default: <dataset-dir>/hla_donor_assignments.json)."
        ),
    )
    return ap.parse_args()


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def _update_stats(
    stats: dict,
    meta: List[dict],
    *,
    dataset: str,
    sample_name: str,
) -> Tuple[bool, bool]:
    donor_changed = False
    donor_dropped = False
    for info in meta:
        stats["alleles_total"] += 1
        raw = str(info.get("raw") or "").strip()
        raw_label = raw or (info.get("coerced") or "").strip()
        raw_key = (info.get("coerced") or raw_label).strip()
        if info.get("status") == "dropped":
            donor_dropped = True
            stats["alleles_dropped"] += 1
            reason = info.get("drop_reason") or "unknown"
            stats["drop_reasons"][reason] = stats["drop_reasons"].get(reason, 0) + 1
            stats["dropped_counts"][raw_key] = stats["dropped_counts"].get(raw_key, 0) + 1
            logger.info(
                "[%s][%s] dropped allele '%s' (reason=%s)",
                dataset,
                sample_name,
                raw_label,
                reason,
            )
            continue

        rules = info.get("change_rules") or []
        if rules:
            donor_changed = True
            stats["alleles_changed"] += 1
            for rule in rules:
                stats["change_rules"][rule] = stats["change_rules"].get(rule, 0) + 1
            normalized = info.get("normalized") or raw_label
            key = f"{raw_key} -> {normalized}"
            stats["changed_pairs"][key] = stats["changed_pairs"].get(key, 0) + 1
            logger.info(
                "[%s][%s] normalized allele '%s' -> '%s' (rules=%s)",
                dataset,
                sample_name,
                raw_label,
                normalized,
                ",".join(rules),
            )

    return donor_changed, donor_dropped


def normalize_entry(entry: dict, *, dataset: str, stats: dict) -> dict:
    sample_name = str(entry.get("sample_name") or "").strip()
    hla_i, hla_ii, hla_types, meta = normalize_hla_entry(entry)
    changed, dropped = _update_stats(
        stats,
        meta,
        dataset=dataset,
        sample_name=sample_name,
    )
    if changed:
        stats["donors_with_changes"] += 1
    if dropped:
        stats["donors_with_drops"] += 1
    hla_types = hla_i + hla_ii
    num_hla = len(hla_types)
    status = str(entry.get("status") or "")
    flags = donor_hla_typing_flags(hla_types, ignore_dra1_locus=True)

    return {
        "donor_file": sample_name,
        "path": f"{config.data.tcrdist_processed_subdir_name}/{sample_name}.tsv",
        "used_samples": [f"{config.data.tcrdist_processed_subdir_name}/{sample_name}.tsv"],
        "status": status,
        "hla_i": hla_i,
        "hla_ii": hla_ii,
        "hla_types": hla_types,
        "num_hla": num_hla,
        **flags,
    }


def main() -> None:
    args = parse_args()
    _configure_logging()
    dataset_dir = Path(args.dataset_dir)
    out_root = processed_dataset_root(dataset_dir)
    in_path = out_root / config.data.hla_assignments_filename
    if not in_path.exists():
        raise SystemExit(f"Missing hla_assignments.json: {in_path}")

    with in_path.open("r", encoding="utf-8") as fh:
        entries = json.load(fh)
    if not isinstance(entries, list):
        raise SystemExit("hla_assignments.json must be a list of objects.")

    dataset_name = dataset_dir.name
    stats = {
        "dataset": dataset_name,
        "donors_total": 0,
        "donors_with_changes": 0,
        "donors_with_drops": 0,
        "donors_empty_after_drop": 0,
        "alleles_total": 0,
        "alleles_dropped": 0,
        "alleles_changed": 0,
        "drop_reasons": {},
        "change_rules": {},
        "changed_pairs": {},
        "dropped_counts": {},
    }
    out_entries: List[dict] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        stats["donors_total"] += 1
        normalized = normalize_entry(e, dataset=dataset_name, stats=stats)
        if e.get("hla_types") or e.get("hla_i") or e.get("hla_ii"):
            if not normalized.get("hla_types"):
                stats["donors_empty_after_drop"] += 1
        out_entries.append(normalized)

    out_path = (
        Path(args.out) if args.out else out_root / config.data.hla_donor_assignments_filename
    )
    out_path.write_text(json.dumps(out_entries, indent=2), encoding="utf-8")
    summary_path = out_root / "hla_donor_assignments_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Wrote %s with %d donors", out_path, len(out_entries))
    logger.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
