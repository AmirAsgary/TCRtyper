#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate Emerson HLA reconstructed from HLA_features.txt and Adaptive Biotechnologies export v1 for consistency

Compare:
  - two-digit HLA-A/B tags from sample_overview.tsv
  - against two-digit truncations of the four-digit HLA-A/B alleles
    stored in hla_assignments.json.
"""

import argparse
import csv
import json
import logging
import re
from pathlib import Path

from tcrtyper.config import config
from tcrtyper.dataset_processing.path_utils import (
    processed_dataset_root,
    sample_overview_path,
)

logger = logging.getLogger(__name__)

TAG_SPLIT = re.compile(r"\s*,\s*")


def _two_digit_from_overview_tag(tag):
    """
    From overview tag (e.g., 'HLA-A*01', 'HLA-A*02:01', 'HLA-B*38'),
    return standardized 'HLA-A*NN' / 'HLA-B*NN' (two digits).
    """
    if not tag:
        return None
    u = tag.strip().upper()
    m = re.match(r"^HLA-(A|B)\*([0-9]+)", u)
    if not m:
        return None
    locus = m.group(1)
    digits = m.group(2)
    two = digits[:2] if len(digits) >= 2 else digits.zfill(2)
    return f"HLA-{locus}*{two}"


def _two_digit_from_assignment_allele(a):
    """
    From assignment allele (normalized 4-digit like 'HLA-A*0201' or with suffix),
    return 'HLA-A*NN' / 'HLA-B*NN' (two digits), else None for non A/B.
    """
    if not a:
        return None
    u = a.strip().upper()
    m = re.match(r"^HLA-(A|B)\*([0-9]+)", u)
    if not m:
        return None
    locus = m.group(1)
    digits = m.group(2)
    two = digits.zfill(2) if len(digits) < 2 else digits[:2]
    return f"HLA-{locus}*{two}"


def load_overview_ab2(overview_tsv):
    """
    Returns mapping: sample_name -> {'A2': set([...]), 'B2': set([...])}
    Only HLA-A*NN and HLA-B*NN extracted from 'sample_tags'.
    """
    out = {}
    with open(overview_tsv, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        req = {"sample_name", "sample_tags"}
        missing = req.difference(reader.fieldnames or [])
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise SystemExit(f"Missing required columns in {overview_tsv}: {missing_cols}")
        for row in reader:
            if not row:
                continue
            name = (row.get("sample_name") or "").strip()
            tags_raw = row.get("sample_tags") or ""
            A2, B2 = set(), set()
            for tok in TAG_SPLIT.split(tags_raw.strip()):
                if not tok:
                    continue
                td = _two_digit_from_overview_tag(tok)
                if not td:
                    continue
                if td.startswith("HLA-A*"):
                    A2.add(td)
                elif td.startswith("HLA-B*"):
                    B2.add(td)
            out[name] = {"A2": A2, "B2": B2}
    return out


def load_assignments_ab2(assignments_json):
    """
    Returns mapping: sample_name -> {'A2': set([...]), 'B2': set([...])}
    by truncating normalized 4-digit A/B in hla_assignments.json to two-digit.
    """
    out = {}
    with open(assignments_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    for entry in data:
        name = (entry.get("sample_name") or "").strip()
        A2, B2 = set(), set()
        for a in (entry.get("hla_i") or []):
            td = _two_digit_from_assignment_allele(a)
            if not td:
                continue
            if td.startswith("HLA-A*"):
                A2.add(td)
            elif td.startswith("HLA-B*"):
                B2.add(td)
        out[name] = {"A2": A2, "B2": B2}
    return out


def validate_ab_two_digit(assignments_json, overview_tsv, out_json=None):
    """
    Compare overview two-digit HLA-A/B against two-digit truncations of the
    four-digit HLA-A/B inferred from features and stored in hla_assignments.json.

    Writes a validation JSON with a summary and per-sample details.

    NOTE: If you only need a quick aggregate check, the per_sample payload,
    histogram, and only_in_* lists below are optional and can be removed to
    slim down the file format.
    """
    assn = load_assignments_ab2(assignments_json)
    ov = load_overview_ab2(overview_tsv)

    names_assn = set(assn.keys())
    names_ov = set(ov.keys())
    overlap = sorted(names_assn & names_ov)
    only_assn = sorted(names_assn - names_ov)
    only_ov = sorted(names_ov - names_assn)

    # Optional detailed histogram; remove if you only care about per-sample mismatches.
    hist = {k: 0 for k in range(0, 5)}
    per_sample = {}

    for name in overlap:
        A2_ov = ov[name]["A2"]
        B2_ov = ov[name]["B2"]
        A2_as = assn[name]["A2"]
        B2_as = assn[name]["B2"]

        mA = len(A2_ov & A2_as)
        mB = len(B2_ov & B2_as)
        mt = max(0, min(4, mA + mB))
        hist[mt] += 1

        # Optional detailed per-sample payload; can be dropped to reduce JSON size.
        per_sample[name] = {
            "overview_A2": sorted(A2_ov),
            "overview_B2": sorted(B2_ov),
            "assign_A2": sorted(A2_as),
            "assign_B2": sorted(B2_as),
            "matches_A": mA,
            "matches_B": mB,
            "matches_total": mt,
        }

    summary = {
        "n_overlap": len(overlap),
        "n_only_in_overview": len(only_ov),
        "n_only_in_assignments": len(only_assn),
        # Optional histogram; can be removed if unused.
        "match_histogram_0to4": {str(k): hist[k] for k in range(0, 5)},
        "notes": "Counts are donors with exactly k matching A/B two-digit alleles (0..4).",
    }

    # only_in_* lists are truncated to 2000 entries; if you never inspect them,
    # they can be removed entirely.
    payload = {
        "summary": summary,
        "per_sample": per_sample,
        "only_in_overview": only_ov[:2000],
        "only_in_assignments": only_assn[:2000],
    }

    if out_json is None:
        out_json = (
            assignments_json.parent
            / config.data.emerson_hla_ab_2digit_validation_filename
        )
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    logger.info(
        "[validate-ab2] overlap=%d, only_overview=%d, only_assignments=%d",
        summary["n_overlap"],
        summary["n_only_in_overview"],
        summary["n_only_in_assignments"],
    )
    logger.info("[validate-ab2] matches (0..4): %s", summary["match_histogram_0to4"])
    logger.info("[validate-ab2] wrote %s", out_json)

    if only_ov:
        logger.warning(
            "[validate-ab2] %d samples present only in overview (first 10): %s",
            len(only_ov),
            only_ov[:10],
        )
    if only_assn:
        logger.warning(
            "[validate-ab2] %d samples present only in assignments (first 10): %s",
            len(only_assn),
            only_assn[:10],
        )

    return out_json


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate Emerson HLA-A/B two-digit typing in sample_overview.tsv "
            "against four-digit assignments from hla_assignments.json."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Dataset root containing hla_assignments.json.",
    )
    parser.add_argument(
        "--overview",
        required=True,
        help=(
            "Path to sample_overview.tsv (relative to --root/metadata or absolute)."
        ),
    )
    parser.add_argument(
        "--assignments",
        default=None,
        help=(
            "Optional path to hla_assignments.json (relative to --root or absolute). "
            f"Default: <root>/{config.data.hla_assignments_filename}."
        ),
    )
    parser.add_argument(
        "--validation-out",
        default=None,
        help=(
            "Optional path for the validation JSON (relative to --root if not absolute). "
            f"Default: <root>/{config.data.emerson_hla_ab_2digit_validation_filename}."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (can be used multiple times).",
    )
    return parser.parse_args()


def _configure_logging(verbosity):
    level = logging.DEBUG if verbosity and verbosity > 0 else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    args = _parse_args()
    _configure_logging(args.verbose)

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Dataset root not found: {root}")
    out_root = processed_dataset_root(root)
    out_root.mkdir(parents=True, exist_ok=True)

    overview_tsv = Path(args.overview)
    if not overview_tsv.is_absolute():
        candidate = root / overview_tsv
        if candidate.exists():
            overview_tsv = candidate
        else:
            overview_tsv = sample_overview_path(root)
    if not overview_tsv.exists():
        raise SystemExit(f"overview not found: {overview_tsv}")

    if args.assignments:
        assignments_json = Path(args.assignments)
        if not assignments_json.is_absolute():
            assignments_json = root / assignments_json
    else:
        assignments_json = out_root / config.data.hla_assignments_filename

    if not assignments_json.exists():
        raise SystemExit(f"assignments not found for validation: {assignments_json}")

    if args.validation_out:
        out_json = Path(args.validation_out)
        if not out_json.is_absolute():
            out_json = out_root / out_json
    else:
        out_json = out_root / config.data.emerson_hla_ab_2digit_validation_filename

    validate_ab_two_digit(assignments_json, overview_tsv, out_json)


if __name__ == "__main__":
    main()
