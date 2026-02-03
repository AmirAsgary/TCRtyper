#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emerson HLA json builder from DeWitt et al. HLA_features.txt

IMPORTANT:
    The DeWitt paper uses numerical sort on the P*.tsv files, and Adaptive
    Biotechnologies v1 export MISSES the P00001 file, which causes a mismatch.
    This Emerson parser should be used on the raw Adaptive dump with all P*.tsv
    files present. If P00001 is missing, it will be caught in downstream
    validation.
"""

import argparse
import gzip
import io
import json
import logging
import re
from pathlib import Path
from typing import Optional

from tcrtyper.config import config
from tcrtyper.dataset_processing.hla_utils import (
    HLAClass,
    build_hla_entry,
    classify_hla_tag,
    coerce_hla_allele,
)
from tcrtyper.dataset_processing.path_utils import processed_dataset_root

logger = logging.getLogger(__name__)

P_NAME_RE = re.compile(r"^P(\d+)\.tsv$", re.IGNORECASE)

# It works, don't touch the regex itself
FEATURE_LINE_RE = re.compile(
    r"^\s*(?:\d+\s+)?feature:\s*(?P<allele>\S+)\s+"
    r"num_positives:\s*(?P<num_pos>\d+)\s+positives:\s*(?P<pos_list>[0-9\s]*)\s+"
    r"num_negatives:\s*(?P<num_neg>\d+)\s+negatives:\s*(?P<neg_list>[0-9\s]*)\s*$",
    flags=re.MULTILINE,
)

def list_discovery_samples(dump_root: Path):
    """Find P*.tsv files under dump_root and return them sorted by numeric ID."""
    candidates = {}
    for path in dump_root.rglob("*.tsv"):
        m = P_NAME_RE.match(path.name)
        if not m:
            continue
        num = int(m.group(1))
        rel = path.relative_to(dump_root)
        # Prefer the shortest relative path if multiple candidates exist
        if num not in candidates or len(str(rel)) < len(
            str(candidates[num].relative_to(dump_root))
        ):
            candidates[num] = path

    if not candidates:
        raise SystemExit(f"No P*.tsv files found under {dump_root}")

    ordered = [candidates[k] for k in sorted(candidates)]
    logger.debug("Discovered %d P-files under %s", len(ordered), dump_root)
    return ordered


def find_hla_features_file(dump_root: Path) -> Path:
    """Locate HLA_features.txt or HLA_features.txt.gz under dump_root."""
    plain = list(dump_root.rglob(config.data.emerson_hla_features_filename))
    gz = list(dump_root.rglob(config.data.emerson_hla_features_gz_filename))

    if plain:
        plain.sort(key=lambda p: len(str(p.relative_to(dump_root))))
        logger.debug("Using HLA features file %s", plain[0])
        return plain[0]

    if gz:
        gz.sort(key=lambda p: len(str(p.relative_to(dump_root))))
        logger.debug("Using gzipped HLA features file %s", gz[0])
        return gz[0]

    raise SystemExit(
        f"{config.data.emerson_hla_features_filename}(.gz) not found under {dump_root}"
    )


def resolve_hla_features_file(dump_root: Path, features_arg: Optional[str]) -> Path:
    """
    Resolve the HLA features file path, honoring an explicit --features argument
    or falling back to discovery under dump_root.
    """
    if features_arg:
        cand = Path(features_arg)
        if not cand.is_absolute():
            cand = dump_root / cand
        if cand.exists():
            logger.debug("Using explicit HLA features path %s", cand)
            return cand
        if not str(cand).endswith(".gz"):
            gz = Path(str(cand) + ".gz")
            if gz.exists():
                logger.debug("Using explicit gzipped HLA features path %s", gz)
                return gz
        raise SystemExit(f"HLA features file not found: {cand}")

    return find_hla_features_file(dump_root)


def open_maybe_gz(path: Path):
    s = str(path)
    if s.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(s, "rb"), encoding="utf-8")
    return open(s, "r", encoding="utf-8")


def parse_hla_features(features_path: Path, n_samples: int):
    """
    Read HLA_features.txt(.gz) and build per-index allele sets for class I and class II.
    Indices >= n_samples are skipped (when P files are missing).
    Returns (by_I, by_II, n_features_matched).
    """
    with open_maybe_gz(features_path) as f:
        text = f.read()

    by_I: dict[int, set[str]] = {}
    by_II: dict[int, set[str]] = {}
    matched = 0
    any_line = False

    for m in FEATURE_LINE_RE.finditer(text):
        any_line = True
        raw = (m.group("allele") or "").strip()
        cls = classify_hla_tag(raw)
        if cls is None or cls is HLAClass.DROP:
            continue

        norm = coerce_hla_allele(raw)
        if not norm:
            continue

        pos_ids = [int(x) for x in re.findall(r"\d+", m.group("pos_list") or "")]
        for idx in pos_ids:
            if idx < 0 or idx >= n_samples:
                continue
            if cls is HLAClass.I:
                by_I.setdefault(idx, set()).add(norm)
            else:
                by_II.setdefault(idx, set()).add(norm)
        matched += 1

    if not any_line:
        raise SystemExit(f"No 'feature:' lines matched in {features_path}")

    logger.debug(
        "Parsed %d HLA features from %s (n_samples=%d)",
        matched,
        features_path,
        n_samples,
    )
    return by_I, by_II, matched


# TODO single util file with adaptive biotech pipeline
def emit_hla_assignments(dump_root: Path, ordered_p_files, by_I, by_II):
    out_entries = []
    out_root = processed_dataset_root(dump_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for idx, fpath in enumerate(ordered_p_files):
        sample_name = fpath.stem
        rel = str(fpath.relative_to(dump_root))

        hla_i = sorted(by_I.get(idx, set()))
        hla_ii = sorted(by_II.get(idx, set()))

        has_tcr = fpath.exists()
        rel_path = rel if has_tcr else None
        out_entries.append(
            build_hla_entry(
                sample_name=sample_name,
                rel_path=rel_path,
                hla_i=hla_i,
                hla_ii=hla_ii,
                has_tcr=has_tcr,
            )
        )

    out_path = out_root / config.data.hla_assignments_filename
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out_entries, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote %s with %d entries", out_path, len(out_entries))
    return len(out_entries)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Build Emerson hla_assignments.json from a raw dataset dump.",
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Dataset root containing P*.tsv and HLA_features.txt(.gz).",
    )
    parser.add_argument(
        "--features",
        default=None,
        help=(
            "Optional path to HLA_features.txt[.gz] relative to --root "
            "or absolute. If omitted, the script searches under --root."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Log verbosity.",
    )
    return parser.parse_args()


def _configure_logging(verbosity: int):
    level = logging.DEBUG if verbosity and verbosity > 0 else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    args = _parse_args()
    _configure_logging(args.verbose)

    dump_root = Path(args.root).resolve()
    if not dump_root.exists():
        raise SystemExit(f"Dump/dataset folder not found: {dump_root}")

    logger.info("Using dump root %s", dump_root)

    # 1) Discover P*.tsv files
    p_files = list_discovery_samples(dump_root)
    n = len(p_files)

    # 2) Locate features
    features_path = resolve_hla_features_file(dump_root, args.features)

    # 3) Parse features and emit assignments
    by_I, by_II, n_feat = parse_hla_features(features_path, n)
    wrote = emit_hla_assignments(dump_root, p_files, by_I, by_II)

    # 4) Summary: count donors with >=1 HLA tag vs none
    n_with_hla = sum(
        1
        for i in range(wrote)
        if (len(by_I.get(i, ())) + len(by_II.get(i, ()))) > 0
    )
    n_nohla = wrote - n_with_hla

    logger.info("[emerson] discovery P-files found: %d", n)
    logger.info("[emerson] features parsed: %d (HLA_features v1)", n_feat)
    logger.info(
        "[emerson] donors with >=1 HLA tag: %d; donors with no_hla: %d",
        n_with_hla,
        n_nohla,
    )


if __name__ == "__main__":
    main()
