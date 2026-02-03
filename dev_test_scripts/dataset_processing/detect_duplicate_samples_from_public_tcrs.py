#!/usr/bin/env python3
# detect_duplicate_samples_from_public_tcrs.py
#
# Input:
#   - public_tcrs.json
#
# Output:
#   - duplicate_pairs.json  (pairs with pct_shared_of_smaller >= emit_threshold, default 0.085)
#     {
#       "emit_threshold": 0.085,
#       "mark_threshold": 0.15,
#       "total_pairs_considered": ...,
#       "total_pairs_emitted": ...,
#       "unique_duplicates": ...,
#       "duplicates": {
#         "<dup_path>": {
#           "reference": "<kept_path>",
#           "overlap": int,
#           "pct_shared_of_smaller": float,
#           "jaccard": float,
#           "size_dup": int,
#           "size_ref": int,
#           "dataset_dup": str, "dataset_ref": str,
#           "sample_dup": str,  "sample_ref": str,
#           "score": [pct_shared_of_smaller, overlap],
#           "meets_mark_threshold": bool
#         },
#         ...
#       }
#     }
#
#   - Always chooses the smaller repertoire as the "duplicate" for each outlier pair.

import argparse
import json
import logging
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm

from tcrtyper.config import config

logger = logging.getLogger(__name__)

DEFAULT_DUPLICATE_PAIRS_FILENAME = config.data.duplicate_pairs_json_filename


def load_public_index(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _accumulate_pairs_chunk(items):
    """Worker: given [(tcr, donors_list), ...] returns dict[(a,b)] -> count."""
    out: dict[tuple[str, str], int] = defaultdict(int)
    for _, donors in items:
        if not donors:
            continue
        # unique donors, stable order
        ds = sorted(set(donors))
        if len(ds) < 2:
            continue
        for a, b in combinations(ds, 2):
            out[(a, b)] += 1
    return out


def accumulate_pair_counts(
    tcr_to_donors: dict[str, list[str]],
    n_workers: int = 8,
    chunk_size: int = 50_000,
    show_progress: bool = True,
) -> dict[tuple[str, str], int]:
    items = list(tcr_to_donors.items())
    total = len(items)
    chunks = [items[i : i + chunk_size] for i in range(0, total, chunk_size)]

    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    desc = f"Accumulating overlaps in {len(chunks)} chunk(s) with {n_workers} workers"
    with Pool(processes=n_workers) as pool:
        for partial in tqdm(
            pool.imap_unordered(_accumulate_pairs_chunk, chunks),
            total=len(chunks),
            desc=desc,
            disable=not show_progress,
        ):
            for k, v in partial.items():
                pair_counts[k] += v
    return pair_counts


def build_donor_sets(
    tcr_to_donors: dict[str, list[str]],
    show_progress: bool = True,
) -> dict[str, set[str]]:
    donor_sets: dict[str, set[str]] = defaultdict(set)
    for tcr, donors in tqdm(
        tcr_to_donors.items(),
        desc="Building donor->public-TCR sets",
        disable=not show_progress,
    ):
        for d in set(donors):
            donor_sets[d].add(tcr)
    return {d: s for d, s in donor_sets.items()}


def split_ds_sample(path_str: str) -> tuple[str, str]:
    parts = path_str.split("/", 1)
    if len(parts) == 1:
        return "", parts[0]
    return parts[0], parts[1].rsplit(".tsv", 1)[0]


def choose_dup_and_ref(a: str, b: str, size_a: int, size_b: int) -> tuple[str, str]:
    """Return (dup, ref) where dup is the smaller repertoire; tie-break lexicographically."""
    if size_a < size_b:
        return a, b
    if size_b < size_a:
        return b, a
    return (b, a) if a > b else (a, b)


def _parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Detect likely duplicate samples from public-TCR co-occurrence. "
            "Input: public_tcrs.json mapping TCR -> [donor paths]. "
            "Output: duplicate_pairs.json with per-pair overlap metrics."
        )
    )
    ap.add_argument(
        "--public-json",
        default="public_tcrs.json",
        help=(
            "Path to public_tcrs.json (TCR -> [donor paths]) "
            "(default: ./public_tcrs.json)."
        ),
    )
    ap.add_argument(
        "--out-json",
        default=None,
        help=(
            "Where to write the duplicate pairs JSON. "
            f"Default: <dir(public-json)>/{DEFAULT_DUPLICATE_PAIRS_FILENAME}"
        ),
    )
    ap.add_argument(
        "--emit-threshold",
        type=float,
        default=0.085,
        help="Minimum pct_shared_of_smaller to emit into JSON (default 0.085 = 8.5%).",
    )
    ap.add_argument(
        "--mark-threshold",
        type=float,
        default=0.15,
        help=(
            "Stricter threshold at which you'd actually mark/merge duplicates later "
            "(default 0.15 = 15%)."
        ),
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=min(8, cpu_count()),
        help="Parallel workers for accumulation (default: min(8, n_cpu)).",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="TCR items per chunk sent to workers (default 50k).",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable tqdm progress bars.",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging.",
    )
    return ap.parse_args()


def _configure_logging(debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    args = _parse_args()
    _configure_logging(args.debug)

    show_progress = not args.no_progress

    public_json_path = Path(args.public_json).resolve()
    if not public_json_path.exists():
        raise FileNotFoundError(f"public_tcrs.json not found: {public_json_path}")

    if args.out_json:
        out_path = Path(args.out_json).resolve()
    else:
        out_path = public_json_path.parent / DEFAULT_DUPLICATE_PAIRS_FILENAME

    logger.info("Loading public index from %s", public_json_path)
    tcr_to_donors = load_public_index(public_json_path)
    logger.info("Total TCR keys: %d", len(tcr_to_donors))

    # Build donor -> set(TCR)
    donor_sets = build_donor_sets(tcr_to_donors, show_progress=show_progress)
    donors = sorted(donor_sets.keys())
    logger.info("Unique donors: %d", len(donors))

    # Accumulate pair overlaps
    logger.info(
        "Accumulating pair overlaps with %d workers and chunk size %d",
        args.workers,
        args.chunk_size,
    )
    pair_counts = accumulate_pair_counts(
        tcr_to_donors,
        n_workers=args.workers,
        chunk_size=args.chunk_size,
        show_progress=show_progress,
    )
    total_pairs = len(pair_counts)
    logger.info("Non-zero pairs: %d", total_pairs)

    # Prepare sizes for metrics
    sizes = {d: len(s) for d, s in donor_sets.items()}

    out_map: dict[str, dict] = {}
    emit_n = 0

    itr = tqdm(
        pair_counts.items(),
        desc="Scoring pairs for emission",
        total=total_pairs,
        disable=not show_progress,
    )
    for (a, b), overlap in itr:
        sa, sb = sizes.get(a, 0), sizes.get(b, 0)
        if not sa or not sb:
            continue
        denom = min(sa, sb)
        pct_small = overlap / denom
        if pct_small < args.emit_threshold:
            continue

        jacc = overlap / (sa + sb - overlap)

        dup, ref = choose_dup_and_ref(a, b, sa, sb)
        size_dup, size_ref = sizes[dup], sizes[ref]

        ddup, sdup = split_ds_sample(dup)
        dref, sref = split_ds_sample(ref)

        out_map[dup] = {
            "reference": ref,
            "overlap": int(overlap),
            "pct_shared_of_smaller": pct_small,
            "jaccard": jacc,
            "size_dup": int(size_dup),
            "size_ref": int(size_ref),
            "dataset_dup": ddup,
            "dataset_ref": dref,
            "sample_dup": sdup,
            "sample_ref": sref,
            "score": [pct_small, int(overlap)],
            "meets_mark_threshold": bool(pct_small >= args.mark_threshold),
        }
        emit_n += 1

    result = {
        "emit_threshold": args.emit_threshold,
        "mark_threshold": args.mark_threshold,
        "total_pairs_considered": total_pairs,
        "total_pairs_emitted": emit_n,
        "unique_duplicates": len(out_map),
        "duplicates": out_map,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    logger.info("Wrote duplicate pairs JSON: %s", out_path)
    logger.info(
        "pairs_emitted >= %.3f: %d; unique duplicates (by smaller sample): %d",
        args.emit_threshold,
        emit_n,
        len(out_map),
    )


if __name__ == "__main__":
    main()
