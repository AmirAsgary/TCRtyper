#!/usr/bin/env python3
"""
filter_clusters_by_size_stream.py

Stream a sorted mmseqs linclust/cluster createtsv output (rep \t member \t ...),
and write a new TSV containing only clusters with size >= min-size.

Assumptions:
  - Input is sorted by the first column (representative ID).
  - All members of a cluster form a contiguous block.
  - Each representative appears only once as a cluster label.
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from tcrtyper.config import config


def parse_args():
    ap = argparse.ArgumentParser(
        description="Stream-filter mmseqs cluster TSV to remove small clusters."
    )
    ap.add_argument(
        "base",
        nargs="?",
        help=(
            "Base directory containing the mmseqs cluster TSV (e.g. the mmseqs folder). "
            "If omitted together with in_tsv/out_tsv, config defaults are used."
        ),
    )
    ap.add_argument(
        "in_tsv",
        nargs="?",
        help="Input cluster TSV path (relative to base).",
    )
    ap.add_argument(
        "out_tsv",
        nargs="?",
        help="Output cluster TSV path (relative to base).",
    )
    ap.add_argument(
        "--min-size",
        type=int,
        default=2,
        help="Minimum cluster size to keep (default: 2 = drop singletons).",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    # Either fully explicit (base + in_tsv + out_tsv) or fully defaulted
    if args.base and args.in_tsv and args.out_tsv:
        base = Path(args.base).resolve()
        in_path = base / args.in_tsv
        out_path = base / args.out_tsv
    elif not args.base and not args.in_tsv and not args.out_tsv:
        base = (
            Path(config.data.base_dir).resolve()
            / config.data.train_export_root_name
            / "mmseqs"
        )
        in_path = base / "linclust_100id_bioid_clusters_all.tsv"
        out_path = base / "linclust_100id_bioid_clusters_ge2.tsv"
    else:
        raise SystemExit(
            "Provide either zero positional arguments (to use config defaults) "
            "or all three: base, in_tsv, out_tsv."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.is_file():
        raise SystemExit(f"Input TSV not found: {in_path}")

    total_rows = 0
    written_rows = 0
    clusters_total = 0
    clusters_kept = 0

    current_rep = None
    current_lines: list[str] = []
    current_size = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for raw_line in tqdm(
            fin,
            desc="Filtering clusters",
            unit="line",
            disable=args.no_progress,
        ):
            if not raw_line.strip():
                continue
            total_rows += 1

            rep = raw_line.split("\t", 1)[0]

            if current_rep is None:
                current_rep = rep
                current_lines = [raw_line]
                current_size = 1
                clusters_total = 1
                continue

            if rep == current_rep:
                current_lines.append(raw_line)
                current_size += 1
            else:
                # flush previous cluster
                if current_size >= args.min_size:
                    for l in current_lines:
                        fout.write(l)
                        written_rows += 1
                    clusters_kept += 1

                clusters_total += 1
                current_rep = rep
                current_lines = [raw_line]
                current_size = 1

        # flush last cluster
        if current_rep is not None and current_size >= args.min_size:
            for l in current_lines:
                fout.write(l)
                written_rows += 1
            clusters_kept += 1

    print(f"Input file: {in_path}")
    print(f"Output file: {out_path}")
    print(f"Total lines (members, non-empty): {total_rows}")
    print(f"Clusters total: {clusters_total}")
    print(f"Clusters kept (size >= {args.min_size}): {clusters_kept}")
    print(f"Lines written: {written_rows}")


if __name__ == "__main__":
    main()
