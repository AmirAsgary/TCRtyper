#!/usr/bin/env python3
# join_clusters_with_metadata_simple.py

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from tcrtyper.config import config


def parse_args():
    ap = argparse.ArgumentParser(
        description="Join mmseqs clusters with metadata by seq_id (both sorted by seq_id)."
    )
    ap.add_argument(
        "base",
        help="Base directory (parent of processed root).",
    )
    ap.add_argument(
        "--processed-root",
        default=config.data.train_export_root_name,
        help=(
            "Processed root under base "
            f"(default: {config.data.train_export_root_name})."
        ),
    )
    ap.add_argument(
        "--clusters",
        default="mmseqs/cluster_members_by_seq.sorted.tsv",
        help="Clusters-by-seq sorted TSV (relative to processed root).",
    )
    ap.add_argument(
        "--metadata",
        default="mmseqs/all_bioid_metadata.sorted.tsv",
        help="Metadata sorted TSV, headerless (relative to processed root).",
    )
    ap.add_argument(
        "--out-full",
        default="mmseqs/cluster_members_full.tsv",
        help="Output TSV with cluster + metadata (relative to processed root).",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    base = Path(args.base).resolve()
    processed_root = base / args.processed_root

    clusters_path = processed_root / args.clusters
    metadata_path = processed_root / args.metadata
    out_path = processed_root / args.out_full
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not clusters_path.is_file():
        raise SystemExit(f"Clusters file not found: {clusters_path}")
    if not metadata_path.is_file():
        raise SystemExit(f"Metadata file not found: {metadata_path}")

    matches = 0
    c_only = 0
    m_only = 0

    # clusters: seq_id \t rep_id (no header)
    # metadata: headerless, sorted by seq_id
    #   0: seq_id
    #   1: dataset
    #   2: sample_file
    #   3: row_index
    #   4: cdr3aa
    #   5: cdr2aa_gapped
    #   6: cdr1aa_gapped
    #   7: cdr2.5aa_gapped
    #   8: v_b
    #   9: j_b
    #  10: count
    with clusters_path.open("r", encoding="utf-8", newline="") as c_fh, \
         metadata_path.open("r", encoding="utf-8", newline="") as m_fh, \
         out_path.open("w", encoding="utf-8", newline="") as out_fh:

        c_reader = csv.reader(c_fh, delimiter="\t")
        m_reader = csv.reader(m_fh, delimiter="\t")
        out_writer = csv.writer(out_fh, delimiter="\t", lineterminator="\n")

        try:
            c_row = next(c_reader)  # [seq_id, rep_id]
        except StopIteration:
            c_row = None

        try:
            m_row = next(m_reader)  # [seq_id, dataset, sample_file, ...]
        except StopIteration:
            m_row = None

        with tqdm(
            desc="Joining clusters & metadata",
            unit="pair",
            disable=args.no_progress,
        ) as pbar:
            while c_row is not None and m_row is not None:
                if not c_row:
                    try:
                        c_row = next(c_reader)
                    except StopIteration:
                        c_row = None
                    continue
                if not m_row:
                    try:
                        m_row = next(m_reader)
                    except StopIteration:
                        m_row = None
                    continue

                c_id = c_row[0]
                rep_id = c_row[1]
                m_id = m_row[0]

                if c_id == m_id:
                    dataset = m_row[1]
                    sample_file = m_row[2]
                    cdr3aa = m_row[4]
                    cdr2 = m_row[5]
                    cdr1 = m_row[6]
                    cdr25 = m_row[7]
                    v_b = m_row[8] if len(m_row) > 8 else ""
                    j_b = m_row[9] if len(m_row) > 9 else ""

                    out_writer.writerow(
                        [
                            rep_id,
                            c_id,
                            dataset,
                            sample_file,
                            cdr3aa,
                            cdr2,
                            cdr1,
                            cdr25,
                            v_b,
                            j_b,
                        ]
                    )

                    matches += 1
                    pbar.update(1)

                    try:
                        c_row = next(c_reader)
                    except StopIteration:
                        c_row = None
                    try:
                        m_row = next(m_reader)
                    except StopIteration:
                        m_row = None

                elif c_id < m_id:
                    c_only += 1
                    try:
                        c_row = next(c_reader)
                    except StopIteration:
                        c_row = None
                else:
                    m_only += 1
                    try:
                        m_row = next(m_reader)
                    except StopIteration:
                        m_row = None

    print(f"Matches written: {matches}")
    print(f"Cluster IDs without metadata: {c_only}")
    print(f"Metadata rows without cluster: {m_only}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
