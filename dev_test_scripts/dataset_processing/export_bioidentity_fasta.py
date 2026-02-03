#!/usr/bin/env python3
"""
export_bioidentity_fasta.py

Scan <base>/<processed_root>/<dataset>/<samples_subdir>/*.tsv, build a FASTA file
of "bio-identity" sequences suitable for mmseqs linclust:

    bio_identity_seq = cdr3aa + cdr2aa_gapped + cdr1aa_gapped + cdr2.5aa_gapped

Alignment dots '.' are kept as-is; mmseqs will treat them as unknown residues,
and the 100% identity + 100% coverage settings enforce strict equality (including dots).
"""

import argparse
import csv
from pathlib import Path
from typing import Iterator, Tuple

from tqdm import tqdm

from tcrtyper.config import config

MISSING_STRINGS = {"", "nan", "none", "na", "n/a"}

REQUIRED_LOOP_COLS = [
    "cdr3aa",
    "cdr2aa_gapped",
    "cdr1aa_gapped",
    "cdr2.5aa_gapped",
]


def norm(s: object | None) -> str | None:
    """Normalize loop strings; return None for missing."""
    if s is None:
        return None
    x = str(s).strip()
    if x.lower() in MISSING_STRINGS:
        return None
    return x


def get_cell(row: list[str], idx: int | None) -> str | None:
    if idx is None or idx >= len(row):
        return None
    return row[idx]


def iter_sample_files(processed_root: Path) -> Iterator[Tuple[str, Path]]:
    """
    Yield (dataset_name, file_path) for all sample TSVs under:
        <processed_root>/<dataset>/<train_samples_subdir_name>/*.tsv
    """
    samples_subdir = config.data.train_samples_subdir_name
    for ds_dir in sorted(p for p in processed_root.iterdir() if p.is_dir()):
        smp_dir = ds_dir / samples_subdir
        if not smp_dir.is_dir():
            continue
        for f in sorted(smp_dir.glob("*.tsv")):
            if f.is_file():
                yield ds_dir.name, f


def main():
    ap = argparse.ArgumentParser(
        description="Export 4-loop bio-identity sequences to FASTA for mmseqs linclust."
    )
    ap.add_argument(
        "--base",
        default=config.data.base_dir,
        help=f"Base directory (default: {config.data.base_dir}).",
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
        "--out-fasta",
        default="mmseqs/all_bioid.faa",
        help="Output FASTA path relative to processed root.",
    )
    ap.add_argument(
        "--out-meta",
        default="mmseqs/all_bioid_metadata.tsv",
        help="Output metadata TSV relative to processed root.",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    args = ap.parse_args()

    base = Path(args.base).resolve()
    processed_root = base / args.processed_root
    if not processed_root.is_dir():
        raise SystemExit(f"Processed root not found: {processed_root}")

    out_fasta = processed_root / args.out_fasta
    out_meta = processed_root / args.out_meta
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    files = list(iter_sample_files(processed_root))
    if not files:
        raise SystemExit(f"No sample TSVs found under {processed_root}")

    pbar = tqdm(
        files,
        desc="Exporting bio-identities",
        unit="file",
        disable=args.no_progress,
    )

    seq_counter = 0
    total_rows = 0
    kept_rows = 0
    skipped_missing_loops = 0

    with out_fasta.open("w", encoding="utf-8") as ffa, \
         out_meta.open("w", encoding="utf-8", newline="") as fmeta:

        mw = csv.writer(fmeta, delimiter="\t", lineterminator="\n")
        write_fa = ffa.write
        write_meta = mw.writerow
        norm_loop = norm
        get_cell_local = get_cell

        write_meta(
            [
                "seq_id",
                "dataset",
                "sample_file",
                "row_index",
                "cdr3aa",
                "cdr2aa_gapped",
                "cdr1aa_gapped",
                "cdr2.5aa_gapped",
                "v_b",
                "j_b",
                "count",
            ]
        )

        for ds_name, fpath in pbar:
            with fpath.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.reader(fh, delimiter="\t")
                header = next(reader, None)
                if not header:
                    continue
                name_to_idx = {name: idx for idx, name in enumerate(header)}
                missing_cols = [c for c in REQUIRED_LOOP_COLS if c not in name_to_idx]
                if missing_cols:
                    continue
                cdr3_idx = name_to_idx["cdr3aa"]
                cdr2_idx = name_to_idx["cdr2aa_gapped"]
                cdr1_idx = name_to_idx["cdr1aa_gapped"]
                cdr25_idx = name_to_idx["cdr2.5aa_gapped"]
                v_b_idx = name_to_idx.get("v_b")
                j_b_idx = name_to_idx.get("j_b")
                count_idx = name_to_idx.get("count")

                for row_idx, row in enumerate(reader):
                    total_rows += 1

                    cdr3aa = norm_loop(get_cell_local(row, cdr3_idx))
                    cdr2aa = norm_loop(get_cell_local(row, cdr2_idx))
                    cdr1aa = norm_loop(get_cell_local(row, cdr1_idx))
                    cdr25aa = norm_loop(get_cell_local(row, cdr25_idx))
                    if None in (cdr3aa, cdr2aa, cdr1aa, cdr25aa):
                        skipped_missing_loops += 1
                        continue

                    # Build bio-identity sequence: CDR3 + CDR2 + CDR1 + CDR2.5
                    seq = (cdr3aa + cdr2aa + cdr1aa + cdr25aa).upper()
                    if not seq:
                        skipped_missing_loops += 1
                        continue

                    seq_counter += 1
                    seq_id = f"S{seq_counter}"

                    # FASTA entry
                    write_fa(f">{seq_id}\n")
                    write_fa(seq + "\n")

                    v_b = get_cell_local(row, v_b_idx) or ""
                    j_b = get_cell_local(row, j_b_idx) or ""
                    count = get_cell_local(row, count_idx) or ""

                    write_meta(
                        [
                            seq_id,
                            ds_name,
                            fpath.name,
                            row_idx,
                            cdr3aa,
                            cdr2aa,
                            cdr1aa,
                            cdr25aa,
                            v_b,
                            j_b,
                            count,
                        ]
                    )

                    kept_rows += 1

    print(f"Total rows seen: {total_rows}")
    print(f"Rows kept with all 4 loops: {kept_rows}")
    print(f"Rows skipped (missing any loop): {skipped_missing_loops}")
    print(f"FASTA written to: {out_fasta}")
    print(f"Metadata written to: {out_meta}")


if __name__ == "__main__":
    main()
