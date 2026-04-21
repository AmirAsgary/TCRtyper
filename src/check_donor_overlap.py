#!/usr/bin/env python3
"""
check_donor_overlap.py — Sanity check train vs valid H5 donor indices.
=========================================================================
Verifies that train and validation H5 files use disjoint donor subsets
of the same donor_hla_matrix.npz. Reports:
  - total donors in each H5
  - overlap (should be zero for a proper train/valid split)
  - donor index bounds relative to donor_hla_matrix.npz
  - which HLA alleles are present only in train, only in valid, or in both
  - rare HLAs (n_donors < rare_threshold) in each set
=========================================================================
USAGE:
    python check_donor_overlap.py \\
        --train_h5 /path/to/train_ds.h5 \\
        --valid_h5 /path/to/valid_ds.h5 \\
        --donor_matrix_path /path/to/donor_hla_matrix.npz \\
        --hla_to_id /path/to/hla_to_id.json \\
        --output_dir /path/to/out \\
        --rare_threshold 5
=========================================================================
"""
import os
import sys
import json
import argparse
import numpy as np
import h5py
from pathlib import Path


def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Check donor overlap between train and valid H5.")
    p.add_argument("--train_h5", required=True, help="Train H5 path.")
    p.add_argument("--valid_h5", required=True, help="Valid H5 path.")
    p.add_argument("--donor_matrix_path", required=True,
                   help="donor_hla_matrix.npz path.")
    p.add_argument("--hla_to_id", required=True, help="hla_to_id.json path.")
    p.add_argument("--output_dir", required=True, help="Output directory.")
    p.add_argument("--rare_threshold", type=int, default=5,
                   help="HLAs with < this many donors are 'rare' (default: 5).")
    return p.parse_args()


def collect_unique_donors(h5_path):
    """Read all donor indices from clusters/donors/indices and return unique set.
    Args:
        h5_path: path to an H5 with clusters/donors CSR group.
    Returns:
        unique_donors: np.ndarray of unique donor indices (int).
        max_donor:     max donor index found.
    """
    with h5py.File(h5_path, "r") as f:
        if "clusters" not in f or "donors" not in f["clusters"]:
            raise KeyError(f"{h5_path}: missing clusters/donors group")
        donor_indices = f["clusters"]["donors"]["indices"]
        # Stream in large blocks; accumulate set
        seen = set()
        chunk = 5_000_000
        total = donor_indices.shape[0]
        max_donor = -1
        for s in range(0, total, chunk):
            e = min(s + chunk, total)
            block = np.asarray(donor_indices[s:e])
            if block.size > 0:
                u = np.unique(block)
                seen.update(int(x) for x in u)
                max_donor = max(max_donor, int(u.max()))
    return np.array(sorted(seen), dtype=np.int64), max_donor


def collect_allele_coverage(donor_set, donor_hla):
    """Return the set of allele indices carried by at least one donor in donor_set.
    Args:
        donor_set: np.ndarray of donor indices.
        donor_hla: (D, A) donor HLA matrix.
    Returns:
        np.ndarray of allele indices present in at least one donor.
    """
    sub = donor_hla[donor_set]
    col_has = sub.sum(axis=0) > 0
    return np.where(col_has)[0]


def main():
    """Run the donor overlap check."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 60)
    print("Donor overlap sanity check")
    print("=" * 60)
    print(f"  Train H5:     {args.train_h5}")
    print(f"  Valid H5:     {args.valid_h5}")
    print(f"  Donor matrix: {args.donor_matrix_path}")
    print(f"  HLA JSON:     {args.hla_to_id}")
    print(f"  Output dir:   {args.output_dir}")
    # ── load donor matrix ───────────────────────────────────────────
    donor_hla = np.load(args.donor_matrix_path)["donor_hla_matrix"]
    D, A = donor_hla.shape
    print(f"\n  Donor HLA matrix: {D} donors x {A} alleles")
    # ── load HLA name map (invert to id -> name) ────────────────────
    with open(args.hla_to_id, "r") as f:
        hla_to_id = json.load(f)
    id_to_hla = {int(v): k for k, v in hla_to_id.items()}
    if len(id_to_hla) != A:
        print(f"  WARNING: hla_to_id has {len(id_to_hla)} entries but "
              f"donor matrix has {A} alleles")
    # ── count per-allele donor frequencies ──────────────────────────
    n_donors_per_allele = donor_hla.sum(axis=0).astype(np.int64)
    rare_mask = n_donors_per_allele < args.rare_threshold
    print(f"  Rare alleles (< {args.rare_threshold} donors): "
          f"{int(rare_mask.sum())}/{A}")
    # ── collect donors from both H5s ────────────────────────────────
    print(f"\n  Scanning train H5 donors...")
    train_donors, train_max = collect_unique_donors(args.train_h5)
    print(f"    Unique donors in train: {len(train_donors):,}")
    print(f"    Max donor index:        {train_max}")
    print(f"  Scanning valid H5 donors...")
    valid_donors, valid_max = collect_unique_donors(args.valid_h5)
    print(f"    Unique donors in valid: {len(valid_donors):,}")
    print(f"    Max donor index:        {valid_max}")
    # ── bounds check ────────────────────────────────────────────────
    if train_max >= D:
        print(f"\n  ERROR: train max donor index {train_max} >= "
              f"donor matrix D={D}. Mismatched matrix!")
    if valid_max >= D:
        print(f"\n  ERROR: valid max donor index {valid_max} >= "
              f"donor matrix D={D}. Mismatched matrix!")
    # ── overlap ─────────────────────────────────────────────────────
    train_set = set(train_donors.tolist())
    valid_set = set(valid_donors.tolist())
    overlap = sorted(train_set & valid_set)
    print(f"\n  Train ∩ Valid donors: {len(overlap)}")
    if len(overlap) > 0:
        print(f"  WARNING: non-zero overlap! First 20: {overlap[:20]}")
    else:
        print(f"  OK: disjoint donor sets.")
    # ── allele coverage in each donor set ───────────────────────────
    train_alleles = collect_allele_coverage(train_donors, donor_hla)
    valid_alleles = collect_allele_coverage(valid_donors, donor_hla)
    train_allele_set = set(train_alleles.tolist())
    valid_allele_set = set(valid_alleles.tolist())
    only_train = sorted(train_allele_set - valid_allele_set)
    only_valid = sorted(valid_allele_set - train_allele_set)
    both = sorted(train_allele_set & valid_allele_set)
    print(f"\n  Allele coverage:")
    print(f"    In train only:  {len(only_train)}")
    print(f"    In valid only:  {len(only_valid)}")
    print(f"    In both:        {len(both)}")
    print(f"    Not in either:  {A - len(train_allele_set | valid_allele_set)}")
    # Rare alleles in each set
    rare_in_train = sorted(train_allele_set & set(np.where(rare_mask)[0].tolist()))
    rare_in_valid = sorted(valid_allele_set & set(np.where(rare_mask)[0].tolist()))
    print(f"\n  Rare alleles (< {args.rare_threshold} donors):")
    print(f"    Present in train: {len(rare_in_train)}")
    print(f"    Present in valid: {len(rare_in_valid)}")
    # Preview some allele names
    if only_valid:
        names = [id_to_hla.get(a, f"idx_{a}") for a in only_valid[:10]]
        print(f"\n  First 10 alleles present only in valid (cannot evaluate): "
              f"{names}")
    # ── save JSON report ────────────────────────────────────────────
    report = {
        "train_h5": str(args.train_h5),
        "valid_h5": str(args.valid_h5),
        "donor_matrix_path": str(args.donor_matrix_path),
        "donor_matrix_shape": [int(D), int(A)],
        "rare_threshold": args.rare_threshold,
        "train_donor_count": int(len(train_donors)),
        "train_max_donor_idx": int(train_max),
        "valid_donor_count": int(len(valid_donors)),
        "valid_max_donor_idx": int(valid_max),
        "train_valid_donor_overlap": int(len(overlap)),
        "overlap_donor_ids": overlap[:100],
        "train_allele_count": int(len(train_allele_set)),
        "valid_allele_count": int(len(valid_allele_set)),
        "alleles_only_in_train": [int(a) for a in only_train],
        "alleles_only_in_valid": [int(a) for a in only_valid],
        "alleles_in_both": int(len(both)),
        "rare_alleles_in_train": int(len(rare_in_train)),
        "rare_alleles_in_valid": int(len(rare_in_valid)),
        "bounds_ok": bool(train_max < D and valid_max < D),
    }
    out_path = Path(args.output_dir) / "donor_overlap_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {out_path}")
    # ── exit status ─────────────────────────────────────────────────
    if not report["bounds_ok"]:
        print("\n  FAIL: donor index bounds mismatch.")
        sys.exit(2)
    if report["train_valid_donor_overlap"] > 0:
        print("\n  FAIL: train/valid donors overlap.")
        sys.exit(3)
    print("\n  PASS: train and valid donor sets are disjoint and within bounds.")


if __name__ == "__main__":
    main()
