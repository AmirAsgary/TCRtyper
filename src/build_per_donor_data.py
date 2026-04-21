#!/usr/bin/env python3
"""
build_per_donor_data.py — Invert CSR, build per-donor data bundles.
=========================================================================
Takes valid H5 donor membership (clusters/donors CSR) and the TCR matches
(from match_tcrs.py), then produces one NPZ per donor containing all the
information needed for MAP optimization:
  tcr_pos:    (T,) int32 — index into used_train_idx array
  confidence: (T,) float32
  gamma_mat:  (T, A) float32 — inherited gammas from matched train TCRs
  q_vec:      (T,) float32 — inherited q_i from matched train TCRs
Writes per-donor bundles AND a global meta NPZ with used_train_idx,
gamma_used, q_used (the shared pool of training gammas).
VECTORIZED CSR inversion: no per-TCR Python loops. Uses np.repeat on the
element mask + lexsort to group by donor.
=========================================================================
USAGE:
    python build_per_donor_data.py \\
        --train_h5 /path/to/train.h5 \\
        --valid_h5 /path/to/valid.h5 \\
        --matches /path/to/matches.npz \\
        --output_dir /path/to/per_donor \\
        --min_confidence 0.01 \\
        --min_tcrs_per_donor 50
=========================================================================
"""
import os
import sys
import time
import argparse
import numpy as np
import h5py
from pathlib import Path
from scipy.sparse import csr_matrix


def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Build per-donor data bundles for MAP optimization.")
    p.add_argument("--train_h5", required=True)
    p.add_argument("--valid_h5", required=True)
    p.add_argument("--matches", required=True, help="matches.npz from match_tcrs.py")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--min_confidence", type=float, default=0.8,
                   help="Drop TCRs below this confidence (default: 0.5).")
    p.add_argument("--min_tcrs_per_donor", type=int, default=50,
                   help="Skip donors with fewer matched TCRs (default: 50).")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    """Run per-donor data build."""
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_donor_dir = out_dir / "per_donor"
    per_donor_dir.mkdir(exist_ok=True)
    meta_path = out_dir / "meta.npz"
    if meta_path.exists() and not args.force:
        print(f"[SKIP] {meta_path} exists (use --force)")
        return
    print("=" * 60)
    print("Build per-donor data bundles")
    print("=" * 60)
    print(f"  Train H5:   {args.train_h5}")
    print(f"  Valid H5:   {args.valid_h5}")
    print(f"  Matches:    {args.matches}")
    print(f"  Output dir: {out_dir}")
    t0 = time.time()
    # ── load matches ────────────────────────────────────────────────
    m = np.load(args.matches)
    nn_idx = m["nn_idx"]             # (n_valid,) train index (-1 if none)
    confidence = m["confidence"]     # (n_valid,)
    n_valid = len(nn_idx)
    print(f"\n  Total valid TCRs: {n_valid:,}")
    # ── filter out low-confidence matches ───────────────────────────
    keep_valid = (confidence >= args.min_confidence) & (nn_idx >= 0)
    valid_kept = np.where(keep_valid)[0].astype(np.int64)
    n_kept = len(valid_kept)
    print(f"  TCRs with conf >= {args.min_confidence}: {n_kept:,} "
          f"({100*n_kept/n_valid:.1f}%)")
    # ── unique used training TCRs ───────────────────────────────────
    used_train_idx, inverse = np.unique(nn_idx[valid_kept], return_inverse=True)
    n_used_train = len(used_train_idx)
    print(f"  Unique training TCRs referenced: {n_used_train:,}")
    # valid_to_pos[valid_tcr_idx in kept] = position in used_train_idx
    # ── load gammas + q_i for used training TCRs ────────────────────
    print(f"\n  Loading gammas + q_i for referenced training TCRs...")
    with h5py.File(args.train_h5, "r") as f:
        clusters = f["clusters"]
        if "q_i" not in clusters:
            print("ERROR: train H5 missing clusters/q_i.")
            sys.exit(1)
        n_alleles = int(f.attrs.get("num_alleles", 0))
        if n_alleles == 0:
            n_alleles = int(np.max(
                clusters["z_probs"]["indices"][:10000])) + 1
        print(f"    n_alleles: {n_alleles}")
        # Bulk-load ENTIRE z_probs CSR into RAM (one sequential read).
        # For filtered H5: ~1 GB indices + ~1 GB data. Much faster than
        # 2.9M individual random h5py reads (which takes hours).
        t_load = time.time()
        print(f"    Bulk loading z_probs CSR into RAM...")
        zp_indptr_full = np.asarray(clusters["z_probs"]["indptr"][:])
        print(f"      indptr: {zp_indptr_full.nbytes/1e9:.2f} GB "
              f"({time.time()-t_load:.1f}s)")
        t_load = time.time()
        zp_indices_full = np.asarray(clusters["z_probs"]["indices"][:])
        print(f"      indices: {zp_indices_full.nbytes/1e9:.2f} GB "
              f"({time.time()-t_load:.1f}s)")
        t_load = time.time()
        zp_data_full = np.asarray(clusters["z_probs"]["data"][:])
        print(f"      data: {zp_data_full.nbytes/1e9:.2f} GB "
              f"({time.time()-t_load:.1f}s)")
        t_load = time.time()
        print(f"    Bulk loading q_i...")
        q_i_full = np.asarray(clusters["q_i"][:])
        print(f"      q_i: {q_i_full.nbytes/1e9:.3f} GB "
              f"({time.time()-t_load:.1f}s)")
    # q_used: fancy index into q_i_full
    print(f"\n    Extracting q_i for {len(used_train_idx):,} used TCRs...")
    q_used = q_i_full[used_train_idx].astype(np.float32, copy=False)
    del q_i_full
    # Vectorized gamma extraction via np.repeat + gather
    print(f"    Extracting gammas (vectorized gather)...")
    t_gather = time.time()
    starts = zp_indptr_full[used_train_idx].astype(np.int64)
    ends = zp_indptr_full[used_train_idx + 1].astype(np.int64)
    nnz_per_tcr = ends - starts
    total_nnz = int(nnz_per_tcr.sum())
    print(f"      total nnz across used TCRs: {total_nnz:,}")
    # Build flat gather index: for each k in [0, total_nnz), which position
    # in zp_indices_full to read. Fully vectorized.
    new_indptr = np.zeros(len(used_train_idx) + 1, dtype=np.int64)
    new_indptr[1:] = np.cumsum(nnz_per_tcr)
    tcr_id_of_k = np.repeat(
        np.arange(len(used_train_idx), dtype=np.int64), nnz_per_tcr)
    offset_within = np.arange(total_nnz, dtype=np.int64) - new_indptr[:-1][tcr_id_of_k]
    gather_src = starts[tcr_id_of_k] + offset_within
    flat_indices = zp_indices_full[gather_src].astype(np.int64, copy=False)
    flat_data = zp_data_full[gather_src].astype(np.float32, copy=False)
    del zp_indices_full, zp_data_full, zp_indptr_full
    del tcr_id_of_k, offset_within, gather_src
    print(f"      gather done in {time.time()-t_gather:.1f}s")
    # Build dense gamma matrix via scipy CSR
    print(f"    Densifying to (n_used, n_alleles) = "
          f"({len(used_train_idx):,}, {n_alleles})...")
    t_dense = time.time()
    gamma_used = csr_matrix(
        (flat_data, flat_indices, new_indptr),
        shape=(len(used_train_idx), n_alleles),
    ).toarray().astype(np.float32)
    del flat_data, flat_indices, new_indptr
    print(f"      densify done in {time.time()-t_dense:.1f}s")
    print(f"    gamma_used shape: {gamma_used.shape} "
          f"({gamma_used.nbytes/1e9:.2f} GB)")
    # ── load valid donor CSR ────────────────────────────────────────
    print(f"\n  Loading valid donor CSR...")
    with h5py.File(args.valid_h5, "r") as f:
        vd_indptr = np.asarray(f["clusters/donors/indptr"][:])
        vd_indices_ds = f["clusters/donors/indices"]
        total_nnz_donors = int(vd_indptr[-1])
        vd_indices = np.asarray(vd_indices_ds[:total_nnz_donors])
    print(f"    donor CSR total entries: {len(vd_indices):,}")
    # ── VECTORIZED CSR inversion ────────────────────────────────────
    # For each kept valid TCR, get its (donor, tcr_pos_in_used, confidence)
    # entries by using np.repeat over donor list lengths.
    print(f"\n  Inverting CSR (vectorized)...")
    t_inv = time.time()
    # Row lengths per valid TCR (how many donors it was observed in)
    row_lengths = (vd_indptr[1:] - vd_indptr[:-1]).astype(np.int64)
    # Kept mask at row level, then we take only those rows
    # To avoid building a full element-level mask over all rows, we can
    # directly compute repeated arrays for the kept rows only.
    kept_row_lengths = row_lengths[valid_kept]
    # For each kept valid TCR, we need its sub-range in vd_indices.
    kept_starts = vd_indptr[valid_kept]
    kept_ends = vd_indptr[valid_kept + 1]
    # Build flat expanded arrays: (total_kept_entries,)
    total_kept_entries = int(kept_row_lengths.sum())
    print(f"    Total (tcr, donor) pairs after filter: "
          f"{total_kept_entries:,}")
    # flat_donor_id[e] = donor index for entry e
    # flat_tcr_pos[e] = position in used_train_idx
    # flat_conf[e] = confidence
    # Use np.repeat + gather to build them fully vectorized:
    #   repeat_indices = np.repeat(valid_kept_position, kept_row_lengths)
    # where valid_kept_position = np.arange(n_kept)
    valid_kept_pos = np.arange(n_kept, dtype=np.int64)
    repeat_pos = np.repeat(valid_kept_pos, kept_row_lengths)
    # Gather donor ids from the contiguous per-valid-TCR slices
    # Build a flat index array that, for each element e, points into vd_indices.
    # Row e belongs to valid_kept_pos[repeat_pos[e]]; its offset within the
    # row is e - cumsum_of_row_lengths_up_to_that_row.
    # Easier: use np.concatenate over per-TCR slices — but that's a loop.
    # Trick: build offsets by subtracting np.repeat(new_row_starts).
    new_row_starts = np.zeros(n_kept + 1, dtype=np.int64)
    new_row_starts[1:] = np.cumsum(kept_row_lengths)
    global_e = np.arange(total_kept_entries, dtype=np.int64)
    offset_within_row = global_e - new_row_starts[:-1][repeat_pos]
    # Gather from vd_indices using (orig_start + offset_within_row)
    orig_starts_rep = kept_starts[repeat_pos].astype(np.int64)
    gather_idx = orig_starts_rep + offset_within_row
    flat_donor_id = vd_indices[gather_idx].astype(np.int64)
    # For confidence and tcr_pos: they're per-valid-TCR, just repeat.
    flat_conf = np.repeat(
        confidence[valid_kept[valid_kept_pos]], kept_row_lengths
    ).astype(np.float32)
    # inverse maps each kept valid TCR -> position in used_train_idx
    flat_tcr_pos = np.repeat(inverse, kept_row_lengths).astype(np.int32)
    print(f"    Inversion done in {time.time() - t_inv:.1f}s")
    # Sort by donor id to group entries
    print(f"\n  Grouping by donor...")
    t_grp = time.time()
    sort_order = np.argsort(flat_donor_id, kind="stable")
    flat_donor_id = flat_donor_id[sort_order]
    flat_tcr_pos = flat_tcr_pos[sort_order]
    flat_conf = flat_conf[sort_order]
    # Per-donor boundaries
    unique_donors, donor_starts = np.unique(
        flat_donor_id, return_index=True)
    donor_ends = np.concatenate([donor_starts[1:],
                                  np.array([len(flat_donor_id)])])
    print(f"    Donors with any matched TCRs: {len(unique_donors):,}")
    print(f"    Grouping done in {time.time() - t_grp:.1f}s")
    # ── write per-donor NPZs ────────────────────────────────────────
    print(f"\n  Writing per-donor NPZ files...")
    n_donors_written = 0
    n_donors_skipped = 0
    donor_index = {}  # donor_id -> (n_tcrs, filepath)
    t_wr = time.time()
    for i, d in enumerate(unique_donors):
        s, e = int(donor_starts[i]), int(donor_ends[i])
        n_t = e - s
        if n_t < args.min_tcrs_per_donor:
            n_donors_skipped += 1
            continue
        tcr_pos = flat_tcr_pos[s:e]
        conf = flat_conf[s:e]
        # Deduplicate: same train-TCR may appear multiple times for a donor
        # if different valid TCRs all matched to it. Keep unique tcr_pos
        # with max confidence.
        unique_pos, first_idx = np.unique(tcr_pos, return_index=True)
        if len(unique_pos) < len(tcr_pos):
            # Aggregate confidence via max
            max_conf = np.zeros(len(unique_pos), dtype=np.float32)
            np.maximum.at(
                max_conf,
                np.searchsorted(unique_pos, tcr_pos),
                conf)
            tcr_pos = unique_pos.astype(np.int32)
            conf = max_conf
            n_t = len(tcr_pos)
        if n_t < args.min_tcrs_per_donor:
            n_donors_skipped += 1
            continue
        fp = per_donor_dir / f"donor_{int(d):06d}.npz"
        np.savez(
            fp,
            donor_id=np.int64(d),
            tcr_pos=tcr_pos.astype(np.int32),
            confidence=conf.astype(np.float32),
            n_tcrs=np.int32(n_t),
        )
        donor_index[int(d)] = (n_t, str(fp))
        n_donors_written += 1
    print(f"    Wrote {n_donors_written:,} donor bundles in "
          f"{time.time() - t_wr:.1f}s")
    print(f"    Skipped {n_donors_skipped:,} with "
          f"< {args.min_tcrs_per_donor} TCRs")
    # ── write global meta ───────────────────────────────────────────
    print(f"\n  Writing global meta...")
    np.savez(
        meta_path,
        used_train_idx=used_train_idx,
        gamma_used=gamma_used,
        q_used=q_used,
        n_alleles=np.int64(n_alleles),
        donor_ids=np.array(sorted(donor_index.keys()), dtype=np.int64),
        min_confidence=np.float32(args.min_confidence),
        min_tcrs_per_donor=np.int32(args.min_tcrs_per_donor),
    )
    print(f"    Saved: {meta_path}")
    total = time.time() - t0
    print(f"\nDone in {total:.1f}s ({total/60:.1f}min)")
    print(f"  Per-donor dir:  {per_donor_dir}")
    print(f"  Meta file:      {meta_path}")
    print(f"  gamma_used:     {gamma_used.shape} "
          f"({gamma_used.nbytes / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()