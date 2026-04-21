#!/usr/bin/env python3
"""
compute_percentile_rank.py — Per-HLA gamma rank normalization stored as CSR.
=========================================================================
Computes the within-HLA percentile rank of each observed gamma value and
stores it as a sparse CSR group `clusters/percentile_rank/{indptr,indices,data}`
with EXACTLY the same structure as `clusters/z_probs`.
For each HLA column a:
    1. Collect all observed gamma values for that HLA (one pass).
    2. Sort them.
    3. Replace each γ_ia with its percentile rank in [0, 1] within that HLA.
After this:
    - Per-HLA gamma distributions are uniform in [0, 1].
    - Common HLAs and rare HLAs have comparable target scales.
    - The NN can no longer learn HLA frequency as a shortcut.
The unsorted gamma values can be recovered at inference time by inverting the
rank → gamma map (saved as a per-HLA lookup table).
=========================================================================
USAGE:
    python compute_percentile_rank.py \\
        --h5_path /path/to/dataset_pval.h5 \\
        --chunk_size 200000
Optional:
    --force                 # overwrite without prompting
    --no_save_lookup        # skip saving rank->gamma lookup tables
=========================================================================
OUTPUT in H5:
    clusters/percentile_rank/indptr   (int64)
    clusters/percentile_rank/indices  (uint32)  — same as z_probs/indices
    clusters/percentile_rank/data     (float32) — values in [0, 1]
Plus optional dense lookup datasets for inference-time inverse mapping:
    clusters/percentile_rank/lookup_indptr   (int64, A+1)
    clusters/percentile_rank/lookup_gammas   (float32, total observed pairs)
=========================================================================
"""
import os
import sys
import time
import argparse
import numpy as np
import h5py
from pathlib import Path


def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Compute per-HLA percentile rank of gammas.")
    p.add_argument("--h5_path", required=True, help="Merged H5 with z_probs.")
    p.add_argument("--chunk_size", type=int, default=200000,
                   help="TCRs per chunk (default: 200000).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing percentile_rank without prompting.")
    p.add_argument("--no_save_lookup", action="store_true",
                   help="Skip saving rank->gamma lookup tables.")
    return p.parse_args()


def confirm_overwrite(h5_path, force):
    """Check if percentile_rank exists; prompt or force-overwrite."""
    with h5py.File(h5_path, "r") as f:
        if "percentile_rank" in f["clusters"]:
            if force:
                return
            print(f"\nWARNING: clusters/percentile_rank already exists.")
            ans = input("Overwrite? [y/N]: ").strip().lower()
            if ans != "y":
                print("Aborted.")
                sys.exit(0)


def pass_one_collect_per_hla_gammas(h5_path, num_alleles, chunk_size):
    """Pass 1: collect all observed (γ_ia, row, col) into per-HLA arrays.
    Returns:
        per_hla_gammas: list of length A; each element is np.ndarray of γ values.
                        Used to compute the sort order for percentile ranks.
        nnz_total:      total number of nonzero entries across all rows.
    """
    print("Pass 1: collecting gammas per HLA...")
    # Pre-count nnz per HLA in a single quick scan over indices
    with h5py.File(h5_path, "r") as f:
        zp_indices_ds = f["clusters"]["z_probs"]["indices"]
        nnz_total = int(zp_indices_ds.shape[0])
        # Streamed bincount over indices to avoid loading 500M ints at once
        per_hla_count = np.zeros(num_alleles, dtype=np.int64)
        idx_chunk = 5_000_000
        for s in range(0, nnz_total, idx_chunk):
            e = min(s + idx_chunk, nnz_total)
            blk = np.asarray(zp_indices_ds[s:e]).astype(np.int64, copy=False)
            per_hla_count += np.bincount(blk, minlength=num_alleles)
    print(f"  Total nonzeros: {nnz_total:,}")
    print(f"  Per-HLA nnz min/median/max: "
          f"{per_hla_count.min()}/{int(np.median(per_hla_count))}/{per_hla_count.max()}")
    # Pre-allocate per-HLA arrays
    per_hla_gammas = [np.empty(c, dtype=np.float32) for c in per_hla_count]
    write_pos = np.zeros(num_alleles, dtype=np.int64)
    # Stream gamma data and indices in chunks; bucket by HLA
    with h5py.File(h5_path, "r") as f:
        zp_indices_ds = f["clusters"]["z_probs"]["indices"]
        zp_data_ds = f["clusters"]["z_probs"]["data"]
        idx_chunk = 5_000_000
        t0 = time.time()
        for s in range(0, nnz_total, idx_chunk):
            e = min(s + idx_chunk, nnz_total)
            idx_blk = np.asarray(zp_indices_ds[s:e]).astype(np.int64, copy=False)
            dat_blk = np.asarray(zp_data_ds[s:e])
            # Sort this block by HLA index to enable bulk per-HLA copies
            order = np.argsort(idx_blk, kind="stable")
            idx_sorted = idx_blk[order]
            dat_sorted = dat_blk[order]
            # Find boundaries between HLAs in the sorted block
            edges = np.searchsorted(idx_sorted,
                                    np.arange(num_alleles + 1, dtype=np.int64))
            for a in range(num_alleles):
                lo, hi = int(edges[a]), int(edges[a + 1])
                if hi > lo:
                    n = hi - lo
                    wp = write_pos[a]
                    per_hla_gammas[a][wp:wp + n] = dat_sorted[lo:hi]
                    write_pos[a] = wp + n
            elapsed = time.time() - t0
            rate = e / elapsed if elapsed > 0 else 0
            print(f"  [pass1] {e:,}/{nnz_total:,} "
                  f"({100*e/nnz_total:.1f}%) | {rate/1e6:.1f}M nnz/s")
    return per_hla_gammas, nnz_total, per_hla_count


def build_rank_lookup_per_hla(per_hla_gammas):
    """For each HLA, sort gammas and build a structure that maps γ -> rank in [0,1].
    Returns:
        sorted_gammas: list of length A; each is sorted np.ndarray.
        Used for: rank = (searchsorted_left + searchsorted_right - 1) / (2*(N-1))
    """
    print("Pass 2a: sorting per-HLA gammas...")
    sorted_gammas = []
    for a in range(len(per_hla_gammas)):
        g = per_hla_gammas[a]
        if len(g) == 0:
            sorted_gammas.append(np.empty(0, dtype=np.float32))
        else:
            sg = np.sort(g)
            sorted_gammas.append(sg)
    print("  Done sorting.")
    return sorted_gammas


def gamma_to_rank(gammas, sorted_for_hla):
    """Vectorized γ -> percentile rank in [0, 1] using midrank for ties.
    Args:
        gammas:         (n,) gamma values for one HLA, in original row order.
        sorted_for_hla: (N_a,) sorted gammas for that HLA.
    Returns:
        ranks: (n,) float32 in [0, 1].
    """
    N = len(sorted_for_hla)
    if N == 0:
        return np.zeros_like(gammas, dtype=np.float32)
    if N == 1:
        return np.full_like(gammas, 0.5, dtype=np.float32)
    # midrank: average of left and right insertion points
    left = np.searchsorted(sorted_for_hla, gammas, side="left")
    right = np.searchsorted(sorted_for_hla, gammas, side="right")
    midrank = (left + right - 1) * 0.5
    return (midrank / (N - 1)).astype(np.float32)


def pass_two_write_ranks(h5_path, sorted_per_hla, num_alleles, chunk_size):
    """Pass 2b: stream z_probs CSR and write ranked CSR to clusters/percentile_rank."""
    print("Pass 2b: streaming and writing percentile_rank CSR...")
    with h5py.File(h5_path, "r+") as f:
        clusters = f["clusters"]
        zp_indptr = clusters["z_probs"]["indptr"]
        zp_indices = clusters["z_probs"]["indices"]
        zp_data = clusters["z_probs"]["data"]
        n_clusters = zp_indptr.shape[0] - 1
        nnz_total = int(zp_indices.shape[0])
        # Remove old percentile_rank if present
        if "percentile_rank" in clusters:
            del clusters["percentile_rank"]
        pr_grp = clusters.create_group("percentile_rank")
        # Mirror z_probs CSR layout exactly
        pr_indptr = pr_grp.create_dataset(
            "indptr", shape=(n_clusters + 1,), dtype=np.int64,
            chunks=(min(chunk_size + 1, n_clusters + 1),),
            compression="gzip", compression_opts=4,
        )
        pr_indices = pr_grp.create_dataset(
            "indices", shape=(nnz_total,), dtype=zp_indices.dtype,
            chunks=(min(100_000, nnz_total),) if nnz_total > 0 else None,
            compression="gzip", compression_opts=4 if nnz_total > 0 else None,
        )
        pr_data = pr_grp.create_dataset(
            "data", shape=(nnz_total,), dtype=np.float32,
            chunks=(min(100_000, nnz_total),) if nnz_total > 0 else None,
            compression="gzip", compression_opts=4 if nnz_total > 0 else None,
        )
        # Copy indptr identically and indices identically
        pr_indptr[:] = zp_indptr[:]
        # Copy indices in chunks
        idx_chunk = 5_000_000
        for s in range(0, nnz_total, idx_chunk):
            e = min(s + idx_chunk, nnz_total)
            pr_indices[s:e] = zp_indices[s:e]
        # Stream data chunk-by-chunk; convert γ -> rank per HLA
        t0 = time.time()
        for s in range(0, nnz_total, idx_chunk):
            e = min(s + idx_chunk, nnz_total)
            idx_blk = np.asarray(zp_indices[s:e]).astype(np.int64, copy=False)
            dat_blk = np.asarray(zp_data[s:e])
            out_blk = np.empty_like(dat_blk, dtype=np.float32)
            # Sort by HLA index for bulk per-HLA conversion
            order = np.argsort(idx_blk, kind="stable")
            idx_sorted = idx_blk[order]
            dat_sorted = dat_blk[order]
            ranks_sorted = np.empty_like(dat_sorted, dtype=np.float32)
            edges = np.searchsorted(idx_sorted,
                                    np.arange(num_alleles + 1, dtype=np.int64))
            for a in range(num_alleles):
                lo, hi = int(edges[a]), int(edges[a + 1])
                if hi > lo:
                    ranks_sorted[lo:hi] = gamma_to_rank(
                        dat_sorted[lo:hi], sorted_per_hla[a])
            # Inverse the order back
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(len(order))
            out_blk[:] = ranks_sorted[inv_order]
            pr_data[s:e] = out_blk
            elapsed = time.time() - t0
            rate = e / elapsed if elapsed > 0 else 0
            print(f"  [pass2b] {e:,}/{nnz_total:,} "
                  f"({100*e/nnz_total:.1f}%) | {rate/1e6:.1f}M nnz/s")
        # Save lookup tables for inference-time inverse mapping
        return pr_grp


def save_lookup_tables(h5_path, sorted_per_hla, num_alleles):
    """Save per-HLA sorted gammas as lookup table for rank -> gamma inversion."""
    print("Saving lookup tables (rank -> gamma)...")
    counts = np.array([len(g) for g in sorted_per_hla], dtype=np.int64)
    lookup_indptr = np.zeros(num_alleles + 1, dtype=np.int64)
    lookup_indptr[1:] = np.cumsum(counts)
    total = int(lookup_indptr[-1])
    flat = np.empty(total, dtype=np.float32)
    for a in range(num_alleles):
        lo, hi = int(lookup_indptr[a]), int(lookup_indptr[a + 1])
        flat[lo:hi] = sorted_per_hla[a]
    with h5py.File(h5_path, "r+") as f:
        pr_grp = f["clusters"]["percentile_rank"]
        if "lookup_indptr" in pr_grp:
            del pr_grp["lookup_indptr"]
        if "lookup_gammas" in pr_grp:
            del pr_grp["lookup_gammas"]
        pr_grp.create_dataset(
            "lookup_indptr", data=lookup_indptr, dtype=np.int64,
            compression="gzip", compression_opts=4,
        )
        pr_grp.create_dataset(
            "lookup_gammas", data=flat, dtype=np.float32,
            compression="gzip", compression_opts=4,
        )
        pr_grp.attrs["lookup_description"] = (
            "lookup_gammas[lookup_indptr[a]:lookup_indptr[a+1]] is sorted gammas "
            "for HLA a; rank r in [0,1] maps to "
            "lookup_gammas[lookup_indptr[a] + round(r*(N_a-1))]")
    print(f"  Saved {total:,} sorted gamma values across {num_alleles} HLAs.")


def main():
    """Run the percentile rank pipeline."""
    args = parse_args()
    print("=" * 60)
    print("Compute percentile rank pipeline")
    print("=" * 60)
    print(f"  H5: {args.h5_path}")
    confirm_overwrite(args.h5_path, args.force)
    # Get num_alleles
    with h5py.File(args.h5_path, "r") as f:
        num_alleles = int(f.attrs.get("num_alleles", 0))
        if num_alleles == 0:
            num_alleles = int(np.max(f["clusters"]["z_probs"]["indices"][:1000])) + 1
    print(f"  num_alleles: {num_alleles}")
    t0 = time.time()
    # Pass 1: collect gammas per HLA
    per_hla_gammas, nnz_total, per_hla_count = pass_one_collect_per_hla_gammas(
        args.h5_path, num_alleles, args.chunk_size)
    # Sort per HLA
    sorted_per_hla = build_rank_lookup_per_hla(per_hla_gammas)
    del per_hla_gammas
    # Pass 2: write ranked CSR
    pass_two_write_ranks(
        args.h5_path, sorted_per_hla, num_alleles, args.chunk_size)
    # Save lookup
    if not args.no_save_lookup:
        save_lookup_tables(args.h5_path, sorted_per_hla, num_alleles)
    # Tag metadata
    with h5py.File(args.h5_path, "r+") as f:
        pr_grp = f["clusters"]["percentile_rank"]
        pr_grp.attrs["description"] = (
            "Per-HLA percentile rank of gammas in [0, 1]. Computed within each "
            "HLA column independently to remove HLA-frequency bias from gamma scale.")
        pr_grp.attrs["mirror_of"] = "clusters/z_probs"
    total = time.time() - t0
    print(f"\nDone in {total:.1f}s ({total/60:.1f}min)")
    print(f"  Wrote: clusters/percentile_rank/{{indptr, indices, data}}")
    if not args.no_save_lookup:
        print(f"  Wrote: clusters/percentile_rank/{{lookup_indptr, lookup_gammas}}")


if __name__ == "__main__":
    main()