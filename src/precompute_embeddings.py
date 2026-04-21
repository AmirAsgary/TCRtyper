#!/usr/bin/env python3
"""
precompute_embeddings.py — Build pooled CDR embeddings for one H5 file.
=========================================================================
Loads clusters/cdr_freq from an H5 file and builds a fixed-size pooled
embedding per cluster using length-invariant summaries:
  - Mean profile per CDR region: (21,)
  - N positional bins per CDR region: (n_bins, 21)
Concatenated across 4 CDRs → embedding dim = 4 * 21 * (1 + n_bins).
For n_bins=5 (default) → 504 dim.
VECTORIZED — no Python per-cluster loops. Bulk reads of full cdr_freq
arrays, then numpy segment operations on stacked rows.
=========================================================================
USAGE:
    python precompute_embeddings.py \\
        --h5_path /path/to/h5 \\
        --output /path/to/emb.npz \\
        --n_bins 5 \\
        --chunk_size 500000
Output NPZ contains:
    embedding: (n_clusters, D_emb) float32, L2-normalized
    n_bins:    scalar
    cdr_lengths: (n_clusters, 4) int32 — original sequence lengths per CDR
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
        description="Precompute pooled CDR embeddings from H5.")
    p.add_argument("--h5_path", required=True, help="Input H5 with cdr_freq.")
    p.add_argument("--output", required=True, help="Output NPZ path.")
    p.add_argument("--n_bins", type=int, default=5,
                   help="Positional bins per CDR (default: 5).")
    p.add_argument("--chunk_size", type=int, default=500000,
                   help="Clusters per chunk (default: 500000).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output.")
    return p.parse_args()


def pooled_embed_chunk(freq_flat, indptr_chunk, n_bins):
    """Compute pooled embedding for a chunk of clusters, fully vectorized.
    Args:
        freq_flat:     (total_rows, 21) — concatenated profile rows for the chunk.
        indptr_chunk:  (n_chunk+1,) — cluster boundaries in freq_flat (local, start=0).
        n_bins:        number of positional bins.
    Returns:
        emb: (n_chunk, 21 * (1 + n_bins)) float32 — embedding for this CDR only.
        lengths: (n_chunk,) int32 — sequence length per cluster.
    """
    n_chunk = indptr_chunk.shape[0] - 1
    D_per_cdr = 21 * (1 + n_bins)
    emb = np.zeros((n_chunk, D_per_cdr), dtype=np.float32)
    lengths = (indptr_chunk[1:] - indptr_chunk[:-1]).astype(np.int32)
    total_rows = int(indptr_chunk[-1])
    if total_rows == 0:
        return emb, lengths
    # ── Mean profile per cluster (vectorized segment sum) ─────────────
    # np.add.at with indices obtained from np.repeat is fastest for sum.
    row_to_cluster = np.repeat(
        np.arange(n_chunk, dtype=np.int64),
        lengths.astype(np.int64))
    mean_prof = np.zeros((n_chunk, 21), dtype=np.float32)
    np.add.at(mean_prof, row_to_cluster, freq_flat)
    # Divide by length (avoid div-by-zero)
    safe_len = np.maximum(lengths, 1).astype(np.float32)
    mean_prof /= safe_len[:, None]
    emb[:, :21] = mean_prof
    # ── Positional bins per cluster (vectorized) ──────────────────────
    # Compute bin index for each row: (row_position_in_cluster * n_bins) // length
    # We need position within cluster for each row.
    # row_offset_in_cluster = arange(total_rows) - indptr_chunk[row_to_cluster]
    global_row_idx = np.arange(total_rows, dtype=np.int64)
    pos_in_cluster = global_row_idx - indptr_chunk[:-1][row_to_cluster]
    # Bin index in [0, n_bins-1]
    cluster_len_per_row = lengths[row_to_cluster].astype(np.int64)
    # Clamp length to >=1 to avoid div-by-zero for empty clusters
    # (empty clusters have no rows anyway so this branch is unused there).
    safe_len_per_row = np.maximum(cluster_len_per_row, 1)
    bin_idx = np.minimum(
        (pos_in_cluster * n_bins) // safe_len_per_row,
        n_bins - 1).astype(np.int64)
    # Compound segment id = cluster_id * n_bins + bin_idx
    compound = row_to_cluster * n_bins + bin_idx
    # Sum frequencies per (cluster, bin)
    bin_sum = np.zeros((n_chunk * n_bins, 21), dtype=np.float32)
    np.add.at(bin_sum, compound, freq_flat)
    # Count rows per (cluster, bin)
    bin_count = np.zeros(n_chunk * n_bins, dtype=np.float32)
    np.add.at(bin_count, compound, 1.0)
    safe_count = np.maximum(bin_count, 1.0)
    bin_mean = bin_sum / safe_count[:, None]  # (n_chunk*n_bins, 21)
    bin_mean = bin_mean.reshape(n_chunk, n_bins, 21)
    emb[:, 21:] = bin_mean.reshape(n_chunk, n_bins * 21)
    return emb, lengths


def build_embedding(h5_path, n_bins, chunk_size):
    """Build the full pooled embedding for all clusters in an H5 file.
    Returns:
        emb:         (n_clusters, 4 * 21 * (1 + n_bins)) float32, L2-normalized
        cdr_lengths: (n_clusters, 4) int32 — length of each CDR per cluster
    """
    cdr_names = ["cdr1", "cdr2", "cdr25", "cdr3"]
    D_per_cdr = 21 * (1 + n_bins)
    D_emb = 4 * D_per_cdr
    with h5py.File(h5_path, "r") as f:
        clusters = f["clusters"]
        if "cdr_freq" not in clusters:
            raise KeyError(f"{h5_path}: missing clusters/cdr_freq")
        cdr_freq_grp = clusters["cdr_freq"]
        # Discover cluster count from any cdr indptr
        ip_any = cdr_freq_grp[f"{cdr_names[0]}_indptr"]
        n_clusters = int(ip_any.shape[0] - 1)
        print(f"  {h5_path}: {n_clusters:,} clusters")
        emb = np.zeros((n_clusters, D_emb), dtype=np.float32)
        cdr_lengths = np.zeros((n_clusters, 4), dtype=np.int32)
        # Process each CDR region independently
        for c_idx, nm in enumerate(cdr_names):
            print(f"  [{nm}] building...")
            ip_full = np.asarray(cdr_freq_grp[f"{nm}_indptr"][:])
            fr_ds = cdr_freq_grp[f"{nm}_freq"]
            base = c_idx * D_per_cdr
            t0 = time.time()
            for cs in range(0, n_clusters, chunk_size):
                ce = min(cs + chunk_size, n_clusters)
                ip_chunk = ip_full[cs:ce + 1]
                fs, fe = int(ip_chunk[0]), int(ip_chunk[-1])
                if fe > fs:
                    freq_flat = np.asarray(fr_ds[fs:fe])
                else:
                    freq_flat = np.empty((0, 21), dtype=np.float32)
                # Local indptr starting at 0
                ip_local = (ip_chunk - fs).astype(np.int64)
                chunk_emb, chunk_lens = pooled_embed_chunk(
                    freq_flat, ip_local, n_bins)
                emb[cs:ce, base:base + D_per_cdr] = chunk_emb
                cdr_lengths[cs:ce, c_idx] = chunk_lens
                elapsed = time.time() - t0
                rate = (ce - 0) / elapsed if elapsed > 0 else 0
                print(f"    [{nm}] {ce:,}/{n_clusters:,} "
                      f"({100*ce/n_clusters:.1f}%) | "
                      f"{rate:,.0f} clusters/s")
    # L2 normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    emb /= norms
    return emb, cdr_lengths


def main():
    """Run embedding precomputation."""
    args = parse_args()
    if Path(args.output).exists() and not args.force:
        print(f"[SKIP] {args.output} exists (use --force to overwrite)")
        return
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Precompute CDR pooled embeddings")
    print("=" * 60)
    print(f"  Input:  {args.h5_path}")
    print(f"  Output: {args.output}")
    print(f"  n_bins: {args.n_bins}")
    t0 = time.time()
    emb, cdr_lengths = build_embedding(
        args.h5_path, args.n_bins, args.chunk_size)
    total = time.time() - t0
    print(f"\n  Built {emb.shape} in {total:.1f}s ({total/60:.1f}min)")
    print(f"  Saving to {args.output}...")
    np.savez(
        args.output,
        embedding=emb,
        n_bins=np.int32(args.n_bins),
        cdr_lengths=cdr_lengths,
    )
    print(f"  Done.")


if __name__ == "__main__":
    main()
