#!/usr/bin/env python3
"""
compute_S_i.py — Compute per-TCR confidence score S_i and store in H5.
=========================================================================
Formula (from the project):
    p_{n,i,a} = x_{n,a} * gamma_{i,a}                      # masked binding mass
    e_{n,i}   = sum_a p_{n,i,a}                            # total mass for donor n
    inner_n   = sum_a p_{n,i,a} * log(p_{n,i,a} / e_{n,i}) # = -e_{n,i} * H(q_n)
    S_i       = (1/N_i) * sum_n y_{n,i} * inner_n
where x is donor_hla_matrix (D, A), gamma is z_probs (I, A), y_{n,i} is the
indicator that TCR i appears in donor n. The sum over n only runs over donors
where y_{n,i}=1, which are the entries in clusters/donors/indices.
S_i is always <= 0. Closer to 0 = peaked (confident) binding hypothesis.
=========================================================================
USAGE:
    python compute_S_i.py \\
        --h5_path /path/to/dataset_pval.h5 \\
        --donor_matrix_path /path/to/donor_hla_matrix.npz \\
        --rare_threshold 5 \\
        --chunk_size 5000 \\
        --gpu
The script writes two new datasets into clusters/:
    clusters/S_i                      (float32, shape=(num_clusters,))
    clusters/S_i_without_rare_alleles (float32, shape=(num_clusters,))
A rare allele is one with fewer donors than --rare_threshold (default 5).
If the datasets already exist, the script asks for permission to overwrite.
Use --force to skip the prompt.
=========================================================================
"""
import os
import sys
import time
import argparse
import numpy as np
import h5py
from scipy.sparse import csr_matrix


def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Compute per-TCR S_i and store in H5.")
    p.add_argument("--h5_path", required=True, help="Merged H5 with z_probs.")
    p.add_argument("--donor_matrix_path", required=True,
                   help="NPZ with key 'donor_hla_matrix' shape (D, A).")
    p.add_argument("--rare_threshold", type=int, default=5,
                   help="HLAs with < this many donors are 'rare' (default: 5).")
    p.add_argument("--chunk_size", type=int, default=5000,
                   help="TCRs per chunk (default: 5000).")
    p.add_argument("--gpu", action="store_true",
                   help="Use TensorFlow GPU acceleration.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing S_i datasets without prompting.")
    return p.parse_args()


def maybe_setup_gpu(use_gpu):
    """Configure GPU memory growth and return tf module if available, else None."""
    if not use_gpu:
        return None
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            print("[GPU] No GPU detected, falling back to CPU.")
            return None
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"[GPU] Using {len(gpus)} GPU(s)")
        return tf
    except Exception as e:
        print(f"[GPU] TF init failed ({e}), falling back to CPU.")
        return None


def confirm_overwrite(h5_path, force):
    """Check if S_i datasets exist; prompt or force-overwrite."""
    with h5py.File(h5_path, "r") as f:
        clusters = f["clusters"]
        existing = []
        if "S_i" in clusters:
            existing.append("S_i")
        if "S_i_without_rare_alleles" in clusters:
            existing.append("S_i_without_rare_alleles")
    if existing and not force:
        print(f"\nWARNING: these datasets already exist in clusters/:")
        for n in existing:
            print(f"  - {n}")
        ans = input("Overwrite? [y/N]: ").strip().lower()
        if ans != "y":
            print("Aborted.")
            sys.exit(0)
    return existing


def compute_S_chunk_cpu(gamma_chunk, donor_idx_flat, row_lengths, donor_hla):
    """CPU computation of S_i for one chunk of TCRs (vectorized).
    Args:
        gamma_chunk:   (n_chunk, A) dense gammas for the chunk.
        donor_idx_flat:(P,) flat array of donor indices for all (TCR,donor) pairs.
        row_lengths:   (n_chunk,) number of donors per TCR in the chunk.
        donor_hla:     (D, A) donor HLA matrix.
    Returns:
        S_chunk: (n_chunk,) float32 S_i values.
    """
    # n_chunk = len(row_lengths)
    P = donor_idx_flat.shape[0]
    if P == 0:
        return np.zeros(len(row_lengths), dtype=np.float32)
    # Map each pair to its TCR row index in the chunk
    row_idx = np.repeat(np.arange(len(row_lengths), dtype=np.int64), row_lengths)
    # Gather donor HLA rows: (P, A)
    x_pairs = donor_hla[donor_idx_flat]
    # Gather gamma rows for each pair: (P, A)
    gamma_pairs = gamma_chunk[row_idx]
    # p = x * gamma: (P, A)
    p = x_pairs * gamma_pairs
    # e = sum_a p: (P,)
    e = p.sum(axis=1)
    # log term, safe at p=0 and e=0
    safe_e = np.where(e > 0, e, 1.0)
    safe_p = np.where(p > 0, p, 1.0)
    log_q = np.log(safe_p / safe_e[:, None])
    log_term = np.where(p > 0, p * log_q, 0.0)
    inner_per_pair = log_term.sum(axis=1)  # (P,)
    # Sum per TCR using bincount, then divide by N_i
    sum_per_tcr = np.bincount(row_idx, weights=inner_per_pair,
                              minlength=len(row_lengths))
    safe_N = np.where(row_lengths > 0, row_lengths, 1)
    S_chunk = (sum_per_tcr / safe_N).astype(np.float32)
    return S_chunk


def compute_S_chunk_gpu(tf, gamma_chunk, donor_idx_flat, row_lengths, donor_hla_tf):
    """GPU/TF computation of S_i for one chunk."""
    n_chunk = len(row_lengths)
    P = int(donor_idx_flat.shape[0])
    if P == 0:
        return np.zeros(n_chunk, dtype=np.float32)
    row_idx_np = np.repeat(np.arange(n_chunk, dtype=np.int32), row_lengths)
    # Move to TF
    gamma_tf = tf.constant(gamma_chunk, dtype=tf.float32)
    donor_idx_tf = tf.constant(donor_idx_flat, dtype=tf.int32)
    row_idx_tf = tf.constant(row_idx_np, dtype=tf.int32)
    # Gather HLA rows for donor pairs: (P, A)
    x_pairs = tf.gather(donor_hla_tf, donor_idx_tf)
    # Gather gamma rows: (P, A)
    gamma_pairs = tf.gather(gamma_tf, row_idx_tf)
    # p, e
    p = x_pairs * gamma_pairs
    e = tf.reduce_sum(p, axis=1)
    safe_e = tf.where(e > 0, e, tf.ones_like(e))
    safe_p = tf.where(p > 0, p, tf.ones_like(p))
    log_q = tf.math.log(safe_p / safe_e[:, None])
    log_term = tf.where(p > 0, p * log_q, tf.zeros_like(p))
    inner_per_pair = tf.reduce_sum(log_term, axis=1)  # (P,)
    # Segment sum per TCR
    sum_per_tcr = tf.math.unsorted_segment_sum(inner_per_pair, row_idx_tf, n_chunk)
    N_i_tf = tf.constant(row_lengths, dtype=tf.float32)
    safe_N = tf.where(N_i_tf > 0, N_i_tf, tf.ones_like(N_i_tf))
    S_chunk = sum_per_tcr / safe_N
    return S_chunk.numpy().astype(np.float32)


def main():
    """Run the S_i computation pipeline."""
    args = parse_args()
    print("=" * 60)
    print("Compute S_i pipeline")
    print("=" * 60)
    print(f"  H5:           {args.h5_path}")
    print(f"  Donor matrix: {args.donor_matrix_path}")
    print(f"  Rare thresh:  < {args.rare_threshold} donors")
    print(f"  Chunk size:   {args.chunk_size} TCRs")
    # ── overwrite check ─────────────────────────────────────────────
    confirm_overwrite(args.h5_path, args.force)
    # ── GPU setup ───────────────────────────────────────────────────
    tf = maybe_setup_gpu(args.gpu)
    use_gpu = tf is not None
    print(f"  Compute on:   {'GPU' if use_gpu else 'CPU'}")
    # ── load donor HLA matrix ───────────────────────────────────────
    donor_hla = np.load(args.donor_matrix_path)["donor_hla_matrix"].astype(np.float32)
    D, A = donor_hla.shape
    print(f"  Donor matrix: {D} donors x {A} alleles")
    # Per-allele donor counts (used for rare-allele mask)
    n_a_per_allele = donor_hla.sum(axis=0)
    rare_mask = n_a_per_allele < args.rare_threshold  # (A,)
    print(f"  Rare alleles: {int(rare_mask.sum())}/{A} "
          f"(< {args.rare_threshold} donors)")
    # Build masked donor matrix (rare alleles zeroed out)
    donor_hla_masked = donor_hla.copy()
    donor_hla_masked[:, rare_mask] = 0.0
    # Push donor matrices to GPU once
    donor_hla_tf = donor_hla_tf_masked = None
    if use_gpu:
        donor_hla_tf = tf.constant(donor_hla, dtype=tf.float32)
        donor_hla_tf_masked = tf.constant(donor_hla_masked, dtype=tf.float32)
    # ── open H5 for read+write ──────────────────────────────────────
    h5 = h5py.File(args.h5_path, "r+")
    clusters = h5["clusters"]
    n_clusters = clusters["cluster_id"].shape[0]
    print(f"  Total TCRs:   {n_clusters:,}")
    # CSR handles
    zp_indptr = clusters["z_probs"]["indptr"]
    zp_indices = clusters["z_probs"]["indices"]
    zp_data = clusters["z_probs"]["data"]
    dn_indptr = clusters["donors"]["indptr"]
    dn_indices = clusters["donors"]["indices"]
    # ── allocate output datasets (overwrite if exist) ───────────────
    for name in ("S_i", "S_i_without_rare_alleles"):
        if name in clusters:
            del clusters[name]
        clusters.create_dataset(
            name, shape=(n_clusters,), dtype=np.float32,
            chunks=(min(args.chunk_size, n_clusters),),
            compression="gzip", compression_opts=4,
        )
    # ── chunked pass ────────────────────────────────────────────────
    t0 = time.time()
    for cs in range(0, n_clusters, args.chunk_size):
        ce = min(cs + args.chunk_size, n_clusters)
        n_chunk = ce - cs
        # Read gammas (CSR -> dense)
        zp_ip = np.asarray(zp_indptr[cs:ce + 1])
        zp_s, zp_e = int(zp_ip[0]), int(zp_ip[-1])
        zp_ip_local = zp_ip - zp_s
        gamma_chunk = csr_matrix(
            (np.asarray(zp_data[zp_s:zp_e]),
             np.asarray(zp_indices[zp_s:zp_e]),
             zp_ip_local),
            shape=(n_chunk, A),
        ).toarray().astype(np.float32)
        # Read donor indices (CSR-style ragged)
        dn_ip = np.asarray(dn_indptr[cs:ce + 1])
        dn_s, dn_e = int(dn_ip[0]), int(dn_ip[-1])
        donor_idx_flat = np.asarray(dn_indices[dn_s:dn_e]).astype(np.int64)
        row_lengths = (dn_ip[1:] - dn_ip[:-1]).astype(np.int64)
        # Compute S_i with full donor matrix
        if use_gpu:
            S_chunk = compute_S_chunk_gpu(
                tf, gamma_chunk, donor_idx_flat, row_lengths, donor_hla_tf)
        else:
            S_chunk = compute_S_chunk_cpu(
                gamma_chunk, donor_idx_flat, row_lengths, donor_hla)
        # Compute S_i with rare alleles masked out
        if use_gpu:
            S_chunk_masked = compute_S_chunk_gpu(
                tf, gamma_chunk, donor_idx_flat, row_lengths, donor_hla_tf_masked)
        else:
            S_chunk_masked = compute_S_chunk_cpu(
                gamma_chunk, donor_idx_flat, row_lengths, donor_hla_masked)
        # Write to H5
        clusters["S_i"][cs:ce] = S_chunk
        clusters["S_i_without_rare_alleles"][cs:ce] = S_chunk_masked
        # Progress
        elapsed = time.time() - t0
        rate = ce / elapsed if elapsed > 0 else 0
        eta = (n_clusters - ce) / rate if rate > 0 else 0
        print(f"  [{ce:>10,}/{n_clusters:,}] "
              f"{100*ce/n_clusters:5.1f}% | "
              f"{rate:,.0f} TCRs/s | ETA {eta/60:.1f}min")
    # ── attach metadata as HDF5 attrs ───────────────────────────────
    clusters["S_i"].attrs["formula"] = (
        "S_i = (1/N_i) * sum_n y_ni * sum_a (x_na*gamma_ia) * "
        "log((x_na*gamma_ia)/(sum_a' x_na'*gamma_ia'))")
    clusters["S_i"].attrs["range"] = "<= 0; closer to 0 = more confident"
    clusters["S_i_without_rare_alleles"].attrs["rare_threshold"] = args.rare_threshold
    clusters["S_i_without_rare_alleles"].attrs["n_rare_alleles_masked"] = int(rare_mask.sum())
    h5.close()
    total = time.time() - t0
    print(f"\nDone in {total:.1f}s ({total/60:.1f}min)")
    print(f"  Wrote: clusters/S_i  and  clusters/S_i_without_rare_alleles")


if __name__ == "__main__":
    main()