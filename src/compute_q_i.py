#!/usr/bin/env python3
"""
compute_q_i.py — Compute per-TCR detection probability q_i and store in H5.
=========================================================================
For each TCR i, q_i is the probability of the TCR being sequenced in a
donor given that the donor carries a binding HLA. Derived in the TCRtyper
paper as the MAP estimate under a Beta(beta, beta) prior:
    q_i = N_i / (N_i + tilde_N_i + beta - 1)
where:
    N_i       = number of donors where y_ni = 1 (from clusters/donors)
    tilde_N_i = sum over donors where y_ni = 0 of p_ni
    p_ni      = 1 - prod_a (1 - x_na * gamma_ia)
    x_na      = donor_hla_matrix[n, a]
    gamma_ia  = clusters/z_probs[i, a]
Uses log-space product to avoid 3D tensor materialization:
    log(1 - p_ni) = sum_a log(1 - x_na * gamma_ia + eps)
=========================================================================
USAGE:
    python compute_q_i.py \\
        --h5_path /path/to/filtered.h5 \\
        --donor_matrix_path /path/to/donor_hla_matrix.npz \\
        --beta 2.0 \\
        --chunk_size 1000 \\
        --gpu
Optional:
    --force    # overwrite existing clusters/q_i
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
    p = argparse.ArgumentParser(
        description="Compute per-TCR q_i and store in H5.")
    p.add_argument("--h5_path", required=True, help="H5 with z_probs + donors.")
    p.add_argument("--donor_matrix_path", required=True,
                   help="donor_hla_matrix.npz path.")
    p.add_argument("--beta", type=float, default=2.0,
                   help="Beta prior parameter (default: 2.0).")
    p.add_argument("--chunk_size", type=int, default=1000,
                   help="TCRs per chunk (default: 1000).")
    p.add_argument("--gpu", action="store_true", help="Use TF GPU.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing clusters/q_i without prompting.")
    return p.parse_args()


def maybe_setup_gpu(use_gpu):
    """Configure GPU memory growth. Return tf module or None."""
    if not use_gpu:
        return None
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            print("[GPU] No GPU detected, using CPU.")
            return None
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"[GPU] Using {len(gpus)} GPU(s)")
        return tf
    except Exception as e:
        print(f"[GPU] TF init failed ({e}), using CPU.")
        return None


def confirm_overwrite(h5_path, force):
    """Check if clusters/q_i exists; prompt or force-overwrite."""
    with h5py.File(h5_path, "r") as f:
        if "q_i" in f["clusters"]:
            if force:
                return
            print(f"\nWARNING: clusters/q_i already exists.")
            ans = input("Overwrite? [y/N]: ").strip().lower()
            if ans != "y":
                print("Aborted.")
                sys.exit(0)


def compute_chunk_cpu(gamma_chunk, donor_hla, y_chunk, beta):
    """CPU computation of q_i for a chunk of TCRs.
    Args:
        gamma_chunk: (B, A) dense gammas.
        donor_hla:   (D, A) donor HLA matrix.
        y_chunk:     (B, D) binary observed matrix.
        beta:        Beta prior parameter.
    Returns:
        q_chunk: (B,) float32 q values.
    """
    B, A = gamma_chunk.shape
    D = donor_hla.shape[0]
    eps = 1e-12
    # Log-space: log(1 - p_ni) = sum_a log(1 - x_na * gamma_ia)
    # Loop over HLA to keep memory bounded
    log_one_minus_p = np.zeros((B, D), dtype=np.float32)
    # Vectorized over A for efficiency: we need (B, D, A) accumulation
    # but we can batch alleles to keep memory in check.
    allele_batch = 64
    for a_start in range(0, A, allele_batch):
        a_end = min(a_start + allele_batch, A)
        # (B, k) @ (k, D) -> doesn't work; we need elementwise.
        # Use broadcasting: (B, 1, k) * (1, D, k) -> (B, D, k)
        g_slice = gamma_chunk[:, a_start:a_end]  # (B, k)
        x_slice = donor_hla[:, a_start:a_end]    # (D, k)
        prod = g_slice[:, None, :] * x_slice[None, :, :]  # (B, D, k)
        log_term = np.log(np.clip(1.0 - prod, eps, None))  # (B, D, k)
        log_one_minus_p += log_term.sum(axis=2).astype(np.float32)
    p_ni = 1.0 - np.exp(log_one_minus_p)  # (B, D)
    N_i = y_chunk.sum(axis=1).astype(np.float64)
    tilde_N_i = ((1.0 - y_chunk) * p_ni).sum(axis=1).astype(np.float64)
    q = N_i / (N_i + tilde_N_i + beta - 1.0)
    return q.astype(np.float32)


def compute_chunk_gpu(tf, gamma_chunk, donor_hla_tf, y_chunk, beta):
    """GPU computation of q_i for a chunk of TCRs."""
    B, A = gamma_chunk.shape
    D = int(donor_hla_tf.shape[0])
    eps = 1e-12
    gamma_tf = tf.constant(gamma_chunk, dtype=tf.float32)
    y_tf = tf.constant(y_chunk, dtype=tf.float32)
    # Compute log(1 - x_na * gamma_ia) summed over a
    # Batch alleles to bound memory: (B, allele_batch, D)
    allele_batch = 64
    log_one_minus_p = tf.zeros((B, D), dtype=tf.float32)
    for a_start in range(0, A, allele_batch):
        a_end = min(a_start + allele_batch, A)
        g_slice = gamma_tf[:, a_start:a_end]              # (B, k)
        x_slice = donor_hla_tf[:, a_start:a_end]          # (D, k)
        prod = g_slice[:, None, :] * x_slice[None, :, :]  # (B, D, k)
        log_term = tf.math.log(tf.maximum(1.0 - prod, eps))
        log_one_minus_p += tf.reduce_sum(log_term, axis=2)
    p_ni = 1.0 - tf.exp(log_one_minus_p)
    N_i = tf.reduce_sum(y_tf, axis=1)
    tilde_N_i = tf.reduce_sum((1.0 - y_tf) * p_ni, axis=1)
    q = N_i / (N_i + tilde_N_i + beta - 1.0)
    return q.numpy().astype(np.float32)


def build_y_chunk(donor_indices_flat, row_lengths, D):
    """Build dense (B, D) binary matrix from ragged donor indices.
    Args:
        donor_indices_flat: (P,) flat donor indices for all (TCR, donor) pairs.
        row_lengths:        (B,) number of donors per TCR.
        D:                  total donor count.
    Returns:
        y: (B, D) float32 binary matrix.
    """
    B = len(row_lengths)
    y = np.zeros((B, D), dtype=np.float32)
    row_idx = np.repeat(np.arange(B, dtype=np.int64), row_lengths)
    y[row_idx, donor_indices_flat] = 1.0
    return y


def main():
    """Run q_i computation pipeline."""
    args = parse_args()
    print("=" * 60)
    print("Compute q_i pipeline")
    print("=" * 60)
    print(f"  H5:           {args.h5_path}")
    print(f"  Donor matrix: {args.donor_matrix_path}")
    print(f"  Beta prior:   {args.beta}")
    print(f"  Chunk size:   {args.chunk_size}")
    confirm_overwrite(args.h5_path, args.force)
    tf = maybe_setup_gpu(args.gpu)
    use_gpu = tf is not None
    print(f"  Compute on:   {'GPU' if use_gpu else 'CPU'}")
    # ── load donor matrix ───────────────────────────────────────────
    donor_hla = np.load(args.donor_matrix_path)["donor_hla_matrix"].astype(np.float32)
    D, A = donor_hla.shape
    print(f"  Donor matrix: {D} donors x {A} alleles")
    donor_hla_tf = None
    if use_gpu:
        donor_hla_tf = tf.constant(donor_hla, dtype=tf.float32)
    # ── open H5 ─────────────────────────────────────────────────────
    h5 = h5py.File(args.h5_path, "r+")
    clusters = h5["clusters"]
    n_clusters = clusters["cluster_id"].shape[0]
    print(f"  Total TCRs:   {n_clusters:,}")
    zp_indptr = clusters["z_probs"]["indptr"]
    zp_indices = clusters["z_probs"]["indices"]
    zp_data = clusters["z_probs"]["data"]
    dn_indptr = clusters["donors"]["indptr"]
    dn_indices = clusters["donors"]["indices"]
    if "q_i" in clusters:
        del clusters["q_i"]
    clusters.create_dataset(
        "q_i", shape=(n_clusters,), dtype=np.float32,
        chunks=(min(args.chunk_size, n_clusters),),
        compression="gzip", compression_opts=4)
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
        # Read donor indices for this chunk
        dn_ip = np.asarray(dn_indptr[cs:ce + 1])
        dn_s, dn_e = int(dn_ip[0]), int(dn_ip[-1])
        donor_idx_flat = np.asarray(dn_indices[dn_s:dn_e]).astype(np.int64)
        row_lengths = (dn_ip[1:] - dn_ip[:-1]).astype(np.int64)
        # Build binary y_chunk (B, D)
        y_chunk = build_y_chunk(donor_idx_flat, row_lengths, D)
        # Compute q_i
        if use_gpu:
            q_chunk = compute_chunk_gpu(
                tf, gamma_chunk, donor_hla_tf, y_chunk, args.beta)
        else:
            q_chunk = compute_chunk_cpu(
                gamma_chunk, donor_hla, y_chunk, args.beta)
        # Write
        clusters["q_i"][cs:ce] = q_chunk
        # Progress
        elapsed = time.time() - t0
        rate = ce / elapsed if elapsed > 0 else 0
        eta = (n_clusters - ce) / rate if rate > 0 else 0
        print(f"  [{ce:>10,}/{n_clusters:,}] "
              f"{100*ce/n_clusters:5.1f}% | "
              f"{rate:,.0f} TCRs/s | ETA {eta/60:.1f}min")
    clusters["q_i"].attrs["beta"] = args.beta
    clusters["q_i"].attrs["formula"] = (
        "q_i = N_i / (N_i + tilde_N_i + beta - 1)")
    h5.close()
    total = time.time() - t0
    print(f"\nDone in {total:.1f}s ({total/60:.1f}min)")
    print(f"  Wrote: clusters/q_i")


if __name__ == "__main__":
    main()
