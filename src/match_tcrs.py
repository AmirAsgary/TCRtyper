#!/usr/bin/env python3
"""
match_tcrs.py — Match valid TCRs to nearest training TCRs.
=========================================================================
Inputs: precomputed train + valid embeddings (from precompute_embeddings.py).
Steps:
  1. Build exact match hash on cdr3aa sequences (if available), else via
     argmax consensus of CDR frequency profile.
  2. GPU cosine similarity search for unmatched TCRs, chunked to fit GPU.
  3. Compute confidence = sigmoid(alpha * (sim - threshold)).
Output: NPZ with arrays `nn_idx`, `nn_sim`, `confidence`, `n_exact`.
=========================================================================
USAGE:
    python match_tcrs.py \\
        --train_emb /path/to/train_emb.npz \\
        --valid_emb /path/to/valid_emb.npz \\
        --train_h5 /path/to/train.h5 \\
        --valid_h5 /path/to/valid.h5 \\
        --output /path/to/matches.npz \\
        --sim_threshold 0.9 \\
        --alpha_conf 20 \\
        --gpu \\
        --chunk_size 1000
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
    p = argparse.ArgumentParser(description="Match valid TCRs to train.")
    p.add_argument("--train_emb", required=True, help="Train embedding NPZ.")
    p.add_argument("--valid_emb", required=True, help="Valid embedding NPZ.")
    p.add_argument("--train_h5", required=True,
                   help="Train H5 (for cdr3aa exact hash if present).")
    p.add_argument("--valid_h5", required=True,
                   help="Valid H5 (for cdr3aa exact hash if present).")
    p.add_argument("--output", required=True, help="Output NPZ path.")
    p.add_argument("--sim_threshold", type=float, default=0.9)
    p.add_argument("--alpha_conf", type=float, default=20.0)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--chunk_size", type=int, default=100000,
                   help="Test TCRs per similarity chunk. Will be auto-clipped "
                        "to int32-safe value. Default: 100000.")
    p.add_argument("--fp32", action="store_true",
                   help="Use FP32 instead of FP16 (slower but more precise).")
    p.add_argument("--force", action="store_true")
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
        print(f"[GPU] init failed ({e}), using CPU.")
        return None


def build_cdr3_hashes(h5_path):
    """Build exact-match hash keys from cdr3aa strings in tcrs/loops/cdr3aa.
    Uses the first TCR row per cluster as a proxy for the cluster sequence.
    Returns None if tcrs/ group not present.
    Returns:
        hashes: list[bytes] of length n_clusters, or None.
    """
    with h5py.File(h5_path, "r") as f:
        if "tcrs" not in f:
            return None
        if "loops" not in f["tcrs"] or "cdr3aa" not in f["tcrs"]["loops"]:
            return None
        if "clusters/tcr_indptr" not in f:
            return None
        tcr_indptr = np.asarray(f["clusters/tcr_indptr"][:])
        n_clusters = tcr_indptr.shape[0] - 1
        # Use first TCR row of each cluster as the cluster's representative.
        # tcr_indptr[i] is the first TCR row index of cluster i.
        first_rows = tcr_indptr[:-1].astype(np.int64)
        cdr3_ds = f["tcrs/loops/cdr3aa"]
        # Bulk read all first rows. Fancy indexing is slow in h5py; read all.
        all_cdr3 = np.asarray(cdr3_ds[:])
        reps = all_cdr3[first_rows]
    # Convert to bytes for hashing
    hashes = [b if isinstance(b, (bytes, bytearray)) else str(b).encode("utf-8")
              for b in reps]
    return hashes


def build_consensus_hashes(emb):
    """Fallback exact-match via argmax bytes of embedding. Collision-prone
    but same across both H5s only if the pooled embedding is identical."""
    # Quantize to 3 decimal places, use bytes as key
    q = np.round(emb, 3).astype(np.float32)
    return [row.tobytes() for row in q]


def similarity_search_gpu(tf, test_emb, train_emb, chunk_size,
                          use_fp16=True):
    """Argmax cosine similarity via BLOCKED matmul with running best.
    The key optimization: we block BOTH test and train dimensions so each
    matmul result stays well under the int32 limit (2^31 elements). This
    lets us use large block sizes that saturate Tensor Cores.
    For each pair (test_block, train_block):
        sim = test_block @ train_block.T        # (B_test, B_train)
        block_best_sim = max(sim, axis=1)        # (B_test,)
        block_best_idx = argmax(sim, axis=1) + train_offset
        update running_best[test_block_range] where block_best_sim > running
    With block=40000, each matmul is 1.6B elements = 3.2 GB FP16.
    On A100 this saturates TensorCores (needs large matrices to be fast).
    Args:
        test_emb:   (Nt, D) L2-normalized.
        train_emb:  (Ntr, D) L2-normalized.
        chunk_size: block size for BOTH dims. Auto-clipped to int32 safe.
        use_fp16:   use FP16 tensor cores.
    Returns:
        best_idx: (Nt,) int64
        best_sim: (Nt,) float32
    """
    Nt = test_emb.shape[0]
    Ntr = train_emb.shape[0]
    # Cap block to int32 safety: a*b < 2^31. With square blocks a=b=46340.
    # Use a bit less headroom for safety. Also cap to avoid GPU OOM.
    max_square = int((2**31 - 1) ** 0.5 * 0.9)  # ~41835
    if chunk_size <= 0 or chunk_size > max_square:
        chunk_size = max_square
    # Also respect GPU memory: FP16 block = chunk^2 * 2 bytes
    # For 40k block: 3.2 GB. A100 can easily handle this with train/test
    # embeddings already resident (~6 GB total).
    dtype_np = np.float16 if use_fp16 else np.float32
    dtype_tf = tf.float16 if use_fp16 else tf.float32
    bytes_per = 2 if use_fp16 else 4
    block_gb = chunk_size * chunk_size * bytes_per / 1e9
    print(f"  [sim] test={Nt:,} train={Ntr:,} block={chunk_size:,} | "
          f"dtype={'fp16' if use_fp16 else 'fp32'} | "
          f"per-block result: {block_gb:.2f} GB")
    # Pre-cast and upload both embeddings ONCE.
    print(f"  [sim] Uploading embeddings to GPU...")
    test_tf_full = tf.constant(test_emb.astype(dtype_np, copy=False),
                                dtype=dtype_tf)
    train_tf_full = tf.constant(train_emb.astype(dtype_np, copy=False),
                                 dtype=dtype_tf)
    n_test_blocks = (Nt + chunk_size - 1) // chunk_size
    n_train_blocks = (Ntr + chunk_size - 1) // chunk_size
    total_blocks = n_test_blocks * n_train_blocks
    print(f"  [sim] {n_test_blocks} test blocks x {n_train_blocks} "
          f"train blocks = {total_blocks} matmuls")
    # Running-best arrays on GPU; updated after each (test, train) block.
    # We allocate them per test block only (small, (chunk,)).
    best_idx = np.empty(Nt, dtype=np.int64)
    best_sim = np.empty(Nt, dtype=np.float32)
    # Compile the inner block once
    @tf.function(jit_compile=False)
    def _block_op(test_block, train_block, train_offset, running_sim, running_idx):
        # sim: (B_test, B_train)
        sim = tf.matmul(test_block, train_block, transpose_b=True)
        # Argmax within block
        block_idx = tf.argmax(sim, axis=1, output_type=tf.int32)  # (B_test,)
        block_max = tf.reduce_max(sim, axis=1)                    # (B_test,)
        block_max_f32 = tf.cast(block_max, tf.float32)
        # Running update: where block_max > running_sim, replace
        better = block_max_f32 > running_sim
        new_sim = tf.where(better, block_max_f32, running_sim)
        new_idx = tf.where(
            better,
            tf.cast(block_idx, tf.int64) + tf.cast(train_offset, tf.int64),
            running_idx)
        return new_sim, new_idx
    t0 = time.time()
    last_log = t0
    blocks_done = 0
    for ts in range(0, Nt, chunk_size):
        te = min(ts + chunk_size, Nt)
        B_test = te - ts
        test_block = test_tf_full[ts:te]
        # Initialize running best for this test block
        running_sim = tf.fill((B_test,), -1e9)
        running_idx = tf.fill((B_test,), tf.constant(-1, tf.int64))
        for trs in range(0, Ntr, chunk_size):
            tre = min(trs + chunk_size, Ntr)
            train_block = train_tf_full[trs:tre]
            running_sim, running_idx = _block_op(
                test_block, train_block, trs, running_sim, running_idx)
            blocks_done += 1
        # Pull results for this test block (one host sync per test block)
        best_idx[ts:te] = running_idx.numpy()
        best_sim[ts:te] = running_sim.numpy()
        now = time.time()
        if now - last_log > 5.0 or te == Nt:
            elapsed = now - t0
            rate = te / elapsed if elapsed > 0 else 0
            eta = (Nt - te) / rate if rate > 0 else 0
            frac_blocks = blocks_done / total_blocks
            print(f"  [sim] {te:,}/{Nt:,} ({100*te/Nt:.1f}%) | "
                  f"{rate:,.0f} TCRs/s | "
                  f"blocks {blocks_done}/{total_blocks} ({100*frac_blocks:.1f}%) | "
                  f"ETA {eta/60:.1f}min")
            last_log = now
    return best_idx, best_sim


def similarity_search_cpu(test_emb, train_emb, chunk_size):
    """CPU fallback."""
    Nt = test_emb.shape[0]
    train_t = train_emb.T
    best_idx = np.empty(Nt, dtype=np.int64)
    best_sim = np.empty(Nt, dtype=np.float32)
    for s in range(0, Nt, chunk_size):
        e = min(s + chunk_size, Nt)
        sim = test_emb[s:e] @ train_t
        best_idx[s:e] = sim.argmax(axis=1)
        best_sim[s:e] = sim.max(axis=1)
    return best_idx, best_sim


def main():
    """Run TCR matching pipeline."""
    args = parse_args()
    if Path(args.output).exists() and not args.force:
        print(f"[SKIP] {args.output} exists (use --force)")
        return
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Match valid TCRs to train")
    print("=" * 60)
    # Load embeddings
    print(f"  Loading train embedding: {args.train_emb}")
    tr = np.load(args.train_emb)
    train_emb = tr["embedding"]
    print(f"    shape: {train_emb.shape}")
    print(f"  Loading valid embedding: {args.valid_emb}")
    vd = np.load(args.valid_emb)
    valid_emb = vd["embedding"]
    print(f"    shape: {valid_emb.shape}")
    n_train = train_emb.shape[0]
    n_valid = valid_emb.shape[0]
    # Exact match via cdr3aa
    print(f"\n  Building exact-match hashes...")
    train_hashes = build_cdr3_hashes(args.train_h5)
    valid_hashes = build_cdr3_hashes(args.valid_h5)
    if train_hashes is not None and valid_hashes is not None:
        print(f"    Using cdr3aa strings from tcrs/loops/cdr3aa")
    else:
        print(f"    tcrs/loops/cdr3aa not available, falling back to "
              f"embedding-based hashes (lower hit rate)")
        train_hashes = build_consensus_hashes(train_emb)
        valid_hashes = build_consensus_hashes(valid_emb)
    hash_to_train = {}
    for i, h in enumerate(train_hashes):
        if h not in hash_to_train:
            hash_to_train[h] = i
    exact_idx = np.full(n_valid, -1, dtype=np.int64)
    # Vectorize via dict lookup
    for i, h in enumerate(valid_hashes):
        hit = hash_to_train.get(h, -1)
        exact_idx[i] = hit
    n_exact = int((exact_idx >= 0).sum())
    print(f"  Exact matches: {n_exact:,}/{n_valid:,} "
          f"({100*n_exact/n_valid:.2f}%)")
    # Similarity search for unmatched
    nn_idx = np.full(n_valid, -1, dtype=np.int64)
    nn_sim = np.zeros(n_valid, dtype=np.float32)
    exact_mask = exact_idx >= 0
    nn_idx[exact_mask] = exact_idx[exact_mask]
    nn_sim[exact_mask] = 1.0
    unmatched = ~exact_mask
    n_unmatched = int(unmatched.sum())
    if n_unmatched > 0:
        print(f"\n  Similarity search for {n_unmatched:,} unmatched...")
        tf = maybe_setup_gpu(args.gpu)
        q_emb = valid_emb[unmatched]
        if tf is not None:
            bi, bs = similarity_search_gpu(
                tf, q_emb, train_emb, args.chunk_size,
                use_fp16=not args.fp32)
        else:
            bi, bs = similarity_search_cpu(
                q_emb, train_emb, args.chunk_size)
        nn_idx[unmatched] = bi
        nn_sim[unmatched] = bs
    # Confidence weights
    confidence = 1.0 / (1.0 + np.exp(
        -args.alpha_conf * (nn_sim - args.sim_threshold)))
    confidence[exact_mask] = 1.0
    n_high_conf = int((confidence > 0.8).sum())
    n_discarded = int((confidence <= 0.8).sum())
    print(f"\n  Confidence stats:")
    print(f"    Mean:                  {confidence.mean():.3f}")
    print(f"    > 0.8 (strong match):  {n_high_conf:,}")
    print(f"    <= 0.8 (will drop):   {n_discarded:,}")
    # Save
    np.savez(
        args.output,
        nn_idx=nn_idx,
        nn_sim=nn_sim,
        confidence=confidence.astype(np.float32),
        n_exact=np.int64(n_exact),
        sim_threshold=np.float32(args.sim_threshold),
        alpha_conf=np.float32(args.alpha_conf),
    )
    print(f"\n  Saved: {args.output}")


if __name__ == "__main__":
    main()