#!/usr/bin/env python3
"""
predict_donor_genotypes.py — MAP HLA genotype prediction from TCR-seq.
=========================================================================
For each donor in a test H5, predict their HLA genotype by:
  1. Matching test TCRs to training TCRs (exact + similarity search)
  2. Running MAP optimization of the TCRtyper likelihood over the
     continuous donor genotype x_n in [0, 1]^A
  3. Applying gene-count regularizer (max 2 alleles per gene)
  4. Masking rare HLAs (< rare_threshold donors in training)
  5. Reporting top-K candidate pairs per gene
Similarity metric (default): pooled CDR frequency profile cosine.
For each CDR region, compute mean profile (21,) + position-binned profile
(5, 21), concatenate across 4 CDRs → 504-dim embedding. Length-invariant.
Optional: hybrid (cosine ANN + CDR3 edit distance re-rank) via --similarity hybrid.
=========================================================================
USAGE:
    python predict_donor_genotypes.py \\
        --train_h5 /path/to/filtered.h5 \\
        --valid_h5 /path/to/valid_ds.h5 \\
        --donor_matrix_path /path/to/donor_hla_matrix.npz \\
        --hla_to_id /path/to/hla_to_id.json \\
        --output_dir /path/to/out \\
        --gpu
Flags:
    --similarity {pooled, hybrid}     Default: pooled
    --sim_threshold 0.9               Cosine similarity threshold for NN.
    --alpha_conf 20                   Confidence sigmoid sharpness.
    --topk 5                          Top-K candidate pairs per gene.
    --add_absent                      Include absent-TCR likelihood term.
    --rare_threshold 5                Mask HLAs with < this many donors.
    --lambda_reg 10.0                 Gene-count regularizer strength.
    --n_iters 300                     MAP optimization steps.
    --lr 0.1                          Optimization learning rate.
    --test_chunk_size 10000           Test TCRs per similarity search chunk.
=========================================================================
"""
import os
import sys
import re
import json
import time
import argparse
import numpy as np
import h5py
from pathlib import Path
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="MAP HLA genotype prediction from TCR-seq.")
    p.add_argument("--train_h5", required=True,
                   help="Filtered training H5 with z_probs, q_i, cdr_freq.")
    p.add_argument("--valid_h5", required=True,
                   help="Test/valid H5 with donors + cdr_freq.")
    p.add_argument("--donor_matrix_path", required=True,
                   help="donor_hla_matrix.npz (for rare HLA masking).")
    p.add_argument("--hla_to_id", required=True,
                   help="hla_to_id.json for gene grouping.")
    p.add_argument("--output_dir", required=True, help="Output directory.")
    p.add_argument("--gpu", action="store_true", help="Use TF GPU.")
    p.add_argument("--similarity", choices=["pooled", "hybrid"],
                   default="pooled",
                   help="Similarity method (default: pooled).")
    p.add_argument("--sim_threshold", type=float, default=0.9,
                   help="Cosine threshold for accepting NN match (default: 0.9).")
    p.add_argument("--alpha_conf", type=float, default=20.0,
                   help="Confidence sigmoid sharpness (default: 20).")
    p.add_argument("--topk", type=int, default=5,
                   help="Top-K candidate pairs per gene (default: 5).")
    p.add_argument("--rare_threshold", type=int, default=5,
                   help="Mask HLAs with < this many donors (default: 5).")
    p.add_argument("--lambda_reg", type=float, default=10.0,
                   help="Gene-count regularizer strength (default: 10).")
    p.add_argument("--n_iters", type=int, default=300,
                   help="MAP optimization iters (default: 300).")
    p.add_argument("--lr", type=float, default=0.1,
                   help="Optimization LR (default: 0.1).")
    p.add_argument("--add_absent", action="store_true",
                   help="Include absent-TCR term in likelihood.")
    p.add_argument("--train_chunk_size", type=int, default=50000,
                   help="Training TCRs per chunk (default: 50000).")
    p.add_argument("--test_chunk_size", type=int, default=1000,
                   help="Test TCRs per similarity-search chunk (default: 1000). "
                        "Lower if you OOM. Result tensor is "
                        "chunk_size * n_train * 4 bytes.")
    p.add_argument("--n_position_bins", type=int, default=5,
                   help="Number of positional bins for CDR embedding (default: 5).")
    return p.parse_args()


# -------------------------------------------------------------------------
# GPU setup
# -------------------------------------------------------------------------

def maybe_setup_gpu(use_gpu):
    """Configure GPU memory growth. Return tf module or None."""
    if not use_gpu:
        return None
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            print("[GPU] No GPU, using CPU.")
            return None
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"[GPU] Using {len(gpus)} GPU(s)")
        return tf
    except Exception as e:
        print(f"[GPU] init failed ({e}), using CPU.")
        return None


# -------------------------------------------------------------------------
# HLA gene grouping
# -------------------------------------------------------------------------

def parse_hla_gene(name):
    """Extract gene symbol from HLA name like 'HLA-A*02:01' -> 'A'.
    Recognizes class I (A, B, C) and class II (DPA1, DPB1, DQA1, DQB1,
    DRA1, DRB1, etc.).
    """
    m = re.match(r"HLA-([ABC]|D[PQR][AB]\d?)\*", name)
    if m:
        return m.group(1)
    return None


def build_gene_mapping(hla_to_id_path):
    """Parse hla_to_id.json and build gene -> list of allele IDs.
    Returns:
        gene_to_alleles: dict[str, List[int]]
        id_to_gene:      np.ndarray of gene names per allele id
        id_to_name:      dict[int, str]
    """
    with open(hla_to_id_path, "r") as f:
        hla_to_id = json.load(f)
    A = max(hla_to_id.values()) + 1
    id_to_name = {int(v): k for k, v in hla_to_id.items()}
    id_to_gene = np.array([""] * A, dtype=object)
    gene_to_alleles = {}
    for name, idx in hla_to_id.items():
        g = parse_hla_gene(name)
        if g is None:
            continue
        id_to_gene[idx] = g
        gene_to_alleles.setdefault(g, []).append(int(idx))
    return gene_to_alleles, id_to_gene, id_to_name


# -------------------------------------------------------------------------
# CDR freq embedding (length-invariant pooled representation)
# -------------------------------------------------------------------------

def embed_cdrs_pooled(cdr_freq_dict, n_clusters, n_bins):
    """Build fixed-size pooled embedding from per-cluster CDR freq profiles.
    For each of 4 CDRs and each cluster:
      - mean profile: (21,) — mean AA composition over length
      - bin profile:  (n_bins, 21) — mean within each positional bin
    Flattened and concatenated across 4 CDRs.
    Final dim = 4 * 21 * (1 + n_bins)
    Args:
        cdr_freq_dict: dict with keys cdr1, cdr2, cdr25, cdr3 -> RaggedAccessor.
        n_clusters:    total clusters in the reader.
        n_bins:        number of positional bins.
    Returns:
        emb: (n_clusters, D_emb) float32 L2-normalized embedding.
    """
    cdr_names = ["cdr1", "cdr2", "cdr25", "cdr3"]
    D_per_cdr = 21 * (1 + n_bins)
    D_emb = 4 * D_per_cdr
    emb = np.zeros((n_clusters, D_emb), dtype=np.float32)
    for c_idx, nm in enumerate(cdr_names):
        acc = cdr_freq_dict[nm]
        # acc is a RaggedClusterAccessor; iterate per cluster
        base = c_idx * D_per_cdr
        for i in range(n_clusters):
            prof = acc[i]  # (L_i, 21)
            L = prof.shape[0]
            if L == 0:
                continue
            # Mean profile
            mean_prof = prof.mean(axis=0)  # (21,)
            emb[i, base:base + 21] = mean_prof
            # Positional bins
            if L >= 1:
                bin_idx = np.minimum(
                    (np.arange(L) * n_bins // L).astype(np.int64),
                    n_bins - 1)
                for b in range(n_bins):
                    m = bin_idx == b
                    if m.sum() > 0:
                        emb[i, base + 21 + b * 21:base + 21 + (b + 1) * 21] = (
                            prof[m].mean(axis=0))
    # L2 normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    emb = emb / norms
    return emb


def build_embedding_for_h5(h5_path, n_bins, chunk_size, tag=""):
    """Stream through H5 and build the pooled embedding for all clusters."""
    from utils import PublicTcrHlaCsrReaderChunk
    with PublicTcrHlaCsrReaderChunk(
        h5_path, include_counts=False, include_donors=False,
        include_cdr_freq=True, include_z_probs=False,
    ) as reader:
        n_clusters = reader.num_clusters
        D_emb = 4 * 21 * (1 + n_bins)
        emb = np.zeros((n_clusters, D_emb), dtype=np.float32)
        t0 = time.time()
        written = 0
        for chunk in reader.iter_cluster_chunks(chunk_rows=chunk_size):
            n_chunk = chunk.n_clusters
            chunk_emb = embed_cdrs_pooled(chunk.cdr_freq, n_chunk, n_bins)
            emb[written:written + n_chunk] = chunk_emb
            written += n_chunk
            elapsed = time.time() - t0
            rate = written / elapsed if elapsed > 0 else 0
            print(f"  [{tag}] {written:,}/{n_clusters:,} "
                  f"({100*written/n_clusters:.1f}%) | {rate:,.0f} TCRs/s")
    return emb


# -------------------------------------------------------------------------
# Exact hash matching + similarity search
# -------------------------------------------------------------------------

def build_seq_hash(cdr_freq_dict, n_clusters):
    """Build hash key per cluster for exact-match lookup.
    Uses bytes of concatenated argmax indices across CDRs (consensus).
    Fast and collision-safe enough for exact matches.
    """
    cdr_names = ["cdr1", "cdr2", "cdr25", "cdr3"]
    hashes = []
    for i in range(n_clusters):
        parts = []
        for nm in cdr_names:
            prof = cdr_freq_dict[nm][i]
            if prof.shape[0] == 0:
                parts.append(b"")
            else:
                consensus = prof.argmax(axis=1).astype(np.uint8).tobytes()
                parts.append(consensus)
        hashes.append(b"|".join(parts))
    return hashes


def similarity_search_gpu(tf, test_emb, train_emb, test_chunk_size):
    """Compute argmax cosine similarity for each test emb against all train.
    Memory-safe: keeps train_emb on GPU once, processes test in small chunks.
    The result tensor (chunk, n_train) float32 dominates memory.
    For chunk=1000, n_train=3M: 12 GB per call. Lower chunk if OOM.
    Args:
        test_emb:  (Nt, D) L2-normalized.
        train_emb: (Ntr, D) L2-normalized.
    Returns:
        best_idx: (Nt,) int64 — index into train.
        best_sim: (Nt,) float32 — cosine similarity of best match.
    """
    Nt = test_emb.shape[0]
    Ntr = train_emb.shape[0]
    print(f"  [sim] test={Nt:,} train={Ntr:,} "
          f"chunk={test_chunk_size:,} | "
          f"per-chunk result: "
          f"{test_chunk_size * Ntr * 4 / 1e9:.1f} GB")
    train_tf = tf.constant(train_emb, dtype=tf.float32)
    train_t = tf.transpose(train_tf)  # (D, Ntr)
    best_idx = np.empty(Nt, dtype=np.int64)
    best_sim = np.empty(Nt, dtype=np.float32)
    t0 = time.time()
    last_log = t0
    n_steps = (Nt + test_chunk_size - 1) // test_chunk_size
    for step_i, s in enumerate(range(0, Nt, test_chunk_size)):
        e = min(s + test_chunk_size, Nt)
        q = tf.constant(test_emb[s:e], dtype=tf.float32)
        sim = tf.matmul(q, train_t)  # (chunk, Ntr)
        bi = tf.argmax(sim, axis=1, output_type=tf.int64)
        bs = tf.reduce_max(sim, axis=1)
        best_idx[s:e] = bi.numpy()
        best_sim[s:e] = bs.numpy()
        # Free GPU intermediates between iterations
        del q, sim, bi, bs
        now = time.time()
        if now - last_log > 5.0 or step_i == n_steps - 1:
            elapsed = now - t0
            rate = e / elapsed if elapsed > 0 else 0
            eta = (Nt - e) / rate if rate > 0 else 0
            print(f"  [sim] {e:,}/{Nt:,} ({100*e/Nt:.1f}%) | "
                  f"{rate:,.0f} TCRs/s | ETA {eta/60:.1f}min")
            last_log = now
    return best_idx, best_sim


def similarity_search_cpu(test_emb, train_emb, test_chunk_size):
    """CPU fallback for similarity search."""
    Nt = test_emb.shape[0]
    train_t = train_emb.T
    best_idx = np.empty(Nt, dtype=np.int64)
    best_sim = np.empty(Nt, dtype=np.float32)
    t0 = time.time()
    for s in range(0, Nt, test_chunk_size):
        e = min(s + test_chunk_size, Nt)
        sim = test_emb[s:e] @ train_t
        best_idx[s:e] = sim.argmax(axis=1)
        best_sim[s:e] = sim.max(axis=1)
        elapsed = time.time() - t0
        print(f"  [sim] {e:,}/{Nt:,} ({100*e/Nt:.1f}%) | {elapsed:.1f}s")
    return best_idx, best_sim


def compute_confidence(sim, threshold, alpha):
    """Map similarity -> [0,1] confidence via sigmoid."""
    return 1.0 / (1.0 + np.exp(-alpha * (sim - threshold)))


# -------------------------------------------------------------------------
# MAP optimization (donor by donor)
# -------------------------------------------------------------------------

def run_map_optimization(tf, gamma_mat, q_vec, conf_vec, y_vec,
                         gene_group_mat, valid_mask, args):
    """Run MAP optimization for a single donor.
    Args:
        gamma_mat:      (T, A) gammas for matched TCRs (confidence-weighted
                        will be applied via conf_vec).
        q_vec:          (T,) q_i per TCR.
        conf_vec:       (T,) confidence weights in [0, 1].
        y_vec:          (T,) binary observed mask (1 if TCR in donor).
        gene_group_mat: (G, A) binary matrix, row g is 1 for alleles in gene g.
        valid_mask:     (A,) float 1 if allele is non-rare and predictable.
        args:           parsed args.
    Returns:
        x_final: (A,) numpy float32 in [0,1].
    """
    T, A = gamma_mat.shape
    if T == 0:
        return np.zeros(A, dtype=np.float32)
    # Init logits; start near uniform low
    init = np.full(A, -3.0, dtype=np.float32)
    theta = tf.Variable(init, dtype=tf.float32)
    gamma_tf = tf.constant(gamma_mat, dtype=tf.float32)
    q_tf = tf.constant(q_vec, dtype=tf.float32)
    conf_tf = tf.constant(conf_vec, dtype=tf.float32)
    y_tf = tf.constant(y_vec, dtype=tf.float32)
    gene_tf = tf.constant(gene_group_mat, dtype=tf.float32)
    valid_tf = tf.constant(valid_mask, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    add_absent = bool(args.add_absent)
    lambda_reg = float(args.lambda_reg)
    eps = 1e-12
    @tf.function
    def step():
        with tf.GradientTape() as tape:
            x = tf.sigmoid(theta) * valid_tf            # (A,)
            # p_ni = 1 - prod_a (1 - x_na * gamma_ia)
            log_one_minus = tf.reduce_sum(
                tf.math.log(tf.maximum(
                    1.0 - x[None, :] * gamma_tf, eps)),
                axis=1)                                  # (T,)
            p = 1.0 - tf.exp(log_one_minus)              # (T,)
            qp = q_tf * p                                # (T,)
            qp = tf.clip_by_value(qp, eps, 1.0 - eps)
            term_obs = y_tf * tf.math.log(qp)
            if add_absent:
                term_abs = (1.0 - y_tf) * tf.math.log(1.0 - qp)
                log_lik = term_obs + term_abs
            else:
                log_lik = term_obs
            weighted_ll = tf.reduce_sum(conf_tf * log_lik)
            # Gene regularizer: penalize sum_a x_a > 2 per gene
            gene_sums = tf.linalg.matvec(gene_tf, x)      # (G,)
            reg = lambda_reg * tf.reduce_sum(
                tf.maximum(0.0, gene_sums - 2.0))
            loss = -weighted_ll + reg
        grads = tape.gradient(loss, [theta])
        optimizer.apply_gradients(zip(grads, [theta]))
        return loss
    for _ in range(args.n_iters):
        step()
    x_final = (tf.sigmoid(theta) * valid_tf).numpy()
    return x_final


# -------------------------------------------------------------------------
# Top-K candidate pair enumeration per gene
# -------------------------------------------------------------------------

def enumerate_topk_pairs(x_scores, gene_to_alleles, id_to_name, topk,
                         top_candidates=4):
    """For each gene, take top `top_candidates` alleles by x score, then
    enumerate all pairs, score by x_a * x_b (simple), return top-K.
    Returns:
        result: dict[gene, list of (allele_name1, allele_name2, score)]
    """
    out = {}
    for gene, allele_ids in gene_to_alleles.items():
        scores = x_scores[allele_ids]
        order = np.argsort(-scores)[:top_candidates]
        top_ids = [allele_ids[i] for i in order]
        top_scores = scores[order]
        pairs = []
        for i in range(len(top_ids)):
            for j in range(i, len(top_ids)):
                a, b = top_ids[i], top_ids[j]
                score = float(top_scores[i] * top_scores[j])
                pairs.append({
                    "allele_a": id_to_name.get(a, f"idx_{a}"),
                    "allele_b": id_to_name.get(b, f"idx_{b}"),
                    "allele_a_id": a,
                    "allele_b_id": b,
                    "score_a": float(top_scores[i]),
                    "score_b": float(top_scores[j]),
                    "pair_score": score,
                })
        pairs.sort(key=lambda p: -p["pair_score"])
        out[gene] = pairs[:topk]
    return out


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    """Run MAP genotype prediction pipeline."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 60)
    print("MAP donor genotype prediction")
    print("=" * 60)
    print(f"  Train H5:     {args.train_h5}")
    print(f"  Valid H5:     {args.valid_h5}")
    print(f"  Similarity:   {args.similarity}")
    print(f"  rare_thresh:  {args.rare_threshold}")
    print(f"  lambda_reg:   {args.lambda_reg}")
    print(f"  add_absent:   {args.add_absent}")
    tf = maybe_setup_gpu(args.gpu)
    use_gpu = tf is not None
    print(f"  Compute on:   {'GPU' if use_gpu else 'CPU'}")
    # ── sanity check: required datasets in train H5 ─────────────────
    with h5py.File(args.train_h5, "r") as f:
        required = [
            "clusters/z_probs", "clusters/q_i",
            "clusters/donors", "clusters/cdr_freq",
        ]
        for r in required:
            if r not in f:
                print(f"ERROR: train H5 missing {r}. "
                      f"Run compute_q_i.py first if q_i is missing.")
                sys.exit(1)
    with h5py.File(args.valid_h5, "r") as f:
        if "clusters/cdr_freq" not in f:
            print(f"ERROR: valid H5 missing clusters/cdr_freq.")
            sys.exit(1)
    # ── load HLA metadata ───────────────────────────────────────────
    gene_to_alleles, id_to_gene, id_to_name = build_gene_mapping(args.hla_to_id)
    print(f"\n  HLA genes found: {sorted(gene_to_alleles.keys())}")
    # ── load donor matrix & build rare mask ─────────────────────────
    donor_hla = np.load(args.donor_matrix_path)["donor_hla_matrix"]
    D, A = donor_hla.shape
    n_donors_per_allele = donor_hla.sum(axis=0)
    rare_mask = n_donors_per_allele < args.rare_threshold
    valid_mask = (~rare_mask).astype(np.float32)
    print(f"  Rare alleles masked: {int(rare_mask.sum())}/{A}")
    # Build gene group matrix (G, A)
    genes = sorted(gene_to_alleles.keys())
    G = len(genes)
    gene_group_mat = np.zeros((G, A), dtype=np.float32)
    for gi, g in enumerate(genes):
        for a in gene_to_alleles[g]:
            gene_group_mat[gi, a] = 1.0
    # ── build embeddings ────────────────────────────────────────────
    print(f"\n  Building training embedding...")
    train_emb = build_embedding_for_h5(
        args.train_h5, args.n_position_bins, args.train_chunk_size, tag="train")
    print(f"    train_emb: {train_emb.shape}")
    print(f"\n  Building valid embedding...")
    valid_emb = build_embedding_for_h5(
        args.valid_h5, args.n_position_bins, args.train_chunk_size, tag="valid")
    print(f"    valid_emb: {valid_emb.shape}")
    # ── exact hash matching ─────────────────────────────────────────
    print(f"\n  Computing exact sequence hashes...")
    # For speed, use argmax consensus bytes as hash
    from utils import PublicTcrHlaCsrReaderChunk
    with PublicTcrHlaCsrReaderChunk(
        args.train_h5, include_counts=False, include_donors=False,
        include_cdr_freq=True,
    ) as reader:
        n_train = reader.num_clusters
        train_hashes = []
        for chunk in reader.iter_cluster_chunks(chunk_rows=args.train_chunk_size):
            train_hashes.extend(build_seq_hash(chunk.cdr_freq, chunk.n_clusters))
    hash_to_train_idx = {}
    for i, h in enumerate(train_hashes):
        if h not in hash_to_train_idx:
            hash_to_train_idx[h] = i
    with PublicTcrHlaCsrReaderChunk(
        args.valid_h5, include_counts=False, include_donors=False,
        include_cdr_freq=True,
    ) as reader:
        n_valid = reader.num_clusters
        valid_hashes = []
        for chunk in reader.iter_cluster_chunks(chunk_rows=args.train_chunk_size):
            valid_hashes.extend(build_seq_hash(chunk.cdr_freq, chunk.n_clusters))
    exact_match_idx = np.full(n_valid, -1, dtype=np.int64)
    for i, h in enumerate(valid_hashes):
        if h in hash_to_train_idx:
            exact_match_idx[i] = hash_to_train_idx[h]
    n_exact = int((exact_match_idx >= 0).sum())
    print(f"  Exact matches: {n_exact:,}/{n_valid:,} "
          f"({100*n_exact/n_valid:.2f}%)")
    # ── similarity search for unmatched ─────────────────────────────
    unmatched = exact_match_idx < 0
    nn_idx = np.full(n_valid, -1, dtype=np.int64)
    nn_sim = np.zeros(n_valid, dtype=np.float32)
    nn_sim[exact_match_idx >= 0] = 1.0
    nn_idx[exact_match_idx >= 0] = exact_match_idx[exact_match_idx >= 0]
    n_unmatched = int(unmatched.sum())
    if n_unmatched > 0:
        print(f"\n  Similarity search for {n_unmatched:,} unmatched TCRs...")
        q_emb = valid_emb[unmatched]
        if use_gpu:
            bi, bs = similarity_search_gpu(
                tf, q_emb, train_emb, args.test_chunk_size)
        else:
            bi, bs = similarity_search_cpu(
                q_emb, train_emb, args.test_chunk_size)
        nn_idx[unmatched] = bi
        nn_sim[unmatched] = bs
    # Confidence weights
    confidence = compute_confidence(nn_sim, args.sim_threshold, args.alpha_conf)
    confidence[exact_match_idx >= 0] = 1.0
    n_high_conf = int((confidence > 0.5).sum())
    n_discarded = int((confidence <= 0.01).sum())
    print(f"\n  Confidence weights:")
    print(f"    Mean confidence:   {confidence.mean():.3f}")
    print(f"    > 0.5:            {n_high_conf:,}")
    print(f"    <= 0.01 (discard): {n_discarded:,}")
    # ── load training gammas + q_i (dense per mapped TCR) ──────────
    print(f"\n  Loading training gammas + q_i for matched TCRs...")
    # Get unique train indices that are actually referenced
    used_train_idx = np.unique(nn_idx[confidence > 0.01])
    used_train_idx = used_train_idx[used_train_idx >= 0]
    print(f"    Unique training TCRs used: {len(used_train_idx):,}")
    # Map from train idx -> position in used array
    train_idx_to_pos = {int(t): i for i, t in enumerate(used_train_idx)}
    # Load gammas for these TCRs (sorted reads for speed)
    gamma_used = np.zeros((len(used_train_idx), A), dtype=np.float32)
    q_used = np.zeros(len(used_train_idx), dtype=np.float32)
    with h5py.File(args.train_h5, "r") as f:
        zp_indptr = np.asarray(f["clusters/z_probs/indptr"][:])
        zp_indices = f["clusters/z_probs/indices"]
        zp_data = f["clusters/z_probs/data"]
        q_i_ds = f["clusters/q_i"]
        for i, t in enumerate(used_train_idx):
            s, e = int(zp_indptr[t]), int(zp_indptr[t + 1])
            if e > s:
                gamma_used[i, np.asarray(zp_indices[s:e])] = np.asarray(zp_data[s:e])
            q_used[i] = float(q_i_ds[t])
    # ── load valid donor membership: which donors each valid TCR is in ──
    print(f"\n  Loading valid donor membership...")
    with h5py.File(args.valid_h5, "r") as f:
        vd_indptr = np.asarray(f["clusters/donors/indptr"][:])
        vd_indices = f["clusters/donors/indices"]
        n_valid_total_entries = vd_indptr[-1]
        vd_all = np.asarray(vd_indices[:n_valid_total_entries])
    # ── build per-donor TCR lists ───────────────────────────────────
    print(f"  Building per-donor TCR lists...")
    donor_tcrs = {}  # donor_id -> list of (used_idx, confidence)
    for t_valid in range(n_valid):
        tr_idx = nn_idx[t_valid]
        conf = confidence[t_valid]
        if tr_idx < 0 or conf < 0.01:
            continue
        pos = train_idx_to_pos.get(int(tr_idx), -1)
        if pos < 0:
            continue
        lo, hi = int(vd_indptr[t_valid]), int(vd_indptr[t_valid + 1])
        for d in vd_all[lo:hi]:
            donor_tcrs.setdefault(int(d), []).append((pos, float(conf)))
    print(f"    Donors with ≥1 matched TCR: {len(donor_tcrs):,}")
    # ── MAP optimization per donor ──────────────────────────────────
    print(f"\n  Running MAP optimization per donor...")
    if tf is None:
        print("ERROR: TF required for optimization step (even on CPU).")
        sys.exit(1)
    predictions = {}
    t0 = time.time()
    for d_idx, donor_id in enumerate(sorted(donor_tcrs.keys())):
        entries = donor_tcrs[donor_id]
        tcr_positions = np.array([e[0] for e in entries], dtype=np.int64)
        tcr_confs = np.array([e[1] for e in entries], dtype=np.float32)
        # All these TCRs were observed in donor_id → y=1
        y_vec = np.ones(len(tcr_positions), dtype=np.float32)
        gamma_mat = gamma_used[tcr_positions]
        q_vec = q_used[tcr_positions]
        x_final = run_map_optimization(
            tf, gamma_mat, q_vec, tcr_confs, y_vec,
            gene_group_mat, valid_mask, args)
        topk = enumerate_topk_pairs(
            x_final, gene_to_alleles, id_to_name, args.topk)
        predictions[int(donor_id)] = {
            "n_matched_tcrs": int(len(tcr_positions)),
            "mean_confidence": float(tcr_confs.mean()),
            "x_scores": x_final.tolist(),
            "topk_pairs_per_gene": topk,
        }
        if (d_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    [{d_idx+1}/{len(donor_tcrs)}] "
                  f"{elapsed:.1f}s | {(d_idx+1)/elapsed:.1f} donors/s")
    # ── save predictions ────────────────────────────────────────────
    results = {
        "config": {
            "train_h5": str(args.train_h5),
            "valid_h5": str(args.valid_h5),
            "similarity": args.similarity,
            "sim_threshold": args.sim_threshold,
            "alpha_conf": args.alpha_conf,
            "topk": args.topk,
            "rare_threshold": args.rare_threshold,
            "lambda_reg": args.lambda_reg,
            "n_iters": args.n_iters,
            "lr": args.lr,
            "add_absent": args.add_absent,
        },
        "gene_list": genes,
        "num_alleles": A,
        "valid_allele_mask": valid_mask.tolist(),
        "rare_allele_count": int(rare_mask.sum()),
        "match_stats": {
            "n_valid_tcrs": int(n_valid),
            "n_exact_matches": int(n_exact),
            "n_high_confidence": int(n_high_conf),
            "n_discarded": int(n_discarded),
            "mean_confidence": float(confidence.mean()),
        },
        "predictions": predictions,
    }
    out_path = Path(args.output_dir) / "predictions.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    # ── always-useful diagnostic plots ──────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(nn_sim[nn_sim > 0], bins=50, color="steelblue", alpha=0.8)
    ax[0].axvline(args.sim_threshold, color="red", linestyle="--",
                  label=f"threshold={args.sim_threshold}")
    ax[0].set_xlabel("Cosine similarity to nearest training TCR")
    ax[0].set_ylabel("Count")
    ax[0].set_title("Match similarity distribution")
    ax[0].legend()
    ax[1].hist(confidence, bins=50, color="steelblue", alpha=0.8)
    ax[1].set_xlabel("Confidence weight")
    ax[1].set_ylabel("Count")
    ax[1].set_title("Confidence distribution")
    fig.tight_layout()
    fig.savefig(Path(args.output_dir) / "match_diagnostics.png", dpi=150)
    plt.close(fig)
    print(f"  Diagnostics plot saved.")
    total = time.time() - t0
    print(f"\nDone. Total prediction time: {total:.1f}s")


if __name__ == "__main__":
    main()