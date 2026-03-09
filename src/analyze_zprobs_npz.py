#!/usr/bin/env python3
"""
Analysis Pipeline for TFRecord-based MLE z_probs Results (NPZ input).
=====================================================================
Mirrors analyze_zprobs.py but reads from the NPZ file produced by
run_inference_tfrecord() instead of an H5 with sparse CSR arrays.
NPZ keys expected:
  z_logits      (N, A) float32
  z_probs       (N, A) float32
  binder_dense  (N, A) float32   — binary co-occurrence mask
  donor_indices (N, D) int32     — padded with pad_token
  n_donors      (N,)   int32
Analyses:
  1. Donor Explanation:  What fraction of each TCR's donors is explained?
  2. HLA Diversity:      Are some HLAs dominating the predictions?
  3. Entropy:            How concentrated are the z_probs per TCR?
  4. Donor Bins:         How do metrics vary across donor-count groups?
Usage:
    python analyze_zprobs_npz.py \
        --npz_path /path/to/predictions.npz \
        --donor_matrix_path /path/to/donor_hla_matrix.npz \
        --output_dir /path/to/output \
        --keep_only_upperthan_n_donors 5 \
        --gpu \
        --all
"""
import os
import sys
import json
import time
import math
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import logging
# ---------------------------------------------------------------------------
# Logging & Hardware
# ---------------------------------------------------------------------------
def setup_logging(output_dir, analysis_folder_name):
    """Configure logging to file and stdout inside the analysis folder."""
    log_dir = Path(output_dir) / analysis_folder_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "analysis.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)
def _early_gpu_config(log):
    """Set GPU memory growth before TF context is locked."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            log.info(f"[HW] {len(gpus)} GPU(s) detected: {[g.name for g in gpus]}")
            return True
        else:
            log.warning("[HW] No GPU detected. Falling back to CPU.")
            return False
    except (ImportError, RuntimeError) as e:
        log.warning(f"[HW] GPU config error: {e}. Falling back to CPU.")
        return False
# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments for the NPZ analysis pipeline."""
    p = argparse.ArgumentParser(
        description="Analysis pipeline for TFRecord-based MLE z_probs (NPZ).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # I/O
    p.add_argument("--npz_path", required=True,
                   help="NPZ from run_inference_tfrecord (z_probs, binder_dense, etc).")
    p.add_argument("--donor_matrix_path", required=True,
                   help="Donor HLA matrix (.npz with key 'donor_hla_matrix').")
    p.add_argument("--output_dir", required=True, help="Output directory.")
    # Hardware
    p.add_argument("--gpu", action="store_true",
                   help="Use TensorFlow GPU acceleration for heavy math.")
    # Donor count filter
    p.add_argument("--keep_only_upperthan_n_donors", type=int, default=None,
                   help="Only analyse TCRs with n_donors >= this value.")
    # Pad token (must match training)
    p.add_argument("--pad_token", type=int, default=-1,
                   help="Pad token used in donor_indices (default: -1).")
    # Analysis selection
    p.add_argument("--all", action="store_true", help="Run all analyses.")
    p.add_argument("--donor_explanation", action="store_true",
                   help="Analysis 1: Donor explanation fractions.")
    p.add_argument("--hla_diversity", action="store_true",
                   help="Analysis 2: HLA diversity / dominance.")
    p.add_argument("--entropy", action="store_true",
                   help="Analysis 3: Entropy and distribution metrics.")
    p.add_argument("--donor_bins", action="store_true",
                   help="Analysis 4: Metrics grouped by donor count.")
    # Thresholds
    p.add_argument("--threshold_step", type=float, default=0.05)
    p.add_argument("--threshold_min", type=float, default=0.0)
    p.add_argument("--threshold_max", type=float, default=1.0)
    # Explanation levels
    p.add_argument("--explanation_levels", type=int, nargs="+",
                   default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # Donor bin edges
    p.add_argument("--donor_bin_edges", type=int, nargs="+",
                   default=[1, 2, 6, 11, 26, 51, 101, 251, 501])
    # Performance
    p.add_argument("--chunk_size", type=int, default=200000,
                   help="Processing chunk size (default: 200000).")
    return p.parse_args()
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_analysis_folder_name(min_donors):
    """Return analysis subfolder name, optionally suffixed with donor filter."""
    if min_donors is not None:
        return f"analysis_npz_{min_donors}"
    return "analysis_npz"
def make_dirs(output_dir, analysis_folder_name):
    """Create analysis directory structure."""
    base = Path(output_dir) / analysis_folder_name
    dirs = {
        "base": base,
        "donor_explanation": base / "donor_explanation",
        "hla_diversity": base / "hla_diversity",
        "entropy": base / "entropy",
        "donor_bins": base / "donor_bins",
        "additional": base / "additional",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
def build_thresholds(args):
    """Build array of z-prob thresholds from CLI arguments."""
    return np.arange(args.threshold_min, args.threshold_max + 1e-9, args.threshold_step)
def load_npz_data(npz_path, log):
    """Load and validate all arrays from the inference NPZ file.
    Returns dict with keys: z_logits, z_probs, binder_dense, donor_indices, n_donors.
    """
    log.info(f"Loading NPZ: {npz_path}")
    data = np.load(npz_path)
    required = ["z_probs", "binder_dense", "donor_indices", "n_donors"]
    for key in required:
        if key not in data:
            log.error(f"NPZ missing required key '{key}'. Found: {list(data.keys())}")
            sys.exit(1)
    result = {
        "z_logits": data["z_logits"] if "z_logits" in data else None,
        "z_probs": data["z_probs"],
        "binder_dense": data["binder_dense"],
        "donor_indices": data["donor_indices"],
        "n_donors": data["n_donors"],
    }
    N, A = result["z_probs"].shape
    log.info(f"  Loaded {N:,} clusters x {A} alleles")
    log.info(f"  donor_indices shape: {result['donor_indices'].shape}")
    if result["z_logits"] is not None:
        log.info(f"  z_logits available (enables logit-level analysis)")
    return result
# ---------------------------------------------------------------------------
# Pass 1: Compute all per-TCR metrics in chunked passes over NPZ arrays
# ---------------------------------------------------------------------------
def compute_metrics(args, npz_data, thresholds, do_expl, do_hla, do_entropy,
                    dirs, log, use_gpu=False):
    """Single chunked pass over in-memory NPZ arrays. Produces metrics.h5
    identical in structure to the H5 version (minus cluster_id)."""
    if use_gpu:
        import tensorflow as tf
    # ── load donor HLA matrix ────────────────────────────────────────
    donor_hla = np.load(args.donor_matrix_path)["donor_hla_matrix"]
    num_donors_total, num_alleles = donor_hla.shape
    donor_hla_T = donor_hla.T.astype(np.float32)  # (A, D)
    donor_hla_T_tf = tf.constant(donor_hla_T, dtype=tf.float32) if use_gpu else None
    n_thresh = len(thresholds)
    log.info(f"Donor HLA matrix: {num_donors_total} donors x {num_alleles} alleles")
    # ── extract arrays from NPZ data ────────────────────────────────
    z_probs_all = npz_data["z_probs"]
    binder_all = npz_data["binder_dense"]
    donor_idx_all = npz_data["donor_indices"]
    n_donors_all = npz_data["n_donors"]
    total_clusters = z_probs_all.shape[0]
    # ── apply donor filter ───────────────────────────────────────────
    if args.keep_only_upperthan_n_donors is not None:
        min_d = args.keep_only_upperthan_n_donors
        keep_mask = n_donors_all >= min_d
        n_kept = int(keep_mask.sum())
        log.info(f"Donor filter: keeping n_donors >= {min_d}")
        log.info(f"  {n_kept:,} / {total_clusters:,} pass "
                 f"({100.0 * n_kept / total_clusters:.2f}%)")
        z_probs_all = z_probs_all[keep_mask]
        binder_all = binder_all[keep_mask]
        donor_idx_all = donor_idx_all[keep_mask]
        n_donors_all = n_donors_all[keep_mask]
    output_clusters = z_probs_all.shape[0]
    log.info(f"Output clusters: {output_clusters:,}")
    # ── allocate output HDF5 ─────────────────────────────────────────
    metrics_path = dirs["base"] / "metrics.h5"
    chunk_h5 = min(args.chunk_size, output_clusters)
    with h5py.File(metrics_path, "w") as out:
        out.attrs["source_npz"] = str(args.npz_path)
        out.attrs["total_clusters"] = output_clusters
        out.attrs["num_alleles"] = num_alleles
        if args.keep_only_upperthan_n_donors is not None:
            out.attrs["min_donors_filter"] = args.keep_only_upperthan_n_donors
        out.create_dataset("thresholds", data=thresholds.astype(np.float32))
        out.create_dataset("n_donors", shape=(output_clusters,), dtype="int32",
                           chunks=(chunk_h5,), compression="gzip", compression_opts=4)
        out.create_dataset("n_active_at_thresh",
                           shape=(output_clusters, n_thresh), dtype="uint16",
                           chunks=(chunk_h5, n_thresh), compression="gzip", compression_opts=4)
        if do_expl:
            out.create_dataset("explanation_fractions",
                               shape=(output_clusters, n_thresh), dtype="float16",
                               chunks=(chunk_h5, n_thresh), compression="gzip", compression_opts=4)
            out.create_dataset("explanation_auc",
                               shape=(output_clusters,), dtype="float32",
                               chunks=(chunk_h5,), compression="gzip", compression_opts=4)
        if do_entropy:
            for name in ["entropy", "gini", "max_z_prob", "mean_z_prob_nonzero", "min_z_prob_nonzero"]:
                out.create_dataset(name, shape=(output_clusters,), dtype="float32",
                                   chunks=(chunk_h5,), compression="gzip", compression_opts=4)
            out.create_dataset("n_active_alleles", shape=(output_clusters,), dtype="uint16",
                               chunks=(chunk_h5,), compression="gzip", compression_opts=4)
    # ── global accumulators ──────────────────────────────────────────
    hla_binding_counts = np.zeros((n_thresh, num_alleles), dtype=np.int64) if do_hla else None
    hla_candidate_counts = np.zeros(num_alleles, dtype=np.int64) if do_hla else None
    # ── reservoir sampling for heatmaps ──────────────────────────────
    edges = sorted(args.donor_bin_edges) + [2147483647]
    num_bins = len(edges) - 1
    sampled_zprobs = {b: [] for b in range(num_bins)}
    sampled_counts = {b: 0 for b in range(num_bins)}
    RESERVOIR_SIZE = 100
    # ── chunked processing ───────────────────────────────────────────
    t0 = time.time()
    out = h5py.File(metrics_path, "a")
    pad_token = args.pad_token
    try:
        for cs in range(0, output_clusters, args.chunk_size):
            ce = min(cs + args.chunk_size, output_clusters)
            n_chunk = ce - cs
            # ── slice chunk arrays ───────────────────────────────────
            z_probs = z_probs_all[cs:ce]
            binder = binder_all[cs:ce]
            donors = donor_idx_all[cs:ce]
            ndon = n_donors_all[cs:ce]
            out["n_donors"][cs:ce] = ndon
            # ── GPU tensors for this chunk ───────────────────────────
            z_probs_tf = None
            if use_gpu:
                z_probs_tf = tf.constant(z_probs, dtype=tf.float32)
            # ── n_active_at_thresh ───────────────────────────────────
            active_at_t = np.zeros((n_chunk, n_thresh), dtype=np.uint16)
            if use_gpu:
                for t_idx, t in enumerate(thresholds):
                    z_disc_tf = tf.cast(z_probs_tf > t, tf.uint16)
                    active_at_t[:, t_idx] = tf.reduce_sum(z_disc_tf, axis=1).numpy()
            else:
                for t_idx, t in enumerate(thresholds):
                    active_at_t[:, t_idx] = (z_probs > t).sum(axis=1).astype(np.uint16)
            out["n_active_at_thresh"][cs:ce] = active_at_t
            # ── analysis 1: donor explanation ────────────────────────
            if do_expl:
                # Build ragged-style indptr from padded donor_indices
                valid_mask = donors != pad_token
                lengths = valid_mask.sum(axis=1).astype(np.int64)
                max_d = int(lengths.max()) if n_chunk > 0 else 0
                n_donors_safe = np.maximum(ndon.astype(np.float32), 1.0)
                fracs = np.zeros((n_chunk, n_thresh), dtype=np.float32)
                if max_d > 0:
                    # Build dense donor pad + mask (already padded from NPZ)
                    arange_n = np.arange(n_chunk)[:, None]
                    if use_gpu:
                        for t_idx, t in enumerate(thresholds):
                            z_disc_tf = tf.cast(z_probs_tf > t, tf.float32)
                            overlap_tf = tf.matmul(z_disc_tf, donor_hla_T_tf)
                            overlap = overlap_tf.numpy()
                            # Gather overlap at donor positions
                            safe_donors = np.maximum(donors, 0)
                            gathered = overlap[arange_n, safe_donors]
                            explained = (gathered > 0) & valid_mask
                            fracs[:, t_idx] = explained.sum(axis=1) / n_donors_safe
                    else:
                        for t_idx, t in enumerate(thresholds):
                            z_disc = (z_probs > t).astype(np.float32)
                            overlap = z_disc @ donor_hla_T
                            safe_donors = np.maximum(donors, 0)
                            gathered = overlap[arange_n, safe_donors]
                            explained = (gathered > 0) & valid_mask
                            fracs[:, t_idx] = explained.sum(axis=1) / n_donors_safe
                auc = np.trapz(fracs, thresholds, axis=1).astype(np.float32)
                out["explanation_fractions"][cs:ce] = fracs.astype(np.float16)
                out["explanation_auc"][cs:ce] = auc
            # ── analysis 2: HLA diversity accumulation ───────────────
            if do_hla:
                # binder_dense serves as candidate mask (same as counts > 0)
                hla_candidate_counts += (binder > 0.5).sum(axis=0).astype(np.int64)
                if use_gpu:
                    for t_idx, t in enumerate(thresholds):
                        z_disc_tf = tf.cast(z_probs_tf > t, tf.int64)
                        hla_binding_counts[t_idx] += tf.reduce_sum(z_disc_tf, axis=0).numpy()
                else:
                    for t_idx, t in enumerate(thresholds):
                        hla_binding_counts[t_idx] += (z_probs > t).sum(axis=0).astype(np.int64)
            # ── analysis 3: entropy ──────────────────────────────────
            if do_entropy:
                if use_gpu:
                    ent, gini, max_z, mean_z, min_z, n_active = _compute_entropy_chunk_gpu(z_probs_tf)
                else:
                    ent, gini, max_z, mean_z, min_z, n_active = _compute_entropy_chunk(z_probs)
                out["entropy"][cs:ce] = ent
                out["gini"][cs:ce] = gini
                out["max_z_prob"][cs:ce] = max_z
                out["mean_z_prob_nonzero"][cs:ce] = mean_z
                out["min_z_prob_nonzero"][cs:ce] = min_z
                out["n_active_alleles"][cs:ce] = n_active
            # ── reservoir sampling for heatmaps ──────────────────────
            chunk_bin_idx = np.digitize(ndon, edges) - 1
            chunk_bin_idx = np.clip(chunk_bin_idx, 0, num_bins - 1)
            for b in np.unique(chunk_bin_idx):
                mask_b = (chunk_bin_idx == b)
                z_b = z_probs[mask_b]
                n_new = len(z_b)
                n_old = sampled_counts[b]
                if n_old < RESERVOIR_SIZE:
                    take = min(n_new, RESERVOIR_SIZE - n_old)
                    sampled_zprobs[b].extend(z_b[:take])
                    n_old += take
                    remain = n_new - take
                    z_remain = z_b[take:]
                else:
                    remain = n_new
                    z_remain = z_b
                if remain > 0:
                    rand_vals = np.random.rand(remain)
                    thresh_prob = float(RESERVOIR_SIZE) / (n_old + np.arange(1, remain + 1))
                    replace_mask = rand_vals < thresh_prob
                    for idx in np.where(replace_mask)[0]:
                        target = np.random.randint(0, RESERVOIR_SIZE)
                        sampled_zprobs[b][target] = z_remain[idx]
                sampled_counts[b] += n_new
            # ── progress ─────────────────────────────────────────────
            elapsed = time.time() - t0
            rate = ce / elapsed if elapsed > 0 else 0
            log.info(f"  [{ce:>10,}/{output_clusters:,}] "
                     f"{100*ce/output_clusters:5.1f}% | {rate:,.0f} clusters/s")
        # ── save reservoir samples ───────────────────────────────────
        grp = out.create_group("sampled_zprobs")
        for b in range(num_bins):
            if len(sampled_zprobs[b]) > 0:
                grp.create_dataset(str(b),
                                   data=np.array(sampled_zprobs[b], dtype=np.float32),
                                   compression="gzip")
    finally:
        out.close()
    total_time = time.time() - t0
    log.info(f"Metrics pass complete: {total_time:.1f}s ({total_time/60:.1f}min)")
    return {
        "metrics_path": str(metrics_path),
        "total_clusters": output_clusters,
        "num_alleles": num_alleles,
        "num_donors_total": num_donors_total,
        "hla_binding_counts": hla_binding_counts,
        "hla_candidate_counts": hla_candidate_counts,
        "donor_hla_abundance": donor_hla.sum(axis=0),
        "elapsed_seconds": round(total_time, 2),
    }
# ---------------------------------------------------------------------------
# Mathematical Engine Helpers (CPU & GPU)
# ---------------------------------------------------------------------------
def _compute_entropy_chunk(z_probs):
    """CPU: Compute entropy, gini, max/mean/min z-prob, n_active."""
    active = z_probs > 0
    n_active = active.sum(axis=1).astype(np.uint16)
    z_sum = z_probs.sum(axis=1, keepdims=True)
    z_sum_safe = np.maximum(z_sum, 1e-10)
    z_norm = z_probs / z_sum_safe
    log_z = np.log2(np.maximum(z_norm, 1e-10))
    entropy = -(z_norm * log_z).sum(axis=1).astype(np.float32)
    entropy[z_sum.ravel() == 0] = 0.0
    A = z_probs.shape[1]
    z_sorted = np.sort(z_norm, axis=1)
    index = np.arange(1, A + 1).astype(np.float32)
    numer = 2.0 * (z_sorted * index[None, :]).sum(axis=1) - (A + 1) * z_sorted.sum(axis=1)
    denom = A * np.maximum(z_sorted.sum(axis=1), 1e-10)
    gini = (numer / denom).astype(np.float32)
    gini[z_sum.ravel() == 0] = 0.0
    max_z = z_probs.max(axis=1).astype(np.float32)
    n_active_safe = np.maximum(n_active.astype(np.float32), 1.0)
    mean_z = (z_probs.sum(axis=1) / n_active_safe).astype(np.float32)
    z_safe_min = np.where(z_probs > 0, z_probs, 1.0)
    min_z = z_safe_min.min(axis=1).astype(np.float32)
    min_z[n_active == 0] = 0.0
    return entropy, gini, max_z, mean_z, min_z, n_active
def _compute_entropy_chunk_gpu(z_probs_tf):
    """GPU: Compute entropy, gini, max/mean/min z-prob, n_active."""
    import tensorflow as tf
    active = z_probs_tf > 0
    n_active = tf.reduce_sum(tf.cast(active, tf.int32), axis=1)
    z_sum = tf.reduce_sum(z_probs_tf, axis=1, keepdims=True)
    z_sum_safe = tf.maximum(z_sum, 1e-10)
    z_norm = z_probs_tf / z_sum_safe
    log2_z = tf.math.log(tf.maximum(z_norm, 1e-10)) / tf.math.log(2.0)
    entropy = -tf.reduce_sum(z_norm * log2_z, axis=1)
    is_zero = tf.squeeze(z_sum, axis=[-1]) == 0
    entropy = tf.where(is_zero, 0.0, entropy)
    A = float(z_probs_tf.shape[1])
    z_sorted = tf.sort(z_norm, axis=1)
    index = tf.range(1, A + 1, dtype=tf.float32)
    numer = 2.0 * tf.reduce_sum(z_sorted * index[tf.newaxis, :], axis=1) - (A + 1) * tf.reduce_sum(z_sorted, axis=1)
    denom = A * tf.maximum(tf.reduce_sum(z_sorted, axis=1), 1e-10)
    gini = numer / denom
    gini = tf.where(is_zero, 0.0, gini)
    max_z = tf.reduce_max(z_probs_tf, axis=1)
    n_active_safe = tf.maximum(tf.cast(n_active, tf.float32), 1.0)
    mean_z = tf.reduce_sum(z_probs_tf, axis=1) / n_active_safe
    z_safe_min = tf.where(z_probs_tf > 0, z_probs_tf, 1.0)
    min_z = tf.reduce_min(z_safe_min, axis=1)
    min_z = tf.where(n_active == 0, 0.0, min_z)
    return (entropy.numpy(), gini.numpy(), max_z.numpy(),
            mean_z.numpy(), min_z.numpy(), n_active.numpy().astype(np.uint16))
# ---------------------------------------------------------------------------
# Analysis 1: Donor Explanation Plots
# ---------------------------------------------------------------------------
def plot_donor_explanation(metrics_path, thresholds, levels, dirs, log):
    """Line plot: fraction of TCRs with >= N% donors explained per threshold."""
    log.info("Generating donor explanation plots...")
    n_thresh = len(thresholds)
    n_levels = len(levels)
    with h5py.File(metrics_path, "r") as f:
        total = f.attrs["total_clusters"]
        fracs_ds = f["explanation_fractions"]
        counts = np.zeros((n_thresh, n_levels), dtype=np.int64)
        chunk = 100_000
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            block = fracs_ds[start:end].astype(np.float32)
            for l_idx, level in enumerate(levels):
                threshold_val = level / 100.0
                counts[:, l_idx] += (block >= threshold_val).sum(axis=0)
    curves = counts / total
    aucs = np.trapz(curves, thresholds, axis=0)
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, n_levels))
    for l_idx, level in enumerate(levels):
        ax.plot(thresholds, curves[:, l_idx], color=cmap[l_idx], linewidth=2,
                label=f"≥{level}% explained (AUC={aucs[l_idx]:.3f})")
    ax.set_xlabel("z-prob Threshold", fontsize=13)
    ax.set_ylabel("Fraction of TCRs", fontsize=13)
    ax.set_title("Donor Explanation Across Thresholds", fontsize=15)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(dirs["donor_explanation"] / "explanation_auc_curves.png", dpi=200)
    plt.close(fig)
    import csv
    csv_path = dirs["donor_explanation"] / "explanation_auc_values.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["threshold"] + [f"frac_geq_{l}pct" for l in levels]
        writer.writerow(header)
        for t_idx, t in enumerate(thresholds):
            row = [f"{t:.3f}"] + [f"{curves[t_idx, l_idx]:.6f}" for l_idx in range(n_levels)]
            writer.writerow(row)
        writer.writerow(["AUC"] + [f"{a:.6f}" for a in aucs])
    log.info(f"  Explanation plot saved: {dirs['donor_explanation']}")
    return {"curves": curves, "aucs": dict(zip(levels, aucs.tolist()))}
# ---------------------------------------------------------------------------
# Analysis 2: HLA Diversity Plots
# ---------------------------------------------------------------------------
def plot_hla_diversity(global_stats, thresholds, dirs, log):
    """HLA diversity: which alleles dominate predictions?"""
    log.info("Generating HLA diversity plots...")
    hla_binding = global_stats["hla_binding_counts"]
    hla_candidates = global_stats["hla_candidate_counts"]
    hla_abundance = global_stats["donor_hla_abundance"]
    total_tcrs = global_stats["total_clusters"]
    num_donors = global_stats["num_donors_total"]
    abundance_frac = hla_abundance / num_donors
    abundance_safe = np.maximum(abundance_frac, 1e-10)
    raw_frac = hla_binding / total_tcrs
    enrichment = raw_frac / abundance_safe[None, :]
    candidates_safe = np.maximum(hla_candidates, 1).astype(np.float64)
    cond_frac = hla_binding / candidates_safe[None, :]
    data_path = dirs["hla_diversity"] / "hla_diversity_data.h5"
    with h5py.File(data_path, "w") as f:
        f["thresholds"] = thresholds.astype(np.float32)
        f["enrichment"] = enrichment.astype(np.float32)
        f["conditional_binding_fraction"] = cond_frac.astype(np.float32)
        f["hla_abundance"] = hla_abundance
        f["hla_candidate_counts"] = hla_candidates
        f["hla_binding_counts"] = hla_binding
        f.attrs["total_tcrs"] = total_tcrs
        f.attrs["num_donors"] = num_donors
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    valid = hla_abundance > 0
    enrich_valid = enrichment[:, valid]
    p25, p50, p75 = np.percentile(enrich_valid, [25, 50, 75], axis=1)
    p5, p95 = np.percentile(enrich_valid, [5, 95], axis=1)
    ax.fill_between(thresholds, p5, p95, alpha=0.15, color="C0", label="5-95th pctl")
    ax.fill_between(thresholds, p25, p75, alpha=0.3, color="C0", label="25-75th pctl")
    ax.plot(thresholds, p50, color="C0", linewidth=2, label="Median")
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Expected (=1)")
    ax.set_xlabel("z-prob Threshold", fontsize=12)
    ax.set_ylabel("Enrichment (obs/expected)", fontsize=12)
    ax.set_title("HLA Binding Enrichment", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(thresholds[0], thresholds[-1])
    ax = axes[1]
    valid_cand = hla_candidates > 10
    cond_valid = cond_frac[:, valid_cand]
    p25, p50, p75 = np.percentile(cond_valid, [25, 50, 75], axis=1)
    p5, p95 = np.percentile(cond_valid, [5, 95], axis=1)
    ax.fill_between(thresholds, p5, p95, alpha=0.15, color="C1", label="5-95th pctl")
    ax.fill_between(thresholds, p25, p75, alpha=0.3, color="C1", label="25-75th pctl")
    ax.plot(thresholds, p50, color="C1", linewidth=2, label="Median")
    ax.set_xlabel("z-prob Threshold", fontsize=12)
    ax.set_ylabel("Binding Fraction (among candidates)", fontsize=12)
    ax.set_title("Conditional HLA Binding", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(thresholds[0], thresholds[-1])
    fig.tight_layout()
    fig.savefig(dirs["hla_diversity"] / "hla_diversity_summary.png", dpi=200)
    plt.close(fig)
    t_idx_05 = np.argmin(np.abs(thresholds - 0.5))
    enrich_at_05 = enrichment[t_idx_05, :]
    valid_idx = np.where(hla_abundance > 0)[0]
    sorted_idx = valid_idx[np.argsort(enrich_at_05[valid_idx])]
    fig, ax = plt.subplots(figsize=(12, 6))
    n_show = min(20, len(sorted_idx) // 2)
    show_idx = np.concatenate([sorted_idx[:n_show], sorted_idx[-n_show:]])
    colors = ["#d62728" if enrich_at_05[i] > 1.5 else
              "#2ca02c" if enrich_at_05[i] < 0.5 else "#1f77b4" for i in show_idx]
    ax.barh(range(len(show_idx)), enrich_at_05[show_idx], color=colors, height=0.7)
    ax.set_yticks(range(len(show_idx)))
    ax.set_yticklabels([f"HLA_{i}" for i in show_idx], fontsize=8)
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Enrichment at threshold=0.5", fontsize=12)
    ax.set_title("Most Over/Under-represented HLAs", fontsize=13)
    fig.tight_layout()
    fig.savefig(dirs["hla_diversity"] / "hla_top_bottom_enrichment.png", dpi=200)
    plt.close(fig)
    cv_per_thresh = np.std(enrich_valid, axis=1) / np.maximum(np.mean(enrich_valid, axis=1), 1e-10)
    log.info(f"  HLA diversity plots saved: {dirs['hla_diversity']}")
    return {"enrichment_cv": cv_per_thresh.tolist()}
# ---------------------------------------------------------------------------
# Analysis 3: Entropy Plots
# ---------------------------------------------------------------------------
def plot_entropy(metrics_path, dirs, log):
    """Distribution plots for entropy, gini, max/mean z-prob, and donor counts."""
    log.info("Generating entropy plots...")
    with h5py.File(metrics_path, "r") as f:
        entropy = f["entropy"][:]
        gini = f["gini"][:]
        max_z = f["max_z_prob"][:]
        mean_z = f["mean_z_prob_nonzero"][:]
        n_active = f["n_active_alleles"][:]
        n_donors = f["n_donors"][:]
    metrics = {
        "Shannon Entropy (bits)": entropy,
        "Gini Index": gini,
        "Max z-prob": max_z,
        "Mean z-prob (nonzero)": mean_z,
        "Num Active Alleles (z>0.01)": n_active.astype(np.float32),
    }
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for idx, (name, vals) in enumerate(metrics.items()):
        ax = axes[idx]
        valid = vals[vals > 0] if "Entropy" in name or "Gini" in name else vals
        ax.hist(valid, bins=100, color="C0", alpha=0.7, edgecolor="none")
        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Distribution of {name}", fontsize=12)
        med, mn = np.median(valid), np.mean(valid)
        ax.axvline(med, color="red", linestyle="--", alpha=0.7, label=f"Median={med:.3f}")
        ax.axvline(mn, color="orange", linestyle="--", alpha=0.7, label=f"Mean={mn:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    ax = axes[5]
    log_donors = np.log10(np.maximum(n_donors, 1).astype(np.float64))
    ax.hist(log_donors, bins=100, color="C2", alpha=0.7, edgecolor="none")
    ax.set_xlabel("log10(n_donors)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Donor Counts", fontsize=12)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(dirs["entropy"] / "metric_distributions.png", dpi=200)
    plt.close(fig)
    stats = {}
    for name, vals in metrics.items():
        valid = vals[vals > 0] if "Entropy" in name else vals
        stats[name] = {
            "mean": float(np.mean(valid)),
            "median": float(np.median(valid)),
            "std": float(np.std(valid)),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "q25": float(np.percentile(valid, 25)),
            "q75": float(np.percentile(valid, 75)),
        }
    log.info(f"  Entropy plots saved: {dirs['entropy']}")
    return stats
# ---------------------------------------------------------------------------
# Analysis 4: Donor Bin Analysis
# ---------------------------------------------------------------------------
def analyze_donor_bins(metrics_path, thresholds, levels, bin_edges, dirs, log):
    """Group TCRs by donor count, compare explanation AUCs, entropy, z_probs."""
    log.info("Generating donor bin analysis...")
    with h5py.File(metrics_path, "r") as f:
        n_donors = f["n_donors"][:]
        expl_auc = f["explanation_auc"][:] if "explanation_auc" in f else None
        fracs = f["explanation_fractions"][:] if "explanation_fractions" in f else None
        entropy = f["entropy"][:] if "entropy" in f else None
        max_z = f["max_z_prob"][:] if "max_z_prob" in f else None
        mean_z = f["mean_z_prob_nonzero"][:] if "mean_z_prob_nonzero" in f else None
        min_z = f["min_z_prob_nonzero"][:] if "min_z_prob_nonzero" in f else None
        n_active_t = f["n_active_at_thresh"][:] if "n_active_at_thresh" in f else None
    edges = sorted(bin_edges) + [2147483647]
    bin_labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1] - 1
        if i == len(edges) - 2:
            bin_labels.append(f"{lo}+")
        elif lo == hi:
            bin_labels.append(f"{lo}")
        else:
            bin_labels.append(f"{lo}-{hi}")
    bin_idx = np.digitize(n_donors, edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_labels) - 1)
    bin_counts = np.bincount(bin_idx, minlength=len(bin_labels))
    nonempty = bin_counts > 0
    active_labels = [l for l, ne in zip(bin_labels, nonempty) if ne]
    active_indices = np.where(nonempty)[0]
    counts_active = bin_counts[active_indices]
    # ── Plot 1: Min, Mean, Max z_probs across bins ───────────────────
    if max_z is not None and mean_z is not None and min_z is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        means_of_max = [max_z[bin_idx == bi].mean() for bi in active_indices]
        means_of_mean = [mean_z[bin_idx == bi].mean() for bi in active_indices]
        means_of_min = [min_z[bin_idx == bi].mean() for bi in active_indices]
        ax.plot(active_labels, means_of_max, marker='o', linestyle='-', label='Max z-prob')
        ax.plot(active_labels, means_of_mean, marker='s', linestyle='-', label='Mean z-prob (nonzero)')
        ax.plot(active_labels, means_of_min, marker='^', linestyle='-', label='Min z-prob (nonzero)')
        ax.set_xlabel("Donor Count Group", fontsize=12)
        ax.set_ylabel("Average Probability", fontsize=12)
        ax.set_title("Average Min, Mean, and Max z-probs per Donor Group", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(dirs["donor_bins"] / "donor_bin_zprobs_summary.png", dpi=200)
        plt.close(fig)
    # ── Plot 2: Explanation AUC curves per bin ───────────────────────
    if fracs is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, bi in enumerate(active_indices):
            mean_curve = fracs[bin_idx == bi].mean(axis=0)
            ax.plot(thresholds, mean_curve, linewidth=2,
                    label=f"{active_labels[i]} (n={counts_active[i]:,})")
        ax.set_xlabel("z-prob Threshold", fontsize=12)
        ax.set_ylabel("Mean Fraction of Donors Explained", fontsize=12)
        ax.set_title("Average Donor Explanation per Donor Group", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(thresholds[0], thresholds[-1])
        ax.legend(title="Donor Bin", bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        fig.savefig(dirs["donor_bins"] / "donor_bin_explanation_curves.png", dpi=200)
        plt.close(fig)
    # ── Plot 3: Entropy boxplot per bin ──────────────────────────────
    if entropy is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        box_data = [entropy[bin_idx == bi] for bi in active_indices]
        bp = ax.boxplot(box_data, patch_artist=True, showfliers=False,
                        medianprops=dict(color="red", linewidth=2))
        ax.set_xticks(range(1, len(active_labels) + 1))
        ax.set_xticklabels(active_labels)
        for patch in bp["boxes"]:
            patch.set_facecolor("C2")
            patch.set_alpha(0.5)
        ax.set_xlabel("Donor Count Group", fontsize=12)
        ax.set_ylabel("Entropy (bits)", fontsize=12)
        ax.set_title("z-prob Entropy Distribution per Donor Group", fontsize=14)
        ax.grid(True, alpha=0.2, axis="y")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(dirs["donor_bins"] / "donor_bin_entropy_boxplot.png", dpi=200)
        plt.close(fig)
    # ── Plot 4: Dual-axis threshold vs fraction/counts grid ──────────
    if n_active_t is not None:
        cols = 3
        rows = math.ceil(len(active_indices) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        if rows * cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        for i, bi in enumerate(active_indices):
            ax1 = axes[i]
            bin_active = n_active_t[bin_idx == bi]
            frac_at_least_one = (bin_active > 0).mean(axis=0)
            mean_hla_count = bin_active.mean(axis=0)
            ax1.plot(thresholds, frac_at_least_one, color='k', linewidth=2,
                     label="Frac TCRs ≥1 HLA")
            ax1.set_xlabel("z-prob Threshold")
            ax1.set_ylabel("Fraction of TCRs (≥1 HLA)", color='k')
            ax1.tick_params(axis='y', labelcolor='k')
            ax1.set_title(f"Donor Bin: {active_labels[i]}")
            ax1.grid(True, alpha=0.2)
            ax2 = ax1.twinx()
            ax2.plot(thresholds, mean_hla_count, color='r', linewidth=2,
                     label="Mean HLA Count")
            ax2.set_ylabel("Mean absolute HLA count", color='r')
            ax2.tick_params(axis='y', labelcolor='r')
        for j in range(len(active_indices), len(axes)):
            axes[j].axis('off')
        fig.tight_layout()
        fig.savefig(dirs["donor_bins"] / "donor_bin_dual_axis_thresholds.png", dpi=200)
        plt.close(fig)
    # ── Plot 5: Heatmap of sampled z_probs per bin ───────────────────
    with h5py.File(metrics_path, "r") as f:
        if "sampled_zprobs" in f:
            sampled_grp = f["sampled_zprobs"]
            cols = 3
            rows = math.ceil(len(active_indices) / cols)
            fig, axes_hm = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
            if rows * cols == 1:
                axes_hm = [axes_hm]
            else:
                axes_hm = axes_hm.flatten()
            im = None
            for i, bi in enumerate(active_indices):
                ax = axes_hm[i]
                if str(bi) in sampled_grp:
                    z_sample = sampled_grp[str(bi)][:]
                    sort_idx = np.argsort(z_sample.max(axis=1))[::-1]
                    z_sample = z_sample[sort_idx]
                    im = ax.imshow(z_sample, aspect='auto', cmap='magma',
                                   vmin=0, vmax=1, interpolation='none')
                    ax.set_title(f"Bin: {active_labels[i]} (n={len(z_sample)})", fontsize=13)
                    ax.set_xlabel("HLAs")
                    if i % cols == 0:
                        ax.set_ylabel("TCRs")
                    ax.set_xticks([])
                else:
                    ax.axis('off')
            for j in range(len(active_indices), len(axes_hm)):
                axes_hm[j].axis('off')
            fig.tight_layout()
            if im is not None:
                cbar = fig.colorbar(im, ax=axes_hm, shrink=0.6,
                                    location='right', pad=0.02)
                cbar.set_label("z-prob", fontsize=12)
            fig.savefig(dirs["donor_bins"] / "donor_bin_zprob_heatmaps.png", dpi=200)
            plt.close(fig)
    # ── Save stats CSV ───────────────────────────────────────────────
    import csv
    csv_path = dirs["donor_bins"] / "donor_bin_stats.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["bin", "n_tcrs"]
        if expl_auc is not None:
            header += ["expl_auc_median", "expl_auc_mean", "expl_auc_std"]
        if entropy is not None:
            header += ["entropy_median", "entropy_mean", "entropy_std"]
        w.writerow(header)
        for i, bi in enumerate(active_indices):
            mask = bin_idx == bi
            row = [active_labels[i], int(mask.sum())]
            if expl_auc is not None:
                v = expl_auc[mask]
                row += [f"{np.median(v):.4f}", f"{np.mean(v):.4f}", f"{np.std(v):.4f}"]
            if entropy is not None:
                v = entropy[mask]
                row += [f"{np.median(v):.4f}", f"{np.mean(v):.4f}", f"{np.std(v):.4f}"]
            w.writerow(row)
    log.info(f"  Donor bin analysis saved: {dirs['donor_bins']}")
    return {"bin_labels": active_labels, "bin_counts": counts_active.tolist()}
# ---------------------------------------------------------------------------
# Additional: Logit Distribution (NPZ exclusive — not available in H5)
# ---------------------------------------------------------------------------
def plot_logit_distributions(npz_data, dirs, log, min_donors=None):
    """Plot logit distributions — only possible with NPZ (H5 stores sigmoid only)."""
    z_logits = npz_data.get("z_logits")
    if z_logits is None:
        log.info("  Skipping logit plots (z_logits not in NPZ)")
        return None
    log.info("Generating logit distribution plots (NPZ exclusive)...")
    binder = npz_data["binder_dense"]
    n_donors = npz_data["n_donors"]
    # Apply filter
    if min_donors is not None:
        mask = n_donors >= min_donors
        z_logits = z_logits[mask]
        binder = binder[mask]
    active_mask = binder > 0.5
    inactive_mask = ~active_mask
    logits_active = z_logits[active_mask].flatten()
    logits_inactive = z_logits[inactive_mask].flatten()
    # Subsample for plotting if too many points
    max_pts = 5_000_000
    if len(logits_active) > max_pts:
        logits_active = np.random.choice(logits_active, max_pts, replace=False)
    if len(logits_inactive) > max_pts:
        logits_inactive = np.random.choice(logits_inactive, max_pts, replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Panel 1: Overlapping histograms
    ax = axes[0]
    bins = np.linspace(min(logits_inactive.min(), logits_active.min()),
                       max(logits_inactive.max(), logits_active.max()), 200)
    ax.hist(logits_inactive, bins=bins, alpha=0.5, density=True,
            color="C0", label=f"Inactive (n={len(logits_inactive):,})")
    ax.hist(logits_active, bins=bins, alpha=0.5, density=True,
            color="C3", label=f"Active (n={len(logits_active):,})")
    ax.set_xlabel("Logit value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Logit Distributions: Active vs Inactive", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    # Panel 2: Per-TCR mean logit (active vs inactive)
    ax = axes[1]
    mean_active_logit = np.where(active_mask.sum(axis=1) > 0,
                                  (z_logits * active_mask).sum(axis=1) / np.maximum(active_mask.sum(axis=1), 1),
                                  0.0)
    mean_inactive_logit = np.where(inactive_mask.sum(axis=1) > 0,
                                    (z_logits * inactive_mask).sum(axis=1) / np.maximum(inactive_mask.sum(axis=1), 1),
                                    0.0)
    ax.hist(mean_active_logit, bins=100, alpha=0.5, density=True,
            color="C3", label="Mean active logit")
    ax.hist(mean_inactive_logit, bins=100, alpha=0.5, density=True,
            color="C0", label="Mean inactive logit")
    ax.set_xlabel("Mean logit per TCR", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Per-TCR Mean Logits", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    # Panel 3: Logit gap (max active - max inactive)
    ax = axes[2]
    max_active = np.where(active_mask.any(axis=1),
                           np.where(active_mask, z_logits, -1e9).max(axis=1),
                           0.0)
    max_inactive = np.where(inactive_mask.any(axis=1),
                             np.where(inactive_mask, z_logits, -1e9).max(axis=1),
                             0.0)
    gap = max_active - max_inactive
    ax.hist(gap, bins=100, color="C2", alpha=0.7, edgecolor="none")
    med_gap = np.median(gap)
    ax.axvline(med_gap, color="red", linestyle="--", label=f"Median={med_gap:.2f}")
    ax.set_xlabel("Max Active Logit − Max Inactive Logit", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Logit Separation Gap", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(dirs["additional"] / "logit_distributions.png", dpi=200)
    plt.close(fig)
    stats = {
        "active_logit_mean": float(logits_active.mean()),
        "active_logit_std": float(logits_active.std()),
        "inactive_logit_mean": float(logits_inactive.mean()),
        "inactive_logit_std": float(logits_inactive.std()),
        "logit_gap_median": float(med_gap),
    }
    log.info(f"  Logit plots saved: {dirs['additional']}")
    return stats
# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def save_summary(args, thresholds, global_stats, expl_stats, hla_stats,
                 entropy_stats, bin_stats, logit_stats, dirs, log):
    """Save overall summary JSON with all analysis results and metadata."""
    summary = {
        "source_npz": str(args.npz_path),
        "donor_matrix_path": str(args.donor_matrix_path),
        "total_clusters": global_stats["total_clusters"],
        "num_alleles": global_stats["num_alleles"],
        "num_donors": global_stats["num_donors_total"],
        "min_donors_filter": args.keep_only_upperthan_n_donors,
        "thresholds": thresholds.tolist(),
        "compute_time_seconds": global_stats["elapsed_seconds"],
    }
    if expl_stats:
        summary["donor_explanation_aucs"] = expl_stats.get("aucs", {})
    if hla_stats:
        summary["hla_enrichment_cv_per_threshold"] = hla_stats.get("enrichment_cv", [])
    if entropy_stats:
        summary["entropy_summary"] = entropy_stats
    if bin_stats:
        summary["donor_bin_counts"] = dict(zip(
            bin_stats.get("bin_labels", []),
            bin_stats.get("bin_counts", []),
        ))
    if logit_stats:
        summary["logit_distributions"] = logit_stats
    path = dirs["additional"] / "summary_statistics.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"  Summary saved: {path}")
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Entry point: parse args, load NPZ, run selected analyses."""
    args = parse_args()
    do_expl = args.all or args.donor_explanation
    do_hla = args.all or args.hla_diversity
    do_entropy = args.all or args.entropy
    do_bins = args.all or args.donor_bins
    if not any([do_expl, do_hla, do_entropy, do_bins]):
        print("No analysis selected. Use --all or individual flags.")
        print("  --donor_explanation  --hla_diversity  --entropy  --donor_bins")
        sys.exit(1)
    if do_bins:
        do_expl = True
        do_entropy = True
    analysis_folder = get_analysis_folder_name(args.keep_only_upperthan_n_donors)
    dirs = make_dirs(args.output_dir, analysis_folder)
    log = setup_logging(args.output_dir, analysis_folder)
    use_gpu = False
    if args.gpu:
        use_gpu = _early_gpu_config(log)
    thresholds = build_thresholds(args)
    log.info("=" * 60)
    log.info("MLE z-probs Analysis Pipeline (NPZ Input)")
    log.info("=" * 60)
    log.info(f"  NPZ path:    {args.npz_path}")
    log.info(f"  Output dir:  {args.output_dir}")
    log.info(f"  GPU support: {'Enabled' if use_gpu else 'Disabled'}")
    log.info(f"  Analysis folder: {analysis_folder}")
    if args.keep_only_upperthan_n_donors is not None:
        log.info(f"  Donor filter: n_donors >= {args.keep_only_upperthan_n_donors}")
    else:
        log.info(f"  Donor filter: None (all TCRs)")
    log.info(f"  Thresholds:  {thresholds[0]:.2f} to {thresholds[-1]:.2f} "
             f"(step={args.threshold_step}, n={len(thresholds)})")
    log.info(f"  Chunk size:  {args.chunk_size:,}")
    log.info(f"  Analyses:    expl={do_expl} hla={do_hla} "
             f"entropy={do_entropy} bins={do_bins}")
    log.info("=" * 60)
    # ── load NPZ into memory ─────────────────────────────────────────
    npz_data = load_npz_data(args.npz_path, log)
    t0 = time.time()
    global_stats = compute_metrics(
        args, npz_data, thresholds, do_expl, do_hla, do_entropy,
        dirs, log, use_gpu=use_gpu,
    )
    metrics_path = global_stats["metrics_path"]
    expl_stats = hla_stats = entropy_stats = bin_stats = logit_stats = None
    if do_expl:
        expl_stats = plot_donor_explanation(
            metrics_path, thresholds, args.explanation_levels, dirs, log)
    if do_hla:
        hla_stats = plot_hla_diversity(global_stats, thresholds, dirs, log)
    if do_entropy:
        entropy_stats = plot_entropy(metrics_path, dirs, log)
    if do_bins:
        bin_stats = analyze_donor_bins(
            metrics_path, thresholds, args.explanation_levels,
            args.donor_bin_edges, dirs, log)
    # ── NPZ exclusive: logit distribution analysis ───────────────────
    logit_stats = plot_logit_distributions(
        npz_data, dirs, log,
        min_donors=args.keep_only_upperthan_n_donors)
    save_summary(args, thresholds, global_stats, expl_stats, hla_stats,
                 entropy_stats, bin_stats, logit_stats, dirs, log)
    total = time.time() - t0
    log.info(f"\nAll analyses complete in {total:.1f}s ({total/60:.1f}min)")
    log.info(f"Results in: {dirs['base']}")
if __name__ == "__main__":
    main()