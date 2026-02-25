#!/usr/bin/env python3
"""
Analysis Pipeline for MLE z_probs Results.

Reads the merged H5 (with z_probs) and produces per-TCR metrics, plots,
and statistical reports. Processes data in memory-efficient chunks.

Analyses:
  1. Donor Explanation:  What fraction of each TCR's donors is explained?
  2. HLA Diversity:      Are some HLAs dominating the predictions?
  3. Entropy:            How concentrated are the z_probs per TCR?
  4. Donor Bins:         How do metrics vary across donor-count groups?

Usage:
    python analyze_zprobs.py \
        --h5_path /path/to/merged.h5 \
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
    """
    Configure logging to write to a file inside the analysis folder and to stdout.
    """
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
    """
    Set GPU memory growth before TF context is locked.
    Returns True if GPU is successfully configured, False otherwise.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            log.info(f"[HW] {len(gpus)} GPU(s) detected for TF operations: {[g.name for g in gpus]}")
            return True
        else:
            log.warning("[HW] No GPU detected by TensorFlow. Falling back to CPU.")
            return False
    except ImportError:
        log.warning("[HW] TensorFlow not installed. Falling back to CPU.")
        return False
    except RuntimeError as e:
        log.warning(f"[HW] GPU config error: {e}. Falling back to CPU.")
        return False

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments for the analysis pipeline."""
    p = argparse.ArgumentParser(
        description="Analysis pipeline for MLE z_probs results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # I/O
    p.add_argument("--h5_path", required=True, help="Merged H5 with z_probs.")
    p.add_argument("--donor_matrix_path", required=True,
                   help="Donor HLA matrix (.npz with key 'donor_hla_matrix').")
    p.add_argument("--output_dir", required=True, help="Output directory.")
    
    # Hardware
    p.add_argument("--gpu", action="store_true", 
                   help="Use TensorFlow GPU acceleration for heavy math.")
                   
    # Donor count filter
    p.add_argument("--keep_only_upperthan_n_donors", type=int, default=None,
                   help="If set, only analyse TCRs with n_donors >= this value. "
                        "Output folder becomes 'analysis_<value>'.")
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
    p.add_argument("--threshold_step", type=float, default=0.05,
                   help="Threshold step size (default: 0.05).")
    p.add_argument("--threshold_min", type=float, default=0.0,
                   help="Min threshold (default: 0.0).")
    p.add_argument("--threshold_max", type=float, default=1.0,
                   help="Max threshold inclusive (default: 1.0).")
    # Explanation levels for analysis 1 plot
    p.add_argument("--explanation_levels", type=int, nargs="+",
                   default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                   help="N%% levels for explanation curves (default: 10 20 ... 100).")
    # Donor bin edges for analysis 4
    p.add_argument("--donor_bin_edges", type=int, nargs="+",
                   default=[1, 2, 6, 11, 26, 51, 101, 251, 501],
                   help="Donor bin edges (default: 1 2 6 11 26 51 101 251 501).")
    # Performance
    p.add_argument("--chunk_size", type=int, default=200000,
                   help="Processing chunk size (default: 200000). Larger = faster, more RAM.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_analysis_folder_name(min_donors):
    """Return the analysis subfolder name, optionally suffixed with the donor filter."""
    if min_donors is not None:
        return f"analysis_{min_donors}"
    return "analysis"


def make_dirs(output_dir, analysis_folder_name):
    """Create analysis directory structure under the given folder name."""
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


def build_donor_filter_mask(h5_path, min_donors, log):
    """Build a boolean mask over all clusters selecting those with n_donors >= min_donors."""
    with h5py.File(h5_path, "r") as f:
        total = f["clusters"]["n_donors"].shape[0]
        if min_donors is None:
            return None, total
        n_donors = f["clusters"]["n_donors"][:]
    mask = n_donors >= min_donors
    n_kept = int(mask.sum())
    log.info(f"Donor filter: keeping TCRs with n_donors >= {min_donors}")
    log.info(f"  {n_kept:,} / {total:,} clusters pass filter ({100.0 * n_kept / total:.2f}%)")
    return mask, total


# ---------------------------------------------------------------------------
# Pass 1: Compute all per-TCR metrics in a single H5 scan
# ---------------------------------------------------------------------------

def compute_metrics(args, thresholds, do_expl, do_hla, do_entropy, dirs, log,
                    donor_filter_mask, use_gpu=False):
    """Single pass through the H5. Reads raw CSR arrays directly and converts
    to dense via scipy.sparse (vectorized, no Python loops)."""
    from scipy.sparse import csr_matrix
    if use_gpu:
        import tensorflow as tf
        
    # Load donor HLA matrix
    donor_hla = np.load(args.donor_matrix_path)["donor_hla_matrix"]
    num_donors_total, num_alleles = donor_hla.shape
    donor_hla_T = donor_hla.T.astype(np.float32)  # (A, D) for matmul
    
    # Push donor HLA matrix to GPU once
    donor_hla_T_tf = None
    if use_gpu:
        donor_hla_T_tf = tf.constant(donor_hla_T, dtype=tf.float32)
        
    n_thresh = len(thresholds)
    log.info(f"Donor HLA matrix: {num_donors_total} donors x {num_alleles} alleles")
    
    # Open source H5 and get dimensions
    src = h5py.File(args.h5_path, "r")
    clusters_grp = src["clusters"]
    cluster_id_ds = clusters_grp["cluster_id"]
    n_donors_ds = clusters_grp["n_donors"]
    total_clusters = cluster_id_ds.shape[0]
    
    # Determine output size (filtered or full)
    if donor_filter_mask is not None:
        output_clusters = int(donor_filter_mask.sum())
    else:
        output_clusters = total_clusters
    log.info(f"Total clusters in H5: {total_clusters:,} | Output clusters (after filter): {output_clusters:,}")
             
    # CSR datasets for z_probs
    zp_indptr = clusters_grp["z_probs"]["indptr"]
    zp_indices = clusters_grp["z_probs"]["indices"]
    zp_data = clusters_grp["z_probs"]["data"]
    
    # CSR datasets for counts (HLA diversity)
    counts_indptr = counts_indices = counts_data = None
    if do_hla:
        counts_indptr = clusters_grp["counts"]["indptr"]
        counts_indices = clusters_grp["counts"]["indices"]
        counts_data = clusters_grp["counts"]["data"]
        
    # CSR datasets for donors (explanation)
    donors_indptr = donors_indices = None
    if do_expl:
        donors_indptr = clusters_grp["donors"]["indptr"]
        donors_indices = clusters_grp["donors"]["indices"]
        
    # --- Allocate output HDF5 ---
    metrics_path = dirs["base"] / "metrics.h5"
    chunk_h5 = min(args.chunk_size, output_clusters)
    with h5py.File(metrics_path, "w") as out:
        out.attrs["source_h5"] = str(args.h5_path)
        out.attrs["total_clusters"] = output_clusters
        out.attrs["num_alleles"] = num_alleles
        if args.keep_only_upperthan_n_donors is not None:
            out.attrs["min_donors_filter"] = args.keep_only_upperthan_n_donors
        out.create_dataset("thresholds", data=thresholds.astype(np.float32))
        out.create_dataset("cluster_id", shape=(output_clusters,), dtype="int64",
                           chunks=(chunk_h5,), compression="gzip", compression_opts=4)
        out.create_dataset("n_donors", shape=(output_clusters,), dtype="int32",
                           chunks=(chunk_h5,), compression="gzip", compression_opts=4)
        
        # New: absolute count of positive HLAs per threshold for the dual-axis plot
        out.create_dataset("n_active_at_thresh", shape=(output_clusters, n_thresh), dtype="uint16",
                           chunks=(chunk_h5, n_thresh), compression="gzip", compression_opts=4)
                           
        if do_expl:
            out.create_dataset("explanation_fractions",
                               shape=(output_clusters, n_thresh), dtype="float16",
                               chunks=(chunk_h5, n_thresh),
                               compression="gzip", compression_opts=4)
            out.create_dataset("explanation_auc",
                               shape=(output_clusters,), dtype="float32",
                               chunks=(chunk_h5,), compression="gzip", compression_opts=4)
        if do_entropy:
            for name in ["entropy", "gini", "max_z_prob", "mean_z_prob_nonzero", "min_z_prob_nonzero"]:
                out.create_dataset(name, shape=(output_clusters,), dtype="float32",
                                   chunks=(chunk_h5,), compression="gzip", compression_opts=4)
            out.create_dataset("n_active_alleles", shape=(output_clusters,), dtype="uint16",
                               chunks=(chunk_h5,), compression="gzip", compression_opts=4)
                               
    # --- Global accumulators for HLA diversity & Reservoir Sampling ---
    hla_binding_counts = np.zeros((n_thresh, num_alleles), dtype=np.int64) if do_hla else None
    hla_candidate_counts = np.zeros(num_alleles, dtype=np.int64) if do_hla else None
    
    # Setup reservoir for Heatmaps
    edges = sorted(args.donor_bin_edges) + [2147483647]
    num_bins = len(edges) - 1
    sampled_zprobs = {b: [] for b in range(num_bins)}
    sampled_counts = {b: 0 for b in range(num_bins)}
    
    # --- Process chunks ---
    t0 = time.time()
    processed_input = 0
    write_cursor = 0
    out = h5py.File(metrics_path, "a")
    
    try:
        for cs in range(0, total_clusters, args.chunk_size):
            ce = min(cs + args.chunk_size, total_clusters)
            n_raw = ce - cs
            
            cids = cluster_id_ds[cs:ce]
            ndon = n_donors_ds[cs:ce]
            
            if donor_filter_mask is not None:
                chunk_mask = donor_filter_mask[cs:ce]
                n_keep = int(chunk_mask.sum())
                if n_keep == 0:
                    processed_input += n_raw
                    continue
            else:
                chunk_mask = None
                n_keep = n_raw
                
            zp_ip = zp_indptr[cs:ce + 1]
            zp_s = int(zp_ip[0])
            zp_e = int(zp_ip[-1])
            zp_ip_local = np.asarray(zp_ip) - zp_s
            z_probs_full = csr_matrix(
                (np.asarray(zp_data[zp_s:zp_e]),
                 np.asarray(zp_indices[zp_s:zp_e]),
                 zp_ip_local),
                shape=(n_raw, num_alleles),
            ).toarray().astype(np.float32)
            
            z_probs_full_tf = None
            if use_gpu and do_expl:
                z_probs_full_tf = tf.constant(z_probs_full, dtype=tf.float32)
            
            if chunk_mask is not None:
                z_probs = z_probs_full[chunk_mask]
                cids_out = cids[chunk_mask]
                ndon_out = ndon[chunk_mask]
            else:
                z_probs = z_probs_full
                cids_out = cids
                ndon_out = ndon
                
            # Reservoir Sampling per Bin for Heatmaps (Max 100 per bin)
            chunk_bin_idx = np.digitize(ndon_out, edges) - 1
            chunk_bin_idx = np.clip(chunk_bin_idx, 0, num_bins - 1)
            
            for b in np.unique(chunk_bin_idx):
                mask_b = (chunk_bin_idx == b)
                z_b = z_probs[mask_b]
                n_new = len(z_b)
                n_old = sampled_counts[b]
                
                if n_old < 100:
                    take = min(n_new, 100 - n_old)
                    sampled_zprobs[b].extend(z_b[:take])
                    n_old += take
                    remain = n_new - take
                    z_remain = z_b[take:]
                else:
                    remain = n_new
                    z_remain = z_b
                    
                if remain > 0:
                    rand_vals = np.random.rand(remain)
                    thresholds_prob = 100.0 / (n_old + np.arange(1, remain + 1))
                    replace_mask = rand_vals < thresholds_prob
                    replace_indices = np.where(replace_mask)[0]
                    for idx in replace_indices:
                        target_idx = np.random.randint(0, 100)
                        sampled_zprobs[b][target_idx] = z_remain[idx]
                        
                sampled_counts[b] += n_new
                
            ws = write_cursor
            we = write_cursor + n_keep
            out["cluster_id"][ws:we] = cids_out
            out["n_donors"][ws:we] = ndon_out
            
            # --- New: n_active_at_thresh calculation ---
            active_at_t = np.zeros((n_keep, n_thresh), dtype=np.uint16)
            if use_gpu:
                z_probs_filtered_tf = tf.constant(z_probs, dtype=tf.float32)
                for t_idx, t in enumerate(thresholds):
                    z_disc_tf = tf.cast(z_probs_filtered_tf > t, tf.uint16)
                    active_at_t[:, t_idx] = tf.reduce_sum(z_disc_tf, axis=1).numpy()
            else:
                for t_idx, t in enumerate(thresholds):
                    active_at_t[:, t_idx] = (z_probs > t).sum(axis=1).astype(np.uint16)
            out["n_active_at_thresh"][ws:we] = active_at_t
            
            # --- Analysis 1: Donor explanation ---
            if do_expl:
                dn_ip = donors_indptr[cs:ce + 1]
                dn_s = int(dn_ip[0])
                dn_e = int(dn_ip[-1])
                dn_ip_local = np.asarray(dn_ip) - dn_s
                dn_idx = np.asarray(donors_indices[dn_s:dn_e])
                
                if use_gpu:
                    fracs_full = _compute_explanation_fast_gpu(
                        z_probs_full_tf, dn_idx, dn_ip_local, ndon,
                        donor_hla_T_tf, thresholds,
                    )
                else:
                    fracs_full = _compute_explanation_fast(
                        z_probs_full, dn_idx, dn_ip_local, ndon,
                        donor_hla_T, thresholds,
                    )
                    
                if chunk_mask is not None:
                    fracs = fracs_full[chunk_mask]
                else:
                    fracs = fracs_full
                    
                auc = np.trapz(fracs, thresholds, axis=1).astype(np.float32)
                out["explanation_fractions"][ws:we] = fracs.astype(np.float16)
                out["explanation_auc"][ws:we] = auc
                
            # --- Analysis 2: HLA diversity accumulation ---
            if do_hla:
                ct_ip = counts_indptr[cs:ce + 1]
                ct_s = int(ct_ip[0])
                ct_e = int(ct_ip[-1])
                ct_ip_local = np.asarray(ct_ip) - ct_s
                counts_dense_full = csr_matrix(
                    (np.asarray(counts_data[ct_s:ct_e]),
                     np.asarray(counts_indices[ct_s:ct_e]),
                     ct_ip_local),
                    shape=(n_raw, num_alleles),
                ).toarray()
                if chunk_mask is not None:
                    counts_dense = counts_dense_full[chunk_mask]
                else:
                    counts_dense = counts_dense_full
                    
                hla_candidate_counts += (counts_dense > 0).sum(axis=0).astype(np.int64)
                
                if use_gpu:
                    for t_idx, t in enumerate(thresholds):
                        z_disc_tf = tf.cast(z_probs_filtered_tf > t, tf.int64)
                        hla_binding_counts[t_idx] += tf.reduce_sum(z_disc_tf, axis=0).numpy()
                else:
                    for t_idx, t in enumerate(thresholds):
                        hla_binding_counts[t_idx] += (z_probs > t).sum(axis=0).astype(np.int64)
                        
            # --- Analysis 3: Entropy ---
            if do_entropy:
                if use_gpu:
                    ent, gini, max_z, mean_z, min_z, n_active = _compute_entropy_chunk_gpu(z_probs_filtered_tf)
                else:
                    ent, gini, max_z, mean_z, min_z, n_active = _compute_entropy_chunk(z_probs)
                    
                out["entropy"][ws:we] = ent
                out["gini"][ws:we] = gini
                out["max_z_prob"][ws:we] = max_z
                out["mean_z_prob_nonzero"][ws:we] = mean_z
                out["min_z_prob_nonzero"][ws:we] = min_z
                out["n_active_alleles"][ws:we] = n_active
                
            write_cursor = we
            processed_input += n_raw
            elapsed = time.time() - t0
            rate = processed_input / elapsed if elapsed > 0 else 0
            log.info(f"  [{processed_input:>10,}/{total_clusters:,}] "
                     f"{100*processed_input/total_clusters:5.1f}% | "
                     f"{rate:,.0f} clusters/s | written {write_cursor:,}/{output_clusters:,}")
                     
        # After Chunk Loop: Save the Reservoir Z-Probs for Heatmaps
        grp = out.create_group("sampled_zprobs")
        for b in range(num_bins):
            if len(sampled_zprobs[b]) > 0:
                grp.create_dataset(str(b), data=np.array(sampled_zprobs[b], dtype=np.float32), compression="gzip")
                
    finally:
        out.close()
        src.close()
        
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
# Mathematical Engine Helpers (CPU & GPU Variants)
# ---------------------------------------------------------------------------

def _compute_explanation_fast(z_probs, donor_indices_flat, donor_indptr,
                              n_donors_arr, donor_hla_T, thresholds):
    """CPU fallback: Fast donor explanation using vectorized NumPy operations."""
    n = z_probs.shape[0]
    n_thresh = len(thresholds)
    fracs = np.zeros((n, n_thresh), dtype=np.float32)
    lengths = donor_indptr[1:] - donor_indptr[:-1]
    max_d = int(lengths.max()) if n > 0 else 0
    if max_d == 0:
        return fracs
        
    row_idx = np.repeat(np.arange(n), lengths)
    col_idx = np.concatenate([np.arange(l) for l in lengths]) if len(lengths) > 0 else np.array([], dtype=np.int32)
    
    donor_pad = np.zeros((n, max_d), dtype=np.int32)
    donor_mask = np.zeros((n, max_d), dtype=bool)
    donor_pad[row_idx, col_idx] = donor_indices_flat
    donor_mask[row_idx, col_idx] = True
    
    n_donors_safe = np.maximum(n_donors_arr.astype(np.float32), 1.0)
    arange_n = np.arange(n)[:, None]
    
    for t_idx, t in enumerate(thresholds):
        z_disc = (z_probs > t).astype(np.float32)
        overlap = z_disc @ donor_hla_T
        gathered = overlap[arange_n, donor_pad]
        explained = (gathered > 0) & donor_mask
        fracs[:, t_idx] = explained.sum(axis=1) / n_donors_safe
    return fracs


def _compute_explanation_fast_gpu(z_probs_tf, donor_indices_flat, donor_indptr,
                                  n_donors_arr, donor_hla_T_tf, thresholds):
    """GPU Fast Path: Fast donor explanation using TensorFlow GPU matrix operations."""
    import tensorflow as tf
    n = z_probs_tf.shape[0]
    n_thresh = len(thresholds)
    fracs = np.zeros((n, n_thresh), dtype=np.float32)
    
    lengths = donor_indptr[1:] - donor_indptr[:-1]
    max_d = int(lengths.max()) if n > 0 else 0
    if max_d == 0:
        return fracs
        
    row_idx = np.repeat(np.arange(n), lengths)
    col_idx = np.concatenate([np.arange(l) for l in lengths]) if len(lengths) > 0 else np.array([], dtype=np.int32)
    
    donor_pad = np.zeros((n, max_d), dtype=np.int32)
    donor_mask = np.zeros((n, max_d), dtype=bool)
    donor_pad[row_idx, col_idx] = donor_indices_flat
    donor_mask[row_idx, col_idx] = True
    
    n_donors_safe = np.maximum(n_donors_arr.astype(np.float32), 1.0)
    arange_n = np.arange(n)[:, None]
    
    for t_idx, t in enumerate(thresholds):
        z_disc_tf = tf.cast(z_probs_tf > t, tf.float32)
        overlap_tf = tf.matmul(z_disc_tf, donor_hla_T_tf)
        overlap = overlap_tf.numpy()
        
        gathered = overlap[arange_n, donor_pad]
        explained = (gathered > 0) & donor_mask
        fracs[:, t_idx] = explained.sum(axis=1) / n_donors_safe
        
    return fracs


def _compute_entropy_chunk(z_probs):
    """CPU fallback: Compute entropy and related metrics."""
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
    
    # Min of non-zero
    z_safe_min = np.where(z_probs > 0, z_probs, 1.0)
    min_z = z_safe_min.min(axis=1).astype(np.float32)
    min_z[n_active == 0] = 0.0
    
    return entropy, gini, max_z, mean_z, min_z, n_active


def _compute_entropy_chunk_gpu(z_probs_tf):
    """GPU Fast Path: Compute entropy and related metrics using TensorFlow."""
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
    """Line plot: fraction of TCRs with >= N% donors explained, for each threshold."""
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
            block = fracs_ds[start:end].astype(np.float32)  # (chunk, T)
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
    """Group TCRs by donor count, compare explanation AUCs, entropy, and z_probs."""
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
        if i == len(edges) - 2: bin_labels.append(f"{lo}+")
        elif lo == hi: bin_labels.append(f"{lo}")
        else: bin_labels.append(f"{lo}-{hi}")
        
    bin_idx = np.digitize(n_donors, edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_labels) - 1)
    bin_counts = np.bincount(bin_idx, minlength=len(bin_labels))
    
    nonempty = bin_counts > 0
    active_labels = [l for l, ne in zip(bin_labels, nonempty) if ne]
    active_indices = np.where(nonempty)[0]
    counts_active = bin_counts[active_indices]
    
    # --- Plot 1: Min, Mean, and Max z_probs across Bins ---
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

    # --- Plot 2: Explanation AUC curves per Bin ---
    if fracs is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, bi in enumerate(active_indices):
            mean_curve = fracs[bin_idx == bi].mean(axis=0)
            ax.plot(thresholds, mean_curve, linewidth=2, label=f"{active_labels[i]} (n={counts_active[i]:,})")
            
        ax.set_xlabel("z-prob Threshold", fontsize=12)
        ax.set_ylabel("Mean Fraction of Donors Explained", fontsize=12)
        ax.set_title("Average Donor Explanation per Donor Group", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(thresholds[0], thresholds[-1])
        ax.legend(title="Donor Bin", bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        fig.savefig(dirs["donor_bins"] / "donor_bin_explanation_curves.png", dpi=200)
        plt.close(fig)

    # --- Plot 3: Entropy Boxplot per Bin ---
    if entropy is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        box_data = [entropy[bin_idx == bi] for bi in active_indices]
        # Bypassing Matplotlib 3.9 deprecation warnings by avoiding the 'labels' argument
        bp = ax.boxplot(box_data, patch_artist=True, showfliers=False, medianprops=dict(color="red", linewidth=2))
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

    # --- Plot 4: Dual-Axis Threshold vs Fraction/Counts Grid Plot ---
    if n_active_t is not None:
        cols = 3
        rows = math.ceil(len(active_indices) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        if rows * cols == 1: axes = [axes]
        else: axes = axes.flatten()
        
        for i, bi in enumerate(active_indices):
            ax1 = axes[i]
            bin_active = n_active_t[bin_idx == bi]
            
            frac_at_least_one = (bin_active > 0).mean(axis=0)
            mean_hla_count = bin_active.mean(axis=0)
            
            ax1.plot(thresholds, frac_at_least_one, color='k', linewidth=2, label="Frac TCRs ≥1 HLA")
            ax1.set_xlabel("z-prob Threshold")
            ax1.set_ylabel("Fraction of TCRs (≥1 HLA)", color='k')
            ax1.tick_params(axis='y', labelcolor='k')
            ax1.set_title(f"Donor Bin: {active_labels[i]}")
            ax1.grid(True, alpha=0.2)
            
            ax2 = ax1.twinx()
            ax2.plot(thresholds, mean_hla_count, color='r', linewidth=2, label="Mean HLA Count")
            ax2.set_ylabel("Mean absolute HLA count", color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
        for j in range(len(active_indices), len(axes)):
            axes[j].axis('off')
            
        fig.tight_layout()
        fig.savefig(dirs["donor_bins"] / "donor_bin_dual_axis_thresholds.png", dpi=200)
        plt.close(fig)

    # --- Plot 5: Heatmap of Sampled z_probs per Bin ---
    with h5py.File(metrics_path, "r") as f:
        if "sampled_zprobs" in f:
            sampled_grp = f["sampled_zprobs"]
            cols = 3
            rows = math.ceil(len(active_indices) / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
            if rows * cols == 1: axes = [axes]
            else: axes = axes.flatten()
            
            im = None
            for i, bi in enumerate(active_indices):
                ax = axes[i]
                if str(bi) in sampled_grp:
                    z_sample = sampled_grp[str(bi)][:]
                    # Sort TCRs by max z-prob for a much cleaner visualization
                    sort_idx = np.argsort(z_sample.max(axis=1))[::-1]
                    z_sample = z_sample[sort_idx]
                    
                    im = ax.imshow(z_sample, aspect='auto', cmap='magma', vmin=0, vmax=1, interpolation='none')
                    ax.set_title(f"Bin: {active_labels[i]} (n={len(z_sample)})", fontsize=13)
                    ax.set_xlabel("HLAs")
                    if i % cols == 0:
                        ax.set_ylabel("TCRs")
                    ax.set_xticks([])
                else:
                    ax.axis('off')
                    
            for j in range(len(active_indices), len(axes)):
                axes[j].axis('off')
                
            fig.tight_layout()
            if im is not None:
                cbar = fig.colorbar(im, ax=axes, shrink=0.6, location='right', pad=0.02)
                cbar.set_label("z-prob", fontsize=12)
                
            fig.savefig(dirs["donor_bins"] / "donor_bin_zprob_heatmaps.png", dpi=200)
            plt.close(fig)

    # --- Save general stats CSV ---
    import csv
    csv_path = dirs["donor_bins"] / "donor_bin_stats.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["bin", "n_tcrs"]
        if expl_auc is not None: header += ["expl_auc_median", "expl_auc_mean", "expl_auc_std"]
        if entropy is not None: header += ["entropy_median", "entropy_mean", "entropy_std"]
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
# Additional statistics
# ---------------------------------------------------------------------------

def save_summary(args, thresholds, global_stats, expl_stats, hla_stats,
                 entropy_stats, bin_stats, dirs, log):
    """Save overall summary JSON with all analysis results and metadata."""
    summary = {
        "source_h5": str(args.h5_path),
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
    path = dirs["additional"] / "summary_statistics.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"  Summary saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Entry point: parse args, build filter mask, run selected analyses."""
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
    log.info("MLE z-probs Analysis Pipeline")
    log.info("=" * 60)
    log.info(f"  H5 path:     {args.h5_path}")
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
    
    donor_filter_mask, _ = build_donor_filter_mask(
        args.h5_path, args.keep_only_upperthan_n_donors, log
    )
    
    t0 = time.time()
    global_stats = compute_metrics(
        args, thresholds, do_expl, do_hla, do_entropy, dirs, log,
        donor_filter_mask, use_gpu=use_gpu,
    )
    metrics_path = global_stats["metrics_path"]
    
    expl_stats = hla_stats = entropy_stats = bin_stats = None
    if do_expl:
        expl_stats = plot_donor_explanation(
            metrics_path, thresholds, args.explanation_levels, dirs, log
        )
    if do_hla:
        hla_stats = plot_hla_diversity(global_stats, thresholds, dirs, log)
    if do_entropy:
        entropy_stats = plot_entropy(metrics_path, dirs, log)
    if do_bins:
        bin_stats = analyze_donor_bins(
            metrics_path, thresholds, args.explanation_levels,
            args.donor_bin_edges, dirs, log,
        )
        
    save_summary(args, thresholds, global_stats, expl_stats, hla_stats,
                 entropy_stats, bin_stats, dirs, log)
                 
    total = time.time() - t0
    log.info(f"\nAll analyses complete in {total:.1f}s ({total/60:.1f}min)")
    log.info(f"Results in: {dirs['base']}")

if __name__ == "__main__":
    main()