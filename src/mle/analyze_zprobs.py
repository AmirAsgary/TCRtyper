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
    python analyze_zprobs.py \\
        --h5_path /path/to/merged.h5 \\
        --donor_matrix_path /path/to/donor_hla_matrix.npz \\
        --output_dir /path/to/output \\
        --all
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir):
    log_dir = Path(output_dir) / "analysis"
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Analysis pipeline for MLE z_probs results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # I/O
    p.add_argument("--h5_path", required=True, help="Merged H5 with z_probs.")
    p.add_argument("--donor_matrix_path", required=True,
                   help="Donor HLA matrix (.npz with key 'donor_hla_matrix').")
    p.add_argument("--output_dir", required=True, help="Output directory.")

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


def make_dirs(output_dir):
    """Create analysis directory structure."""
    base = Path(output_dir) / "analysis"
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
    return np.arange(args.threshold_min, args.threshold_max + 1e-9, args.threshold_step)


# ---------------------------------------------------------------------------
# Pass 1: Compute all per-TCR metrics in a single H5 scan
# ---------------------------------------------------------------------------

def compute_metrics(args, thresholds, do_expl, do_hla, do_entropy, dirs, log):
    """
    Single pass through the H5. Reads raw CSR arrays directly and converts
    to dense via scipy.sparse (vectorized, no Python loops).
    """
    from scipy.sparse import csr_matrix

    donor_hla = np.load(args.donor_matrix_path)["donor_hla_matrix"]
    num_donors_total, num_alleles = donor_hla.shape
    donor_hla_T = donor_hla.T.astype(np.float32)  # (A, D) for matmul
    n_thresh = len(thresholds)

    log.info(f"Donor HLA matrix: {num_donors_total} donors x {num_alleles} alleles")

    # Open source H5 and get dimensions
    src = h5py.File(args.h5_path, "r")
    clusters_grp = src["clusters"]
    cluster_id_ds = clusters_grp["cluster_id"]
    n_donors_ds = clusters_grp["n_donors"]
    total_clusters = cluster_id_ds.shape[0]
    log.info(f"Total clusters: {total_clusters:,}")

    # CSR datasets
    zp_indptr = clusters_grp["z_probs"]["indptr"]
    zp_indices = clusters_grp["z_probs"]["indices"]
    zp_data = clusters_grp["z_probs"]["data"]

    counts_indptr = counts_indices = counts_data = None
    if do_hla:
        counts_indptr = clusters_grp["counts"]["indptr"]
        counts_indices = clusters_grp["counts"]["indices"]
        counts_data = clusters_grp["counts"]["data"]

    donors_indptr = donors_indices = None
    if do_expl:
        donors_indptr = clusters_grp["donors"]["indptr"]
        donors_indices = clusters_grp["donors"]["indices"]

    # --- Allocate output HDF5 ---
    metrics_path = dirs["base"] / "metrics.h5"
    chunk_h5 = min(args.chunk_size, total_clusters)
    with h5py.File(metrics_path, "w") as out:
        out.attrs["source_h5"] = str(args.h5_path)
        out.attrs["total_clusters"] = total_clusters
        out.attrs["num_alleles"] = num_alleles
        out.create_dataset("thresholds", data=thresholds.astype(np.float32))
        out.create_dataset("cluster_id", shape=(total_clusters,), dtype="int64",
                           chunks=(chunk_h5,), compression="gzip", compression_opts=4)
        out.create_dataset("n_donors", shape=(total_clusters,), dtype="int32",
                           chunks=(chunk_h5,), compression="gzip", compression_opts=4)
        if do_expl:
            out.create_dataset("explanation_fractions",
                               shape=(total_clusters, n_thresh), dtype="float16",
                               chunks=(chunk_h5, n_thresh),
                               compression="gzip", compression_opts=4)
            out.create_dataset("explanation_auc",
                               shape=(total_clusters,), dtype="float32",
                               chunks=(chunk_h5,), compression="gzip", compression_opts=4)
        if do_entropy:
            for name in ["entropy", "gini", "max_z_prob", "mean_z_prob_nonzero"]:
                out.create_dataset(name, shape=(total_clusters,), dtype="float32",
                                   chunks=(chunk_h5,), compression="gzip", compression_opts=4)
            out.create_dataset("n_active_alleles", shape=(total_clusters,), dtype="uint16",
                               chunks=(chunk_h5,), compression="gzip", compression_opts=4)

    # --- Global accumulators for HLA diversity ---
    hla_binding_counts = np.zeros((n_thresh, num_alleles), dtype=np.int64) if do_hla else None
    hla_candidate_counts = np.zeros(num_alleles, dtype=np.int64) if do_hla else None

    # --- Process chunks ---
    t0 = time.time()
    processed = 0
    out = h5py.File(metrics_path, "a")

    try:
        for cs in range(0, total_clusters, args.chunk_size):
            ce = min(cs + args.chunk_size, total_clusters)
            n = ce - cs

            # Identifiers (direct H5 slice — fast)
            cids = cluster_id_ds[cs:ce]
            ndon = n_donors_ds[cs:ce]
            out["cluster_id"][cs:ce] = cids
            out["n_donors"][cs:ce] = ndon

            # z_probs: CSR -> dense via scipy (vectorized, no Python loop)
            zp_ip = zp_indptr[cs:ce + 1]
            zp_s = int(zp_ip[0])
            zp_e = int(zp_ip[-1])
            zp_ip_local = np.asarray(zp_ip) - zp_s
            z_probs = csr_matrix(
                (np.asarray(zp_data[zp_s:zp_e]),
                 np.asarray(zp_indices[zp_s:zp_e]),
                 zp_ip_local),
                shape=(n, num_alleles),
            ).toarray().astype(np.float32)

            # --- Analysis 1: Donor explanation ---
            if do_expl:
                dn_ip = donors_indptr[cs:ce + 1]
                dn_s = int(dn_ip[0])
                dn_e = int(dn_ip[-1])
                dn_ip_local = np.asarray(dn_ip) - dn_s
                dn_idx = np.asarray(donors_indices[dn_s:dn_e])

                fracs = _compute_explanation_fast(
                    z_probs, dn_idx, dn_ip_local, ndon,
                    donor_hla_T, thresholds,
                )
                auc = np.trapz(fracs, thresholds, axis=1).astype(np.float32)
                out["explanation_fractions"][cs:ce] = fracs.astype(np.float16)
                out["explanation_auc"][cs:ce] = auc

            # --- Analysis 2: HLA diversity accumulation ---
            if do_hla:
                ct_ip = counts_indptr[cs:ce + 1]
                ct_s = int(ct_ip[0])
                ct_e = int(ct_ip[-1])
                ct_ip_local = np.asarray(ct_ip) - ct_s
                counts_dense = csr_matrix(
                    (np.asarray(counts_data[ct_s:ct_e]),
                     np.asarray(counts_indices[ct_s:ct_e]),
                     ct_ip_local),
                    shape=(n, num_alleles),
                ).toarray()
                hla_candidate_counts += (counts_dense > 0).sum(axis=0).astype(np.int64)
                for t_idx, t in enumerate(thresholds):
                    hla_binding_counts[t_idx] += (z_probs > t).sum(axis=0).astype(np.int64)

            # --- Analysis 3: Entropy ---
            if do_entropy:
                ent, gini, max_z, mean_z, n_active = _compute_entropy_chunk(z_probs)
                out["entropy"][cs:ce] = ent
                out["gini"][cs:ce] = gini
                out["max_z_prob"][cs:ce] = max_z
                out["mean_z_prob_nonzero"][cs:ce] = mean_z
                out["n_active_alleles"][cs:ce] = n_active

            processed += n
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            log.info(f"  [{processed:>10,}/{total_clusters:,}] "
                     f"{100*processed/total_clusters:5.1f}% | "
                     f"{rate:,.0f} clusters/s | chunk [{cs}, {ce})")
    finally:
        out.close()
        src.close()

    total_time = time.time() - t0
    log.info(f"Metrics pass complete: {total_time:.1f}s ({total_time/60:.1f}min)")

    return {
        "metrics_path": str(metrics_path),
        "total_clusters": total_clusters,
        "num_alleles": num_alleles,
        "num_donors_total": num_donors_total,
        "hla_binding_counts": hla_binding_counts,
        "hla_candidate_counts": hla_candidate_counts,
        "donor_hla_abundance": donor_hla.sum(axis=0),
        "elapsed_seconds": round(total_time, 2),
    }


def _compute_explanation_fast(z_probs, donor_indices_flat, donor_indptr,
                              n_donors_arr, donor_hla_T, thresholds):
    """
    Fast donor explanation using vectorized operations.
    """
    n = z_probs.shape[0]
    n_thresh = len(thresholds)
    fracs = np.zeros((n, n_thresh), dtype=np.float32)

    lengths = donor_indptr[1:] - donor_indptr[:-1]
    max_d = int(lengths.max()) if n > 0 else 0
    if max_d == 0:
        return fracs

    # Vectorized padding: build row indices, column indices for scatter
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
        overlap = z_disc @ donor_hla_T                        # (n, D)
        gathered = overlap[arange_n, donor_pad]               # (n, max_d)
        explained = (gathered > 0) & donor_mask
        fracs[:, t_idx] = explained.sum(axis=1) / n_donors_safe

    return fracs



def _compute_entropy_chunk(z_probs):
    """
    Compute entropy and related metrics for a chunk.
    Returns: entropy, gini, max_z, mean_z_nonzero, n_active (all 1D arrays).
    """
    # Mask of active alleles (z > 0)
    active = z_probs > 0
    n_active = active.sum(axis=1).astype(np.uint16)

    # Normalize to probability distribution per TCR
    z_sum = z_probs.sum(axis=1, keepdims=True)
    z_sum_safe = np.maximum(z_sum, 1e-10)
    z_norm = z_probs / z_sum_safe

    # Shannon entropy (base 2)
    log_z = np.log2(np.maximum(z_norm, 1e-10))
    entropy = -(z_norm * log_z).sum(axis=1).astype(np.float32)
    entropy[z_sum.ravel() == 0] = 0.0

    # Gini index (on normalized distribution)
    A = z_probs.shape[1]
    z_sorted = np.sort(z_norm, axis=1)
    index = np.arange(1, A + 1).astype(np.float32)
    numer = 2.0 * (z_sorted * index[None, :]).sum(axis=1) - (A + 1) * z_sorted.sum(axis=1)
    denom = A * np.maximum(z_sorted.sum(axis=1), 1e-10)
    gini = (numer / denom).astype(np.float32)
    gini[z_sum.ravel() == 0] = 0.0

    # Max and mean of nonzero z_probs
    max_z = z_probs.max(axis=1).astype(np.float32)
    n_active_safe = np.maximum(n_active.astype(np.float32), 1.0)
    mean_z = (z_probs.sum(axis=1) / n_active_safe).astype(np.float32)

    return entropy, gini, max_z, mean_z, n_active


# ---------------------------------------------------------------------------
# Analysis 1: Donor Explanation Plots
# ---------------------------------------------------------------------------

def plot_donor_explanation(metrics_path, thresholds, levels, dirs, log):
    """
    Line plot: fraction of TCRs with >= N% donors explained, for each threshold.
    """
    log.info("Generating donor explanation plots...")
    n_thresh = len(thresholds)
    n_levels = len(levels)

    # We need to count, for each (threshold, level), how many TCRs qualify.
    # Process the metrics.h5 in chunks to avoid loading 38M x 20 into RAM.
    with h5py.File(metrics_path, "r") as f:
        total = f.attrs["total_clusters"]
        fracs_ds = f["explanation_fractions"]
        # counts[t, l] = number of TCRs with explanation_fraction >= level/100 at threshold t
        counts = np.zeros((n_thresh, n_levels), dtype=np.int64)

        chunk = 100_000
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            block = fracs_ds[start:end].astype(np.float32)  # (chunk, T)
            for l_idx, level in enumerate(levels):
                threshold_val = level / 100.0
                counts[:, l_idx] += (block >= threshold_val).sum(axis=0)

    # Convert to fractions
    curves = counts / total  # (n_thresh, n_levels)

    # Compute AUC for each level
    aucs = np.trapz(curves, thresholds, axis=0)

    # --- Plot ---
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

    # Save CSV
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
    log.info(f"  AUCs: {dict(zip(levels, [round(a,4) for a in aucs]))}")

    return {"curves": curves, "aucs": dict(zip(levels, aucs.tolist()))}


# ---------------------------------------------------------------------------
# Analysis 2: HLA Diversity Plots
# ---------------------------------------------------------------------------

def plot_hla_diversity(global_stats, thresholds, dirs, log):
    """
    HLA diversity: which alleles dominate predictions?
    """
    log.info("Generating HLA diversity plots...")
    hla_binding = global_stats["hla_binding_counts"]  # (T, A)
    hla_candidates = global_stats["hla_candidate_counts"]  # (A,)
    hla_abundance = global_stats["donor_hla_abundance"]  # (A,)
    total_tcrs = global_stats["total_clusters"]
    num_donors = global_stats["num_donors_total"]
    num_alleles = hla_binding.shape[1]
    n_thresh = len(thresholds)

    # Metric 1: Binding fraction normalized by abundance
    # raw_frac[t, a] = hla_binding[t, a] / total_tcrs
    # expected[a] = hla_abundance[a] / num_donors
    # enrichment[t, a] = raw_frac[t, a] / expected[a]
    abundance_frac = hla_abundance / num_donors  # (A,)
    abundance_safe = np.maximum(abundance_frac, 1e-10)
    raw_frac = hla_binding / total_tcrs  # (T, A)
    enrichment = raw_frac / abundance_safe[None, :]  # (T, A)

    # Metric 2: Conditional binding fraction (among candidates)
    # cond_frac[t, a] = hla_binding[t, a] / hla_candidates[a]
    candidates_safe = np.maximum(hla_candidates, 1).astype(np.float64)
    cond_frac = hla_binding / candidates_safe[None, :]  # (T, A)

    # --- Save data ---
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

    # --- Plot 1: Enrichment summary (median + percentile bands) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: enrichment over thresholds
    ax = axes[0]
    # Only use alleles with nonzero abundance
    valid = hla_abundance > 0
    enrich_valid = enrichment[:, valid]
    p25 = np.percentile(enrich_valid, 25, axis=1)
    p50 = np.percentile(enrich_valid, 50, axis=1)
    p75 = np.percentile(enrich_valid, 75, axis=1)
    p5 = np.percentile(enrich_valid, 5, axis=1)
    p95 = np.percentile(enrich_valid, 95, axis=1)

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

    # Right: conditional binding fraction
    ax = axes[1]
    valid_cand = hla_candidates > 10  # only HLAs with enough candidates
    cond_valid = cond_frac[:, valid_cand]
    p25 = np.percentile(cond_valid, 25, axis=1)
    p50 = np.percentile(cond_valid, 50, axis=1)
    p75 = np.percentile(cond_valid, 75, axis=1)
    p5 = np.percentile(cond_valid, 5, axis=1)
    p95 = np.percentile(cond_valid, 95, axis=1)

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

    # --- Plot 2: Top/bottom HLAs at threshold=0.5 ---
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

    # Coefficient of variation across HLAs
    cv_per_thresh = np.std(enrich_valid, axis=1) / np.maximum(np.mean(enrich_valid, axis=1), 1e-10)
    log.info(f"  Enrichment CV at t=0.5: {cv_per_thresh[t_idx_05]:.3f}")
    log.info(f"  HLA diversity plots saved: {dirs['hla_diversity']}")

    return {"enrichment_cv": cv_per_thresh.tolist()}


# ---------------------------------------------------------------------------
# Analysis 3: Entropy Plots
# ---------------------------------------------------------------------------

def plot_entropy(metrics_path, dirs, log):
    """Distribution plots for entropy and related metrics."""
    log.info("Generating entropy plots...")

    with h5py.File(metrics_path, "r") as f:
        total = f.attrs["total_clusters"]
        # Read all at once — these are (N,) float32, ~150MB each. Acceptable.
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
        # Add stats
        med = np.median(valid)
        mn = np.mean(valid)
        ax.axvline(med, color="red", linestyle="--", alpha=0.7, label=f"Median={med:.3f}")
        ax.axvline(mn, color="orange", linestyle="--", alpha=0.7, label=f"Mean={mn:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    # Last panel: n_donors distribution
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

    # Summary stats
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
    """Group TCRs by donor count, compare explanation AUCs and entropy."""
    log.info("Generating donor bin analysis...")

    with h5py.File(metrics_path, "r") as f:
        n_donors = f["n_donors"][:]
        expl_auc = f["explanation_auc"][:] if "explanation_auc" in f else None
        entropy = f["entropy"][:] if "entropy" in f else None
        gini = f["gini"][:] if "gini" in f else None
        max_z = f["max_z_prob"][:] if "max_z_prob" in f else None

    # Build bin labels
    edges = sorted(bin_edges) + [int(n_donors.max()) + 1]
    bin_labels = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1] - 1
        if i == len(edges) - 2:
            bin_labels.append(f"{lo}+")
        elif lo == hi:
            bin_labels.append(f"{lo}")
        else:
            bin_labels.append(f"{lo}-{hi}")

    bin_idx = np.digitize(n_donors, edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_labels) - 1)

    # Count per bin
    bin_counts = np.bincount(bin_idx, minlength=len(bin_labels))

    # Filter out empty bins
    nonempty = bin_counts > 0
    active_labels = [l for l, ne in zip(bin_labels, nonempty) if ne]
    active_indices = np.where(nonempty)[0]

    # --- Plot A: Donor bin distribution ---
    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = np.arange(len(active_labels))
    counts_active = bin_counts[active_indices]
    ax.bar(x_pos, counts_active, color="C0", alpha=0.7, edgecolor="none")
    for i, c in enumerate(counts_active):
        ax.text(i, c, f"{c:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(active_labels, rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("Donor Count Group", fontsize=12)
    ax.set_ylabel("Number of TCRs", fontsize=12)
    ax.set_title("TCR Distribution by Donor Count", fontsize=14)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(dirs["donor_bins"] / "donor_bin_distribution.png", dpi=200)
    plt.close(fig)

    # --- Plot B: Explanation AUC by donor bin ---
    if expl_auc is not None:
        fig, ax = plt.subplots(figsize=(12, 6))
        box_data = [expl_auc[bin_idx == bi] for bi in active_indices]
        bp = ax.boxplot(box_data, labels=active_labels, patch_artist=True,
                        showfliers=False, medianprops=dict(color="red", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor("C0")
            patch.set_alpha(0.5)
        ax.set_xlabel("Donor Count Group", fontsize=12)
        ax.set_ylabel("Explanation AUC", fontsize=12)
        ax.set_title("Donor Explanation AUC by Donor Group", fontsize=14)
        ax.grid(True, alpha=0.2, axis="y")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(dirs["donor_bins"] / "donor_bin_explanation_auc.png", dpi=200)
        plt.close(fig)

    # --- Plot C: Entropy by donor bin ---
    if entropy is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        ax = axes[0]
        box_data = [entropy[bin_idx == bi] for bi in active_indices]
        bp = ax.boxplot(box_data, labels=active_labels, patch_artist=True,
                        showfliers=False, medianprops=dict(color="red", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor("C1")
            patch.set_alpha(0.5)
        ax.set_xlabel("Donor Count Group", fontsize=12)
        ax.set_ylabel("Entropy (bits)", fontsize=12)
        ax.set_title("z-prob Entropy by Donor Group", fontsize=13)
        ax.grid(True, alpha=0.2, axis="y")
        plt.sca(ax)
        plt.xticks(rotation=45, ha="right")

        ax = axes[1]
        if gini is not None:
            box_data = [gini[bin_idx == bi] for bi in active_indices]
            bp = ax.boxplot(box_data, labels=active_labels, patch_artist=True,
                            showfliers=False, medianprops=dict(color="red", linewidth=2))
            for patch in bp["boxes"]:
                patch.set_facecolor("C2")
                patch.set_alpha(0.5)
            ax.set_xlabel("Donor Count Group", fontsize=12)
            ax.set_ylabel("Gini Index", fontsize=12)
            ax.set_title("Gini Index by Donor Group", fontsize=13)
            ax.grid(True, alpha=0.2, axis="y")
            plt.sca(ax)
            plt.xticks(rotation=45, ha="right")

        fig.tight_layout()
        fig.savefig(dirs["donor_bins"] / "donor_bin_entropy_gini.png", dpi=200)
        plt.close(fig)

    # --- Plot D: Max z-prob by donor bin ---
    if max_z is not None:
        fig, ax = plt.subplots(figsize=(12, 6))
        box_data = [max_z[bin_idx == bi] for bi in active_indices]
        bp = ax.boxplot(box_data, labels=active_labels, patch_artist=True,
                        showfliers=False, medianprops=dict(color="red", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor("C3")
            patch.set_alpha(0.5)
        ax.set_xlabel("Donor Count Group", fontsize=12)
        ax.set_ylabel("Max z-prob", fontsize=12)
        ax.set_title("Maximum z-probability by Donor Group", fontsize=14)
        ax.grid(True, alpha=0.2, axis="y")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(dirs["donor_bins"] / "donor_bin_max_zprob.png", dpi=200)
        plt.close(fig)

    # --- Save stats CSV ---
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
# Additional statistics
# ---------------------------------------------------------------------------

def save_summary(args, thresholds, global_stats, expl_stats, hla_stats,
                 entropy_stats, bin_stats, dirs, log):
    """Save overall summary JSON."""
    summary = {
        "source_h5": str(args.h5_path),
        "donor_matrix_path": str(args.donor_matrix_path),
        "total_clusters": global_stats["total_clusters"],
        "num_alleles": global_stats["num_alleles"],
        "num_donors": global_stats["num_donors_total"],
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
    args = parse_args()

    # Resolve which analyses to run
    do_expl = args.all or args.donor_explanation
    do_hla = args.all or args.hla_diversity
    do_entropy = args.all or args.entropy
    do_bins = args.all or args.donor_bins

    if not any([do_expl, do_hla, do_entropy, do_bins]):
        print("No analysis selected. Use --all or individual flags.")
        print("  --donor_explanation  --hla_diversity  --entropy  --donor_bins")
        sys.exit(1)

    # Donor bins require both explanation and entropy
    if do_bins:
        do_expl = True
        do_entropy = True

    dirs = make_dirs(args.output_dir)
    log = setup_logging(args.output_dir)
    thresholds = build_thresholds(args)

    log.info("=" * 60)
    log.info("MLE z-probs Analysis Pipeline")
    log.info("=" * 60)
    log.info(f"  H5 path:     {args.h5_path}")
    log.info(f"  Output dir:  {args.output_dir}")
    log.info(f"  Thresholds:  {thresholds[0]:.2f} to {thresholds[-1]:.2f} "
             f"(step={args.threshold_step}, n={len(thresholds)})")
    log.info(f"  Chunk size:  {args.chunk_size:,}")
    log.info(f"  Analyses:    expl={do_expl} hla={do_hla} "
             f"entropy={do_entropy} bins={do_bins}")
    log.info("=" * 60)

    # ---- Phase 1: Single pass metrics computation ----
    t0 = time.time()
    global_stats = compute_metrics(
        args, thresholds, do_expl, do_hla, do_entropy, dirs, log
    )
    metrics_path = global_stats["metrics_path"]

    # ---- Phase 2: Plotting ----
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

    # ---- Save summary ----
    save_summary(args, thresholds, global_stats, expl_stats, hla_stats,
                 entropy_stats, bin_stats, dirs, log)

    total = time.time() - t0
    log.info(f"\nAll analyses complete in {total:.1f}s ({total/60:.1f}min)")
    log.info(f"Results in: {dirs['base']}")


if __name__ == "__main__":
    main()