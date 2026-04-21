#!/usr/bin/env python3
"""
Patch for analyze_zprobs.py — adds --hla_gamma_boxplot analysis.
Three integration points in analyze_zprobs.py:
  1. parse_args():  add the flag
  2. make_dirs():   add the output subdirectory
  3. main():        wire up the dispatch
  4. Add two new functions below: compute_hla_gamma_reservoir() and plot_hla_gamma_boxplot()
=========================================================================
STEP 1 — In parse_args(), after the --donor_bins argument, add:
=========================================================================
    p.add_argument("--hla_gamma_boxplot", action="store_true",
                   help="Analysis 5: Per-HLA boxplot of observed gamma values "
                        "with background HLA frequencies.")
    p.add_argument("--boxplot_reservoir_size", type=int, default=5000,
                   help="Max gamma samples per HLA for boxplot (default: 5000).")
=========================================================================
STEP 2 — In make_dirs(), add to the dirs dict:
=========================================================================
        "hla_gamma_boxplot": base / "hla_gamma_boxplot",
=========================================================================
STEP 3 — In main(), after the do_bins lines, add:
=========================================================================
    do_boxplot = args.all or args.hla_gamma_boxplot
    # ... (update the 'if not any(...)' check to include do_boxplot)
    # ... after compute_metrics call, add:
    boxplot_stats = None
    if do_boxplot:
        boxplot_stats = run_hla_gamma_boxplot(args, dirs, log, donor_filter_mask)
    # ... update save_summary if desired
=========================================================================
STEP 4 — Add these two functions to the file (before main):
=========================================================================
"""
import os
import sys
import time
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Analysis 5: Per-HLA gamma boxplot with reservoir sampling
# ---------------------------------------------------------------------------

def compute_hla_gamma_reservoir(args, log, donor_filter_mask):
    """Chunked pass through H5 collecting observed gamma values per HLA.
    Uses reservoir sampling to bound memory at O(A * reservoir_size).
    Reads both z_probs and counts CSR; only collects gamma where counts > 0.
    Args:
        args:              parsed CLI arguments (needs h5_path, donor_matrix_path,
                           chunk_size, boxplot_reservoir_size).
        log:               logger instance.
        donor_filter_mask: optional boolean mask over all clusters.
    Returns:
        dict with keys:
            reservoir:   dict[int -> np.ndarray] of sampled gammas per HLA.
            hla_freq:    (A,) HLA background frequencies from donor matrix.
            num_alleles: int.
            tcr_counts:  (A,) total observed TCR count per HLA.
    """
    # --- Load donor HLA matrix for background frequencies ---
    donor_hla = np.load(args.donor_matrix_path)["donor_hla_matrix"]
    num_donors_total, num_alleles = donor_hla.shape
    # HLA frequency = fraction of donors carrying each allele
    hla_freq = donor_hla.sum(axis=0).astype(np.float64) / num_donors_total
    log.info(f"  Donor matrix: {num_donors_total} donors x {num_alleles} alleles")
    # --- Open H5 ---
    src = h5py.File(args.h5_path, "r")
    clusters_grp = src["clusters"]
    total_clusters = clusters_grp["cluster_id"].shape[0]
    # CSR handles for z_probs
    zp_indptr = clusters_grp["z_probs"]["indptr"]
    zp_indices = clusters_grp["z_probs"]["indices"]
    zp_data = clusters_grp["z_probs"]["data"]
    # CSR handles for counts (observed mask)
    ct_indptr = clusters_grp["counts"]["indptr"]
    ct_indices = clusters_grp["counts"]["indices"]
    ct_data = clusters_grp["counts"]["data"]
    # --- Reservoir storage ---
    R = args.boxplot_reservoir_size
    # Pre-allocate fixed-size arrays per HLA (faster than lists)
    reservoir = {a: np.empty(R, dtype=np.float32) for a in range(num_alleles)}
    fill_count = np.zeros(num_alleles, dtype=np.int64)  # how many inserted so far
    tcr_counts = np.zeros(num_alleles, dtype=np.int64)  # total observed per HLA
    # --- RNG for reservoir sampling ---
    rng = np.random.default_rng(seed=42)
    # --- Chunked pass ---
    t0 = time.time()
    chunk_size = args.chunk_size
    for cs in range(0, total_clusters, chunk_size):
        ce = min(cs + chunk_size, total_clusters)
        n_raw = ce - cs
        # ── apply donor filter ───────────────────────────────────
        if donor_filter_mask is not None:
            chunk_mask = donor_filter_mask[cs:ce]
            if chunk_mask.sum() == 0:
                continue
        else:
            chunk_mask = None
        # ── read z_probs CSR → dense ─────────────────────────────
        zp_ip = np.asarray(zp_indptr[cs:ce + 1])
        zp_s, zp_e = int(zp_ip[0]), int(zp_ip[-1])
        zp_ip_local = zp_ip - zp_s
        z_dense = csr_matrix(
            (np.asarray(zp_data[zp_s:zp_e]),
             np.asarray(zp_indices[zp_s:zp_e]),
             zp_ip_local),
            shape=(n_raw, num_alleles),
        ).toarray().astype(np.float32)
        # ── read counts CSR → dense (binary observed mask) ───────
        ct_ip = np.asarray(ct_indptr[cs:ce + 1])
        ct_s, ct_e = int(ct_ip[0]), int(ct_ip[-1])
        ct_ip_local = ct_ip - ct_s
        obs_mask = csr_matrix(
            (np.ones(ct_e - ct_s, dtype=np.float32),
             np.asarray(ct_indices[ct_s:ct_e]),
             ct_ip_local),
            shape=(n_raw, num_alleles),
        ).toarray().astype(np.bool_)
        # ── apply donor filter ───────────────────────────────────
        if chunk_mask is not None:
            z_dense = z_dense[chunk_mask]
            obs_mask = obs_mask[chunk_mask]
        # ── vectorized reservoir sampling per HLA ────────────────
        # Process all HLAs that have observations in this chunk
        any_obs_hla = np.where(obs_mask.any(axis=0))[0]
        for a in any_obs_hla:
            col_mask = obs_mask[:, a]
            gammas = z_dense[col_mask, a]
            n_new = len(gammas)
            tcr_counts[a] += n_new
            n_old = fill_count[a]
            if n_old < R:
                # Still filling reservoir
                take = min(n_new, R - n_old)
                reservoir[a][n_old:n_old + take] = gammas[:take]
                fill_count[a] = n_old + take
                # Remaining elements need reservoir replacement
                if n_new > take:
                    _reservoir_replace(
                        reservoir[a], R, n_old + take,
                        gammas[take:], rng)
                    fill_count[a] = n_old + n_new
            else:
                # Reservoir full — replacement sampling
                _reservoir_replace(
                    reservoir[a], R, n_old, gammas, rng)
                fill_count[a] = n_old + n_new
        # ── progress ─────────────────────────────────────────────
        elapsed = time.time() - t0
        rate = ce / elapsed if elapsed > 0 else 0
        log.info(f"  [boxplot] [{ce:>10,}/{total_clusters:,}] "
                 f"{100*ce/total_clusters:5.1f}% | {rate:,.0f} clusters/s")
    src.close()
    # Trim reservoirs to actual fill
    for a in range(num_alleles):
        n = min(int(fill_count[a]), R)
        reservoir[a] = reservoir[a][:n]
    total_time = time.time() - t0
    log.info(f"  Reservoir pass done: {total_time:.1f}s | "
             f"Mean samples/HLA: {np.mean([len(reservoir[a]) for a in range(num_alleles)]):.0f}")
    return {
        "reservoir": reservoir,
        "hla_freq": hla_freq.astype(np.float32),
        "num_alleles": num_alleles,
        "tcr_counts": tcr_counts,
    }


def _reservoir_replace(buf, R, total_seen, new_items, rng):
    """Vectorized reservoir replacement for a single HLA column.
    Modifies buf in-place. total_seen is count BEFORE these new_items.
    Args:
        buf:        (R,) reservoir array.
        R:          reservoir capacity.
        total_seen: total items seen before new_items.
        new_items:  (n_new,) array of new gamma values.
        rng:        numpy Generator for random numbers.
    """
    n_new = len(new_items)
    # For each new item, probability of inclusion = R / (total_seen + j + 1)
    indices = np.arange(1, n_new + 1, dtype=np.float64) + total_seen
    probs = float(R) / indices
    # Random draws — accept if rand < prob
    rand_vals = rng.random(n_new)
    accept = rand_vals < probs
    accept_indices = np.where(accept)[0]
    if len(accept_indices) > 0:
        # Random replacement positions in reservoir
        replace_pos = rng.integers(0, R, size=len(accept_indices))
        buf[replace_pos] = new_items[accept_indices]


def plot_hla_gamma_boxplot(reservoir_data, dirs, log):
    """Plot per-HLA boxplot of observed gammas with background frequency overlay.
    Creates a wide PNG (dpi=300) suitable for ~440 HLAs.
    Args:
        reservoir_data: dict from compute_hla_gamma_reservoir().
        dirs:           directory dict with 'hla_gamma_boxplot' key.
        log:            logger instance.
    Returns:
        dict with summary statistics.
    """
    reservoir = reservoir_data["reservoir"]
    hla_freq = reservoir_data["hla_freq"]
    num_alleles = reservoir_data["num_alleles"]
    tcr_counts = reservoir_data["tcr_counts"]
    # --- Collect boxplot data in index order ---
    bp_data = [reservoir[a] for a in range(num_alleles)]
    x_positions = np.arange(num_alleles)
    # --- Figure dimensions: ~0.12 inch per HLA, min 20 inches ---
    fig_width = max(20.0, num_alleles * 0.12)
    fig_height = 8.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    # --- Boxplot (no fliers — we draw dots separately) ---
    bp = ax.boxplot(
        bp_data,
        positions=x_positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.0),
        boxprops=dict(facecolor="lightblue", edgecolor="gray", linewidth=0.5),
        whiskerprops=dict(color="gray", linewidth=0.5),
        capprops=dict(color="gray", linewidth=0.5),
        manage_ticks=False,
    )
    # --- Transparent dots (strip plot) via scatter ---
    # Sub-sample for plotting speed: max 500 dots per HLA
    MAX_DOTS = 500
    for a in range(num_alleles):
        vals = reservoir[a]
        if len(vals) == 0:
            continue
        if len(vals) > MAX_DOTS:
            idx = np.random.default_rng(a).choice(len(vals), MAX_DOTS, replace=False)
            vals = vals[idx]
        # Jitter x position slightly for visibility
        jitter = np.random.default_rng(a + 1000).uniform(-0.2, 0.2, size=len(vals))
        ax.scatter(
            x_positions[a] + jitter, vals,
            s=2, alpha=0.15, color="steelblue", linewidths=0, zorder=2,
        )
    # --- HLA background frequency as red points ---
    ax.scatter(
        x_positions, hla_freq,
        s=18, color="red", marker="o", zorder=5,
        label="HLA background freq",
    )
    # --- Formatting ---
    ax.set_xlim(-0.5, num_alleles - 0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("HLA allele index", fontsize=12)
    ax.set_ylabel("γ (gamma) — observed TCR-HLA pairs only", fontsize=12)
    ax.set_title(
        f"Per-HLA gamma distribution (observed pairs) | {num_alleles} alleles | "
        f"reservoir={len(reservoir[0])} max/HLA",
        fontsize=13,
    )
    # Tick every 10 HLAs to avoid clutter
    tick_step = max(1, num_alleles // 50)
    tick_pos = np.arange(0, num_alleles, tick_step)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_pos, fontsize=6, rotation=90)
    ax.legend(loc="upper right", fontsize=10, markerscale=2)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    # --- Save ---
    out_path = dirs["hla_gamma_boxplot"] / "hla_gamma_boxplot.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Boxplot saved: {out_path}")
    log.info(f"    Figure size: {fig_width:.1f} x {fig_height:.1f} inches @ 300 dpi")
    # --- Summary stats ---
    median_gammas = np.array([
        float(np.median(reservoir[a])) if len(reservoir[a]) > 0 else np.nan
        for a in range(num_alleles)
    ])
    mean_gammas = np.array([
        float(np.mean(reservoir[a])) if len(reservoir[a]) > 0 else np.nan
        for a in range(num_alleles)
    ])
    stats = {
        "num_alleles": num_alleles,
        "alleles_with_data": int(np.sum(tcr_counts > 0)),
        "alleles_without_data": int(np.sum(tcr_counts == 0)),
        "median_gamma_per_hla_mean": float(np.nanmean(median_gammas)),
        "median_gamma_per_hla_std": float(np.nanstd(median_gammas)),
        "mean_tcr_count_per_hla": float(np.mean(tcr_counts)),
        "min_tcr_count": int(np.min(tcr_counts)),
        "max_tcr_count": int(np.max(tcr_counts)),
    }
    log.info(f"    Alleles with data: {stats['alleles_with_data']}/{num_alleles}")
    log.info(f"    Mean median gamma: {stats['median_gamma_per_hla_mean']:.4f}")
    log.info(f"    TCR counts/HLA: mean={stats['mean_tcr_count_per_hla']:.0f} "
             f"min={stats['min_tcr_count']} max={stats['max_tcr_count']}")
    return stats


def run_hla_gamma_boxplot(args, dirs, log, donor_filter_mask):
    """Top-level dispatcher for Analysis 5: HLA gamma boxplot.
    Args:
        args:              parsed CLI namespace.
        dirs:              directory dict from make_dirs().
        log:               logger.
        donor_filter_mask: optional boolean mask.
    Returns:
        dict with summary statistics.
    """
    log.info("-" * 60)
    log.info("Analysis 5: Per-HLA gamma boxplot")
    log.info("-" * 60)
    t0 = time.time()
    reservoir_data = compute_hla_gamma_reservoir(args, log, donor_filter_mask)
    stats = plot_hla_gamma_boxplot(reservoir_data, dirs, log)
    elapsed = time.time() - t0
    log.info(f"  Total boxplot analysis time: {elapsed:.1f}s")
    return stats


# =========================================================================
# Quick standalone test (can also be run independently)
# =========================================================================
if __name__ == "__main__":
    import argparse as _ap
    p = _ap.ArgumentParser(description="Standalone HLA gamma boxplot")
    p.add_argument("--h5_path", required=True)
    p.add_argument("--donor_matrix_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--chunk_size", type=int, default=200000)
    p.add_argument("--boxplot_reservoir_size", type=int, default=5000)
    p.add_argument("--keep_only_upperthan_n_donors", type=int, default=None)
    args = p.parse_args()
    # --- setup ---
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger(__name__)
    out_dir = Path(args.output_dir) / "hla_gamma_boxplot"
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = {"hla_gamma_boxplot": out_dir}
    # --- donor filter ---
    donor_filter_mask = None
    if args.keep_only_upperthan_n_donors is not None:
        with h5py.File(args.h5_path, "r") as f:
            n_donors = f["clusters"]["n_donors"][:]
        donor_filter_mask = n_donors >= args.keep_only_upperthan_n_donors
        log.info(f"Filter: {donor_filter_mask.sum():,}/{len(n_donors):,} "
                 f"TCRs with n_donors >= {args.keep_only_upperthan_n_donors}")
    # --- run ---
    run_hla_gamma_boxplot(args, dirs, log, donor_filter_mask)