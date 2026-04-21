#!/usr/bin/env python3
"""
compare_gamma_vs_rank.py — Side-by-side per-HLA boxplots: gamma vs rank.
=========================================================================
Produces a 2-panel PNG comparing the original gamma distribution against
the percentile-rank normalized version, per HLA. Useful for confirming
that rank normalization removed the per-HLA scale bias before NN training.
PANELS:
  Top:    γ_ia distribution per HLA (observed pairs only)
  Bottom: percentile_rank distribution per HLA (observed pairs only)
Both panels overlay red dots showing HLA background frequency from the
donor matrix. After rank normalization, the bottom panel should be
roughly uniform [0,1] for every HLA, while the top panel will still
show the MLE bias.
=========================================================================
USAGE:
    python compare_gamma_vs_rank.py \\
        --h5_path /path/to/filtered.h5 \\
        --donor_matrix_path /path/to/donor_hla_matrix.npz \\
        --output_dir /path/to/out \\
        --reservoir_size 5000
=========================================================================
"""
import os
import sys
import time
import argparse
import numpy as np
import h5py
from pathlib import Path
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="2-panel per-HLA boxplot: gamma vs percentile rank.")
    p.add_argument("--h5_path", required=True,
                   help="Filtered H5 with z_probs AND percentile_rank.")
    p.add_argument("--donor_matrix_path", required=True,
                   help="Donor HLA matrix NPZ for background frequencies.")
    p.add_argument("--output_dir", required=True, help="Output directory.")
    p.add_argument("--chunk_size", type=int, default=200000,
                   help="Chunk size (default: 200000).")
    p.add_argument("--reservoir_size", type=int, default=5000,
                   help="Max samples per HLA per panel (default: 5000).")
    return p.parse_args()


def reservoir_replace(buf, R, total_seen, new_items, rng):
    """Vectorized reservoir replacement, modifies buf in place."""
    n_new = len(new_items)
    indices = np.arange(1, n_new + 1, dtype=np.float64) + total_seen
    probs = float(R) / indices
    rand_vals = rng.random(n_new)
    accept = rand_vals < probs
    accept_idx = np.where(accept)[0]
    if len(accept_idx) > 0:
        replace_pos = rng.integers(0, R, size=len(accept_idx))
        buf[replace_pos] = new_items[accept_idx]


def gather_reservoirs(args):
    """Single chunked pass: collect per-HLA reservoir for both gamma and rank.
    Returns:
        dict with gamma_reservoir, rank_reservoir, hla_freq, num_alleles, tcr_counts.
    """
    # ── load donor matrix ───────────────────────────────────────────
    donor_hla = np.load(args.donor_matrix_path)["donor_hla_matrix"]
    D, A = donor_hla.shape
    hla_freq = donor_hla.sum(axis=0).astype(np.float64) / D
    print(f"  Donor matrix: {D} donors x {A} alleles")
    # ── open H5 ─────────────────────────────────────────────────────
    src = h5py.File(args.h5_path, "r")
    clusters = src["clusters"]
    if "percentile_rank" not in clusters:
        print("ERROR: clusters/percentile_rank not found. "
              "Run compute_percentile_rank.py first.")
        sys.exit(1)
    n_clusters = clusters["cluster_id"].shape[0]
    print(f"  Filtered H5: {n_clusters:,} TCRs")
    # CSR handles
    zp_indptr = clusters["z_probs"]["indptr"]
    zp_indices = clusters["z_probs"]["indices"]
    zp_data = clusters["z_probs"]["data"]
    pr_indptr = clusters["percentile_rank"]["indptr"]
    pr_indices = clusters["percentile_rank"]["indices"]
    pr_data = clusters["percentile_rank"]["data"]
    ct_indptr = clusters["counts"]["indptr"]
    ct_indices = clusters["counts"]["indices"]
    ct_data = clusters["counts"]["data"]
    # ── reservoir storage (one per panel) ───────────────────────────
    R = args.reservoir_size
    gamma_res = np.zeros((A, R), dtype=np.float32)
    rank_res = np.zeros((A, R), dtype=np.float32)
    fill = np.zeros(A, dtype=np.int64)
    tcr_counts = np.zeros(A, dtype=np.int64)
    rng = np.random.default_rng(seed=42)
    # ── chunked pass ────────────────────────────────────────────────
    t0 = time.time()
    for cs in range(0, n_clusters, args.chunk_size):
        ce = min(cs + args.chunk_size, n_clusters)
        n_raw = ce - cs
        # Read gammas
        zp_ip = np.asarray(zp_indptr[cs:ce + 1])
        zp_s, zp_e = int(zp_ip[0]), int(zp_ip[-1])
        zp_ip_local = zp_ip - zp_s
        gamma_dense = csr_matrix(
            (np.asarray(zp_data[zp_s:zp_e]),
             np.asarray(zp_indices[zp_s:zp_e]),
             zp_ip_local),
            shape=(n_raw, A),
        ).toarray().astype(np.float32)
        # Read ranks
        pr_ip = np.asarray(pr_indptr[cs:ce + 1])
        pr_s, pr_e = int(pr_ip[0]), int(pr_ip[-1])
        pr_ip_local = pr_ip - pr_s
        rank_dense = csr_matrix(
            (np.asarray(pr_data[pr_s:pr_e]),
             np.asarray(pr_indices[pr_s:pr_e]),
             pr_ip_local),
            shape=(n_raw, A),
        ).toarray().astype(np.float32)
        # Read counts → observed mask
        ct_ip = np.asarray(ct_indptr[cs:ce + 1])
        ct_s, ct_e = int(ct_ip[0]), int(ct_ip[-1])
        ct_ip_local = ct_ip - ct_s
        obs_mask = csr_matrix(
            (np.ones(ct_e - ct_s, dtype=np.float32),
             np.asarray(ct_indices[ct_s:ct_e]),
             ct_ip_local),
            shape=(n_raw, A),
        ).toarray().astype(np.bool_)
        # Per-HLA reservoir update
        any_obs = np.where(obs_mask.any(axis=0))[0]
        for a in any_obs:
            col = obs_mask[:, a]
            g_vals = gamma_dense[col, a]
            r_vals = rank_dense[col, a]
            n_new = len(g_vals)
            tcr_counts[a] += n_new
            n_old = fill[a]
            if n_old < R:
                take = min(n_new, R - n_old)
                gamma_res[a, n_old:n_old + take] = g_vals[:take]
                rank_res[a, n_old:n_old + take] = r_vals[:take]
                fill[a] = n_old + take
                if n_new > take:
                    # The same random pattern must be applied to both
                    # arrays so a row reservation in gamma matches rank.
                    tail_g = g_vals[take:]
                    tail_r = r_vals[take:]
                    n_tail = len(tail_g)
                    indices = (np.arange(1, n_tail + 1, dtype=np.float64)
                               + n_old + take)
                    probs = float(R) / indices
                    rand_vals = rng.random(n_tail)
                    accept_idx = np.where(rand_vals < probs)[0]
                    if len(accept_idx) > 0:
                        replace_pos = rng.integers(0, R, size=len(accept_idx))
                        gamma_res[a, replace_pos] = tail_g[accept_idx]
                        rank_res[a, replace_pos] = tail_r[accept_idx]
                    fill[a] = n_old + n_new
            else:
                indices = np.arange(1, n_new + 1, dtype=np.float64) + n_old
                probs = float(R) / indices
                rand_vals = rng.random(n_new)
                accept_idx = np.where(rand_vals < probs)[0]
                if len(accept_idx) > 0:
                    replace_pos = rng.integers(0, R, size=len(accept_idx))
                    gamma_res[a, replace_pos] = g_vals[accept_idx]
                    rank_res[a, replace_pos] = r_vals[accept_idx]
                fill[a] = n_old + n_new
        elapsed = time.time() - t0
        rate = ce / elapsed if elapsed > 0 else 0
        print(f"  [{ce:>10,}/{n_clusters:,}] "
              f"{100*ce/n_clusters:5.1f}% | {rate:,.0f} TCRs/s")
    src.close()
    # Trim reservoirs
    gamma_list = [gamma_res[a, :min(int(fill[a]), R)] for a in range(A)]
    rank_list = [rank_res[a, :min(int(fill[a]), R)] for a in range(A)]
    return {
        "gamma_reservoir": gamma_list,
        "rank_reservoir": rank_list,
        "hla_freq": hla_freq,
        "num_alleles": A,
        "tcr_counts": tcr_counts,
        "n_clusters": n_clusters,
    }


def _draw_panel(ax, reservoirs, hla_freq, A, ylabel, title, max_dots=500):
    """Draw one boxplot panel: boxes, jittered dots, red freq dots."""
    x_pos = np.arange(A)
    bp_data = [r if len(r) > 0 else np.array([0.0]) for r in reservoirs]
    ax.boxplot(
        bp_data, positions=x_pos, widths=0.6,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.0),
        boxprops=dict(facecolor="lightblue", edgecolor="gray", linewidth=0.5),
        whiskerprops=dict(color="gray", linewidth=0.5),
        capprops=dict(color="gray", linewidth=0.5),
        manage_ticks=False,
    )
    for a in range(A):
        vals = reservoirs[a]
        if len(vals) == 0:
            continue
        if len(vals) > max_dots:
            idx = np.random.default_rng(a).choice(len(vals), max_dots, replace=False)
            vals = vals[idx]
        jitter = np.random.default_rng(a + 1000).uniform(-0.2, 0.2, size=len(vals))
        ax.scatter(x_pos[a] + jitter, vals,
                   s=2, alpha=0.15, color="steelblue",
                   linewidths=0, zorder=2)
    ax.scatter(x_pos, hla_freq, s=18, color="red", marker="o", zorder=5,
               label="HLA background freq")
    ax.set_xlim(-0.5, A - 0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    tick_step = max(1, A // 50)
    ax.set_xticks(np.arange(0, A, tick_step))
    ax.set_xticklabels(np.arange(0, A, tick_step), fontsize=6, rotation=90)
    ax.legend(loc="upper right", fontsize=9, markerscale=2)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)


def plot_2panel(stats, args):
    """Render the 2-panel comparison and save as one PNG."""
    A = stats["num_alleles"]
    fig_w = max(20.0, A * 0.12)
    fig_h = 16.0
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)
    _draw_panel(
        axes[0], stats["gamma_reservoir"], stats["hla_freq"], A,
        ylabel="γ (raw gamma)",
        title=f"BEFORE rank normalization | gamma per HLA | "
              f"{stats['n_clusters']:,} TCRs",
    )
    _draw_panel(
        axes[1], stats["rank_reservoir"], stats["hla_freq"], A,
        ylabel="percentile rank in [0, 1]",
        title="AFTER rank normalization | should be ~uniform per HLA",
    )
    axes[1].set_xlabel("HLA allele index", fontsize=12)
    fig.tight_layout()
    out_path = Path(args.output_dir) / "compare_gamma_vs_rank.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    """Run comparison plot pipeline."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 60)
    print("Compare gamma vs percentile rank")
    print("=" * 60)
    print(f"  H5:         {args.h5_path}")
    print(f"  Output dir: {args.output_dir}")
    t0 = time.time()
    stats = gather_reservoirs(args)
    print(f"\nGenerating 2-panel plot...")
    out_path = plot_2panel(stats, args)
    print(f"  Saved: {out_path}")
    total = time.time() - t0
    print(f"Total time: {total:.1f}s ({total/60:.1f}min)")


if __name__ == "__main__":
    main()