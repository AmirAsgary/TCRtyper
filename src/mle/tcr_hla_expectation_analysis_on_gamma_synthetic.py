#!/usr/bin/env python3
"""
TCR-HLA Expectation Analysis using TRAINED gamma (sigmoid(z)) probabilities.

Same computation as tcr_hla_expectation_analysis_synthetic.py but replaces
ground-truth binary z_{ia} with the learned gamma_{ia} = sigmoid(z_logit_{ia}).

For each TCR i in each positive donor n, computes:
    e_{ni} = sum_a  gamma_{ia} * x_{na}

This gives the "expected number of binding HLAs" that the model predicts
for each TCR-donor pair, and can be compared against the ground-truth
version (which should be peaked at integers, typically 1).

Usage:
    # Single result:
    python tcr_hla_expectation_analysis_on_gamma_synthetic.py \
        --results_dir outputs/synthetic/b100_n35 \
        --data_dir data/autotcr/synthetic/binder_set/b100/n35/N100000 \
        --donor_matrix data/autotcr/donor_hla_matrix.npz \
        --output_dir results/gamma_expectation/b100_n35

    # Auto-discover all bX_nY results:
    python tcr_hla_expectation_analysis_on_gamma_synthetic.py \
        --results_base_dir outputs/cbscratch/synthetic_03_10_2026 \
        --data_base_dir data/autotcr/synthetic/binder_set \
        --donor_matrix data/autotcr/donor_hla_matrix.npz \
        --output_dir results/gamma_expectation

    # Explicit config CSV:
    python tcr_hla_expectation_analysis_on_gamma_synthetic.py \
        --df config.csv \
        --output_dir results/gamma_expectation
"""
import os, sys, json, re, argparse, csv
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils import (
    pad_list_to_array,
    PublicTcrHlaCsrReader,
    PublicTcrHlaCsrReaderChunk,
    NumpyEncoder,
)


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="TCR-HLA expectation analysis using trained gamma (sigmoid(z))")
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--results_dir", type=str,
                     help="Path to a single MLE output directory "
                          "(must contain figures/analysis_arrays.npz)")
    inp.add_argument("--results_base_dir", type=str,
                     help="Root dir containing bX_nY/ MLE output directories "
                          "(auto-discover mode)")
    inp.add_argument("--df", type=str,
                     help="Path to CSV config with columns: "
                          "name, results_dir, data_dir, donor_matrix")
    p.add_argument("--data_dir", type=str,
                   help="Path to single synthetic data directory. "
                        "Required with --results_dir")
    p.add_argument("--data_base_dir", type=str,
                   help="Root dir containing bX/nY/N*/ data directories. "
                        "Required with --results_base_dir")
    p.add_argument("--donor_matrix", type=str,
                   help="Path to donor_hla_matrix.npz. "
                        "Required with --results_dir or --results_base_dir")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for figures and stats")
    p.add_argument("--pad_token", type=float, default=-1.0)
    p.add_argument("--max_e_plot", type=float, default=10.0,
                   help="Upper x-axis limit for e_ni histograms (default: 10.0)")
    p.add_argument("--n_bins", type=int, default=100,
                   help="Number of histogram bins for continuous e_ni (default: 100)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Auto-discovery: match bX_nY output dirs to bX/nY/N*/ data dirs
# ═══════════════════════════════════════════════════════════════════
_BN_OUTPUT_PATTERN = re.compile(r"^b(\d+)_n(\d+)$", re.IGNORECASE)


def _natural_sort_key(name: str):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", name)]


def _find_data_dir(data_base_dir: Path, b_val: str, n_val: str) -> str | None:
    """
    Given data_base_dir and b/n values, find the data directory:
        data_base_dir/b{b_val}/n{n_val}/N*/
    Returns the first N* directory found, or None.
    """
    bn_dir = data_base_dir / f"b{b_val}" / f"n{n_val}"
    if not bn_dir.is_dir():
        return None
    # Find N* subdirectory
    n_dirs = sorted(
        [d for d in bn_dir.iterdir()
         if d.is_dir() and re.match(r"^N\d+$", d.name)],
        key=lambda p: _natural_sort_key(p.name),
    )
    if not n_dirs:
        return None
    # Check the first one has the expected files
    candidate = n_dirs[0]
    if (candidate / "synthetic_binder_sets.npy").exists():
        return str(candidate)
    return None


def discover_configs(results_base_dir: str, data_base_dir: str,
                     donor_matrix: str) -> list[dict]:
    """
    Walk results_base_dir for directories matching bX_nY pattern,
    and map each to data_base_dir/bX/nY/N*/.
    """
    results_base = Path(results_base_dir)
    data_base = Path(data_base_dir)
    configs = []

    for child in sorted(results_base.iterdir(),
                        key=lambda p: _natural_sort_key(p.name)):
        if not child.is_dir():
            continue
        m = _BN_OUTPUT_PATTERN.match(child.name)
        if not m:
            continue
        b_val, n_val = m.group(1), m.group(2)
        # Check that analysis_arrays.npz exists
        arrays_path = child / "figures" / "analysis_arrays.npz"
        if not arrays_path.exists():
            print(f"  SKIP {child.name}: no figures/analysis_arrays.npz")
            continue
        # Find matching data directory
        data_dir = _find_data_dir(data_base, b_val, n_val)
        if data_dir is None:
            print(f"  SKIP {child.name}: no matching data dir "
                  f"at {data_base}/b{b_val}/n{n_val}/N*/")
            continue
        configs.append({
            "name": child.name,
            "results_dir": str(child),
            "data_dir": data_dir,
            "donor_matrix": donor_matrix,
        })

    configs.sort(key=lambda d: _natural_sort_key(d["name"]))
    return configs


# ═══════════════════════════════════════════════════════════════════
# Config loader
# ═══════════════════════════════════════════════════════════════════
def load_config_file(config_path):
    import pandas as pd
    config_path = Path(config_path)
    ext = config_path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(config_path)
    elif ext == ".tsv":
        df = pd.read_csv(config_path, sep="\t")
    elif ext == ".json":
        df = pd.read_json(config_path)
    else:
        raise ValueError(f"Unsupported config format: {ext}")
    required = ["results_dir", "data_dir", "donor_matrix"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════
def load_gamma_and_data(results_dir, data_dir, donor_matrix_path, pad_token=-1.0):
    """
    Load trained gamma probabilities from MLE output and donor data.

    Handles both sparse model (gamma indexed by binder_sets) and
    dense model (gamma is full num_tcrs x num_alleles matrix).
    """
    results_dir = Path(results_dir)
    data_dir = Path(data_dir)

    # Load trained probabilities
    arrays_path = results_dir / "figures" / "analysis_arrays.npz"
    arrays = np.load(arrays_path)
    trained_probs = arrays["trained_probs"]  # (num_tcrs, ?)

    # Load donor data
    donor_indices = np.load(data_dir / "synthetic_donor_indices.npy")
    donor_hla_matrix = np.load(donor_matrix_path)["donor_hla_matrix"]
    num_alleles = donor_hla_matrix.shape[1]
    num_tcrs = trained_probs.shape[0]

    # Determine if sparse or dense model based on shape
    if trained_probs.shape[1] == num_alleles:
        # Dense model: trained_probs is already (num_tcrs, num_alleles)
        gamma_dense = trained_probs.copy()
        print(f"  Detected DENSE model output: gamma shape = {gamma_dense.shape}")
    else:
        # Sparse model: need binder_sets to map to allele IDs
        h5_path = data_dir / "synthetic_tcr_hla_counts.h5"
        try:
            with PublicTcrHlaCsrReaderChunk(str(h5_path)) as reader:
                counts_set, max_all = reader.read_sparse_indices_of_counts()
        except KeyError:
            with PublicTcrHlaCsrReader(str(h5_path)) as reader:
                counts_set, max_all = reader.read_sparse_indices()
        binder_sets = pad_list_to_array(counts_set, max_all, pad_token)

        # Build dense gamma matrix from sparse representation
        gamma_dense = np.zeros((num_tcrs, num_alleles), dtype=np.float32)
        valid_mask = binder_sets != pad_token
        for i in range(num_tcrs):
            valid = valid_mask[i]
            allele_ids = binder_sets[i][valid].astype(int)
            gamma_dense[i, allele_ids] = trained_probs[i][valid]
        print(f"  Detected SPARSE model output: mapped {trained_probs.shape[1]} "
              f"candidates -> {num_alleles} alleles")

    # Load ground truth for comparison stats
    true_hla_set = np.load(data_dir / "synthetic_binder_sets.npy")

    return {
        "gamma_dense": gamma_dense,
        "donor_indices": np.asarray(donor_indices),
        "donor_hla_matrix": donor_hla_matrix,
        "true_hla_set": np.asarray(true_hla_set),
        "num_tcrs": num_tcrs,
        "num_donors": donor_hla_matrix.shape[0],
        "num_alleles": num_alleles,
    }


# ═══════════════════════════════════════════════════════════════════
# Core computation
# ═══════════════════════════════════════════════════════════════════
def compute_e_ni_gamma(data, pad_token=-1.0):
    """
    For every (TCR i, positive donor n) pair compute:
        e_{ni} = sum_a gamma_{ia} * x_{na}   (continuous)

    Also computes ground-truth e_{ni} for comparison.

    Returns
    -------
    all_e_gamma : np.ndarray, 1-D float
        Flat array of gamma-based e_ni values.
    all_e_true : np.ndarray, 1-D int
        Flat array of ground-truth e_ni values (same pairs).
    per_tcr_mean_e_gamma : np.ndarray, shape (num_tcrs,)
    per_tcr_mean_e_true : np.ndarray, shape (num_tcrs,)
    """
    gamma_dense = data["gamma_dense"]
    true_hla_set = data["true_hla_set"]
    donor_indices = data["donor_indices"]
    donor_hla_matrix = data["donor_hla_matrix"]
    num_tcrs = data["num_tcrs"]
    num_alleles = data["num_alleles"]

    # Build ground-truth z matrix
    z_true = np.zeros((num_tcrs, num_alleles), dtype=np.float32)
    for i in range(num_tcrs):
        valid = true_hla_set[i] >= 0
        allele_ids = true_hla_set[i][valid].astype(int)
        z_true[i, allele_ids] = 1.0

    all_e_gamma = []
    all_e_true = []
    per_tcr_mean_e_gamma = np.zeros(num_tcrs, dtype=np.float64)
    per_tcr_mean_e_true = np.zeros(num_tcrs, dtype=np.float64)

    pad_int = int(pad_token)
    for i in range(num_tcrs):
        pos_donors = donor_indices[i]
        valid_mask = pos_donors != pad_int
        pos_donors_valid = pos_donors[valid_mask]
        if len(pos_donors_valid) == 0:
            continue
        x_pos = donor_hla_matrix[pos_donors_valid]  # (n_pos, A)

        e_gamma = x_pos @ gamma_dense[i]  # (n_pos,)
        e_true = x_pos @ z_true[i]        # (n_pos,)

        all_e_gamma.append(e_gamma)
        all_e_true.append(e_true)
        per_tcr_mean_e_gamma[i] = e_gamma.mean()
        per_tcr_mean_e_true[i] = e_true.mean()

    if all_e_gamma:
        all_e_gamma = np.concatenate(all_e_gamma).astype(np.float64)
        all_e_true = np.concatenate(all_e_true).astype(int)
    else:
        all_e_gamma = np.array([], dtype=np.float64)
        all_e_true = np.array([], dtype=int)

    return all_e_gamma, all_e_true, per_tcr_mean_e_gamma, per_tcr_mean_e_true


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════
def plot_single_dataset(all_e_gamma, all_e_true, per_tcr_mean_gamma,
                        per_tcr_mean_true, name, output_path,
                        max_e_plot=10.0, n_bins=100):
    """Four-panel figure for one dataset."""
    os.makedirs(output_path, exist_ok=True)
    if len(all_e_gamma) == 0:
        print(f"  WARNING: no valid pairs for {name}, skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Panel 1: histogram of gamma e_ni (continuous) ---
    ax = axes[0, 0]
    ax.hist(all_e_gamma, bins=n_bins, range=(0, max_e_plot),
            edgecolor="white", color="#4C72B0", alpha=0.85, density=True)
    ax.axvline(1.0, color="red", linestyle="--", lw=1.5, alpha=0.8, label="$e=1$")
    ax.set_xlabel("$e_{ni} = \\sum_a \\gamma_{ia}\\, x_{na}$")
    ax.set_ylabel("Density")
    ax.set_title(f"Gamma-based $e_{{ni}}$ distribution\n{name}")
    ax.legend()
    # Annotation
    mean_e = all_e_gamma.mean()
    median_e = np.median(all_e_gamma)
    frac_lt1 = np.mean(all_e_gamma < 1.0) * 100
    frac_1_2 = np.mean((all_e_gamma >= 1.0) & (all_e_gamma < 2.0)) * 100
    frac_ge2 = np.mean(all_e_gamma >= 2.0) * 100
    ax.text(0.97, 0.95,
            f"mean: {mean_e:.2f}\nmedian: {median_e:.2f}\n"
            f"$e<1$: {frac_lt1:.1f}%\n$1 \\leq e < 2$: {frac_1_2:.1f}%\n"
            f"$e \\geq 2$: {frac_ge2:.1f}%",
            transform=ax.transAxes, va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
    ax.grid(True, alpha=0.3)

    # --- Panel 2: gamma vs true scatter / 2D hist ---
    ax = axes[0, 1]
    if len(all_e_gamma) > 50000:
        # Use 2D histogram for large datasets
        h = ax.hist2d(all_e_true.astype(float), all_e_gamma,
                       bins=[np.arange(-0.5, max(all_e_true.max(), 5) + 1.5, 1),
                             n_bins],
                       range=[[-.5, max(all_e_true.max(), 5) + .5], [0, max_e_plot]],
                       cmap="viridis", cmin=1)
        plt.colorbar(h[3], ax=ax, label="Count")
    else:
        ax.scatter(all_e_true + np.random.normal(0, 0.05, len(all_e_true)),
                   all_e_gamma, alpha=0.1, s=2, c="#4C72B0")
    # Diagonal
    diag_max = min(max_e_plot, all_e_true.max() + 1) if len(all_e_true) else max_e_plot
    ax.plot([0, diag_max], [0, diag_max], "r--", lw=1.5, alpha=0.7, label="$y=x$")
    ax.set_xlabel("Ground-truth $e_{ni}$ (integer)")
    ax.set_ylabel("Gamma-based $e_{ni}$ (continuous)")
    ax.set_title(f"Gamma vs Ground-Truth\n{name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 3: overlaid histograms (gamma vs true) ---
    ax = axes[1, 0]
    bins_int = np.arange(-0.5, min(int(all_e_true.max()), 15) + 1.5, 1)
    ax.hist(all_e_true, bins=bins_int, edgecolor="white", color="#55A868",
            alpha=0.6, density=True, label="Ground-truth $z$")
    ax.hist(all_e_gamma, bins=n_bins, range=(0, max_e_plot),
            edgecolor="white", color="#4C72B0", alpha=0.5, density=True,
            label="Trained $\\gamma$")
    ax.axvline(1.0, color="red", linestyle="--", lw=1.5, alpha=0.7)
    ax.set_xlabel("$e_{ni}$")
    ax.set_ylabel("Density")
    ax.set_title(f"Comparison: Ground-Truth vs Gamma\n{name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 4: per-TCR mean gamma e vs mean true e ---
    ax = axes[1, 1]
    valid = (per_tcr_mean_true > 0) | (per_tcr_mean_gamma > 0)
    if valid.sum() > 0:
        ax.scatter(per_tcr_mean_true[valid], per_tcr_mean_gamma[valid],
                   alpha=0.3, s=8, c="#4C72B0")
        lim = max(per_tcr_mean_true[valid].max(),
                  per_tcr_mean_gamma[valid].max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", lw=1.5, alpha=0.7, label="$y=x$")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    ax.set_xlabel("Per-TCR mean ground-truth $e_{ni}$")
    ax.set_ylabel("Per-TCR mean gamma $e_{ni}$")
    ax.set_title(f"Per-TCR Mean Comparison\n{name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_path, "gamma_e_ni_distribution.png"),
                dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(output_path, "gamma_e_ni_distribution.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Plots saved to {output_path}")


def plot_combined(all_datasets, output_path, max_e_plot=10.0, n_bins=100):
    """Combined figures across all datasets."""
    os.makedirs(output_path, exist_ok=True)
    n_datasets = len(all_datasets)
    if n_datasets == 0:
        return

    cmap = plt.cm.tab10 if n_datasets <= 10 else plt.cm.tab20
    colors = [cmap(i / max(n_datasets - 1, 1)) for i in range(n_datasets)]

    # --- Figure 1: overlaid gamma e_ni densities ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    for (name, d), c in zip(all_datasets.items(), colors):
        e = d["all_e_gamma"]
        if len(e) == 0:
            continue
        ax.hist(e, bins=n_bins, range=(0, max_e_plot), alpha=0.4,
                label=name, color=c, density=True, edgecolor="white")
    ax.axvline(1.0, color="red", linestyle="--", lw=1.5, alpha=0.7)
    ax.set_xlabel("$e_{ni}$ (gamma-based)")
    ax.set_ylabel("Density")
    ax.set_title("Gamma $e_{ni}$ across datasets (normalised)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Boxplot of per-TCR mean gamma e
    ax = axes[1]
    data_for_box, labels = [], []
    for (name, d), c in zip(all_datasets.items(), colors):
        vals = d["per_tcr_mean_gamma"]
        vals = vals[vals > 0]
        if len(vals) > 0:
            data_for_box.append(vals)
            labels.append(name)
    if data_for_box:
        bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="red",
                                       markersize=5))
        for patch, c in zip(bp["boxes"], colors[:len(data_for_box)]):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.axhline(1.0, color="red", linestyle="--", lw=1, alpha=0.7,
                    label="$e=1$")
        ax.set_ylabel("Per-TCR mean gamma $e_{ni}$")
        ax.set_title("Per-TCR mean gamma overlap across datasets")
        ax.legend()
        if len(labels) > 4:
            ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_path, "gamma_e_ni_combined.png"),
                dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(output_path, "gamma_e_ni_combined.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: breakdown bars (e<1, 1<=e<2, e>=2) ---
    fig2, ax2 = plt.subplots(figsize=(max(8, 2.5 * n_datasets), 5))
    names = list(all_datasets.keys())
    frac_lt1, frac_1_2, frac_ge2 = [], [], []
    for name in names:
        e = all_datasets[name]["all_e_gamma"]
        n = max(len(e), 1)
        frac_lt1.append((e < 1.0).sum() / n * 100)
        frac_1_2.append(((e >= 1.0) & (e < 2.0)).sum() / n * 100)
        frac_ge2.append((e >= 2.0).sum() / n * 100)
    x = np.arange(len(names))
    w = 0.25
    ax2.bar(x - w, frac_lt1, w, label="$e<1$", color="#E24A33", edgecolor="white")
    ax2.bar(x, frac_1_2, w, label="$1 \\leq e < 2$", color="#55A868",
            edgecolor="white")
    ax2.bar(x + w, frac_ge2, w, label="$e \\geq 2$", color="#4C72B0",
            edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.set_ylabel("% of TCR–donor pairs")
    ax2.set_title("Gamma $e_{ni}$ breakdown across datasets")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig2.savefig(os.path.join(output_path, "gamma_e_ni_breakdown.png"),
                 dpi=150, bbox_inches="tight")
    fig2.savefig(os.path.join(output_path, "gamma_e_ni_breakdown.pdf"),
                 bbox_inches="tight")
    plt.close(fig2)

    # --- Figure 3: correlation summary (mean gamma e vs mean true e per dataset) ---
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for (name, d), c in zip(all_datasets.items(), colors):
        s = d["stats"]
        if s.get("total_pairs", 0) == 0:
            continue
        ax3.scatter(s["e_true_mean"], s["e_gamma_mean"],
                    s=80, color=c, edgecolor="black", lw=0.5, zorder=3)
        ax3.annotate(name, (s["e_true_mean"], s["e_gamma_mean"]),
                     fontsize=7, ha="left", va="bottom")
    all_vals = [d["stats"]["e_true_mean"] for d in all_datasets.values()
                if d["stats"].get("total_pairs", 0) > 0]
    all_vals += [d["stats"]["e_gamma_mean"] for d in all_datasets.values()
                 if d["stats"].get("total_pairs", 0) > 0]
    if all_vals:
        lim = max(all_vals) * 1.1
        ax3.plot([0, lim], [0, lim], "r--", lw=1.5, alpha=0.7, label="$y=x$")
        ax3.set_xlim(0, lim)
        ax3.set_ylim(0, lim)
    ax3.set_xlabel("Dataset mean ground-truth $e_{ni}$")
    ax3.set_ylabel("Dataset mean gamma $e_{ni}$")
    ax3.set_title("Per-Dataset Mean: Gamma vs Ground-Truth")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3.savefig(os.path.join(output_path, "gamma_vs_true_summary.png"),
                 dpi=150, bbox_inches="tight")
    fig3.savefig(os.path.join(output_path, "gamma_vs_true_summary.pdf"),
                 bbox_inches="tight")
    plt.close(fig3)

    print(f"  Combined plots saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Single / multi dataset runners
# ═══════════════════════════════════════════════════════════════════
def run_single(results_dir, data_dir, donor_matrix, output_path,
               pad_token, max_e_plot, n_bins, name=None):
    if name:
        print(f"\n{'='*60}\nDataset: {name}\n{'='*60}")

    data = load_gamma_and_data(results_dir, data_dir, donor_matrix, pad_token)
    all_e_gamma, all_e_true, per_tcr_mean_gamma, per_tcr_mean_true = \
        compute_e_ni_gamma(data, pad_token)
    label = name or Path(results_dir).name

    # Stats
    stats = {}
    if len(all_e_gamma) > 0:
        stats = {
            "name": label,
            "num_tcrs": int(data["num_tcrs"]),
            "num_donors": int(data["num_donors"]),
            "num_alleles": int(data["num_alleles"]),
            "total_pairs": int(len(all_e_gamma)),
            # Gamma-based
            "e_gamma_mean": float(all_e_gamma.mean()),
            "e_gamma_median": float(np.median(all_e_gamma)),
            "e_gamma_std": float(all_e_gamma.std()),
            "frac_gamma_lt_1": float(np.mean(all_e_gamma < 1.0)),
            "frac_gamma_1_to_2": float(np.mean(
                (all_e_gamma >= 1.0) & (all_e_gamma < 2.0))),
            "frac_gamma_ge_2": float(np.mean(all_e_gamma >= 2.0)),
            "per_tcr_mean_gamma_median": float(np.median(per_tcr_mean_gamma)),
            "per_tcr_mean_gamma_mean": float(np.mean(per_tcr_mean_gamma)),
            # Ground-truth (for comparison)
            "e_true_mean": float(all_e_true.mean()),
            "e_true_median": float(np.median(all_e_true)),
            "frac_true_eq_0": float(np.mean(all_e_true == 0)),
            "frac_true_eq_1": float(np.mean(all_e_true == 1)),
            "frac_true_ge_2": float(np.mean(all_e_true >= 2)),
        }
        print(f"  Total (TCR, pos-donor) pairs: {stats['total_pairs']}")
        print(f"  Gamma e_ni: mean={stats['e_gamma_mean']:.3f}, "
              f"median={stats['e_gamma_median']:.3f}")
        print(f"  True  e_ni: mean={stats['e_true_mean']:.3f}, "
              f"median={stats['e_true_median']:.1f}")
        print(f"  Gamma fractions: e<1={stats['frac_gamma_lt_1']:.3f}, "
              f"1<=e<2={stats['frac_gamma_1_to_2']:.3f}, "
              f"e>=2={stats['frac_gamma_ge_2']:.3f}")
    else:
        print("  WARNING: no valid (TCR, donor) pairs found.")
        stats = {"name": label, "total_pairs": 0}

    plot_single_dataset(all_e_gamma, all_e_true, per_tcr_mean_gamma,
                        per_tcr_mean_true, label, output_path,
                        max_e_plot, n_bins)

    stats_path = os.path.join(output_path, "gamma_e_ni_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, cls=NumpyEncoder)
    print(f"  Stats saved to {stats_path}")

    return {
        "all_e_gamma": all_e_gamma,
        "all_e_true": all_e_true,
        "per_tcr_mean_gamma": per_tcr_mean_gamma,
        "per_tcr_mean_true": per_tcr_mean_true,
        "stats": stats,
    }


def run_multiple(configs, output_dir, pad_token, max_e_plot, n_bins):
    all_datasets = {}
    all_stats = {}

    for cfg in configs:
        name = cfg["name"]
        out = os.path.join(output_dir, name)
        try:
            result = run_single(
                cfg["results_dir"], cfg["data_dir"], cfg["donor_matrix"],
                out, pad_token, max_e_plot, n_bins, name=name)
            all_datasets[name] = result
            all_stats[name] = result["stats"]
        except Exception as e:
            print(f"  ERROR processing {name}: {e}")
            import traceback; traceback.print_exc()
            all_stats[name] = {"error": str(e)}

    if all_datasets:
        plot_combined(all_datasets, output_dir, max_e_plot, n_bins)

    combined_path = os.path.join(output_dir, "gamma_e_ni_all_stats.json")
    with open(combined_path, "w") as f:
        json.dump(all_stats, f, indent=2, cls=NumpyEncoder)
    print(f"\nCombined stats saved to {combined_path}")
    return all_datasets


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.results_dir:
        # --- Single result mode ---
        if not args.data_dir:
            raise ValueError("--data_dir is required when using --results_dir")
        if not args.donor_matrix:
            raise ValueError("--donor_matrix is required when using --results_dir")
        run_single(args.results_dir, args.data_dir, args.donor_matrix,
                   args.output_dir, args.pad_token, args.max_e_plot, args.n_bins)

    elif args.results_base_dir:
        # --- Auto-discover mode ---
        if not args.data_base_dir:
            raise ValueError(
                "--data_base_dir is required when using --results_base_dir")
        if not args.donor_matrix:
            raise ValueError(
                "--donor_matrix is required when using --results_base_dir")
        print(f"Auto-discovering results under {args.results_base_dir} ...")
        print(f"Matching data dirs under {args.data_base_dir} ...")
        configs = discover_configs(
            args.results_base_dir, args.data_base_dir, args.donor_matrix)
        if not configs:
            print("ERROR: No valid (results_dir, data_dir) pairs found.")
            sys.exit(1)
        print(f"Found {len(configs)} datasets:")
        for cfg in configs:
            print(f"  {cfg['name']}: results={cfg['results_dir']}, "
                  f"data={cfg['data_dir']}")
        # Save config for reproducibility
        config_csv_path = os.path.join(args.output_dir,
                                        "auto_discovered_config.csv")
        with open(config_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["name", "results_dir", "data_dir", "donor_matrix"])
            writer.writeheader()
            writer.writerows(configs)
        print(f"Config saved to {config_csv_path}\n")
        run_multiple(configs, args.output_dir, args.pad_token,
                     args.max_e_plot, args.n_bins)

    elif args.df:
        # --- Explicit config file mode ---
        config_df = load_config_file(args.df)
        configs = []
        for idx, row in config_df.iterrows():
            configs.append({
                "name": row.get("name", f"dataset_{idx}"),
                "results_dir": row["results_dir"],
                "data_dir": row["data_dir"],
                "donor_matrix": row["donor_matrix"],
            })
        run_multiple(configs, args.output_dir, args.pad_token,
                     args.max_e_plot, args.n_bins)

    print("\nDone!")


if __name__ == "__main__":
    main()