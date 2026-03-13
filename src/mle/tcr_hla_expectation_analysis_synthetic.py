#!/usr/bin/env python3
"""
TCR-HLA Expectation Analysis for Synthetic Data.

For each TCR i in each positive donor n, computes:
    e_{ni} = sum_a  z_{ia} * x_{na}
where z_{ia} is the ground-truth binary binding label and x_{na} is the
donor HLA binary matrix. Visualises the distribution of e_{ni} to check
whether it is peaked at 1 (i.e. most TCR-donor pairs are explained by
exactly one shared HLA).

Usage:
    # Single dataset:
    python tcr_hla_expectation_analysis_synthetic.py \
        --data_dir /path/to/data \
        --donor_matrix /path/to/donor_hla_matrix.npz \
        --output_dir /path/to/output

    # Auto-discover all bX/nY/N* datasets under a base directory:
    python tcr_hla_expectation_analysis_synthetic.py \
        --base_dir data/autotcr/synthetic/binder_set \
        --donor_matrix data/autotcr/donor_hla_matrix.npz \
        --output_dir /path/to/output

    # Multiple datasets from explicit config file:
    python tcr_hla_expectation_analysis_synthetic.py \
        --df /path/to/config.csv \
        --output_dir /path/to/output
"""
import os, sys, json, re, argparse
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
        description="TCR-HLA ground-truth expectation analysis (e_ni distribution)")
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--data_dir", type=str,
                     help="Path to a single synthetic dataset directory")
    inp.add_argument("--base_dir", type=str,
                     help="Root dir containing bX/nY/N*/ datasets (auto-discover)")
    inp.add_argument("--df", type=str,
                     help="Path to CSV/TSV/JSON config with multiple datasets")
    p.add_argument("--donor_matrix", type=str,
                   help="Path to donor HLA matrix (.npz). "
                        "Required with --data_dir or --base_dir")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for figures and stats")
    p.add_argument("--pad_token", type=float, default=-1.0,
                   help="Padding token value (default: -1.0)")
    p.add_argument("--max_e_plot", type=int, default=20,
                   help="Upper x-axis limit for e_ni histograms (default: 20)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Auto-discovery of datasets under base_dir
# ═══════════════════════════════════════════════════════════════════
_BN_PATTERN = re.compile(r"^[bn]\d+$", re.IGNORECASE)


def _is_bn_dir(name: str) -> bool:
    """Return True if directory name matches bX or nX pattern."""
    return _BN_PATTERN.match(name) is not None


def _natural_sort_key(name: str):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", name)]


def discover_datasets(base_dir: str, donor_matrix: str):
    """
    Walk base_dir looking for synthetic_binder_sets.npy, but only descend
    into directories whose names match bX or nX (case-insensitive).
    Everything else (processed/, json files, etc.) is ignored.

    Returns a list of dicts compatible with the --df config format:
        [{"name": "b200_n50_N100000", "data_dir": "...", "donor_matrix": "..."}, ...]
    """
    base = Path(base_dir)
    datasets = []

    def _walk(current: Path, path_parts: tuple):
        if not current.is_dir():
            return
        # If this directory contains synthetic_binder_sets.npy, register it
        if (current / "synthetic_binder_sets.npy").exists():
            name = "_".join(path_parts) if path_parts else current.name
            datasets.append({
                "name": name,
                "data_dir": str(current),
                "donor_matrix": donor_matrix,
            })
            # Don't return — there might be deeper datasets too (unlikely but safe)

        # Recurse only into children that match bX/nX pattern or NX pattern
        for child in sorted(current.iterdir(), key=lambda p: _natural_sort_key(p.name)):
            if not child.is_dir():
                continue
            # Accept bX, nX at top levels; also accept NX (e.g. N100000)
            # at deeper levels
            child_name = child.name
            if _is_bn_dir(child_name) or re.match(r"^N\d+$", child_name):
                _walk(child, path_parts + (child_name,))

    _walk(base, ())
    datasets.sort(key=lambda d: _natural_sort_key(d["name"]))
    return datasets


# ═══════════════════════════════════════════════════════════════════
# Config loader (mirrors mle_pipeline_dense.py)
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
    missing = [c for c in ("data_dir", "donor_matrix") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


# ═══════════════════════════════════════════════════════════════════
# Data loading (same logic as mle_pipeline_dense.load_data)
# ═══════════════════════════════════════════════════════════════════
def load_data(data_dir, donor_matrix_path, pad_token=-1.0):
    data_dir = Path(data_dir)
    print(f"Loading data from {data_dir} ...")
    true_hla_set = np.load(data_dir / "synthetic_binder_sets.npy")
    donor_indices = np.load(data_dir / "synthetic_donor_indices.npy")
    donor_hla_matrix = np.load(donor_matrix_path)["donor_hla_matrix"]
    num_tcrs = true_hla_set.shape[0]
    num_donors = donor_hla_matrix.shape[0]
    num_alleles = donor_hla_matrix.shape[1]
    print(f"  {num_tcrs} TCRs, {num_donors} donors, {num_alleles} alleles")
    return {
        "true_hla_set": np.asarray(true_hla_set),
        "donor_indices": np.asarray(donor_indices),
        "donor_hla_matrix": donor_hla_matrix,
        "num_tcrs": num_tcrs,
        "num_donors": num_donors,
        "num_alleles": num_alleles,
    }


# ═══════════════════════════════════════════════════════════════════
# Core computation
# ═══════════════════════════════════════════════════════════════════
def compute_e_ni(data, pad_token=-1.0):
    """
    For every (TCR i, positive donor n) pair compute:
        e_{ni} = sum_a z_{ia} * x_{na}

    Returns
    -------
    all_e : np.ndarray, 1-D
        Flat array of e_ni values for every valid (TCR, positive-donor) pair.
    per_tcr_mean_e : np.ndarray, shape (num_tcrs,)
        Mean e_ni across positive donors of each TCR.
    per_tcr_max_e : np.ndarray, shape (num_tcrs,)
        Max e_ni across positive donors of each TCR.
    """
    true_hla_set = data["true_hla_set"]
    donor_indices = data["donor_indices"]
    donor_hla_matrix = data["donor_hla_matrix"]  # (D, A)
    num_tcrs = data["num_tcrs"]
    num_alleles = data["num_alleles"]

    # Build dense ground-truth z matrix: (num_tcrs, num_alleles)
    z = np.zeros((num_tcrs, num_alleles), dtype=np.float32)
    for i in range(num_tcrs):
        valid = true_hla_set[i] >= 0
        allele_ids = true_hla_set[i][valid].astype(int)
        z[i, allele_ids] = 1.0

    all_e = []
    per_tcr_mean_e = np.zeros(num_tcrs, dtype=np.float64)
    per_tcr_max_e = np.zeros(num_tcrs, dtype=np.float64)

    pad_int = int(pad_token)
    for i in range(num_tcrs):
        pos_donors = donor_indices[i]
        valid_mask = pos_donors != pad_int
        pos_donors_valid = pos_donors[valid_mask]
        if len(pos_donors_valid) == 0:
            continue
        # x_{na} for positive donors: (n_pos, A)
        x_pos = donor_hla_matrix[pos_donors_valid]
        # e_{ni} = z_i . x_n  for each positive donor n
        e_vals = x_pos @ z[i]  # (n_pos,)
        all_e.append(e_vals)
        per_tcr_mean_e[i] = e_vals.mean()
        per_tcr_max_e[i] = e_vals.max()

    all_e = np.concatenate(all_e).astype(int) if all_e else np.array([], dtype=int)
    return all_e, per_tcr_mean_e, per_tcr_max_e


# ═══════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════
def plot_single_dataset(all_e, per_tcr_mean_e, name, output_path, max_e_plot=20):
    """Three-panel figure for one dataset."""
    os.makedirs(output_path, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # --- Panel 1: histogram of e_ni (all pairs) ---
    ax = axes[0]
    max_val = int(all_e.max()) if len(all_e) else 0
    bins = np.arange(-0.5, min(max_val, max_e_plot) + 1.5, 1)
    counts, _, patches = ax.hist(all_e, bins=bins, edgecolor="white",
                                  color="#4C72B0", alpha=0.85)
    for patch, left_edge in zip(patches, bins[:-1]):
        if int(round(left_edge + 0.5)) == 1:
            patch.set_facecolor("#E24A33")
    ax.set_xlabel("$e_{ni} = \\sum_a z_{ia}\\, x_{na}$")
    ax.set_ylabel("Count (TCR–donor pairs)")
    ax.set_title(f"Distribution of $e_{{ni}}$\n{name}")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if len(all_e) > 0:
        frac_eq1 = np.mean(all_e == 1) * 100
        frac_ge1 = np.mean(all_e >= 1) * 100
        frac_eq0 = np.mean(all_e == 0) * 100
        ax.text(0.97, 0.95,
                f"$e=0$: {frac_eq0:.1f}%\n$e=1$: {frac_eq1:.1f}%\n$e\\geq1$: {frac_ge1:.1f}%\n"
                f"mean: {all_e.mean():.2f}\nmedian: {np.median(all_e):.1f}",
                transform=ax.transAxes, va="top", ha="right",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="wheat", alpha=0.5))
    ax.grid(True, alpha=0.3)

    # --- Panel 2: histogram of per-TCR mean e ---
    ax = axes[1]
    ax.hist(per_tcr_mean_e[per_tcr_mean_e > 0], bins=50, edgecolor="white",
            color="#55A868", alpha=0.85)
    ax.set_xlabel("Mean $e_{ni}$ across positive donors")
    ax.set_ylabel("Count (TCRs)")
    ax.set_title(f"Per-TCR mean $e_{{ni}}$\n{name}")
    ax.axvline(1.0, color="red", linestyle="--", lw=1.5, label="$e=1$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 3: fraction of e_ni == k, stacked ---
    ax = axes[2]
    if len(all_e) > 0:
        max_k_bar = min(int(all_e.max()), max_e_plot)
        k_vals = np.arange(0, max_k_bar + 1)
        fracs = np.array([(all_e == k).sum() / len(all_e) * 100 for k in k_vals])
        colors = ["#E24A33" if k == 1 else "#4C72B0" for k in k_vals]
        ax.bar(k_vals, fracs, color=colors, edgecolor="white")
        ax.set_xlabel("$e_{ni}$")
        ax.set_ylabel("Percentage of pairs (%)")
        ax.set_title(f"Fraction per $e_{{ni}}$ value\n{name}")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_path, f"e_ni_distribution.png"),
                dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(output_path, f"e_ni_distribution.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Plots saved to {output_path}")


def plot_combined(all_datasets, output_path, max_e_plot=20):
    """Combined overlay figure across all datasets."""
    os.makedirs(output_path, exist_ok=True)
    n_datasets = len(all_datasets)
    if n_datasets == 0:
        return

    # --- Figure 1: overlaid histograms of e_ni ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.cm.tab10 if n_datasets <= 10 else plt.cm.tab20
    colors = [cmap(i / max(n_datasets - 1, 1)) for i in range(n_datasets)]

    ax = axes[0]
    global_max = max(int(d["all_e"].max()) if len(d["all_e"]) else 0
                     for d in all_datasets.values())
    bins = np.arange(-0.5, min(global_max, max_e_plot) + 1.5, 1)
    for (name, d), c in zip(all_datasets.items(), colors):
        if len(d["all_e"]) == 0:
            continue
        ax.hist(d["all_e"], bins=bins, alpha=0.5, label=name, color=c,
                edgecolor="white", density=True)
    ax.set_xlabel("$e_{ni}$")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of $e_{ni}$ (all datasets, normalised)")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Figure 2: box/violin of per-TCR mean e ---
    ax = axes[1]
    data_for_box = []
    labels = []
    for name, d in all_datasets.items():
        vals = d["per_tcr_mean_e"]
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
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.axhline(1.0, color="red", linestyle="--", lw=1, alpha=0.7,
                    label="$e=1$")
        ax.set_ylabel("Mean $e_{ni}$ per TCR")
        ax.set_title("Per-TCR mean overlap across datasets")
        ax.legend()
        if len(labels) > 4:
            ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_path, "e_ni_combined.png"),
                dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(output_path, "e_ni_combined.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: summary bar chart (frac e==0, e==1, e>=2) per dataset ---
    fig2, ax2 = plt.subplots(figsize=(max(8, 2.5 * n_datasets), 5))
    names = list(all_datasets.keys())
    frac_0 = []
    frac_1 = []
    frac_ge2 = []
    for name in names:
        e = all_datasets[name]["all_e"]
        n = max(len(e), 1)
        frac_0.append((e == 0).sum() / n * 100)
        frac_1.append((e == 1).sum() / n * 100)
        frac_ge2.append((e >= 2).sum() / n * 100)
    x = np.arange(len(names))
    w = 0.25
    ax2.bar(x - w, frac_0, w, label="$e=0$ (unexplained)", color="#E24A33",
            edgecolor="white")
    ax2.bar(x, frac_1, w, label="$e=1$ (single HLA)", color="#55A868",
            edgecolor="white")
    ax2.bar(x + w, frac_ge2, w, label="$e \\geq 2$ (multi-HLA)", color="#4C72B0",
            edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.set_ylabel("% of TCR–donor pairs")
    ax2.set_title("Breakdown of $e_{ni}$ across datasets")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig2.savefig(os.path.join(output_path, "e_ni_breakdown.png"),
                 dpi=150, bbox_inches="tight")
    fig2.savefig(os.path.join(output_path, "e_ni_breakdown.pdf"),
                 bbox_inches="tight")
    plt.close(fig2)
    print(f"  Combined plots saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Single / multi dataset runners
# ═══════════════════════════════════════════════════════════════════
def run_single(data_dir, donor_matrix, output_path, pad_token, max_e_plot, name=None):
    if name:
        print(f"\n{'='*60}\nDataset: {name}\n{'='*60}")
    data = load_data(data_dir, donor_matrix, pad_token)
    all_e, per_tcr_mean_e, per_tcr_max_e = compute_e_ni(data, pad_token)
    label = name or Path(data_dir).name

    stats = {}
    if len(all_e) > 0:
        stats = {
            "name": label,
            "num_tcrs": int(data["num_tcrs"]),
            "num_donors": int(data["num_donors"]),
            "num_alleles": int(data["num_alleles"]),
            "total_pairs": int(len(all_e)),
            "e_mean": float(all_e.mean()),
            "e_median": float(np.median(all_e)),
            "e_std": float(all_e.std()),
            "frac_e_eq_0": float((all_e == 0).mean()),
            "frac_e_eq_1": float((all_e == 1).mean()),
            "frac_e_ge_1": float((all_e >= 1).mean()),
            "frac_e_ge_2": float((all_e >= 2).mean()),
            "per_tcr_mean_e_median": float(np.median(per_tcr_mean_e)),
            "per_tcr_mean_e_mean": float(np.mean(per_tcr_mean_e)),
        }
        print(f"  Total (TCR, pos-donor) pairs: {stats['total_pairs']}")
        print(f"  e_ni: mean={stats['e_mean']:.2f}, median={stats['e_median']:.1f}")
        print(f"  Frac e=0: {stats['frac_e_eq_0']:.3f}, "
              f"e=1: {stats['frac_e_eq_1']:.3f}, "
              f"e>=2: {stats['frac_e_ge_2']:.3f}")
    else:
        print("  WARNING: no valid (TCR, donor) pairs found.")
        stats = {"name": label, "total_pairs": 0}

    plot_single_dataset(all_e, per_tcr_mean_e, label, output_path, max_e_plot)

    stats_path = os.path.join(output_path, "e_ni_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, cls=NumpyEncoder)
    print(f"  Stats saved to {stats_path}")

    return {"all_e": all_e, "per_tcr_mean_e": per_tcr_mean_e,
            "per_tcr_max_e": per_tcr_max_e, "stats": stats}


def run_multiple(dataset_configs, output_dir, pad_token, max_e_plot):
    """
    Run analysis for a list of dataset configs.

    Parameters
    ----------
    dataset_configs : list[dict]
        Each dict has keys: name, data_dir, donor_matrix.
    """
    all_datasets = {}
    all_stats = {}

    for cfg in dataset_configs:
        name = cfg["name"]
        data_dir = cfg["data_dir"]
        donor_matrix = cfg["donor_matrix"]
        out = os.path.join(output_dir, name)
        try:
            result = run_single(data_dir, donor_matrix, out,
                                pad_token, max_e_plot, name=name)
            all_datasets[name] = result
            all_stats[name] = result["stats"]
        except Exception as e:
            print(f"  ERROR processing {name}: {e}")
            all_stats[name] = {"error": str(e)}

    # Combined plots
    if all_datasets:
        plot_combined(all_datasets, output_dir, max_e_plot)

    # Save combined stats
    combined_path = os.path.join(output_dir, "e_ni_all_stats.json")
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

    if args.data_dir:
        # --- Single dataset mode ---
        if not args.donor_matrix:
            raise ValueError("--donor_matrix is required when using --data_dir")
        run_single(args.data_dir, args.donor_matrix, args.output_dir,
                   args.pad_token, args.max_e_plot)

    elif args.base_dir:
        # --- Auto-discover mode ---
        if not args.donor_matrix:
            raise ValueError("--donor_matrix is required when using --base_dir")
        print(f"Auto-discovering datasets under {args.base_dir} ...")
        configs = discover_datasets(args.base_dir, args.donor_matrix)
        if not configs:
            print("ERROR: No datasets found. Expected bX/nY/N*/ structure "
                  "with synthetic_binder_sets.npy inside.")
            sys.exit(1)
        print(f"Found {len(configs)} datasets:")
        for cfg in configs:
            print(f"  {cfg['name']}: {cfg['data_dir']}")
        # Save the auto-generated config for reproducibility
        config_csv_path = os.path.join(args.output_dir, "auto_discovered_config.csv")
        import csv
        with open(config_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "data_dir", "donor_matrix"])
            writer.writeheader()
            writer.writerows(configs)
        print(f"Config saved to {config_csv_path}\n")
        run_multiple(configs, args.output_dir, args.pad_token, args.max_e_plot)

    elif args.df:
        # --- Explicit config file mode ---
        import pandas as pd
        config_df = load_config_file(args.df)
        configs = []
        for _, row in config_df.iterrows():
            configs.append({
                "name": row.get("name", f"dataset_{_}"),
                "data_dir": row["data_dir"],
                "donor_matrix": row["donor_matrix"],
            })
        run_multiple(configs, args.output_dir, args.pad_token, args.max_e_plot)

    print("\nDone!")


if __name__ == "__main__":
    main()