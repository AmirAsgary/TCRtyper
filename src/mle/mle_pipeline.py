#!/usr/bin/env python3
"""
TCR-HLA Binding Model Training and Analysis Pipeline.

This script trains a probabilistic model to predict TCR-HLA binding 
using maximum likelihood estimation and performs downstream analysis.

Usage:
    # Single dataset:
    python pipeline.py --data_dir /path/to/data --donor_matrix /path/to/donor_hla_matrix.npz --output_dir /path/to/output

    # Multiple datasets from file:
    python pipeline.py --df /path/to/config.csv --output_dir /path/to/output

    # With all analyses:
    python pipeline.py --data_dir /path/to/data --donor_matrix /path/to/donor_hla_matrix.npz --output_dir /path/to/output --analyze_all
"""
import os, sys, json, argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd

# Import local utilities
from utils import (
    SparseTCRModel, create_dataset, pad_list_to_array, NumpyEncoder,
    assess_explanation_for_donors, analyze_model_predictions,
    evaluate_model_performance, compute_precision_at_k, 
    plot_precision_at_k_heatmap, plot_precision_at_k_curves, save_metrics_json,
    PublicTcrHlaCsrReader, PublicTcrHlaCsrReaderChunk  # add the new reader
)

# ---------- Macro-averaging helpers ----------
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt


def compute_macro_metrics(model, binder_sets, true_hla_set, num_tcrs,
                          num_alleles, pad_token=-1.0):
    """Compute per-TCR ROC-AUC and AP (macro-averaged) in two evaluation modes.

    **full_space** — all ``num_alleles`` are evaluated; non-candidate alleles
    receive prob = 0.  This penalises the model for true positives that were
    never among its candidates (they are tied with non-candidate negatives).

    **candidate_only** — only candidate alleles (those in ``binder_sets``)
    are evaluated per TCR.  This measures how well the model *ranks within
    its candidate set*, independent of candidate selection quality.

    Returns
    -------
    dict  with keys (prefixed by scope ``full_`` or ``cand_``):
        per_tcr_auc_full, per_tcr_ap_full, valid_tcr_indices_full
        per_tcr_auc_cand, per_tcr_ap_cand, valid_tcr_indices_cand
        median_auc_full … / median_auc_cand …   (+ mean, std, q25, q75)
        num_valid_tcrs_full, num_valid_tcrs_cand
        candidate_recall  — fraction of TPs that are in the candidate set
    """
    # Extract candidate probabilities (same as evaluate_model_performance)
    z_logits = model.z_embedding.get_weights()[0]            # (num_tcrs, max_hlas)
    candidate_probs = 1.0 / (1.0 + np.exp(-z_logits))       # sigmoid, numpy
    valid_candidate_mask = (binder_sets != pad_token)         # (num_tcrs, max_hlas)

    # Accumulators for both modes
    full_auc,  full_ap,  full_idx  = [], [], []
    cand_auc,  cand_ap,  cand_idx  = [], [], []
    total_tp, tp_in_candidates = 0, 0

    for i in range(num_tcrs):
        # Ground-truth alleles
        true_alleles = np.asarray(true_hla_set[i])
        true_alleles = true_alleles[true_alleles >= 0].astype(int)
        if len(true_alleles) == 0:
            continue
        true_set = set(true_alleles)
        total_tp += len(true_set)

        # Candidate info for this TCR
        cand_mask_i = valid_candidate_mask[i]
        cand_ids = binder_sets[i][cand_mask_i].astype(int)
        cand_probs_i = candidate_probs[i][cand_mask_i]
        cand_set = set(cand_ids)
        tp_in_candidates += len(true_set & cand_set)

        # ── Full-space evaluation ──
        if len(true_alleles) < num_alleles:          # need both classes
            y_true_full = np.zeros(num_alleles, dtype=np.float32)
            y_pred_full = np.zeros(num_alleles, dtype=np.float32)
            y_true_full[true_alleles] = 1.0
            y_pred_full[cand_ids] = cand_probs_i

            full_auc.append(roc_auc_score(y_true_full, y_pred_full))
            full_ap.append(average_precision_score(y_true_full, y_pred_full))
            full_idx.append(i)

        # ── Candidate-only evaluation ──
        n_cand = len(cand_ids)
        if n_cand >= 2:
            y_true_cand = np.array(
                [1.0 if aid in true_set else 0.0 for aid in cand_ids],
                dtype=np.float32,
            )
            n_pos_cand = int(y_true_cand.sum())
            n_neg_cand = n_cand - n_pos_cand
            if n_pos_cand > 0 and n_neg_cand > 0:
                cand_auc.append(roc_auc_score(y_true_cand, cand_probs_i))
                cand_ap.append(average_precision_score(y_true_cand, cand_probs_i))
                cand_idx.append(i)

    # Convert to arrays
    full_auc = np.asarray(full_auc)
    full_ap  = np.asarray(full_ap)
    cand_auc = np.asarray(cand_auc)
    cand_ap  = np.asarray(cand_ap)

    def _stat_block(auc_arr, ap_arr, suffix):
        out = {}
        for arr, name in [(auc_arr, "auc"), (ap_arr, "ap")]:
            if len(arr) == 0:
                for s in ("median", "mean", "std", "q25", "q75"):
                    out[f"{s}_{name}_{suffix}"] = 0.0
            else:
                out[f"median_{name}_{suffix}"] = float(np.median(arr))
                out[f"mean_{name}_{suffix}"]   = float(np.mean(arr))
                out[f"std_{name}_{suffix}"]    = float(np.std(arr))
                out[f"q25_{name}_{suffix}"]    = float(np.percentile(arr, 25))
                out[f"q75_{name}_{suffix}"]    = float(np.percentile(arr, 75))
        return out

    result = {
        # Raw arrays (for plots / downstream)
        "per_tcr_auc_full":        full_auc,
        "per_tcr_ap_full":         full_ap,
        "valid_tcr_indices_full":  np.array(full_idx),
        "num_valid_tcrs_full":     len(full_idx),
        "per_tcr_auc_cand":        cand_auc,
        "per_tcr_ap_cand":         cand_ap,
        "valid_tcr_indices_cand":  np.array(cand_idx),
        "num_valid_tcrs_cand":     len(cand_idx),
        # What fraction of ground-truth positives are candidates at all
        "candidate_recall":        tp_in_candidates / max(total_tp, 1),
        # Legacy aliases so old key names still work (point to full-space)
        "per_tcr_auc":             full_auc,
        "per_tcr_ap":              full_ap,
        "valid_tcr_indices":       np.array(full_idx),
        "num_valid_tcrs":          len(full_idx),
    }
    result.update(_stat_block(full_auc, full_ap, "full"))
    result.update(_stat_block(cand_auc, cand_ap, "cand"))
    # Legacy un-suffixed stats (point to full-space)
    result.update({
        "median_auc": result.get("median_auc_full", 0.0),
        "mean_auc":   result.get("mean_auc_full", 0.0),
        "std_auc":    result.get("std_auc_full", 0.0),
        "q25_auc":    result.get("q25_auc_full", 0.0),
        "q75_auc":    result.get("q75_auc_full", 0.0),
        "median_ap":  result.get("median_ap_full", 0.0),
        "mean_ap":    result.get("mean_ap_full", 0.0),
        "std_ap":     result.get("std_ap_full", 0.0),
        "q25_ap":     result.get("q25_ap_full", 0.0),
        "q75_ap":     result.get("q75_ap_full", 0.0),
    })
    return result


def plot_macro_metrics(macro_results, output_path):
    """Save distribution plots for per-TCR AUC and AP (both evaluation modes)."""

    def _plot_pair(auc_arr, ap_arr, med_auc, med_ap, label_suffix, fname_suffix):
        """Histogram + boxplot for one evaluation mode."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # AUC histogram
        ax = axes[0]
        ax.hist(auc_arr, bins=50, edgecolor="black", alpha=0.75, color="#4C72B0")
        ax.axvline(med_auc, color="red", linestyle="--", linewidth=1.5,
                   label=f"Median = {med_auc:.4f}")
        ax.set_xlabel("Per-TCR ROC-AUC")
        ax.set_ylabel("Count")
        ax.set_title(f"ROC-AUC Distribution ({label_suffix})")
        ax.legend()

        # AP histogram
        ax = axes[1]
        ax.hist(ap_arr, bins=50, edgecolor="black", alpha=0.75, color="#55A868")
        ax.axvline(med_ap, color="red", linestyle="--", linewidth=1.5,
                   label=f"Median = {med_ap:.4f}")
        ax.set_xlabel("Per-TCR Average Precision")
        ax.set_ylabel("Count")
        ax.set_title(f"AP Distribution ({label_suffix})")
        ax.legend()

        plt.tight_layout()
        fig_file = os.path.join(output_path, f"macro_metrics_distribution_{fname_suffix}.png")
        fig.savefig(fig_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Macro-metrics plot saved to: {fig_file}")

        # Boxplot
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        bp = ax2.boxplot([auc_arr, ap_arr], labels=["ROC-AUC", "Avg Precision"],
                         patch_artist=True, showmeans=True,
                         meanprops=dict(marker="D", markerfacecolor="red",
                                        markersize=6))
        colors = ["#4C72B0", "#55A868"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax2.set_ylabel("Score")
        ax2.set_title(f"Per-TCR Metric Distributions ({label_suffix})")
        ax2.set_ylim(-0.05, 1.05)
        fig2_file = os.path.join(output_path, f"macro_metrics_boxplot_{fname_suffix}.png")
        fig2.savefig(fig2_file, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Macro-metrics boxplot saved to: {fig2_file}")

    # Full-space plots
    if len(macro_results["per_tcr_auc_full"]) > 0:
        _plot_pair(
            macro_results["per_tcr_auc_full"],
            macro_results["per_tcr_ap_full"],
            macro_results["median_auc_full"],
            macro_results["median_ap_full"],
            "Full-Space", "full",
        )

    # Candidate-only plots
    if len(macro_results["per_tcr_auc_cand"]) > 0:
        _plot_pair(
            macro_results["per_tcr_auc_cand"],
            macro_results["per_tcr_ap_cand"],
            macro_results["median_auc_cand"],
            macro_results["median_ap_cand"],
            "Candidate-Only", "cand",
        )

    # Side-by-side comparison boxplot
    has_full = len(macro_results["per_tcr_auc_full"]) > 0
    has_cand = len(macro_results["per_tcr_auc_cand"]) > 0
    if has_full and has_cand:
        fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, metric, title in [
            (axes[0], "auc", "ROC-AUC"),
            (axes[1], "ap", "Average Precision"),
        ]:
            bp = ax.boxplot(
                [macro_results[f"per_tcr_{metric}_full"],
                 macro_results[f"per_tcr_{metric}_cand"]],
                labels=["Full-Space", "Candidate-Only"],
                patch_artist=True, showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
            )
            bp["boxes"][0].set_facecolor("#4C72B0")
            bp["boxes"][0].set_alpha(0.6)
            bp["boxes"][1].set_facecolor("#DD8452")
            bp["boxes"][1].set_alpha(0.6)
            ax.set_ylabel(title)
            ax.set_title(f"Per-TCR {title}")
            ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        fig3_file = os.path.join(output_path, "macro_metrics_comparison.png")
        fig3.savefig(fig3_file, dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"  Macro-metrics comparison plot saved to: {fig3_file}")


# ---------- Original pipeline code (unchanged except for macro integration) ----------


def parse_args():
    parser = argparse.ArgumentParser(description='TCR-HLA Binding Model Training Pipeline')
    ### Data input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--data_dir', type=str, help='Path to single dataset directory')
    input_group.add_argument('--df', type=str, help='Path to CSV/TSV/JSON file with multiple dataset configs')
    parser.add_argument('--donor_matrix', type=str, help='Path to donor HLA matrix (.npz). Required if --data_dir is used')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    ### Training hyperparameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size (default: 512)')
    parser.add_argument('--learning_rate', type=float, default=1.0, help='Learning rate (default: 0.01)')
    parser.add_argument('--l2_reg', type=float, default=1e-5, help='L2 regularization lambda (default: 1e-5)')
    parser.add_argument('--pad_token', type=float, default=-1.0, help='Padding token value (default: -1.0)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold (default: 0.5)')
    # MAP
    parser.add_argument('--beta', type=float, default=4.0, help='Beta hyperparameter (default: 4.0)')
    parser.add_argument('--alpha_0', type=float, default=1.0, help='Alpha_0 MAP hyperparameter (default: 1.0)')
    parser.add_argument('--alpha_1', type=float, default=2.5, help='Alpha_1 MAP hyperparameter (default: 2.5)')
    parser.add_argument('--alpha', type=float, default=2.0, help='Alpha MAP hyperparameter (default: 2.0)')
    parser.add_argument('--B', type=float, default=30.0, help='B MAP hyperparameter (default: 30.0)')
    ### Analysis flags
    parser.add_argument('--analyze_all', action='store_true', help='Run all analysis modules')
    parser.add_argument('--analyze_donors', action='store_true', help='Run donor explanation analysis')
    parser.add_argument('--analyze_predictions', action='store_true', help='Run model predictions analysis')
    parser.add_argument('--analyze_performance', action='store_true', help='Run PR/ROC performance analysis')
    parser.add_argument('--analyze_precision_k', action='store_true', help='Run Precision@k analysis')
    parser.add_argument('--analyze_macro', action='store_true',
                        help='Run macro-averaged per-TCR AUC/AP analysis')
    parser.add_argument('--max_k', type=int, default=20, help='Max k for Precision@k (default: 20)')
    # Other options
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level (default: 1)')

    return parser.parse_args()


def load_config_file(config_path):
    """Load dataset configurations from CSV, TSV, or JSON file.
    Expected columns: data_dir, donor_matrix, [optional: name, l2_reg, epochs, ...]
    """
    config_path = Path(config_path)
    ext = config_path.suffix.lower()
    if ext == '.csv':
        df = pd.read_csv(config_path)
    elif ext == '.tsv':
        df = pd.read_csv(config_path, sep='\t')
    elif ext == '.json':
        df = pd.read_json(config_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv, .tsv, or .json")
    # Validate required columns
    required = ['data_dir', 'donor_matrix']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in config file: {missing}")
    return df


def load_data(data_dir, donor_matrix_path, pad_token=-1.):
    """Load dataset and return all required arrays.
    Supports both old (y_counts) and new (clusters/counts) HDF5 layouts
    by trying PublicTcrHlaCsrReaderChunk first, then falling back to PublicTcrHlaCsrReader.
    """
    data_dir = Path(data_dir)
    print(f"Loading data from {data_dir}...")
    # Load binder sets (ground truth)
    true_hla_set = np.load(data_dir / "synthetic_binder_sets.npy", mmap_mode="r")
    # Load donor indices
    donor_indices = np.load(data_dir / "synthetic_donor_indices.npy", mmap_mode="r")
    # Load counts from H5 file — try new chunk format first, fall back to legacy
    h5_path = data_dir / 'synthetic_tcr_hla_counts.h5'
    try:
        with PublicTcrHlaCsrReaderChunk(str(h5_path)) as reader:
            counts_set, max_all = reader.read_sparse_indices_of_counts()
    except KeyError:
        # Fall back to legacy y_counts layout
        with PublicTcrHlaCsrReader(str(h5_path)) as reader:
            counts_set, max_all = reader.read_sparse_indices()
    # Pad variable-length lists into fixed-width array
    binder_sets = pad_list_to_array(counts_set, max_all, pad_token)
    # Load donor HLA matrix
    donor_hla_matrix = np.load(donor_matrix_path)['donor_hla_matrix']
    num_tcrs = binder_sets.shape[0]
    max_hlas_per_tcr = binder_sets.shape[1]
    num_alleles = donor_hla_matrix.shape[1]
    num_donors = donor_hla_matrix.shape[0]
    print(f"Dataset: {num_tcrs} TCRs, {num_donors} Donors, {num_alleles} Total Alleles, Max {max_hlas_per_tcr} HLAs/TCR")
    return {
        'binder_sets': binder_sets, 'donor_indices': np.array(donor_indices),
        'true_hla_set': np.array(true_hla_set), 'donor_hla_matrix': donor_hla_matrix,
        'num_tcrs': num_tcrs, 'max_hlas_per_tcr': max_hlas_per_tcr,
        'num_alleles': num_alleles, 'num_donors': num_donors
    }

def train_model(data, args, output_path):
    """Train the SparseTCRModel and return trained model + history."""
    os.makedirs(output_path, exist_ok=True)
    print(f"\n{'='*60}\nTraining Model\n{'='*60}")
    # Create dataset
    train_dataset = create_dataset(data['donor_indices'], args.batch_size)
    # Learning rate schedule
    steps_per_epoch = data['num_tcrs'] // args.batch_size
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=args.learning_rate,
        first_decay_steps=steps_per_epoch * 20, t_mul=2.0, m_mul=0.9, alpha=0.01)
    # Initialize model
    model = SparseTCRModel(
        num_tcrs=data['num_tcrs'], max_hlas_per_tcr=data['max_hlas_per_tcr'],
        donor_hla_matrix=data['donor_hla_matrix'], binder_sets=data['binder_sets'],
        beta=args.beta, mode='continuous', pad_token=args.pad_token, l2_reg_lambda=args.l2_reg,
        alpha_0=args.alpha_0, alpha_1=args.alpha_1, alpha=args.alpha, B=args.B)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer)
    # Train
    print("Starting training...")
    history = model.fit(train_dataset, epochs=args.epochs, verbose=args.verbose)
    # Save model and history
    model.save_weights(os.path.join(output_path, 'model.weights.h5'))
    with open(os.path.join(output_path, 'history.json'), 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    print(f"Model saved to: {output_path}")
    return model, history


def run_analysis(model, data, args, output_path):
    """Run all enabled analysis modules."""
    results = {}
    figures_path = os.path.join(output_path, "figures")
    os.makedirs(figures_path, exist_ok=True)
    # Donor explanation analysis
    if args.analyze_all or args.analyze_donors:
        print(f"\n{'='*60}\nDonor Explanation Analysis\n{'='*60}")
        donor_scores, donor_stats = assess_explanation_for_donors(
            model, data['donor_indices'], data['donor_hla_matrix'],
            output_path=figures_path, pad_token=args.pad_token)
        results['donor_stats'] = donor_stats
    else:
        donor_stats = {'mean_fraction_explained_t005': 0.0}
        results['donor_stats'] = donor_stats
    # Model predictions analysis
    if args.analyze_all or args.analyze_predictions:
        print(f"\n{'='*60}\nModel Predictions Analysis\n{'='*60}")
        analysis_results = analyze_model_predictions(
            model, data['binder_sets'], data['num_alleles'],
            threshold=args.threshold, output_path=figures_path, pad_token=args.pad_token)
        results['analysis'] = analysis_results
    else:
        analysis_results = {'coverage': 0.0, 'avg_hlas_per_tcr': 0.0}
        results['analysis'] = analysis_results
    # Performance evaluation (PR/ROC)
    if args.analyze_all or args.analyze_performance:
        print(f"\n{'='*60}\nPerformance Evaluation\n{'='*60}")
        perf_metrics = evaluate_model_performance(
            model=model, binder_sets=data['binder_sets'], true_hla_set=data['true_hla_set'],
            num_total_alleles=data['num_alleles'], output_path=figures_path, pad_token=args.pad_token)
        results['performance'] = perf_metrics
    else:
        perf_metrics = {'auc': 0.0, 'ap': 0.0, 'best_f1': 0.0}
        results['performance'] = perf_metrics

    # ---- Macro-averaged per-TCR metrics (NEW) ----
    if args.analyze_all or args.analyze_macro:
        print(f"\n{'='*60}\nMacro-Averaged Per-TCR Metrics\n{'='*60}")
        macro = compute_macro_metrics(
            model,
            binder_sets=data['binder_sets'],
            true_hla_set=data['true_hla_set'],
            num_tcrs=data['num_tcrs'],
            num_alleles=data['num_alleles'],
            pad_token=args.pad_token,
        )
        # Console report
        cand_recall = macro['candidate_recall']
        print(f"  Candidate recall (TP in candidates / total TP): {cand_recall:.4f}")
        print(f"\n  [Full-space]  ({macro['num_valid_tcrs_full']} TCRs)")
        print(f"    ROC-AUC  — median: {macro['median_auc_full']:.4f}  "
              f"mean: {macro['mean_auc_full']:.4f}  std: {macro['std_auc_full']:.4f}  "
              f"IQR: [{macro['q25_auc_full']:.4f}, {macro['q75_auc_full']:.4f}]")
        print(f"    Avg Prec — median: {macro['median_ap_full']:.4f}  "
              f"mean: {macro['mean_ap_full']:.4f}  std: {macro['std_ap_full']:.4f}  "
              f"IQR: [{macro['q25_ap_full']:.4f}, {macro['q75_ap_full']:.4f}]")
        print(f"\n  [Candidate-only]  ({macro['num_valid_tcrs_cand']} TCRs)")
        print(f"    ROC-AUC  — median: {macro['median_auc_cand']:.4f}  "
              f"mean: {macro['mean_auc_cand']:.4f}  std: {macro['std_auc_cand']:.4f}  "
              f"IQR: [{macro['q25_auc_cand']:.4f}, {macro['q75_auc_cand']:.4f}]")
        print(f"    Avg Prec — median: {macro['median_ap_cand']:.4f}  "
              f"mean: {macro['mean_ap_cand']:.4f}  std: {macro['std_ap_cand']:.4f}  "
              f"IQR: [{macro['q25_ap_cand']:.4f}, {macro['q75_ap_cand']:.4f}]")

        # Plots
        plot_macro_metrics(macro, figures_path)

        # Persist — serialisable subset (drop large numpy arrays)
        macro_serialisable = {k: v for k, v in macro.items()
                              if not isinstance(v, np.ndarray)}
        macro_json_path = os.path.join(output_path, "macro_metrics.json")
        with open(macro_json_path, "w") as f:
            json.dump(macro_serialisable, f, indent=2, cls=NumpyEncoder)
        print(f"  Macro-metrics JSON saved to: {macro_json_path}")

        # Also save the raw per-TCR arrays for downstream consumers
        np.savez_compressed(
            os.path.join(output_path, "macro_metrics_per_tcr.npz"),
            per_tcr_auc_full=macro["per_tcr_auc_full"],
            per_tcr_ap_full=macro["per_tcr_ap_full"],
            valid_tcr_indices_full=macro["valid_tcr_indices_full"],
            per_tcr_auc_cand=macro["per_tcr_auc_cand"],
            per_tcr_ap_cand=macro["per_tcr_ap_cand"],
            valid_tcr_indices_cand=macro["valid_tcr_indices_cand"],
        )

        results['macro'] = macro_serialisable
    else:
        results['macro'] = {}

    # Save final metrics summary (unchanged call — backward compatible)
    if args.analyze_all or args.analyze_donors or args.analyze_predictions or args.analyze_performance:
        save_metrics_json(output_path, perf_metrics, analysis_results, donor_stats, args.threshold)
    return results


def run_precision_at_k(output_path, data_dir, args):
    """Run Precision@k analysis (post-training)."""
    print(f"\n{'='*60}\nPrecision@k Analysis\n{'='*60}")
    try:
        results = compute_precision_at_k(output_path, data_dir, max_k=args.max_k, pad_token=args.pad_token)
        # Save results
        pk_path = os.path.join(output_path, "precision_at_k.json")
        with open(pk_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"Precision@k results saved to: {pk_path}")
        return results
    except Exception as e:
        print(f"Warning: Could not compute Precision@k: {e}")
        return None


def run_single_dataset(args, data_dir, donor_matrix, output_path, name=None):
    """Run the full pipeline for a single dataset."""
    if name:
        print(f"\n{'#'*80}\nProcessing: {name}\n{'#'*80}")
    # Load data
    data = load_data(data_dir, donor_matrix, args.pad_token)
    # Train model
    model, history = train_model(data, args, output_path)
    # Run analysis
    results = run_analysis(model, data, args, output_path)
    # Precision@k analysis (if enabled)
    if args.analyze_all or args.analyze_precision_k:
        pk_results = run_precision_at_k(output_path, data_dir, args)
        if pk_results:
            results['precision_at_k'] = pk_results
    # Summary
    print(f"\n{'='*60}\nPipeline Complete\n{'='*60}")
    print(f"Results saved to: {output_path}")
    return results


def run_multiple_datasets(args):
    """Run pipeline for multiple datasets from config file."""
    config_df = load_config_file(args.df)
    all_results = {}
    for idx, row in config_df.iterrows():
        name = row.get('name', f'dataset_{idx}')
        data_dir = row['data_dir']
        donor_matrix = row['donor_matrix']
        # Override args with row-specific values if present
        row_args = argparse.Namespace(**vars(args))
        for col in ['l2_reg', 'epochs', 'batch_size', 'learning_rate', 'beta']:
            if col in row and pd.notna(row[col]):
                setattr(row_args, col, row[col])
        output_path = os.path.join(args.output_dir, name)
        try:
            results = run_single_dataset(row_args, data_dir, donor_matrix, output_path, name=name)
            all_results[name] = results
        except Exception as e:
            print(f"Error processing {name}: {e}")
            all_results[name] = {'error': str(e)}
    # Save summary of all results
    summary_path = os.path.join(args.output_dir, 'all_results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nAll results summary saved to: {summary_path}")
    # Generate comparison plots if precision@k was run
    if args.analyze_all or args.analyze_precision_k:
        pk_results = {k: v.get('precision_at_k') for k, v in all_results.items() 
                      if isinstance(v, dict) and 'precision_at_k' in v}
        if pk_results:
            plot_precision_at_k_heatmap(pk_results, k_values=[1, 3, 5, 10],
                                        output_path=os.path.join(args.output_dir, 'precision_at_k_heatmap.png'))
            plot_precision_at_k_curves(pk_results, max_k=args.max_k,
                                       output_path=os.path.join(args.output_dir, 'precision_at_k_curves.png'))
    return all_results


def main():
    args = parse_args()
    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # Save config
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    # Run pipeline
    if args.df:
        results = run_multiple_datasets(args)
    else:
        if not args.donor_matrix:
            raise ValueError("--donor_matrix is required when using --data_dir")
        results = run_single_dataset(args, args.data_dir, args.donor_matrix, args.output_dir)
    print("\nDone!")
    return results


if __name__ == '__main__':
    main()