#!/usr/bin/env python3
"""
TCR-HLA Binding Model Training and Analysis Pipeline (Dense / Non-Sparse).

Uses NonSparseTCRModel which evaluates ALL HLA alleles (no candidate masking).
The MAP prior and likelihood push non-binding allele probabilities toward zero
rather than excluding them from the model entirely.

Usage:
    # Single dataset:
    python mle_pipeline_dense.py --data_dir /path/to/data --donor_matrix /path/to/donor_hla_matrix.npz --output_dir /path/to/output

    # Multiple datasets from file:
    python mle_pipeline_dense.py --df /path/to/config.csv --output_dir /path/to/output

    # With all analyses:
    python mle_pipeline_dense.py --data_dir /path/to/data --donor_matrix /path/to/donor_hla_matrix.npz --output_dir /path/to/output --analyze_all
"""
import os, sys, json, argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd

from utils import (
    create_dataset, pad_list_to_array, NumpyEncoder,
    save_metrics_json,
    PublicTcrHlaCsrReader, PublicTcrHlaCsrReaderChunk,
    NonSparseTCRModel
)

# ---------- Macro-averaging helpers ----------
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description='TCR-HLA Binding Model Training Pipeline (Dense / Non-Sparse)')
    ### Data input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--data_dir', type=str,
                             help='Path to single dataset directory')
    input_group.add_argument('--df', type=str,
                             help='Path to CSV/TSV/JSON file with multiple dataset configs')
    parser.add_argument('--donor_matrix', type=str,
                        help='Path to donor HLA matrix (.npz). Required if --data_dir is used')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    ### Training hyperparameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size (default: 512)')
    parser.add_argument('--learning_rate', type=float, default=1.0,
                        help='Learning rate (default: 1.0)')
    parser.add_argument('--pad_token', type=float, default=-1.0,
                        help='Padding token value for donor indices (default: -1.0)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold (default: 0.5)')
    # MAP
    parser.add_argument('--beta', type=float, default=4.0,
                        help='Beta hyperparameter (default: 4.0)')
    parser.add_argument('--alpha_0', type=float, default=1.0,
                        help='Alpha_0 MAP hyperparameter (default: 1.0)')
    parser.add_argument('--alpha_1', type=float, default=2.5,
                        help='Alpha_1 MAP hyperparameter (default: 2.5)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Alpha MAP hyperparameter (default: 1.0, Exponential)')
    parser.add_argument('--B', type=float, default=30.0,
                        help='B MAP hyperparameter (default: 30.0)')
    ### Analysis flags
    parser.add_argument('--analyze_all', action='store_true',
                        help='Run all analysis modules')
    parser.add_argument('--analyze_donors', action='store_true',
                        help='Run donor explanation analysis')
    parser.add_argument('--analyze_predictions', action='store_true',
                        help='Run model predictions analysis')
    parser.add_argument('--analyze_performance', action='store_true',
                        help='Run PR/ROC performance analysis')
    parser.add_argument('--analyze_precision_k', action='store_true',
                        help='Run Precision@k analysis')
    parser.add_argument('--analyze_macro', action='store_true',
                        help='Run macro-averaged per-TCR AUC/AP analysis')
    parser.add_argument('--max_k', type=int, default=20,
                        help='Max k for Precision@k (default: 20)')
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level (default: 1)')
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Config loader
# ═══════════════════════════════════════════════════════════════════
def load_config_file(config_path):
    """Load dataset configurations from CSV, TSV, or JSON file."""
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
    required = ['data_dir', 'donor_matrix']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in config file: {missing}")
    return df


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════
def load_data(data_dir, donor_matrix_path, pad_token=-1.):
    """Load dataset and return all required arrays.

    Returns the same dict as the sparse pipeline so that downstream code
    (multi-dataset summary, precision@k) keeps working.  ``binder_sets``
    and ``max_hlas_per_tcr`` are still loaded — they are needed for
    candidate-only evaluation, even though NonSparseTCRModel doesn't use
    them internally.
    """
    data_dir = Path(data_dir)
    print(f"Loading data from {data_dir}...")
    true_hla_set = np.load(data_dir / "synthetic_binder_sets.npy", mmap_mode="r")
    donor_indices = np.load(data_dir / "synthetic_donor_indices.npy", mmap_mode="r")
    h5_path = data_dir / 'synthetic_tcr_hla_counts.h5'
    try:
        with PublicTcrHlaCsrReaderChunk(str(h5_path)) as reader:
            counts_set, max_all = reader.read_sparse_indices_of_counts()
    except KeyError:
        with PublicTcrHlaCsrReader(str(h5_path)) as reader:
            counts_set, max_all = reader.read_sparse_indices()
    binder_sets = pad_list_to_array(counts_set, max_all, pad_token)
    donor_hla_matrix = np.load(donor_matrix_path)['donor_hla_matrix']
    num_tcrs = binder_sets.shape[0]
    max_hlas_per_tcr = binder_sets.shape[1]
    num_alleles = donor_hla_matrix.shape[1]
    num_donors = donor_hla_matrix.shape[0]
    print(f"Dataset: {num_tcrs} TCRs, {num_donors} Donors, "
          f"{num_alleles} Total Alleles, Max {max_hlas_per_tcr} candidate HLAs/TCR")
    return {
        'binder_sets': binder_sets,
        'donor_indices': np.array(donor_indices),
        'true_hla_set': np.array(true_hla_set),
        'donor_hla_matrix': donor_hla_matrix,
        'num_tcrs': num_tcrs,
        'max_hlas_per_tcr': max_hlas_per_tcr,
        'num_alleles': num_alleles,
        'num_donors': num_donors,
    }


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════
def train_model(data, args, output_path):
    """Train NonSparseTCRModel and return trained model + history."""
    os.makedirs(output_path, exist_ok=True)
    print(f"\n{'='*60}\nTraining Dense Model (NonSparseTCRModel)\n{'='*60}")
    train_dataset = create_dataset(data['donor_indices'], args.batch_size)
    steps_per_epoch = data['num_tcrs'] // args.batch_size
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=args.learning_rate,
        first_decay_steps=steps_per_epoch * 20, t_mul=2.0, m_mul=0.9, alpha=0.01)
    model = NonSparseTCRModel(
        num_tcrs=data['num_tcrs'],
        num_total_alleles=data['num_alleles'],
        donor_hla_matrix=data['donor_hla_matrix'],
        beta=args.beta,
        pad_token=args.pad_token,
        alpha_0=args.alpha_0,
        alpha_1=args.alpha_1,
        alpha=args.alpha,
        B=args.B,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer)
    print("Starting training...")
    history = model.fit(train_dataset, epochs=args.epochs, verbose=args.verbose)
    model.save_weights(os.path.join(output_path, 'model.weights.h5'))
    with open(os.path.join(output_path, 'history.json'), 'w') as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)
    print(f"Model saved to: {output_path}")
    return model, history


# ═══════════════════════════════════════════════════════════════════
# Dense helper: get full (num_tcrs, num_alleles) probability matrix
# ═══════════════════════════════════════════════════════════════════
def _get_dense_probs(model):
    """Return sigmoid(z_logits) as numpy array of shape (num_tcrs, num_alleles)."""
    z_logits = model.z_embedding.get_weights()[0]
    return 1.0 / (1.0 + np.exp(-z_logits))


# ═══════════════════════════════════════════════════════════════════
# Donor Explanation Analysis (dense version)
# ═══════════════════════════════════════════════════════════════════
def assess_explanation_for_donors_dense(model, donor_indices, donor_hla_matrix,
                                        batch_size=1024, output_path=None,
                                        pad_token=-1.):
    """Check if we can explain TCR presence in donors.

    For the dense model every allele has a probability, so the
    explanation score for a (TCR, donor) pair is simply the maximum
    σ(z_ia) over alleles *a* that the donor carries.
    """
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    report_lines = []
    def log(msg):
        print(msg)
        if output_path:
            report_lines.append(msg)

    log(f"\n{'='*60}\nASSESSING DONOR EXPLANATION (Dense)\n{'='*60}")
    num_tcrs = donor_indices.shape[0]
    donor_hla_tensor = tf.constant(donor_hla_matrix, dtype=tf.float32)  # (D, A)
    all_donor_scores = []

    log(f"Processing {num_tcrs} TCRs in batches of {batch_size}...")
    num_batches = int(np.ceil(num_tcrs / batch_size))

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_tcrs)
        tcr_indices = tf.range(start, end, dtype=tf.int32)

        # (batch, num_alleles)
        z_logits = model.z_embedding(tcr_indices)
        z_probs = tf.sigmoid(z_logits)

        # Donors for this batch: (batch, max_donors)
        batch_donors = donor_indices[start:end]
        valid_donor_mask = tf.not_equal(
            batch_donors, tf.cast(pad_token, tf.int32))
        safe_donor_ids = tf.maximum(batch_donors, 0)

        # (batch, max_donors, num_alleles) — binary HLA profile per donor
        batch_donor_hlas = tf.gather(donor_hla_tensor, safe_donor_ids)

        # Explanation score = max_a [ σ(z_ia) * x_na ]
        # z_probs: (batch, 1, num_alleles)  ×  batch_donor_hlas: (batch, max_donors, num_alleles)
        explanation_scores = tf.expand_dims(z_probs, 1) * batch_donor_hlas
        max_score_per_donor = tf.reduce_max(explanation_scores, axis=2)

        scores_masked = tf.where(valid_donor_mask, max_score_per_donor, pad_token)
        all_donor_scores.append(scores_masked.numpy())
        if i % 10 == 0:
            print(f"  Batch {i}/{num_batches} done...", end='\r')

    donor_scores_matrix = np.concatenate(all_donor_scores)
    log("\n\nAnalysis Complete. Generating Report...")

    # ── Curves across thresholds ──
    thresholds = np.linspace(0.01, 0.99, 100)
    curves = {level: [] for level in range(100, 9, -10)}
    curves[1] = []
    total_donors_per_tcr = np.maximum(
        np.sum(donor_scores_matrix != pad_token, axis=1), 1)

    for t in thresholds:
        is_explained = (donor_scores_matrix > t)
        num_explained = np.sum(is_explained, axis=1)
        fraction_explained = num_explained / total_donors_per_tcr
        for level in curves.keys():
            if level == 100:
                perc = np.mean(fraction_explained == 1.0) * 100
            elif level == 1:
                perc = np.mean(num_explained >= 1) * 100
            else:
                perc = np.mean(fraction_explained >= (level / 100.0)) * 100
            curves[level].append(perc)

    # ── Visualization ──
    fig = plt.figure(figsize=(20, 6))
    plt.subplot(1, 3, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, 11))
    for ci, level in enumerate(range(100, 9, -10)):
        label_text = "100% Donors" if level == 100 else f"≥ {level}% Donors"
        plt.plot(thresholds, curves[level], color=colors[ci],
                 linewidth=2, label=label_text)
    plt.plot(thresholds, curves[1], color='grey', linewidth=2,
             linestyle='--', label="≥ 1 Donor")
    plt.title("Explanation Robustness Spectrum")
    plt.xlabel("Binarization Threshold")
    plt.ylabel("% TCRs satisfying condition")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    masked_scores = np.ma.masked_equal(donor_scores_matrix, pad_token)
    min_scores = np.min(masked_scores, axis=1).filled(0.0)
    plt.subplot(1, 3, 2)
    plt.hist(min_scores, bins=50, color='#d65f5f', edgecolor='white', range=(0, 1))
    plt.title("Critical Score Distribution")
    plt.xlabel("Probability")
    plt.ylabel("Count of TCRs")

    check_t = 0.5
    is_explained_check = (donor_scores_matrix > check_t)
    fracs = np.sum(is_explained_check, axis=1) / total_donors_per_tcr
    plt.subplot(1, 3, 3)
    plt.hist(fracs * 100, bins=20, color='#4c72b0', edgecolor='white', range=(0, 100))
    plt.title(f"Fraction of Donors Explained (T={check_t})")
    plt.xlabel("% of Donors Explained")
    plt.ylabel("Count of TCRs")
    plt.tight_layout()

    if output_path:
        fig.savefig(os.path.join(output_path, "donor_explanation_plots.png"),
                    dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(output_path, "donor_explanation_plots.pdf"),
                    bbox_inches='tight')
        plt.close(fig)
        columns = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 1]
        with open(os.path.join(output_path, "donor_explanation_report.txt"), 'w') as f:
            f.write('\n'.join(report_lines))
        curve_data = np.column_stack(
            [thresholds] + [curves[level] for level in columns])
        np.savetxt(os.path.join(output_path, "explanation_curves.csv"),
                   curve_data, delimiter=',',
                   header='threshold,' + ','.join(
                       [f'pct_{c}_donors' for c in columns]),
                   comments='')
        np.savez_compressed(os.path.join(output_path, "donor_scores_matrix.npz"),
                            donor_scores=donor_scores_matrix,
                            thresholds=thresholds,
                            total_donors_per_tcr=total_donors_per_tcr)
    else:
        plt.show()

    perfect_count = np.sum(fracs == 1.0)
    summary_stats = {
        'num_tcrs': num_tcrs,
        'perfect_100pct': int(perfect_count),
        'mean_fraction_explained_t005': float(np.mean(fracs)),
        'median_fraction_explained_t005': float(np.median(fracs)),
    }
    return donor_scores_matrix, summary_stats


# ═══════════════════════════════════════════════════════════════════
# Model Predictions Analysis (dense version)
# ═══════════════════════════════════════════════════════════════════
def analyze_model_predictions_dense(model, true_hla_set, num_total_alleles,
                                     threshold=0.5, output_path=None, pad_token=-1.):
    """Analysis pipeline for the dense model.

    Unlike the sparse version, every allele has a predicted probability
    so there is no candidate mask — the analysis runs over the full
    (num_tcrs, num_alleles) matrix.
    """
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    report_lines = []
    def log(msg):
        print(msg)
        if output_path:
            report_lines.append(msg)

    log(f"\n{'='*50}\nSTARTING MODEL ANALYSIS (Dense)\n{'='*50}")
    trained_probs = _get_dense_probs(model)   # (num_tcrs, num_alleles)
    num_tcrs = trained_probs.shape[0]

    # ── Threshold optimisation ──
    log("\n--- Threshold Optimization Analysis ---")
    threshold_range = np.linspace(0.01, 0.999, 1000)
    coverages, avg_counts = [], []
    idx_99, idx_95, best_tradeoff_idx = -1, -1, -1
    max_tradeoff_score = -float('inf')

    for i, t in enumerate(threshold_range):
        matches = np.sum(trained_probs > t, axis=1)
        cov = np.mean(matches > 0) * 100
        avg = np.mean(matches)
        coverages.append(cov)
        avg_counts.append(avg)
        if cov >= 99.0:
            idx_99 = i
        if cov >= 95.0:
            idx_95 = i
        score = cov - (5.0 * avg)
        if score > max_tradeoff_score:
            max_tradeoff_score, best_tradeoff_idx = score, i

    def print_stat(name, idx):
        if idx >= 0:
            log(f"Strategy: {name:<25} | Threshold: {threshold_range[idx]:.3f} "
                f"| Coverage: {coverages[idx]:.2f}%")
    print_stat("'Strict' (99% Coverage)", idx_99)
    print_stat("'Relaxed' (95% Coverage)", idx_95)
    print_stat("'Balanced' (Elbow Point)", best_tradeoff_idx)

    # ── Current threshold stats ──
    log(f"\n--- Statistics for Threshold ({threshold}) ---")
    final_decisions = (trained_probs > threshold)
    matches_per_tcr = np.sum(final_decisions, axis=1)
    current_coverage = np.mean(matches_per_tcr > 0) * 100
    current_avg = np.mean(matches_per_tcr)
    current_median = np.median(matches_per_tcr)
    current_max = np.max(matches_per_tcr)
    zero_matches = np.sum(matches_per_tcr == 0)
    log(f"Coverage: {current_coverage:.2f}% | Avg HLAs/TCR: {current_avg:.2f} "
        f"| Zero matches: {zero_matches}")

    # ── Visualisations ──
    fig = plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    max_probs = np.max(trained_probs, axis=1)
    plt.hist(max_probs, bins=50, range=(0, 1), color='#4c72b0', edgecolor='white')
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'T={threshold}')
    plt.title("Distribution of Max Probability per TCR")
    plt.legend()

    plt.subplot(2, 2, 2)
    ax1 = plt.gca()
    ax1.plot(threshold_range, coverages, 'b-', linewidth=2, label='Coverage %')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Coverage (%)', color='b')
    ax2 = ax1.twinx()
    ax2.plot(threshold_range, avg_counts, 'r--', linewidth=2, label='Avg HLAs/TCR')
    ax2.set_ylabel('Avg Count', color='r')
    plt.title("Optimization Curve")

    plt.subplot(2, 2, 3)
    chosen_alleles = np.where(final_decisions)
    if len(chosen_alleles[1]) > 0:
        unique_ids, counts = np.unique(chosen_alleles[1], return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        top_n = min(30, len(sorted_indices))
        plt.bar(range(top_n), counts[sorted_indices[:top_n]], color='#55a868')
        plt.title(f"Top {top_n} Predicted Alleles")

    plt.subplot(2, 2, 4)
    sample_size = min(20, num_tcrs)
    random_indices = np.random.choice(num_tcrs, sample_size, replace=False)
    plt.imshow(trained_probs[random_indices], aspect='auto',
               cmap='viridis', vmin=0.0, vmax=1.0)
    plt.colorbar(label="Probability")
    plt.title(f"Binding Probabilities ({num_total_alleles} Alleles)")
    plt.tight_layout()

    if output_path:
        fig.savefig(os.path.join(output_path, "analysis_plots.png"),
                    dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(output_path, "analysis_plots.pdf"),
                    bbox_inches='tight')
        plt.close(fig)
        with open(os.path.join(output_path, "analysis_report.txt"), 'w') as f:
            f.write('\n'.join(report_lines))
        np.savetxt(os.path.join(output_path, "threshold_optimization.csv"),
                   np.column_stack([threshold_range, coverages, avg_counts]),
                   delimiter=',',
                   header='threshold,coverage_percent,avg_hlas_per_tcr',
                   comments='')
        np.savez(os.path.join(output_path, "analysis_arrays.npz"),
                 trained_probs=trained_probs,
                 final_decisions=final_decisions,
                 matches_per_tcr=matches_per_tcr,
                 threshold_range=threshold_range,
                 coverages=np.array(coverages),
                 avg_counts=np.array(avg_counts))
    else:
        plt.show()

    return {
        'coverage': current_coverage,
        'avg_hlas_per_tcr': current_avg,
        'median_hlas_per_tcr': current_median,
        'max_hlas_per_tcr': current_max,
        'tcrs_with_zero_hlas': zero_matches,
        'threshold_95_coverage': threshold_range[idx_95] if idx_95 >= 0 else None,
        'threshold_99_coverage': threshold_range[idx_99] if idx_99 >= 0 else None,
    }


# ═══════════════════════════════════════════════════════════════════
# Performance Evaluation — PR / ROC (dense version)
# ═══════════════════════════════════════════════════════════════════
def evaluate_model_performance_dense(model, binder_sets, true_hla_set,
                                      num_total_alleles=440, output_path=None,
                                      pad_token=-1.):
    """PR / ROC evaluation for the dense model.

    Two scopes are reported:

    **Full-space** — all alleles evaluated directly from the dense
    probability matrix.

    **Candidate-only** — restricted to alleles that appear in
    ``binder_sets`` for each TCR, so results are directly comparable
    with the sparse pipeline.
    """
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    print(f"\n--- Performance Evaluation (PR & ROC) [Dense] ---")

    trained_probs = _get_dense_probs(model)  # (num_tcrs, num_alleles)
    num_tcrs = trained_probs.shape[0]

    # Build true allele sets
    true_allele_sets = []
    for i in range(num_tcrs):
        valid_true = true_hla_set[i] >= 0
        true_allele_sets.append(set(true_hla_set[i][valid_true].astype(int)))

    # ================================================================
    # FULL-SPACE: flatten the dense matrix
    # ================================================================
    y_true_full_flat = np.zeros((num_tcrs, num_total_alleles), dtype=np.int32)
    for i, s in enumerate(true_allele_sets):
        for a in s:
            y_true_full_flat[i, a] = 1
    y_true_f = y_true_full_flat.ravel()
    y_pred_f = trained_probs.ravel()

    total_pos = int(y_true_f.sum())
    total_neg = len(y_true_f) - total_pos

    sorted_idx = np.argsort(-y_pred_f)
    y_true_sorted_f = y_true_f[sorted_idx]
    y_pred_sorted_f = y_pred_f[sorted_idx]

    tps_f = np.concatenate([[0], np.cumsum(y_true_sorted_f)])
    fps_f = np.concatenate([[0], np.cumsum(1 - y_true_sorted_f)])
    tpr_f = tps_f / max(total_pos, 1)
    fpr_f = fps_f / max(total_neg, 1)
    recall_f = tpr_f.copy()
    precision_f = np.divide(tps_f, tps_f + fps_f,
                            out=np.ones_like(tps_f, dtype=np.float64),
                            where=(tps_f + fps_f) > 0)

    roc_auc_full = float(np.trapz(tpr_f, fpr_f))
    ap_full = float(np.sum(np.diff(recall_f) * precision_f[1:]))

    p_cand_f = precision_f[1:]
    r_cand_f = recall_f[1:]
    f1_f = 2 * (p_cand_f * r_cand_f) / (p_cand_f + r_cand_f + 1e-7)
    best_f1_idx_f = int(np.argmax(f1_f))
    best_threshold_f = float(y_pred_sorted_f[best_f1_idx_f]) if best_f1_idx_f < len(y_pred_sorted_f) else 0.5
    best_f1_f = float(f1_f[best_f1_idx_f])

    print(f"[Full-space]  AUC ROC: {roc_auc_full:.5f} | AP: {ap_full:.5f} "
          f"| Best F1: {best_f1_f:.5f}")

    # ================================================================
    # CANDIDATE-ONLY: restrict to binder_sets alleles per TCR
    # ================================================================
    valid_mask = (binder_sets != pad_token)
    pred_allele_ids = binder_sets[valid_mask].astype(int)
    pred_tcr_ids = np.repeat(
        np.arange(num_tcrs), binder_sets.shape[1])[valid_mask.flatten()]
    # Gather probs from the dense matrix at candidate positions
    pred_probs_sparse = trained_probs[pred_tcr_ids, pred_allele_ids]
    y_true_sparse = np.array([
        1 if pred_allele_ids[j] in true_allele_sets[pred_tcr_ids[j]] else 0
        for j in range(len(pred_probs_sparse))
    ], dtype=np.int32)

    total_pos_cand = max(int(y_true_sparse.sum()), 1)
    total_neg_cand = max(int((1 - y_true_sparse).sum()), 1)

    sorted_idx_c = np.argsort(-pred_probs_sparse)
    y_true_sorted_c = y_true_sparse[sorted_idx_c]
    y_pred_sorted_c = pred_probs_sparse[sorted_idx_c]

    tps_c = np.concatenate([[0], np.cumsum(y_true_sorted_c)])
    fps_c = np.concatenate([[0], np.cumsum(1 - y_true_sorted_c)])
    tpr_c = tps_c / total_pos_cand
    fpr_c = fps_c / total_neg_cand
    recall_c = tpr_c.copy()
    precision_c = np.divide(tps_c, tps_c + fps_c,
                            out=np.ones_like(tps_c, dtype=np.float64),
                            where=(tps_c + fps_c) > 0)

    roc_auc_co = float(np.trapz(tpr_c, fpr_c))
    ap_co = float(np.sum(np.diff(recall_c) * precision_c[1:]))
    p_co = precision_c[1:]
    r_co = recall_c[1:]
    f1_co = 2 * (p_co * r_co) / (p_co + r_co + 1e-7)
    best_f1_idx_co = int(np.argmax(f1_co))
    best_threshold_co = float(y_pred_sorted_c[best_f1_idx_co]) if best_f1_idx_co < len(y_pred_sorted_c) else 0.5
    best_f1_co = float(f1_co[best_f1_idx_co])

    total_tp_all = sum(len(s) for s in true_allele_sets)
    tp_in_cand = int(y_true_sparse.sum())
    print(f"[Candidate-only]  AUC ROC: {roc_auc_co:.5f} | AP: {ap_co:.5f} "
          f"| Best F1: {best_f1_co:.5f}")
    print(f"  Candidate positives: {tp_in_cand} | "
          f"Candidate negatives: {int((1-y_true_sparse).sum())}")

    # ================================================================
    # PLOTTING
    # ================================================================
    if output_path:
        # Fig 1: Full-space (backward-compatible filename)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        axes[0].plot(recall_f, precision_f, color='#2ca02c', lw=2,
                     label=f'AP = {ap_full:.3f}')
        axes[0].set_title('Precision-Recall Curve (Full-Space)')
        axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision')
        axes[0].set_xlim([0, 1.02]); axes[0].set_ylim([0, 1.02])
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(fpr_f, tpr_f, color='#1f77b4', lw=2,
                     label=f'AUC = {roc_auc_full:.3f}')
        axes[1].plot([0, 1], [0, 1], 'gray', linestyle='--')
        axes[1].set_title('ROC Curve (Full-Space)')
        axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
        axes[1].set_xlim([0, 1.02]); axes[1].set_ylim([0, 1.02])
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(output_path, "performance_curves.png"),
                    dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(output_path, "performance_curves.pdf"),
                    bbox_inches='tight')
        plt.close(fig)

        # Fig 2: Candidate-only
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
        axes2[0].plot(recall_c, precision_c, color='#d62728', lw=2,
                      label=f'AP = {ap_co:.3f}')
        axes2[0].set_title('Precision-Recall Curve (Candidate-Only)')
        axes2[0].set_xlabel('Recall'); axes2[0].set_ylabel('Precision')
        axes2[0].set_xlim([0, 1.02]); axes2[0].set_ylim([0, 1.02])
        axes2[0].legend(); axes2[0].grid(True, alpha=0.3)
        axes2[1].plot(fpr_c, tpr_c, color='#ff7f0e', lw=2,
                      label=f'AUC = {roc_auc_co:.3f}')
        axes2[1].plot([0, 1], [0, 1], 'gray', linestyle='--')
        axes2[1].set_title('ROC Curve (Candidate-Only)')
        axes2[1].set_xlabel('FPR'); axes2[1].set_ylabel('TPR')
        axes2[1].set_xlim([0, 1.02]); axes2[1].set_ylim([0, 1.02])
        axes2[1].legend(); axes2[1].grid(True, alpha=0.3)
        plt.tight_layout()
        fig2.savefig(os.path.join(output_path,
                     "performance_curves_candidate_only.png"),
                     dpi=150, bbox_inches='tight')
        fig2.savefig(os.path.join(output_path,
                     "performance_curves_candidate_only.pdf"),
                     bbox_inches='tight')
        plt.close(fig2)

        # Fig 3: Comparison overlay
        fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))
        axes3[0].plot(recall_f, precision_f, color='#2ca02c', lw=2,
                      label=f'Full-space  AP={ap_full:.3f}')
        axes3[0].plot(recall_c, precision_c, color='#d62728', lw=2,
                      linestyle='--', label=f'Candidate-only  AP={ap_co:.3f}')
        axes3[0].set_title('PR Curve Comparison')
        axes3[0].set_xlabel('Recall'); axes3[0].set_ylabel('Precision')
        axes3[0].set_xlim([0, 1.02]); axes3[0].set_ylim([0, 1.02])
        axes3[0].legend(); axes3[0].grid(True, alpha=0.3)
        axes3[1].plot(fpr_f, tpr_f, color='#1f77b4', lw=2,
                      label=f'Full-space  AUC={roc_auc_full:.3f}')
        axes3[1].plot(fpr_c, tpr_c, color='#ff7f0e', lw=2,
                      linestyle='--', label=f'Candidate-only  AUC={roc_auc_co:.3f}')
        axes3[1].plot([0, 1], [0, 1], 'gray', linestyle='--', alpha=0.5)
        axes3[1].set_title('ROC Curve Comparison')
        axes3[1].set_xlabel('FPR'); axes3[1].set_ylabel('TPR')
        axes3[1].set_xlim([0, 1.02]); axes3[1].set_ylim([0, 1.02])
        axes3[1].legend(); axes3[1].grid(True, alpha=0.3)
        plt.tight_layout()
        fig3.savefig(os.path.join(output_path,
                     "performance_curves_comparison.png"),
                     dpi=150, bbox_inches='tight')
        fig3.savefig(os.path.join(output_path,
                     "performance_curves_comparison.pdf"),
                     bbox_inches='tight')
        plt.close(fig3)

        # Save curve data
        np.savez(os.path.join(output_path, "curve_data.npz"),
                 precision=precision_f, recall=recall_f,
                 fpr=fpr_f, tpr=tpr_f,
                 y_pred_sorted=y_pred_sorted_f,
                 total_true_positives=total_pos,
                 total_negatives=total_neg,
                 precision_cand=precision_c, recall_cand=recall_c,
                 fpr_cand=fpr_c, tpr_cand=tpr_c,
                 true_positives_in_candidates=tp_in_cand,
                 neg_in_candidates=total_neg_cand)

    return {
        'auc': roc_auc_full, 'ap': ap_full,
        'best_f1': best_f1_f, 'best_threshold': best_threshold_f,
        'auc_cand': roc_auc_co, 'ap_cand': ap_co,
        'best_f1_cand': best_f1_co, 'best_threshold_cand': best_threshold_co,
        'candidate_recall': tp_in_cand / max(total_tp_all, 1),
    }


# ═══════════════════════════════════════════════════════════════════
# Macro-Averaged Per-TCR Metrics (dense version)
# ═══════════════════════════════════════════════════════════════════
def compute_macro_metrics_dense(model, binder_sets, true_hla_set,
                                 num_tcrs, num_alleles, pad_token=-1.0):
    """Per-TCR AUC / AP in two scopes: full-space and candidate-only."""
    trained_probs = _get_dense_probs(model)
    valid_candidate_mask = (binder_sets != pad_token)

    full_auc, full_ap, full_idx = [], [], []
    cand_auc, cand_ap, cand_idx = [], [], []
    total_tp, tp_in_candidates = 0, 0

    for i in range(num_tcrs):
        true_alleles = np.asarray(true_hla_set[i])
        true_alleles = true_alleles[true_alleles >= 0].astype(int)
        if len(true_alleles) == 0:
            continue
        true_set = set(true_alleles)
        total_tp += len(true_set)

        cand_mask_i = valid_candidate_mask[i]
        cand_ids = binder_sets[i][cand_mask_i].astype(int)
        cand_set = set(cand_ids)
        tp_in_candidates += len(true_set & cand_set)

        # Full-space
        if len(true_alleles) < num_alleles:
            y_true_full = np.zeros(num_alleles, dtype=np.float32)
            y_true_full[true_alleles] = 1.0
            full_auc.append(roc_auc_score(y_true_full, trained_probs[i]))
            full_ap.append(average_precision_score(y_true_full, trained_probs[i]))
            full_idx.append(i)

        # Candidate-only
        n_cand = len(cand_ids)
        if n_cand >= 2:
            y_true_cand = np.array(
                [1.0 if a in true_set else 0.0 for a in cand_ids],
                dtype=np.float32)
            cand_probs_i = trained_probs[i, cand_ids]
            n_pos = int(y_true_cand.sum())
            if 0 < n_pos < n_cand:
                cand_auc.append(roc_auc_score(y_true_cand, cand_probs_i))
                cand_ap.append(average_precision_score(y_true_cand, cand_probs_i))
                cand_idx.append(i)

    full_auc = np.asarray(full_auc)
    full_ap = np.asarray(full_ap)
    cand_auc = np.asarray(cand_auc)
    cand_ap = np.asarray(cand_ap)

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
        "per_tcr_auc_full": full_auc, "per_tcr_ap_full": full_ap,
        "valid_tcr_indices_full": np.array(full_idx),
        "num_valid_tcrs_full": len(full_idx),
        "per_tcr_auc_cand": cand_auc, "per_tcr_ap_cand": cand_ap,
        "valid_tcr_indices_cand": np.array(cand_idx),
        "num_valid_tcrs_cand": len(cand_idx),
        "candidate_recall": tp_in_candidates / max(total_tp, 1),
        # Legacy aliases
        "per_tcr_auc": full_auc, "per_tcr_ap": full_ap,
        "valid_tcr_indices": np.array(full_idx),
        "num_valid_tcrs": len(full_idx),
    }
    result.update(_stat_block(full_auc, full_ap, "full"))
    result.update(_stat_block(cand_auc, cand_ap, "cand"))
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


def plot_macro_metrics_dense(macro_results, output_path):
    """Distribution plots for per-TCR AUC / AP (both scopes)."""

    def _plot_pair(auc_arr, ap_arr, med_auc, med_ap, label, fname):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(auc_arr, bins=50, edgecolor="black", alpha=0.75, color="#4C72B0")
        axes[0].axvline(med_auc, color="red", linestyle="--", lw=1.5,
                        label=f"Median = {med_auc:.4f}")
        axes[0].set_xlabel("Per-TCR ROC-AUC"); axes[0].set_ylabel("Count")
        axes[0].set_title(f"ROC-AUC Distribution ({label})"); axes[0].legend()
        axes[1].hist(ap_arr, bins=50, edgecolor="black", alpha=0.75, color="#55A868")
        axes[1].axvline(med_ap, color="red", linestyle="--", lw=1.5,
                        label=f"Median = {med_ap:.4f}")
        axes[1].set_xlabel("Per-TCR Average Precision"); axes[1].set_ylabel("Count")
        axes[1].set_title(f"AP Distribution ({label})"); axes[1].legend()
        plt.tight_layout()
        fig_file = os.path.join(output_path, f"macro_metrics_distribution_{fname}.png")
        fig.savefig(fig_file, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  Macro-metrics plot saved to: {fig_file}")

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        bp = ax2.boxplot([auc_arr, ap_arr], labels=["ROC-AUC", "Avg Precision"],
                         patch_artist=True, showmeans=True,
                         meanprops=dict(marker="D", markerfacecolor="red", markersize=6))
        for patch, c in zip(bp["boxes"], ["#4C72B0", "#55A868"]):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax2.set_ylabel("Score"); ax2.set_ylim(-0.05, 1.05)
        ax2.set_title(f"Per-TCR Distributions ({label})")
        fig2_file = os.path.join(output_path, f"macro_metrics_boxplot_{fname}.png")
        fig2.savefig(fig2_file, dpi=150, bbox_inches="tight"); plt.close(fig2)
        print(f"  Macro-metrics boxplot saved to: {fig2_file}")

    if len(macro_results["per_tcr_auc_full"]) > 0:
        _plot_pair(macro_results["per_tcr_auc_full"],
                   macro_results["per_tcr_ap_full"],
                   macro_results["median_auc_full"],
                   macro_results["median_ap_full"],
                   "Full-Space", "full")

    if len(macro_results["per_tcr_auc_cand"]) > 0:
        _plot_pair(macro_results["per_tcr_auc_cand"],
                   macro_results["per_tcr_ap_cand"],
                   macro_results["median_auc_cand"],
                   macro_results["median_ap_cand"],
                   "Candidate-Only", "cand")

    has_full = len(macro_results["per_tcr_auc_full"]) > 0
    has_cand = len(macro_results["per_tcr_auc_cand"]) > 0
    if has_full and has_cand:
        fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, metric, title in [(axes[0], "auc", "ROC-AUC"),
                                   (axes[1], "ap", "Average Precision")]:
            bp = ax.boxplot(
                [macro_results[f"per_tcr_{metric}_full"],
                 macro_results[f"per_tcr_{metric}_cand"]],
                labels=["Full-Space", "Candidate-Only"],
                patch_artist=True, showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="red", markersize=6))
            bp["boxes"][0].set_facecolor("#4C72B0"); bp["boxes"][0].set_alpha(0.6)
            bp["boxes"][1].set_facecolor("#DD8452"); bp["boxes"][1].set_alpha(0.6)
            ax.set_ylabel(title); ax.set_title(f"Per-TCR {title}")
            ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        fig3_file = os.path.join(output_path, "macro_metrics_comparison.png")
        fig3.savefig(fig3_file, dpi=150, bbox_inches="tight"); plt.close(fig3)
        print(f"  Macro-metrics comparison plot saved to: {fig3_file}")


# ═══════════════════════════════════════════════════════════════════
# Precision@k (dense version)
# ═══════════════════════════════════════════════════════════════════
def compute_precision_at_k_dense(output_path, data_dir, max_k=20, pad_token=-1.):
    """Precision@k / Recall@k using the dense probability matrix.

    Uses *all* alleles as candidates (not just binder_sets), since the
    dense model scores every allele.
    """
    output_dir, data_dir = Path(output_path), Path(data_dir)
    arrays_path = output_dir / "figures" / "analysis_arrays.npz"
    arrays = np.load(arrays_path)
    trained_probs = arrays['trained_probs']  # (num_tcrs, num_alleles)

    true_hla_set = np.load(data_dir / "synthetic_binder_sets.npy")
    num_tcrs = trained_probs.shape[0]

    true_allele_sets = []
    for i in range(num_tcrs):
        valid_mask = true_hla_set[i] >= 0
        true_allele_sets.append(set(true_hla_set[i][valid_mask].astype(int)))

    print(f"Computing Precision@k for k=1 to {max_k} (dense, all alleles)...")
    precision_at_k = {k: [] for k in range(1, max_k + 1)}
    recall_at_k = {k: [] for k in range(1, max_k + 1)}

    for i in range(num_tcrs):
        sorted_allele_ids = np.argsort(-trained_probs[i])
        true_set = true_allele_sets[i]
        num_true = len(true_set)
        hits = 0
        for k in range(1, max_k + 1):
            if k <= len(sorted_allele_ids) and sorted_allele_ids[k - 1] in true_set:
                hits += 1
            precision_at_k[k].append(hits / k)
            recall_at_k[k].append(hits / num_true if num_true > 0 else 0.0)

    results = {
        'mean_precision_at_k': {k: np.mean(v) for k, v in precision_at_k.items()},
        'std_precision_at_k': {k: np.std(v) for k, v in precision_at_k.items()},
        'mean_recall_at_k': {k: np.mean(v) for k, v in recall_at_k.items()},
        'num_tcrs': num_tcrs,
    }
    return results


# ═══════════════════════════════════════════════════════════════════
# Analysis orchestrator
# ═══════════════════════════════════════════════════════════════════
def run_analysis(model, data, args, output_path):
    """Run all enabled analysis modules (dense versions)."""
    results = {}
    figures_path = os.path.join(output_path, "figures")
    os.makedirs(figures_path, exist_ok=True)

    # ── Donor explanation ──
    if args.analyze_all or args.analyze_donors:
        print(f"\n{'='*60}\nDonor Explanation Analysis\n{'='*60}")
        _, donor_stats = assess_explanation_for_donors_dense(
            model, data['donor_indices'], data['donor_hla_matrix'],
            output_path=figures_path, pad_token=args.pad_token)
        results['donor_stats'] = donor_stats
    else:
        donor_stats = {'mean_fraction_explained_t005': 0.0}
        results['donor_stats'] = donor_stats

    # ── Model predictions ──
    if args.analyze_all or args.analyze_predictions:
        print(f"\n{'='*60}\nModel Predictions Analysis\n{'='*60}")
        analysis_results = analyze_model_predictions_dense(
            model, data['true_hla_set'], data['num_alleles'],
            threshold=args.threshold, output_path=figures_path,
            pad_token=args.pad_token)
        results['analysis'] = analysis_results
    else:
        analysis_results = {'coverage': 0.0, 'avg_hlas_per_tcr': 0.0}
        results['analysis'] = analysis_results

    # ── Performance (PR/ROC) ──
    if args.analyze_all or args.analyze_performance:
        print(f"\n{'='*60}\nPerformance Evaluation\n{'='*60}")
        perf_metrics = evaluate_model_performance_dense(
            model=model, binder_sets=data['binder_sets'],
            true_hla_set=data['true_hla_set'],
            num_total_alleles=data['num_alleles'],
            output_path=figures_path, pad_token=args.pad_token)
        results['performance'] = perf_metrics
    else:
        perf_metrics = {'auc': 0.0, 'ap': 0.0, 'best_f1': 0.0}
        results['performance'] = perf_metrics

    # ── Macro-averaged per-TCR metrics ──
    if args.analyze_all or args.analyze_macro:
        print(f"\n{'='*60}\nMacro-Averaged Per-TCR Metrics\n{'='*60}")
        macro = compute_macro_metrics_dense(
            model, binder_sets=data['binder_sets'],
            true_hla_set=data['true_hla_set'],
            num_tcrs=data['num_tcrs'],
            num_alleles=data['num_alleles'],
            pad_token=args.pad_token)
        cand_recall = macro['candidate_recall']
        print(f"  Candidate recall (TP in candidates / total TP): "
              f"{cand_recall:.4f}")
        print(f"\n  [Full-space]  ({macro['num_valid_tcrs_full']} TCRs)")
        print(f"    ROC-AUC  — median: {macro['median_auc_full']:.4f}  "
              f"mean: {macro['mean_auc_full']:.4f}  "
              f"std: {macro['std_auc_full']:.4f}  "
              f"IQR: [{macro['q25_auc_full']:.4f}, {macro['q75_auc_full']:.4f}]")
        print(f"    Avg Prec — median: {macro['median_ap_full']:.4f}  "
              f"mean: {macro['mean_ap_full']:.4f}  "
              f"std: {macro['std_ap_full']:.4f}  "
              f"IQR: [{macro['q25_ap_full']:.4f}, {macro['q75_ap_full']:.4f}]")
        print(f"\n  [Candidate-only]  ({macro['num_valid_tcrs_cand']} TCRs)")
        print(f"    ROC-AUC  — median: {macro['median_auc_cand']:.4f}  "
              f"mean: {macro['mean_auc_cand']:.4f}  "
              f"std: {macro['std_auc_cand']:.4f}  "
              f"IQR: [{macro['q25_auc_cand']:.4f}, {macro['q75_auc_cand']:.4f}]")
        print(f"    Avg Prec — median: {macro['median_ap_cand']:.4f}  "
              f"mean: {macro['mean_ap_cand']:.4f}  "
              f"std: {macro['std_ap_cand']:.4f}  "
              f"IQR: [{macro['q25_ap_cand']:.4f}, {macro['q75_ap_cand']:.4f}]")

        plot_macro_metrics_dense(macro, figures_path)

        macro_serialisable = {k: v for k, v in macro.items()
                              if not isinstance(v, np.ndarray)}
        macro_json_path = os.path.join(output_path, "macro_metrics.json")
        with open(macro_json_path, "w") as f:
            json.dump(macro_serialisable, f, indent=2, cls=NumpyEncoder)
        print(f"  Macro-metrics JSON saved to: {macro_json_path}")

        np.savez_compressed(
            os.path.join(output_path, "macro_metrics_per_tcr.npz"),
            per_tcr_auc_full=macro["per_tcr_auc_full"],
            per_tcr_ap_full=macro["per_tcr_ap_full"],
            valid_tcr_indices_full=macro["valid_tcr_indices_full"],
            per_tcr_auc_cand=macro["per_tcr_auc_cand"],
            per_tcr_ap_cand=macro["per_tcr_ap_cand"],
            valid_tcr_indices_cand=macro["valid_tcr_indices_cand"])

        results['macro'] = macro_serialisable
    else:
        results['macro'] = {}

    # ── Save final metrics (backward-compatible call) ──
    if (args.analyze_all or args.analyze_donors
            or args.analyze_predictions or args.analyze_performance):
        save_metrics_json(output_path, perf_metrics, analysis_results,
                          donor_stats, args.threshold)
    return results


def run_precision_at_k(output_path, data_dir, args):
    """Run Precision@k analysis (post-training)."""
    print(f"\n{'='*60}\nPrecision@k Analysis\n{'='*60}")
    try:
        results = compute_precision_at_k_dense(
            output_path, data_dir, max_k=args.max_k, pad_token=args.pad_token)
        pk_path = os.path.join(output_path, "precision_at_k.json")
        with open(pk_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"Precision@k results saved to: {pk_path}")
        return results
    except Exception as e:
        print(f"Warning: Could not compute Precision@k: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# Single / multi-dataset runners
# ═══════════════════════════════════════════════════════════════════
def run_single_dataset(args, data_dir, donor_matrix, output_path, name=None):
    if name:
        print(f"\n{'#'*80}\nProcessing: {name}\n{'#'*80}")
    data = load_data(data_dir, donor_matrix, args.pad_token)
    model, history = train_model(data, args, output_path)
    results = run_analysis(model, data, args, output_path)
    if args.analyze_all or args.analyze_precision_k:
        pk_results = run_precision_at_k(output_path, data_dir, args)
        if pk_results:
            results['precision_at_k'] = pk_results
    print(f"\n{'='*60}\nPipeline Complete\n{'='*60}")
    print(f"Results saved to: {output_path}")
    return results


def run_multiple_datasets(args):
    from utils import plot_precision_at_k_heatmap, plot_precision_at_k_curves
    config_df = load_config_file(args.df)
    all_results = {}
    for idx, row in config_df.iterrows():
        name = row.get('name', f'dataset_{idx}')
        data_dir = row['data_dir']
        donor_matrix = row['donor_matrix']
        row_args = argparse.Namespace(**vars(args))
        for col in ['epochs', 'batch_size', 'learning_rate', 'beta']:
            if col in row and pd.notna(row[col]):
                setattr(row_args, col, row[col])
        output_path = os.path.join(args.output_dir, name)
        try:
            results = run_single_dataset(
                row_args, data_dir, donor_matrix, output_path, name=name)
            all_results[name] = results
        except Exception as e:
            print(f"Error processing {name}: {e}")
            all_results[name] = {'error': str(e)}
    summary_path = os.path.join(args.output_dir, 'all_results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nAll results summary saved to: {summary_path}")
    if args.analyze_all or args.analyze_precision_k:
        pk_results = {k: v.get('precision_at_k') for k, v in all_results.items()
                      if isinstance(v, dict) and 'precision_at_k' in v}
        if pk_results:
            plot_precision_at_k_heatmap(
                pk_results, k_values=[1, 3, 5, 10],
                output_path=os.path.join(args.output_dir,
                                         'precision_at_k_heatmap.png'))
            plot_precision_at_k_curves(
                pk_results, max_k=args.max_k,
                output_path=os.path.join(args.output_dir,
                                         'precision_at_k_curves.png'))
    return all_results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    if args.df:
        results = run_multiple_datasets(args)
    else:
        if not args.donor_matrix:
            raise ValueError("--donor_matrix is required when using --data_dir")
        results = run_single_dataset(
            args, args.data_dir, args.donor_matrix, args.output_dir)
    print("\nDone!")
    return results


if __name__ == '__main__':
    main()