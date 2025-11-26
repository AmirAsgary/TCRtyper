#!/usr/bin/env python3
"""
TCR-HLA Binding Analysis Pipeline
==================================
Analyzes TCR binding predictions and compares with ground truth HLA typing data.
Supports GPU acceleration via TensorFlow and efficient CPU operations via NumPy.

Author: Computational Immunology Pipeline
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore')

# ============================================================================
# GPU DETECTION AND TENSORFLOW SETUP
# ============================================================================
GPU_AVAILABLE = False
tf = None

def setup_tensorflow():
    """Initialize TensorFlow with optimal GPU settings."""
    global GPU_AVAILABLE, tf
    try:
        import tensorflow as _tf
        tf = _tf
        
        # Enable memory growth to avoid OOM
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            GPU_AVAILABLE = True
            print(f"[INFO] GPU(s) detected: {[gpu.name for gpu in gpus]}")
        else:
            print("[INFO] No GPU detected, using CPU")
    except ImportError:
        print("[WARN] TensorFlow not installed, using NumPy only")
    except Exception as e:
        print(f"[WARN] TensorFlow GPU setup failed: {e}")

setup_tensorflow()


# ============================================================================
# ARGUMENT PARSING
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='TCR-HLA Binding Analysis Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--tcr_file', type=str, required=True,
                        help='Path to TCR probability parquet file')
    parser.add_argument('--gt_file', type=str, required=True,
                        help='Path to ground truth CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Top K TCRs for aggregation')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Binary threshold for HLA prediction')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU if available')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Batch size for GPU processing')
    return parser.parse_args()


# ============================================================================
# DATA LOADING
# ============================================================================
def get_hla_columns(df: pd.DataFrame) -> List[str]:
    """Extract HLA column names from dataframe."""
    return sorted([col for col in df.columns if col.startswith('HLA-')])


def load_data(tcr_file: str, gt_file: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load TCR predictions and ground truth data efficiently."""
    print("[LOAD] Loading ground truth CSV...")
    gt_df = pd.read_csv(gt_file)
    
    print("[LOAD] Loading TCR parquet file (streaming)...")
    # Use pyarrow for efficient reading
    tcr_table = pq.read_table(tcr_file)
    tcr_df = tcr_table.to_pandas()
    
    hla_cols = get_hla_columns(tcr_df)
    
    # Validate columns
    gt_hla_cols = get_hla_columns(gt_df)
    common_hla = sorted(set(hla_cols) & set(gt_hla_cols))
    
    print(f"[LOAD] TCR data shape: {tcr_df.shape}")
    print(f"[LOAD] Ground truth shape: {gt_df.shape}")
    print(f"[LOAD] HLA columns in TCR: {len(hla_cols)}")
    print(f"[LOAD] HLA columns in GT: {len(gt_hla_cols)}")
    print(f"[LOAD] Common HLA columns: {len(common_hla)}")
    print(f"[LOAD] Unique donors in TCR: {tcr_df['donor_id'].nunique()}")
    print(f"[LOAD] Unique donors in GT: {gt_df['donor_id'].nunique()}")
    
    return tcr_df, gt_df, hla_cols, common_hla


# ============================================================================
# AGGREGATION - NUMPY (CPU)
# ============================================================================
def aggregate_mean_numpy(tcr_df: pd.DataFrame, hla_cols: List[str]) -> pd.DataFrame:
    """Aggregate TCR probabilities by donor using mean (vectorized NumPy)."""
    print("[AGG-CPU] Computing mean aggregation...")
    agg_df = tcr_df.groupby('donor_id', sort=False)[hla_cols].mean()
    return agg_df.reset_index()


def aggregate_topk_numpy(tcr_df: pd.DataFrame, hla_cols: List[str], k: int) -> pd.DataFrame:
    """
    Aggregate TCR probabilities by donor using top-k mean.
    Optimized with numpy partition for O(n) top-k selection.
    """
    print(f"[AGG-CPU] Computing top-{k} aggregation...")
    
    donors = tcr_df['donor_id'].unique()
    n_hla = len(hla_cols)
    results = np.zeros((len(donors), n_hla), dtype=np.float32)
    
    # Group once and iterate
    grouped = tcr_df.groupby('donor_id', sort=False)
    donor_to_idx = {d: i for i, d in enumerate(donors)}
    
    for donor, group in grouped:
        data = group[hla_cols].values.astype(np.float32)
        n_tcr = data.shape[0]
        idx = donor_to_idx[donor]
        
        if n_tcr <= k:
            results[idx] = np.mean(data, axis=0)
        else:
            # Use argpartition for O(n) top-k per column
            # Process column-wise for memory efficiency
            for j in range(n_hla):
                col_data = data[:, j]
                top_k_idx = np.argpartition(col_data, -k)[-k:]
                results[idx, j] = np.mean(col_data[top_k_idx])
    
    agg_df = pd.DataFrame(results, columns=hla_cols)
    agg_df.insert(0, 'donor_id', donors)
    return agg_df


# ============================================================================
# AGGREGATION - TENSORFLOW (GPU)
# ============================================================================
def aggregate_mean_gpu(tcr_df: pd.DataFrame, hla_cols: List[str]) -> pd.DataFrame:
    """Aggregate TCR probabilities by donor using mean (TensorFlow GPU)."""
    print("[AGG-GPU] Computing mean aggregation...")
    
    donors = tcr_df['donor_id'].unique()
    donor_to_idx = {d: i for i, d in enumerate(donors)}
    n_donors = len(donors)
    
    # Convert to tensors
    with tf.device('/GPU:0'):
        hla_data = tf.constant(tcr_df[hla_cols].values, dtype=tf.float32)
        donor_indices = tf.constant(
            [donor_to_idx[d] for d in tcr_df['donor_id']], 
            dtype=tf.int32
        )
        
        # Efficient unsorted segment mean
        agg_values = tf.math.unsorted_segment_mean(
            hla_data, donor_indices, n_donors
        )
    
    agg_df = pd.DataFrame(agg_values.numpy(), columns=hla_cols)
    agg_df.insert(0, 'donor_id', donors)
    return agg_df


def aggregate_topk_gpu(tcr_df: pd.DataFrame, hla_cols: List[str], k: int) -> pd.DataFrame:
    """
    Aggregate TCR probabilities by donor using top-k mean (TensorFlow GPU).
    Uses tf.math.top_k for efficient GPU-accelerated top-k selection.
    """
    print(f"[AGG-GPU] Computing top-{k} aggregation...")
    
    donors = tcr_df['donor_id'].unique()
    n_hla = len(hla_cols)
    results = np.zeros((len(donors), n_hla), dtype=np.float32)
    
    grouped = tcr_df.groupby('donor_id', sort=False)
    donor_to_idx = {d: i for i, d in enumerate(donors)}
    
    with tf.device('/GPU:0'):
        for donor, group in grouped:
            data = group[hla_cols].values.astype(np.float32)
            n_tcr = data.shape[0]
            idx = donor_to_idx[donor]
            
            if n_tcr <= k:
                results[idx] = np.mean(data, axis=0)
            else:
                # tf.math.top_k operates on last dimension, so transpose
                # Shape: (n_tcr, n_hla) -> (n_hla, n_tcr)
                data_t = tf.constant(data.T, dtype=tf.float32)
                top_k_vals, _ = tf.math.top_k(data_t, k=k)
                results[idx] = tf.reduce_mean(top_k_vals, axis=1).numpy()
    
    agg_df = pd.DataFrame(results, columns=hla_cols)
    agg_df.insert(0, 'donor_id', donors)
    return agg_df


def aggregate_data(tcr_df: pd.DataFrame, hla_cols: List[str], k: int, 
                   use_gpu: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform both aggregation methods with appropriate backend."""
    if use_gpu and GPU_AVAILABLE and tf is not None:
        agg_mean = aggregate_mean_gpu(tcr_df, hla_cols)
        agg_topk = aggregate_topk_gpu(tcr_df, hla_cols, k)
    else:
        agg_mean = aggregate_mean_numpy(tcr_df, hla_cols)
        agg_topk = aggregate_topk_numpy(tcr_df, hla_cols, k)
    
    return agg_mean, agg_topk


# ============================================================================
# VISUALIZATION - HEATMAPS
# ============================================================================
def plot_heatmap(agg_df: pd.DataFrame, hla_cols: List[str], 
                 output_path: str, title: str):
    """Plot heatmap of aggregated probabilities."""
    print(f"[VIZ] Plotting heatmap: {title[:50]}...")
    
    data = agg_df.set_index('donor_id')[hla_cols]
    
    # Dynamic figure sizing
    n_donors, n_hla = data.shape
    fig_width = min(50, max(15, n_hla * 0.08))
    fig_height = min(30, max(8, n_donors * 0.25))
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    sns.heatmap(
        data, 
        cmap='viridis', 
        ax=ax,
        xticklabels=True if n_hla <= 100 else False,
        yticklabels=True,
        cbar_kws={'label': 'Binding Probability'}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('HLA Alleles', fontsize=12)
    ax.set_ylabel('Donors', fontsize=12)
    
    if n_hla <= 100:
        ax.tick_params(axis='x', labelsize=5, rotation=90)
    ax.tick_params(axis='y', labelsize=7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================================
# ENRICHMENT ANALYSIS
# ============================================================================
def compute_enrichment_vectorized(agg_df: pd.DataFrame, gt_df: pd.DataFrame,
                                   hla_cols: List[str], use_gpu: bool) -> Dict:
    """
    Compute enrichment curves for all patients in a vectorized manner.
    Returns curves and normalized AUC values.
    """
    print("[ENRICH] Computing enrichment curves...")
    
    # Align dataframes by donor_id
    common_donors = sorted(set(agg_df['donor_id']) & set(gt_df['donor_id']))
    
    agg_aligned = agg_df[agg_df['donor_id'].isin(common_donors)].copy()
    agg_aligned = agg_aligned.set_index('donor_id').loc[common_donors]
    
    gt_aligned = gt_df[gt_df['donor_id'].isin(common_donors)].copy()
    gt_aligned = gt_aligned.set_index('donor_id').loc[common_donors]
    
    # Get common HLA columns
    common_hla = [col for col in hla_cols if col in gt_aligned.columns]
    n_hla = len(common_hla)
    n_donors = len(common_donors)
    
    probs_matrix = agg_aligned[common_hla].values.astype(np.float32)
    gt_matrix = gt_aligned[common_hla].values.astype(np.float32)
    
    # Vectorized enrichment computation
    if use_gpu and GPU_AVAILABLE and tf is not None:
        with tf.device('/GPU:0'):
            probs_tf = tf.constant(probs_matrix, dtype=tf.float32)
            gt_tf = tf.constant(gt_matrix, dtype=tf.float32)
            
            # Get sorted indices (descending by probability)
            sorted_indices = tf.argsort(probs_tf, axis=1, direction='DESCENDING')
            
            # Gather ground truth in sorted order
            batch_indices = tf.repeat(
                tf.range(n_donors)[:, tf.newaxis], n_hla, axis=1
            )
            gather_indices = tf.stack([batch_indices, sorted_indices], axis=-1)
            sorted_gt = tf.gather_nd(gt_tf, gather_indices)
            
            # Cumulative sum for enrichment
            cumsum = tf.cumsum(sorted_gt, axis=1).numpy()
            sorted_gt_np = sorted_gt.numpy()
    else:
        # NumPy version
        sorted_indices = np.argsort(-probs_matrix, axis=1)
        sorted_gt_np = np.take_along_axis(gt_matrix, sorted_indices, axis=1)
        cumsum = np.cumsum(sorted_gt_np, axis=1)
    
    # Compute AUCs
    total_positives = np.sum(gt_matrix, axis=1)
    actual_aucs = np.sum(cumsum, axis=1)
    
    # Maximum AUC: if all positives were ranked first
    # max_auc = sum(n_hla - i for i in range(n_pos)) = n_pos * n_hla - n_pos*(n_pos-1)/2
    max_aucs = total_positives * n_hla - (total_positives * (total_positives - 1)) / 2
    
    # Avoid division by zero
    normalized_aucs = np.where(
        max_aucs > 0,
        actual_aucs / max_aucs,
        0.0
    )
    
    # Build curves
    curves = []
    for i in range(n_donors):
        x = np.arange(n_hla + 1)
        y = np.concatenate([[0], cumsum[i]])
        curves.append((x, y))
    
    return {
        'donors': common_donors,
        'curves': curves,
        'aucs': normalized_aucs.tolist(),
        'n_hla': n_hla
    }


def plot_enrichment_curves(results: Dict, output_path: str, title: str):
    """Plot enrichment curves for all patients with color coding."""
    print(f"[VIZ] Plotting enrichment curves...")
    
    n_donors = len(results['donors'])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by AUC value
    cmap = plt.cm.viridis
    aucs = np.array(results['aucs'])
    norm = plt.Normalize(aucs.min(), aucs.max())
    
    for i, (x, y) in enumerate(results['curves']):
        color = cmap(norm(results['aucs'][i]))
        ax.plot(x, y, color=color, alpha=0.6, linewidth=1)
    
    # Add diagonal reference line (random expectation)
    n_hla = results['n_hla']
    mean_positives = np.mean([c[1][-1] for c in results['curves']])
    ax.plot([0, n_hla], [0, mean_positives], 'k--', alpha=0.5, 
            label='Random expectation')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Normalized AUC', fontsize=10)
    
    ax.set_xlabel('Rank (sorted by predicted probability)', fontsize=12)
    ax.set_ylabel('Cumulative Hits (True Positives)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_auc_barplot(results: Dict, output_path: str, title: str):
    """Plot AUC bar plot for all patients, sorted by AUC."""
    print(f"[VIZ] Plotting AUC bar plot...")
    
    # Sort by AUC
    sorted_idx = np.argsort(results['aucs'])[::-1]
    sorted_donors = [results['donors'][i] for i in sorted_idx]
    sorted_aucs = [results['aucs'][i] for i in sorted_idx]
    
    n_donors = len(sorted_donors)
    fig_width = max(12, n_donors * 0.25)
    
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    # Color gradient based on AUC
    colors = plt.cm.viridis(np.array(sorted_aucs))
    
    x = np.arange(n_donors)
    bars = ax.bar(x, sorted_aucs, color=colors, edgecolor='black', linewidth=0.3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_donors, rotation=90, fontsize=7)
    ax.set_xlabel('Donors (sorted by AUC)', fontsize=12)
    ax.set_ylabel('Normalized Enrichment AUC', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=np.mean(sorted_aucs), color='red', linestyle='--', 
               linewidth=2, label=f'Mean AUC: {np.mean(sorted_aucs):.3f}')
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================================
# HLA-LEVEL METRICS
# ============================================================================
def compute_hla_metrics_vectorized(agg_df: pd.DataFrame, gt_df: pd.DataFrame,
                                    hla_cols: List[str], threshold: float,
                                    use_gpu: bool) -> pd.DataFrame:
    """
    Compute metrics for each HLA allele (vectorized where possible).
    Metrics: Precision, Recall, F1, AUC
    """
    print(f"[METRICS] Computing HLA-level metrics (threshold={threshold})...")
    
    # Align dataframes
    common_donors = sorted(set(agg_df['donor_id']) & set(gt_df['donor_id']))
    
    agg_aligned = agg_df[agg_df['donor_id'].isin(common_donors)].copy()
    agg_aligned = agg_aligned.set_index('donor_id').loc[common_donors]
    
    gt_aligned = gt_df[gt_df['donor_id'].isin(common_donors)].copy()
    gt_aligned = gt_aligned.set_index('donor_id').loc[common_donors]
    
    common_hla = [col for col in hla_cols if col in gt_aligned.columns]
    
    probs_matrix = agg_aligned[common_hla].values.astype(np.float32)
    gt_matrix = gt_aligned[common_hla].values.astype(np.float32)
    pred_matrix = (probs_matrix >= threshold).astype(np.float32)
    
    n_donors, n_hla = gt_matrix.shape
    
    metrics_list = []
    
    for j, hla in enumerate(common_hla):
        y_true = gt_matrix[:, j]
        y_prob = probs_matrix[:, j]
        y_pred = pred_matrix[:, j]
        
        n_pos = int(np.sum(y_true))
        n_neg = n_donors - n_pos
        
        # Handle edge cases
        if n_pos == 0 or n_pos == n_donors:
            metrics_list.append({
                'HLA': hla,
                'Precision': np.nan,
                'Recall': np.nan,
                'F1': np.nan,
                'AUC': np.nan,
                'N_positive': n_pos,
                'N_negative': n_neg,
                'N_predicted_positive': int(np.sum(y_pred))
            })
            continue
        
        # Compute metrics
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = np.nan
        
        metrics_list.append({
            'HLA': hla,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc,
            'N_positive': n_pos,
            'N_negative': n_neg,
            'N_predicted_positive': int(np.sum(y_pred))
        })
    
    metrics_df = pd.DataFrame(metrics_list)
    
    # Sort by AUC descending
    metrics_df = metrics_df.sort_values('AUC', ascending=False, na_position='last')
    
    return metrics_df


def plot_hla_metrics_summary(metrics_df: pd.DataFrame, output_path: str, title: str):
    """Plot summary of HLA-level metrics."""
    print(f"[VIZ] Plotting HLA metrics summary...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Filter valid metrics
    valid_df = metrics_df.dropna(subset=['AUC'])
    
    # AUC distribution
    ax = axes[0, 0]
    ax.hist(valid_df['AUC'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(valid_df['AUC'].mean(), color='red', linestyle='--', 
               label=f"Mean: {valid_df['AUC'].mean():.3f}")
    ax.axvline(0.5, color='gray', linestyle=':', label='Random (0.5)')
    ax.set_xlabel('AUC')
    ax.set_ylabel('Count')
    ax.set_title('AUC Distribution')
    ax.legend()
    
    # F1 distribution
    ax = axes[0, 1]
    ax.hist(valid_df['F1'], bins=30, color='forestgreen', edgecolor='black', alpha=0.7)
    ax.axvline(valid_df['F1'].mean(), color='red', linestyle='--',
               label=f"Mean: {valid_df['F1'].mean():.3f}")
    ax.set_xlabel('F1 Score')
    ax.set_ylabel('Count')
    ax.set_title('F1 Score Distribution')
    ax.legend()
    
    # Precision vs Recall scatter
    ax = axes[1, 0]
    scatter = ax.scatter(valid_df['Recall'], valid_df['Precision'], 
                        c=valid_df['AUC'], cmap='viridis', alpha=0.7, s=30)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Recall (colored by AUC)')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.colorbar(scatter, ax=ax, label='AUC')
    
    # N_positive vs AUC
    ax = axes[1, 1]
    ax.scatter(valid_df['N_positive'], valid_df['AUC'], 
              alpha=0.6, s=30, color='purple')
    ax.set_xlabel('Number of Positive Samples')
    ax.set_ylabel('AUC')
    ax.set_title('AUC vs Sample Size')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.7)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_analysis(agg_df: pd.DataFrame, gt_df: pd.DataFrame, hla_cols: List[str],
                 common_hla: List[str], method_name: str, output_dir: Path,
                 threshold: float, use_gpu: bool):
    """Run full analysis pipeline for one aggregation method."""
    print(f"\n{'='*60}")
    print(f"[ANALYSIS] Running analysis for: {method_name}")
    print(f"{'='*60}")
    
    # Save aggregated table
    agg_path = output_dir / f'aggregated_{method_name}.csv'
    agg_df.to_csv(agg_path, index=False)
    print(f"[SAVE] Aggregated table: {agg_path}")
    
    # Plot heatmap
    heatmap_path = output_dir / f'heatmap_{method_name}.png'
    plot_heatmap(agg_df, hla_cols, str(heatmap_path),
                 f'HLA Binding Probabilities ({method_name.replace("_", " ").title()})')
    
    # Enrichment analysis
    enrichment = compute_enrichment_vectorized(agg_df, gt_df, common_hla, use_gpu)
    
    # Save enrichment AUCs
    auc_df = pd.DataFrame({
        'donor_id': enrichment['donors'],
        'enrichment_auc': enrichment['aucs']
    }).sort_values('enrichment_auc', ascending=False)
    auc_path = output_dir / f'enrichment_auc_{method_name}.csv'
    auc_df.to_csv(auc_path, index=False)
    print(f"[SAVE] Enrichment AUCs: {auc_path}")
    
    # Plot enrichment curves
    curves_path = output_dir / f'enrichment_curves_{method_name}.png'
    plot_enrichment_curves(enrichment, str(curves_path),
                          f'Enrichment Curves ({method_name.replace("_", " ").title()})')
    
    # Plot AUC barplot
    barplot_path = output_dir / f'enrichment_auc_barplot_{method_name}.png'
    plot_auc_barplot(enrichment, str(barplot_path),
                    f'Enrichment AUC per Donor ({method_name.replace("_", " ").title()})')
    
    # HLA-level metrics
    hla_metrics = compute_hla_metrics_vectorized(
        agg_df, gt_df, common_hla, threshold, use_gpu
    )
    metrics_path = output_dir / f'hla_metrics_{method_name}.csv'
    hla_metrics.to_csv(metrics_path, index=False)
    print(f"[SAVE] HLA metrics: {metrics_path}")
    
    # Plot HLA metrics summary
    metrics_plot_path = output_dir / f'hla_metrics_summary_{method_name}.png'
    plot_hla_metrics_summary(hla_metrics, str(metrics_plot_path),
                            f'HLA Metrics Summary ({method_name.replace("_", " ").title()})')
    
    # Print summary
    valid_metrics = hla_metrics.dropna(subset=['AUC'])
    print(f"\n[SUMMARY] {method_name.upper()} Results:")
    print(f"  Donors analyzed: {len(enrichment['donors'])}")
    print(f"  HLAs analyzed: {len(common_hla)}")
    print(f"  Mean Enrichment AUC: {np.mean(enrichment['aucs']):.4f}")
    print(f"  HLA Metrics (mean over {len(valid_metrics)} valid HLAs):")
    print(f"    Precision: {valid_metrics['Precision'].mean():.4f}")
    print(f"    Recall:    {valid_metrics['Recall'].mean():.4f}")
    print(f"    F1:        {valid_metrics['F1'].mean():.4f}")
    print(f"    AUC:       {valid_metrics['AUC'].mean():.4f}")
    
    return enrichment, hla_metrics


def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("TCR-HLA BINDING ANALYSIS PIPELINE")
    print("="*70)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[CONFIG] Output directory: {output_dir}")
    print(f"[CONFIG] Top-K: {args.top_k}")
    print(f"[CONFIG] Threshold: {args.threshold}")
    
    # Determine compute backend
    use_gpu = args.use_gpu and GPU_AVAILABLE and tf is not None
    print(f"[CONFIG] Backend: {'TensorFlow GPU' if use_gpu else 'NumPy CPU'}")
    
    # Load data
    tcr_df, gt_df, hla_cols, common_hla = load_data(args.tcr_file, args.gt_file)
    
    # Perform aggregations
    print("\n[AGG] Starting aggregation...")
    agg_mean, agg_topk = aggregate_data(tcr_df, hla_cols, args.top_k, use_gpu)
    
    # Free memory
    del tcr_df
    import gc
    gc.collect()
    
    # Run analysis for both aggregation methods
    results = {}
    
    results['mean'] = run_analysis(
        agg_mean, gt_df, hla_cols, common_hla,
        'mean', output_dir, args.threshold, use_gpu
    )
    
    results[f'top{args.top_k}'] = run_analysis(
        agg_topk, gt_df, hla_cols, common_hla,
        f'top{args.top_k}', output_dir, args.threshold, use_gpu
    )
    
    # Comparative summary
    print("\n" + "="*70)
    print("COMPARATIVE SUMMARY")
    print("="*70)
    print(f"{'Metric':<25} {'Mean Agg':>15} {'Top-K Agg':>15}")
    print("-"*55)
    
    mean_enrich_auc = np.mean(results['mean'][0]['aucs'])
    topk_enrich_auc = np.mean(results[f'top{args.top_k}'][0]['aucs'])
    print(f"{'Mean Enrichment AUC':<25} {mean_enrich_auc:>15.4f} {topk_enrich_auc:>15.4f}")
    
    mean_hla_auc = results['mean'][1]['AUC'].mean()
    topk_hla_auc = results[f'top{args.top_k}'][1]['AUC'].mean()
    print(f"{'Mean HLA AUC':<25} {mean_hla_auc:>15.4f} {topk_hla_auc:>15.4f}")
    
    mean_f1 = results['mean'][1]['F1'].mean()
    topk_f1 = results[f'top{args.top_k}'][1]['F1'].mean()
    print(f"{'Mean HLA F1':<25} {mean_f1:>15.4f} {topk_f1:>15.4f}")
    
    print("\n" + "="*70)
    print(f"[DONE] All outputs saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()