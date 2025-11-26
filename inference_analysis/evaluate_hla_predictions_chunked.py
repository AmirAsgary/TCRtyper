#!/usr/bin/env python3
"""
TCR-HLA Binding Analysis Pipeline (Memory-Optimized for Large Scale)
=====================================================================
Handles 40M+ TCR sequences with chunked processing.
Supports GPU acceleration via TensorFlow and efficient CPU operations via NumPy.

Author: Computational Immunology Pipeline
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Generator
import warnings
import gc
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

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
        tf.get_logger().setLevel('ERROR')
        
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
        description='TCR-HLA Binding Analysis Pipeline (Large Scale)',
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
    parser.add_argument('--chunk_size', type=int, default=500_000,
                        help='Rows per chunk for memory efficiency')
    parser.add_argument('--use_float16', action='store_true', default=False,
                        help='Use float16 to halve memory (slight precision loss)')
    return parser.parse_args()


# ============================================================================
# UTILITIES
# ============================================================================
def get_hla_columns(columns) -> List[str]:
    """Extract HLA column names."""
    return sorted([col for col in columns if col.startswith('HLA-')])


def get_dtype(use_float16: bool):
    """Get appropriate dtype."""
    return np.float16 if use_float16 else np.float32


def format_bytes(n_bytes: int) -> str:
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.2f} PB"


def format_time(seconds: float) -> str:
    """Format seconds to human readable."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


class Timer:
    """Context manager for timing operations."""
    def __init__(self, name: str):
        self.name = name
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"[TIME] {self.name}: {format_time(elapsed)}")


# ============================================================================
# CHUNKED DATA LOADING AND AGGREGATION
# ============================================================================
def get_parquet_metadata(tcr_file: str) -> Tuple[List[str], int]:
    """Get metadata without loading full file."""
    pf = pq.ParquetFile(tcr_file)
    schema = pf.schema_arrow
    columns = [field.name for field in schema]
    n_rows = pf.metadata.num_rows
    return columns, n_rows


def stream_parquet_chunks(tcr_file: str, chunk_size: int, 
                          columns: List[str]) -> Generator[pd.DataFrame, None, None]:
    """Stream parquet file in chunks."""
    pf = pq.ParquetFile(tcr_file)
    
    for batch in pf.iter_batches(batch_size=chunk_size, columns=columns):
        yield batch.to_pandas()


def aggregate_chunked(tcr_file: str, hla_cols: List[str], top_k: int,
                      chunk_size: int, use_float16: bool, 
                      use_gpu: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Memory-efficient chunked aggregation.
    
    Strategy:
    - For MEAN: Accumulate sum and count per donor, compute mean at end
    - For TOP-K: Maintain a heap/sorted list of top-k values per donor per HLA
    
    This approach uses O(n_donors * n_hla * k) memory instead of O(n_tcr * n_hla)
    """
    print("[AGG] Starting chunked aggregation...")
    
    dtype = get_dtype(use_float16)
    n_hla = len(hla_cols)
    columns_to_load = ['donor_id', 'tcr_id'] + hla_cols
    
    # First pass: get unique donors
    print("[AGG] Pass 1: Counting donors and TCRs...")
    donor_counts = {}
    total_rows = 0
    
    with Timer("Donor counting"):
        for chunk in stream_parquet_chunks(tcr_file, chunk_size, ['donor_id']):
            for donor, count in chunk['donor_id'].value_counts().items():
                donor_counts[donor] = donor_counts.get(donor, 0) + count
            total_rows += len(chunk)
            gc.collect()
    
    donors = sorted(donor_counts.keys())
    n_donors = len(donors)
    donor_to_idx = {d: i for i, d in enumerate(donors)}
    
    print(f"[AGG] Found {n_donors} donors, {total_rows:,} total TCR rows")
    print(f"[AGG] Average TCRs per donor: {total_rows // n_donors:,}")
    
    # Initialize accumulators
    # For mean: sum and count
    sum_acc = np.zeros((n_donors, n_hla), dtype=np.float64)  # Use float64 for accumulation
    count_acc = np.zeros(n_donors, dtype=np.int64)
    
    # For top-k: maintain top-k values per donor per HLA
    # Memory: n_donors * n_hla * k * 4 bytes = 53 * 358 * 100 * 4 = ~7.6 MB (very manageable!)
    topk_heaps = np.full((n_donors, n_hla, top_k), -np.inf, dtype=dtype)
    topk_counts = np.zeros((n_donors, n_hla), dtype=np.int32)
    
    # Second pass: aggregate
    print("[AGG] Pass 2: Aggregating chunks...")
    chunk_num = 0
    
    with Timer("Main aggregation"):
        for chunk in stream_parquet_chunks(tcr_file, chunk_size, columns_to_load):
            chunk_num += 1
            
            # Convert to numpy
            donor_ids = chunk['donor_id'].values
            hla_data = chunk[hla_cols].values.astype(dtype)
            
            # Vectorized donor index lookup
            donor_indices = np.array([donor_to_idx[d] for d in donor_ids], dtype=np.int32)
            
            # Update mean accumulators (vectorized)
            if use_gpu and GPU_AVAILABLE and tf is not None:
                with tf.device('/GPU:0'):
                    hla_tf = tf.constant(hla_data, dtype=tf.float32)
                    idx_tf = tf.constant(donor_indices, dtype=tf.int32)
                    
                    chunk_sums = tf.math.unsorted_segment_sum(
                        hla_tf, idx_tf, n_donors
                    ).numpy()
                    
                    chunk_counts = tf.math.unsorted_segment_sum(
                        tf.ones(len(donor_indices), dtype=tf.float32),
                        idx_tf, n_donors
                    ).numpy()
                
                sum_acc += chunk_sums
                count_acc += chunk_counts.astype(np.int64)
            else:
                # NumPy version using bincount-like approach
                np.add.at(sum_acc, donor_indices, hla_data)
                np.add.at(count_acc, donor_indices, 1)
            
            # Update top-k heaps (this is the expensive part)
            # Process by donor to minimize memory
            unique_donors_in_chunk = np.unique(donor_indices)
            
            for d_idx in unique_donors_in_chunk:
                mask = donor_indices == d_idx
                donor_data = hla_data[mask]  # Shape: (n_tcr_for_donor, n_hla)
                
                for j in range(n_hla):
                    col_data = donor_data[:, j]
                    current_heap = topk_heaps[d_idx, j]
                    current_count = topk_counts[d_idx, j]
                    
                    # Combine existing heap with new data
                    combined = np.concatenate([
                        current_heap[:min(current_count, top_k)],
                        col_data
                    ])
                    
                    # Get top-k
                    if len(combined) <= top_k:
                        topk_heaps[d_idx, j, :len(combined)] = combined
                        topk_counts[d_idx, j] = len(combined)
                    else:
                        top_k_vals = np.partition(combined, -top_k)[-top_k:]
                        topk_heaps[d_idx, j, :] = top_k_vals
                        topk_counts[d_idx, j] = top_k
            
            if chunk_num % 10 == 0:
                processed = chunk_num * chunk_size
                pct = min(100, processed / total_rows * 100)
                print(f"[AGG] Processed {processed:,} / {total_rows:,} rows ({pct:.1f}%)")
            
            del chunk, hla_data, donor_ids, donor_indices
            gc.collect()
    
    print("[AGG] Computing final aggregations...")
    
    # Compute mean
    count_acc_safe = np.maximum(count_acc, 1)[:, np.newaxis]
    mean_values = (sum_acc / count_acc_safe).astype(dtype)
    
    # Compute top-k mean
    topk_values = np.zeros((n_donors, n_hla), dtype=dtype)
    for i in range(n_donors):
        for j in range(n_hla):
            k_actual = min(topk_counts[i, j], top_k)
            if k_actual > 0:
                topk_values[i, j] = np.mean(topk_heaps[i, j, :k_actual])
    
    # Build DataFrames
    agg_mean = pd.DataFrame(mean_values, columns=hla_cols)
    agg_mean.insert(0, 'donor_id', donors)
    
    agg_topk = pd.DataFrame(topk_values, columns=hla_cols)
    agg_topk.insert(0, 'donor_id', donors)
    
    # Memory stats
    heap_mem = topk_heaps.nbytes
    print(f"[AGG] Top-K heap memory used: {format_bytes(heap_mem)}")
    
    del sum_acc, count_acc, topk_heaps, topk_counts
    gc.collect()
    
    return agg_mean, agg_topk


# ============================================================================
# OPTIMIZED TOP-K AGGREGATION (ALTERNATIVE: PER-DONOR STREAMING)
# ============================================================================
def aggregate_chunked_v2(tcr_file: str, hla_cols: List[str], top_k: int,
                         chunk_size: int, use_float16: bool,
                         use_gpu: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Alternative chunked aggregation - processes one donor at a time.
    Better for very large top_k values but slower overall.
    Use aggregate_chunked() for most cases.
    """
    print("[AGG-V2] Per-donor streaming aggregation...")
    
    dtype = get_dtype(use_float16)
    n_hla = len(hla_cols)
    
    # Get donor list first
    columns, n_rows = get_parquet_metadata(tcr_file)
    
    donors = set()
    for chunk in stream_parquet_chunks(tcr_file, chunk_size, ['donor_id']):
        donors.update(chunk['donor_id'].unique())
    donors = sorted(donors)
    n_donors = len(donors)
    
    print(f"[AGG-V2] Processing {n_donors} donors...")
    
    mean_values = np.zeros((n_donors, n_hla), dtype=dtype)
    topk_values = np.zeros((n_donors, n_hla), dtype=dtype)
    
    # Process each donor by filtering chunks
    columns_to_load = ['donor_id'] + hla_cols
    
    with Timer("Per-donor aggregation"):
        for i, donor in enumerate(donors):
            donor_data_list = []
            
            for chunk in stream_parquet_chunks(tcr_file, chunk_size, columns_to_load):
                donor_mask = chunk['donor_id'] == donor
                if donor_mask.any():
                    donor_data_list.append(chunk.loc[donor_mask, hla_cols].values)
            
            if donor_data_list:
                donor_data = np.vstack(donor_data_list).astype(dtype)
                n_tcr = donor_data.shape[0]
                
                # Mean
                mean_values[i] = np.mean(donor_data, axis=0)
                
                # Top-K
                if n_tcr <= top_k:
                    topk_values[i] = mean_values[i]
                else:
                    for j in range(n_hla):
                        col = donor_data[:, j]
                        top_k_idx = np.argpartition(col, -top_k)[-top_k:]
                        topk_values[i, j] = np.mean(col[top_k_idx])
            
            if (i + 1) % 10 == 0:
                print(f"[AGG-V2] Processed {i+1}/{n_donors} donors")
            
            gc.collect()
    
    agg_mean = pd.DataFrame(mean_values, columns=hla_cols)
    agg_mean.insert(0, 'donor_id', donors)
    
    agg_topk = pd.DataFrame(topk_values, columns=hla_cols)
    agg_topk.insert(0, 'donor_id', donors)
    
    return agg_mean, agg_topk


# ============================================================================
# VISUALIZATION - HEATMAPS
# ============================================================================
def plot_heatmap(agg_df: pd.DataFrame, hla_cols: List[str], 
                 output_path: str, title: str):
    """Plot heatmap of aggregated probabilities."""
    print(f"[VIZ] Plotting heatmap...")
    
    data = agg_df.set_index('donor_id')[hla_cols]
    n_donors, n_hla = data.shape
    
    fig_width = min(40, max(12, n_hla * 0.06))
    fig_height = min(20, max(6, n_donors * 0.2))
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    sns.heatmap(
        data, cmap='viridis', ax=ax,
        xticklabels=False,  # Too many HLAs
        yticklabels=True,
        cbar_kws={'label': 'Binding Probability', 'shrink': 0.8}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(f'HLA Alleles (n={n_hla})', fontsize=12)
    ax.set_ylabel('Donors', fontsize=12)
    ax.tick_params(axis='y', labelsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================================
# ENRICHMENT ANALYSIS (VECTORIZED)
# ============================================================================
def compute_enrichment_vectorized(agg_df: pd.DataFrame, gt_df: pd.DataFrame,
                                   hla_cols: List[str], use_gpu: bool) -> Dict:
    """Compute enrichment curves for all patients (fully vectorized)."""
    print("[ENRICH] Computing enrichment curves...")
    
    # Align dataframes
    common_donors = sorted(set(agg_df['donor_id']) & set(gt_df['donor_id']))
    
    agg_aligned = agg_df[agg_df['donor_id'].isin(common_donors)].copy()
    agg_aligned = agg_aligned.set_index('donor_id').loc[common_donors]
    
    gt_aligned = gt_df[gt_df['donor_id'].isin(common_donors)].copy()
    gt_aligned = gt_aligned.set_index('donor_id').loc[common_donors]
    
    common_hla = [col for col in hla_cols if col in gt_aligned.columns]
    n_hla = len(common_hla)
    n_donors = len(common_donors)
    
    probs_matrix = agg_aligned[common_hla].values.astype(np.float32)
    gt_matrix = gt_aligned[common_hla].values.astype(np.float32)
    
    # Vectorized enrichment
    if use_gpu and GPU_AVAILABLE and tf is not None:
        with tf.device('/GPU:0'):
            probs_tf = tf.constant(probs_matrix, dtype=tf.float32)
            gt_tf = tf.constant(gt_matrix, dtype=tf.float32)
            
            sorted_indices = tf.argsort(probs_tf, axis=1, direction='DESCENDING')
            
            batch_idx = tf.repeat(tf.range(n_donors)[:, tf.newaxis], n_hla, axis=1)
            gather_idx = tf.stack([batch_idx, sorted_indices], axis=-1)
            sorted_gt = tf.gather_nd(gt_tf, gather_idx)
            
            cumsum = tf.cumsum(sorted_gt, axis=1).numpy()
            sorted_gt_np = sorted_gt.numpy()
    else:
        sorted_indices = np.argsort(-probs_matrix, axis=1)
        sorted_gt_np = np.take_along_axis(gt_matrix, sorted_indices, axis=1)
        cumsum = np.cumsum(sorted_gt_np, axis=1)
    
    # Compute normalized AUCs
    total_positives = np.sum(gt_matrix, axis=1)
    actual_aucs = np.sum(cumsum, axis=1)
    max_aucs = total_positives * n_hla - (total_positives * (total_positives - 1)) / 2
    normalized_aucs = np.where(max_aucs > 0, actual_aucs / max_aucs, 0.0)
    
    # Build curves
    curves = [(np.arange(n_hla + 1), np.concatenate([[0], cumsum[i]])) 
              for i in range(n_donors)]
    
    return {
        'donors': common_donors,
        'curves': curves,
        'aucs': normalized_aucs.tolist(),
        'n_hla': n_hla,
        'total_positives': total_positives.tolist()
    }


def plot_enrichment_curves(results: Dict, output_path: str, title: str):
    """Plot enrichment curves with color coding by AUC."""
    print(f"[VIZ] Plotting enrichment curves...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    cmap = plt.cm.viridis
    aucs = np.array(results['aucs'])
    norm = plt.Normalize(aucs.min(), aucs.max())
    
    for i, (x, y) in enumerate(results['curves']):
        ax.plot(x, y, color=cmap(norm(aucs[i])), alpha=0.7, linewidth=1.5)
    
    # Reference line
    n_hla = results['n_hla']
    mean_pos = np.mean(results['total_positives'])
    ax.plot([0, n_hla], [0, mean_pos], 'k--', alpha=0.5, 
            linewidth=2, label='Random expectation')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Normalized AUC', fontsize=11)
    
    ax.set_xlabel('Rank (sorted by predicted probability)', fontsize=12)
    ax.set_ylabel('Cumulative Hits (True Positives)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_auc_barplot(results: Dict, output_path: str, title: str):
    """Plot AUC bar plot sorted by value."""
    print(f"[VIZ] Plotting AUC bar plot...")
    
    sorted_idx = np.argsort(results['aucs'])[::-1]
    sorted_donors = [results['donors'][i] for i in sorted_idx]
    sorted_aucs = [results['aucs'][i] for i in sorted_idx]
    
    n_donors = len(sorted_donors)
    fig, ax = plt.subplots(figsize=(max(10, n_donors * 0.3), 6))
    
    colors = plt.cm.viridis(np.array(sorted_aucs))
    x = np.arange(n_donors)
    ax.bar(x, sorted_aucs, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_donors, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Donors (sorted by AUC)', fontsize=12)
    ax.set_ylabel('Normalized Enrichment AUC', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=np.mean(sorted_aucs), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(sorted_aucs):.3f}')
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================================
# HLA-LEVEL METRICS
# ============================================================================
def compute_hla_metrics(agg_df: pd.DataFrame, gt_df: pd.DataFrame,
                        hla_cols: List[str], threshold: float) -> pd.DataFrame:
    """Compute per-HLA metrics: Precision, Recall, F1, AUC."""
    print(f"[METRICS] Computing HLA-level metrics (threshold={threshold})...")
    
    common_donors = sorted(set(agg_df['donor_id']) & set(gt_df['donor_id']))
    
    agg_aligned = agg_df[agg_df['donor_id'].isin(common_donors)].set_index('donor_id').loc[common_donors]
    gt_aligned = gt_df[gt_df['donor_id'].isin(common_donors)].set_index('donor_id').loc[common_donors]
    
    common_hla = [col for col in hla_cols if col in gt_aligned.columns]
    
    probs = agg_aligned[common_hla].values
    gt = gt_aligned[common_hla].values
    preds = (probs >= threshold).astype(np.float32)
    
    metrics_list = []
    
    for j, hla in enumerate(common_hla):
        y_true = gt[:, j]
        y_prob = probs[:, j]
        y_pred = preds[:, j]
        
        n_pos = int(np.sum(y_true))
        n_neg = len(y_true) - n_pos
        
        if n_pos == 0 or n_pos == len(y_true):
            metrics_list.append({
                'HLA': hla, 'Precision': np.nan, 'Recall': np.nan,
                'F1': np.nan, 'AUC': np.nan,
                'N_positive': n_pos, 'N_negative': n_neg,
                'N_predicted_positive': int(np.sum(y_pred))
            })
            continue
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = np.nan
        
        metrics_list.append({
            'HLA': hla, 'Precision': prec, 'Recall': rec,
            'F1': f1, 'AUC': auc,
            'N_positive': n_pos, 'N_negative': n_neg,
            'N_predicted_positive': int(np.sum(y_pred))
        })
    
    return pd.DataFrame(metrics_list).sort_values('AUC', ascending=False, na_position='last')


def plot_hla_metrics_summary(metrics_df: pd.DataFrame, output_path: str, title: str):
    """Plot 4-panel summary of HLA metrics."""
    print(f"[VIZ] Plotting HLA metrics summary...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    valid = metrics_df.dropna(subset=['AUC'])
    
    # AUC distribution
    ax = axes[0, 0]
    ax.hist(valid['AUC'], bins=25, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(valid['AUC'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f"Mean: {valid['AUC'].mean():.3f}")
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=2, label='Random (0.5)')
    ax.set_xlabel('AUC', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('AUC Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    
    # F1 distribution
    ax = axes[0, 1]
    ax.hist(valid['F1'], bins=25, color='forestgreen', edgecolor='black', alpha=0.7)
    ax.axvline(valid['F1'].mean(), color='red', linestyle='--',
               linewidth=2, label=f"Mean: {valid['F1'].mean():.3f}")
    ax.set_xlabel('F1 Score', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('F1 Score Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    
    # Precision vs Recall
    ax = axes[1, 0]
    sc = ax.scatter(valid['Recall'], valid['Precision'], c=valid['AUC'], 
                   cmap='viridis', alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision vs Recall', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.colorbar(sc, ax=ax, label='AUC')
    
    # Sample size vs AUC
    ax = axes[1, 1]
    ax.scatter(valid['N_positive'], valid['AUC'], alpha=0.6, s=40, 
              color='purple', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('N Positive Samples', fontsize=11)
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_title('AUC vs Sample Size', fontsize=12, fontweight='bold')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.7)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_analysis(agg_df: pd.DataFrame, gt_df: pd.DataFrame, hla_cols: List[str],
                 method_name: str, output_dir: Path, threshold: float, use_gpu: bool):
    """Run full analysis for one aggregation method."""
    print(f"\n{'='*60}")
    print(f"[ANALYSIS] {method_name.upper()}")
    print(f"{'='*60}")
    
    # Save aggregated table
    agg_path = output_dir / f'aggregated_{method_name}.csv'
    agg_df.to_csv(agg_path, index=False)
    print(f"[SAVE] {agg_path}")
    
    # Heatmap
    plot_heatmap(agg_df, hla_cols, str(output_dir / f'heatmap_{method_name}.png'),
                 f'HLA Binding Probabilities ({method_name.replace("_", " ").title()})')
    
    # Enrichment
    common_hla = [c for c in hla_cols if c in gt_df.columns]
    enrichment = compute_enrichment_vectorized(agg_df, gt_df, common_hla, use_gpu)
    
    auc_df = pd.DataFrame({'donor_id': enrichment['donors'], 
                           'enrichment_auc': enrichment['aucs']})
    auc_df.to_csv(output_dir / f'enrichment_auc_{method_name}.csv', index=False)
    
    plot_enrichment_curves(enrichment, str(output_dir / f'enrichment_curves_{method_name}.png'),
                          f'Enrichment Curves ({method_name.replace("_", " ").title()})')
    
    plot_auc_barplot(enrichment, str(output_dir / f'enrichment_auc_barplot_{method_name}.png'),
                    f'Enrichment AUC ({method_name.replace("_", " ").title()})')
    
    # HLA metrics
    hla_metrics = compute_hla_metrics(agg_df, gt_df, common_hla, threshold)
    hla_metrics.to_csv(output_dir / f'hla_metrics_{method_name}.csv', index=False)
    
    plot_hla_metrics_summary(hla_metrics, str(output_dir / f'hla_metrics_summary_{method_name}.png'),
                            f'HLA Metrics ({method_name.replace("_", " ").title()})')
    
    # Summary stats
    valid_metrics = hla_metrics.dropna(subset=['AUC'])
    print(f"\n[RESULTS] {method_name.upper()}:")
    print(f"  Mean Enrichment AUC: {np.mean(enrichment['aucs']):.4f}")
    print(f"  HLA Metrics (n={len(valid_metrics)}):")
    print(f"    Precision: {valid_metrics['Precision'].mean():.4f}")
    print(f"    Recall:    {valid_metrics['Recall'].mean():.4f}")
    print(f"    F1:        {valid_metrics['F1'].mean():.4f}")
    print(f"    AUC:       {valid_metrics['AUC'].mean():.4f}")
    
    return enrichment, hla_metrics


def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("TCR-HLA ANALYSIS PIPELINE (MEMORY-OPTIMIZED)")
    print("="*70)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    use_gpu = args.use_gpu and GPU_AVAILABLE and tf is not None
    dtype_str = 'float16' if args.use_float16 else 'float32'
    
    print(f"[CONFIG] Output: {output_dir}")
    print(f"[CONFIG] Top-K: {args.top_k}")
    print(f"[CONFIG] Threshold: {args.threshold}")
    print(f"[CONFIG] Chunk size: {args.chunk_size:,}")
    print(f"[CONFIG] Data type: {dtype_str}")
    print(f"[CONFIG] Backend: {'TensorFlow GPU' if use_gpu else 'NumPy CPU'}")
    
    # Load ground truth (small)
    print("\n[LOAD] Loading ground truth...")
    gt_df = pd.read_csv(args.gt_file)
    
    # Get parquet metadata
    columns, n_rows = get_parquet_metadata(args.tcr_file)
    hla_cols = get_hla_columns(columns)
    
    print(f"[LOAD] TCR file: {n_rows:,} rows, {len(hla_cols)} HLA columns")
    print(f"[LOAD] Ground truth: {len(gt_df)} donors")
    
    # Estimate memory
    est_mem = n_rows * len(hla_cols) * (2 if args.use_float16 else 4)
    print(f"[LOAD] Estimated raw data size: {format_bytes(est_mem)}")
    
    # Chunked aggregation
    with Timer("Total aggregation time"):
        agg_mean, agg_topk = aggregate_chunked(
            args.tcr_file, hla_cols, args.top_k,
            args.chunk_size, args.use_float16, use_gpu
        )
    
    print(f"\n[AGG] Aggregated shape: {agg_mean.shape}")
    
    # Run analyses
    results = {}
    
    results['mean'] = run_analysis(
        agg_mean, gt_df, hla_cols, 'mean', output_dir, args.threshold, use_gpu
    )
    
    results[f'top{args.top_k}'] = run_analysis(
        agg_topk, gt_df, hla_cols, f'top{args.top_k}', output_dir, args.threshold, use_gpu
    )
    
    # Comparative summary
    print("\n" + "="*70)
    print("COMPARATIVE SUMMARY")
    print("="*70)
    print(f"{'Metric':<25} {'Mean Agg':>15} {'Top-K Agg':>15}")
    print("-"*55)
    
    m_auc = np.mean(results['mean'][0]['aucs'])
    t_auc = np.mean(results[f'top{args.top_k}'][0]['aucs'])
    print(f"{'Enrichment AUC':<25} {m_auc:>15.4f} {t_auc:>15.4f}")
    
    m_hla = results['mean'][1]['AUC'].mean()
    t_hla = results[f'top{args.top_k}'][1]['AUC'].mean()
    print(f"{'HLA AUC':<25} {m_hla:>15.4f} {t_hla:>15.4f}")
    
    m_f1 = results['mean'][1]['F1'].mean()
    t_f1 = results[f'top{args.top_k}'][1]['F1'].mean()
    print(f"{'HLA F1':<25} {m_f1:>15.4f} {t_f1:>15.4f}")
    
    print("\n" + "="*70)
    print(f"[DONE] All outputs saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()