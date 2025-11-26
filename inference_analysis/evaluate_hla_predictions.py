#!/usr/bin/env python3
"""
Donor-Level HLA Prediction Evaluation (GPU-Accelerated, TensorFlow-based)
Aggregates TCR-level HLA binding probabilities to donor-level predictions
and evaluates performance against ground truth HLA typing.
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, confusion_matrix
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# GPU acceleration imports
import keras
import tensorflow as tf

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU available: {len(gpus)} device(s)")
        USE_GPU = True
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
        USE_GPU = False
else:
    print("No GPU found, using CPU")
    USE_GPU = False


class DonorLevelHLAEvaluator:
    """
    Evaluates TCR-level HLA predictions at the donor level (TensorFlow-accelerated).
    """
    
    def __init__(self, output_dir: str, use_gpu: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu and USE_GPU
        
        # Create subdirectories
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.predictions_dir = self.output_dir / "predictions"
        self.predictions_dir.mkdir(exist_ok=True)
        
        if self.use_gpu:
            print(f"GPU acceleration ENABLED (TensorFlow)")
        else:
            print(f"GPU acceleration DISABLED (using CPU)")
    
    def _to_gpu(self, array: np.ndarray):
        """
        Convert a NumPy array to a TensorFlow tensor (placed on GPU if available).
        """
        if self.use_gpu:
            # Let TensorFlow place on GPU automatically if available.
            # TensorFlow handles device placement for tensors and operations.[web:3][web:4]
            return tf.convert_to_tensor(array)
        return array
    
    def _to_cpu(self, array):
        """
        Convert a TensorFlow tensor back to NumPy if needed.
        """
        if isinstance(array, tf.Tensor):
            return array.numpy()
        return array
        
    def aggregate_tcr_to_donor(
        self,
        parquet_file: str,
        aggregation_method: str = 'max',
        top_k: int = None,
        chunk_size: int = 100000
    ) -> pd.DataFrame:
        """
        Aggregate TCR-level predictions to donor-level predictions (GPU-accelerated via TensorFlow).
        
        Parameters:
        -----------
        parquet_file : str
            Path to parquet file with TCR predictions
        aggregation_method : str
            Method to aggregate: 'max', 'mean', 'top_k_mean', 'weighted_sum'
        top_k : int
            If using 'top_k_mean', number of top TCRs to average
        chunk_size : int
            Number of rows to process at once
            
        Returns:
        --------
        pd.DataFrame : Donor-level predictions (donors × HLA alleles)
        """
        print(f"Aggregating TCR predictions to donor level using '{aggregation_method}'...")
        
        parquet_file_obj = pq.ParquetFile(parquet_file)
        
        # First pass: collect all data by donor (memory efficient)
        donor_data = {}
        hla_cols = None
        
        for batch_idx, batch in enumerate(parquet_file_obj.iter_batches(batch_size=chunk_size)):
            if batch_idx % 1 == 0:
                print(f"  Processing batch {batch_idx}...")
            
            df_chunk = batch.to_pandas()
            
            # Get HLA columns on first batch
            if hla_cols is None:
                hla_cols = [
                    col for col in df_chunk.columns
                    if col not in ['donor_id', 'tcr_id']
                ]
                print(f"  Found {len(hla_cols)} HLA alleles")
            
            # Group by donor and collect indices
            for donor_id in df_chunk['donor_id'].unique():
                donor_mask = df_chunk['donor_id'] == donor_id
                donor_chunk = df_chunk.loc[donor_mask, hla_cols].values
                
                if donor_id not in donor_data:
                    donor_data[donor_id] = []
                donor_data[donor_id].append(donor_chunk.astype(np.float32, copy=False))
        
        print(f"Aggregating predictions for {len(donor_data)} donors"
              f" {'on GPU (TensorFlow)...' if self.use_gpu else ' on CPU...'}")
        
        # Second pass: aggregate
        aggregated = {}
        
        for donor_idx, (donor_id, chunks) in enumerate(donor_data.items()):
            if donor_idx % 1 == 0 and donor_idx > 0:
                print(f"  Processed {donor_idx}/{len(donor_data)} donors")
            
            donor_probs = np.vstack(chunks)  # Shape: (n_tcrs, n_hlas)
            
            if self.use_gpu:
                # TensorFlow GPU path
                t = tf.convert_to_tensor(donor_probs, dtype=tf.float32)
                
                if aggregation_method == 'max':
                    result = tf.reduce_max(t, axis=0)
                
                elif aggregation_method == 'mean':
                    result = tf.reduce_mean(t, axis=0)
                
                elif aggregation_method == 'top_k_mean' and top_k:
                    # Sort along TCR dimension and average top-k per allele
                    sorted_probs = tf.sort(t, axis=0)
                    n_tcrs = tf.shape(sorted_probs)[0]
                    top_k_actual = tf.minimum(top_k, n_tcrs)
                    # Take last top_k_actual rows
                    top_slice = sorted_probs[-top_k_actual:, :]
                    result = tf.reduce_mean(top_slice, axis=0)
                
                elif aggregation_method == 'weighted_sum':
                    # Softmax-like weighting per allele
                    denom = tf.reduce_sum(t, axis=0, keepdims=True) + 1e-10
                    weights = t / denom
                    result = tf.reduce_sum(t * weights, axis=0)
                
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregation_method}")
                
                aggregated[donor_id] = result.numpy()
            
            else:
                # Pure NumPy CPU path
                if aggregation_method == 'max':
                    result = np.max(donor_probs, axis=0)
                    
                elif aggregation_method == 'mean':
                    result = np.mean(donor_probs, axis=0)
                    
                elif aggregation_method == 'top_k_mean' and top_k:
                    sorted_probs = np.sort(donor_probs, axis=0)
                    top_k_actual = min(top_k, donor_probs.shape[0])
                    result = np.mean(sorted_probs[-top_k_actual:, :], axis=0)
                    
                elif aggregation_method == 'weighted_sum':
                    weights = donor_probs / (np.sum(donor_probs, axis=0, keepdims=True) + 1e-10)
                    result = np.sum(donor_probs * weights, axis=0)
                    
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregation_method}")
                
                aggregated[donor_id] = result
        
        # Convert to DataFrame
        donor_pred_df = pd.DataFrame.from_dict(aggregated, orient='index', columns=hla_cols)
        donor_pred_df.index.name = 'donor_id'
        
        print(f"Aggregated shape: {donor_pred_df.shape}")
        return donor_pred_df
    
    def load_ground_truth(self, ground_truth_file: str) -> pd.DataFrame:
        """
        Load ground truth HLA typing.
        
        Expected format: CSV/TSV with columns:
        - donor_id: donor identifier
        - HLA columns: binary (0/1) indicating presence/absence of each allele
        
        Returns:
        --------
        pd.DataFrame : Ground truth labels (donors × HLA alleles)
        """
        print(f"Loading ground truth from {ground_truth_file}...")
        
        if ground_truth_file.endswith('.csv'):
            gt_df = pd.read_csv(ground_truth_file, index_col='donor_id')
        elif ground_truth_file.endswith('.tsv'):
            gt_df = pd.read_csv(ground_truth_file, sep='\t', index_col='donor_id')
        elif ground_truth_file.endswith('.parquet'):
            gt_df = pd.read_parquet(ground_truth_file)
            gt_df = gt_df.set_index('donor_id')
        else:
            raise ValueError("Ground truth file must be CSV, TSV, or Parquet")
        
        print(f"Ground truth shape: {gt_df.shape}")
        return gt_df
    
    def align_predictions_and_truth(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align predictions and ground truth to have same donors and HLA alleles.
        """
        # Find common donors
        common_donors = predictions.index.intersection(ground_truth.index)
        print(f"Common donors: {len(common_donors)}")
        
        # Find common HLA alleles
        common_hlas = predictions.columns.intersection(ground_truth.columns)
        print(f"Common HLA alleles: {len(common_hlas)}")
        
        # Subset both dataframes
        pred_aligned = predictions.loc[common_donors, common_hlas]
        truth_aligned = ground_truth.loc[common_donors, common_hlas]
        
        return pred_aligned, truth_aligned
    
    def compute_metrics_per_allele(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        allele_name: str
    ) -> Dict:
        """
        Compute comprehensive metrics for a single HLA allele.
        """
        metrics = {
            'allele': allele_name,
            'n_positive': int(np.sum(y_true)),
            'n_negative': int(len(y_true) - np.sum(y_true)),
            'prevalence': float(np.mean(y_true))
        }
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['sensitivity'] = metrics['recall']
        
        # Specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['true_negatives'] = int(tn)
        metrics['false_negatives'] = int(fn)
        
        # ROC-AUC
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            metrics['auc_roc'] = auc(fpr, tpr)
            
            # Precision-Recall AUC
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
            metrics['auc_pr'] = auc(recall_curve, precision_curve)
            metrics['avg_precision'] = average_precision_score(y_true, y_scores)
        else:
            metrics['auc_roc'] = np.nan
            metrics['auc_pr'] = np.nan
            metrics['avg_precision'] = np.nan
        
        return metrics
    
    def evaluate_all_alleles(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Evaluate predictions for all HLA alleles (GPU-accelerated confusion counts via TensorFlow).
        """
        print(f"Evaluating {len(predictions.columns)} HLA alleles"
              f" {'on GPU...' if self.use_gpu else ' on CPU...'}")
        
        # Convert to numpy arrays for vectorization
        y_true_all = ground_truth.values.astype(np.int32)  # Shape: (n_donors, n_hlas)
        y_scores_all = predictions.values  # Shape: (n_donors, n_hlas)
        y_pred_all = (y_scores_all >= threshold).astype(np.int32)
        
        if self.use_gpu:
            # Move to TensorFlow for vectorized operations on GPU
            t_true = tf.convert_to_tensor(y_true_all, dtype=tf.int32)
            t_pred = tf.convert_to_tensor(y_pred_all, dtype=tf.int32)
            
            t_true_bool = tf.equal(t_true, 1)
            t_pred_bool = tf.equal(t_pred, 1)
            t_pred_neg_bool = tf.equal(t_pred, 0)
            t_true_neg_bool = tf.equal(t_true, 0)
            
            tp = tf.reduce_sum(
                tf.cast(tf.logical_and(t_true_bool, t_pred_bool), tf.int32),
                axis=0
            ).numpy()
            fp = tf.reduce_sum(
                tf.cast(tf.logical_and(t_true_neg_bool, t_pred_bool), tf.int32),
                axis=0
            ).numpy()
            tn = tf.reduce_sum(
                tf.cast(tf.logical_and(t_true_neg_bool, t_pred_neg_bool), tf.int32),
                axis=0
            ).numpy()
            fn = tf.reduce_sum(
                tf.cast(tf.logical_and(t_true_bool, t_pred_neg_bool), tf.int32),
                axis=0
            ).numpy()
        
        else:
            # CPU vectorized operations
            tp = np.sum((y_true_all == 1) & (y_pred_all == 1), axis=0)
            fp = np.sum((y_true_all == 0) & (y_pred_all == 1), axis=0)
            tn = np.sum((y_true_all == 0) & (y_pred_all == 0), axis=0)
            fn = np.sum((y_true_all == 1) & (y_pred_all == 0), axis=0)
        
        all_metrics = []
        
        # Compute metrics for each allele
        for idx, allele in enumerate(predictions.columns):
            y_true = y_true_all[:, idx]
            y_scores = y_scores_all[:, idx]
            
            n_pos = int(tp[idx] + fn[idx])
            n_neg = int(tn[idx] + fp[idx])
            
            metrics = {
                'allele': allele,
                'n_positive': n_pos,
                'n_negative': n_neg,
                'prevalence': float(np.mean(y_true)),
                'true_positives': int(tp[idx]),
                'false_positives': int(fp[idx]),
                'true_negatives': int(tn[idx]),
                'false_negatives': int(fn[idx])
            }
            
            # Compute derived metrics
            total = len(y_true)
            metrics['accuracy'] = (tp[idx] + tn[idx]) / total if total > 0 else 0.0
            metrics['precision'] = tp[idx] / (tp[idx] + fp[idx]) if (tp[idx] + fp[idx]) > 0 else 0.0
            metrics['recall'] = tp[idx] / (tp[idx] + fn[idx]) if (tp[idx] + fn[idx]) > 0 else 0.0
            metrics['sensitivity'] = metrics['recall']
            metrics['specificity'] = tn[idx] / (tn[idx] + fp[idx]) if (tn[idx] + fp[idx]) > 0 else 0.0
            
            # ROC-AUC (requires sklearn, can't be fully vectorized)
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                metrics['auc_roc'] = auc(fpr, tpr)
                
                # Precision-Recall AUC
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
                metrics['auc_pr'] = auc(recall_curve, precision_curve)
                metrics['avg_precision'] = average_precision_score(y_true, y_scores)
            else:
                metrics['auc_roc'] = np.nan
                metrics['auc_pr'] = np.nan
                metrics['avg_precision'] = np.nan
            
            all_metrics.append(metrics)
        
        metrics_df = pd.DataFrame(all_metrics)
        
        # Save metrics
        metrics_file = self.metrics_dir / "per_allele_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Saved per-allele metrics to {metrics_file}")
        
        return metrics_df
    
    def plot_roc_curves(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame,
        selected_alleles: List[str] = None,
        n_alleles: int = 10
    ):
        """
        Plot ROC curves for selected HLA alleles.
        """
        if selected_alleles is None:
            # Select most prevalent alleles
            prevalences = ground_truth.sum(axis=0) / len(ground_truth)
            selected_alleles = prevalences.nlargest(n_alleles).index.tolist()
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx, allele in enumerate(selected_alleles[:10]):
            ax = axes[idx]
            
            y_true = ground_truth[allele].values
            y_scores = predictions[allele].values
            
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
                ax.plot([0, 1], [0, 1], 'k--', lw=1)
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'{allele}')
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(
                    0.5, 0.5, 'Insufficient data',
                    ha='center', va='center', transform=ax.transAxes
                )
                ax.set_title(f'{allele}')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "roc_curves_top_alleles.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved ROC curves to {self.plots_dir / 'roc_curves_top_alleles.png'}")
    
    def plot_precision_recall_curves(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame,
        selected_alleles: List[str] = None,
        n_alleles: int = 10
    ):
        """
        Plot Precision-Recall curves for selected HLA alleles.
        """
        if selected_alleles is None:
            prevalences = ground_truth.sum(axis=0) / len(ground_truth)
            selected_alleles = prevalences.nlargest(n_alleles).index.tolist()
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx, allele in enumerate(selected_alleles[:10]):
            ax = axes[idx]
            
            y_true = ground_truth[allele].values
            y_scores = predictions[allele].values
            
            if len(np.unique(y_true)) > 1:
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                avg_precision = average_precision_score(y_true, y_scores)
                
                ax.plot(recall, precision, lw=2, label=f'AP = {avg_precision:.3f}')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title(f'{allele}')
                ax.legend(loc='lower left')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(
                    0.5, 0.5, 'Insufficient data',
                    ha='center', va='center', transform=ax.transAxes
                )
                ax.set_title(f'{allele}')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "pr_curves_top_alleles.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved PR curves to {self.plots_dir / 'pr_curves_top_alleles.png'}")
    
    def plot_performance_summary(self, metrics_df: pd.DataFrame):
        """
        Create summary visualizations of performance across all alleles.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. AUC-ROC distribution
        ax = axes[0, 0]
        valid_aucs = metrics_df['auc_roc'].dropna()
        ax.hist(valid_aucs, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(
            valid_aucs.mean(), color='red', linestyle='--',
            label=f'Mean: {valid_aucs.mean():.3f}'
        )
        ax.set_xlabel('AUC-ROC')
        ax.set_ylabel('Number of Alleles')
        ax.set_title('Distribution of AUC-ROC Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Precision vs Prevalence
        ax = axes[0, 1]
        ax.scatter(metrics_df['prevalence'], metrics_df['precision'], alpha=0.6)
        ax.set_xlabel('Allele Prevalence')
        ax.set_ylabel('Precision')
        ax.set_title('Precision vs Allele Prevalence')
        ax.grid(True, alpha=0.3)
        
        # Add correlation
        corr, pval = stats.pearsonr(
            metrics_df['prevalence'].dropna(),
            metrics_df['precision'].dropna()
        )
        ax.text(
            0.05, 0.95, f'ρ = {corr:.3f}\np = {pval:.2e}',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        # 3. Sensitivity vs Specificity
        ax = axes[0, 2]
        ax.scatter(metrics_df['specificity'], metrics_df['sensitivity'], alpha=0.6)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('Specificity')
        ax.set_ylabel('Sensitivity')
        ax.set_title('Sensitivity vs Specificity')
        ax.grid(True, alpha=0.3)
        
        # 4. Performance metrics by HLA locus
        ax = axes[1, 0]
        metrics_df['locus'] = metrics_df['allele'].str.split('*').str[0]
        locus_auc = metrics_df.groupby('locus')['auc_roc'].mean().sort_values()
        locus_auc.plot(kind='barh', ax=ax)
        ax.set_xlabel('Mean AUC-ROC')
        ax.set_title('Performance by HLA Locus')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 5. Number of positives vs AUC
        ax = axes[1, 1]
        ax.scatter(metrics_df['n_positive'], metrics_df['auc_roc'], alpha=0.6)
        ax.set_xlabel('Number of Positive Samples')
        ax.set_ylabel('AUC-ROC')
        ax.set_title('AUC-ROC vs Sample Size')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # 6. Top and bottom performing alleles
        ax = axes[1, 2]
        top_bottom = pd.concat([
            metrics_df.nlargest(5, 'auc_roc')[['allele', 'auc_roc']],
            metrics_df.nsmallest(5, 'auc_roc')[['allele', 'auc_roc']]
        ])
        colors = ['green'] * 5 + ['red'] * 5
        ax.barh(range(len(top_bottom)), top_bottom['auc_roc'].values, color=colors, alpha=0.6)
        ax.set_yticks(range(len(top_bottom)))
        ax.set_yticklabels(top_bottom['allele'].values)
        ax.set_xlabel('AUC-ROC')
        ax.set_title('Top 5 and Bottom 5 Alleles')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "performance_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved performance summary to {self.plots_dir / 'performance_summary.png'}")
    
    def compare_aggregation_methods(
        self,
        parquet_file: str,
        ground_truth: pd.DataFrame,
        methods: List[str] = ['max', 'mean', 'top_k_mean'],
        top_k_values: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Compare different aggregation methods.
        """
        print("\n" + "="*60)
        print("COMPARING AGGREGATION METHODS")
        print("="*60)
        
        results = []
        
        for method in methods:
            if method == 'top_k_mean':
                for k in top_k_values:
                    print(f"\nEvaluating: {method} with k={k}")
                    predictions = self.aggregate_tcr_to_donor(
                        parquet_file,
                        aggregation_method=method,
                        top_k=k
                    )
                    pred_aligned, truth_aligned = self.align_predictions_and_truth(
                        predictions, ground_truth
                    )
                    metrics = self.evaluate_all_alleles(pred_aligned, truth_aligned)
                    
                    results.append({
                        'method': f'{method}_k{k}',
                        'mean_auc_roc': metrics['auc_roc'].mean(),
                        'median_auc_roc': metrics['auc_roc'].median(),
                        'mean_precision': metrics['precision'].mean(),
                        'mean_recall': metrics['recall'].mean(),
                        'mean_accuracy': metrics['accuracy'].mean()
                    })
            else:
                print(f"\nEvaluating: {method}")
                predictions = self.aggregate_tcr_to_donor(
                    parquet_file,
                    aggregation_method=method
                )
                pred_aligned, truth_aligned = self.align_predictions_and_truth(
                    predictions, ground_truth
                )
                metrics = self.evaluate_all_alleles(pred_aligned, truth_aligned)
                
                results.append({
                    'method': method,
                    'mean_auc_roc': metrics['auc_roc'].mean(),
                    'median_auc_roc': metrics['auc_roc'].median(),
                    'mean_precision': metrics['precision'].mean(),
                    'mean_recall': metrics['recall'].mean(),
                    'mean_accuracy': metrics['accuracy'].mean()
                })
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('mean_auc_roc', ascending=False)
        
        # Save comparison
        comparison_file = self.metrics_dir / "aggregation_method_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        print(f"\nSaved aggregation comparison to {comparison_file}")
        
        return comparison_df
    
    def generate_report(self, metrics_df: pd.DataFrame):
        """
        Generate a text report summarizing the evaluation.
        """
        report = []
        report.append("="*70)
        report.append("HLA PREDICTION EVALUATION REPORT")
        report.append("="*70)
        report.append("")
        
        report.append(f"Total number of HLA alleles evaluated: {len(metrics_df)}")
        report.append(f"Total number of donors: {metrics_df['n_positive'].sum() + metrics_df['n_negative'].sum()}")
        report.append("")
        
        report.append("OVERALL PERFORMANCE METRICS:")
        report.append("-" * 70)
        report.append(f"Mean AUC-ROC:        {metrics_df['auc_roc'].mean():.4f} ± {metrics_df['auc_roc'].std():.4f}")
        report.append(f"Median AUC-ROC:      {metrics_df['auc_roc'].median():.4f}")
        report.append(f"Mean Precision:      {metrics_df['precision'].mean():.4f}")
        report.append(f"Mean Recall:         {metrics_df['recall'].mean():.4f}")
        report.append(f"Mean Accuracy:       {metrics_df['accuracy'].mean():.4f}")
        report.append(f"Mean Specificity:    {metrics_df['specificity'].mean():.4f}")
        report.append("")
        
        report.append("TOP 10 BEST PERFORMING ALLELES (by AUC-ROC):")
        report.append("-" * 70)
        top10 = metrics_df.nlargest(10, 'auc_roc')[['allele', 'auc_roc', 'prevalence', 'n_positive']]
        for _, row in top10.iterrows():
            report.append(
                f"  {row['allele']:20s}  AUC: {row['auc_roc']:.4f}  "
                f"Prev: {row['prevalence']:.3f}  N+: {row['n_positive']}"
            )
        report.append("")
        
        report.append("BOTTOM 10 WORST PERFORMING ALLELES (by AUC-ROC):")
        report.append("-" * 70)
        bottom10 = metrics_df.nsmallest(10, 'auc_roc')[['allele', 'auc_roc', 'prevalence', 'n_positive']]
        for _, row in bottom10.iterrows():
            report.append(
                f"  {row['allele']:20s}  AUC: {row['auc_roc']:.4f}  "
                f"Prev: {row['prevalence']:.3f}  N+: {row['n_positive']}"
            )
        report.append("")
        
        # Performance by HLA class
        metrics_df['hla_class'] = metrics_df['allele'].apply(
            lambda x: 'Class I' if x.split('*')[0] in ['A', 'B', 'C'] else 'Class II'
        )
        report.append("PERFORMANCE BY HLA CLASS:")
        report.append("-" * 70)
        for hla_class in ['Class I', 'Class II']:
            class_metrics = metrics_df[metrics_df['hla_class'] == hla_class]
            report.append(f"{hla_class}:")
            report.append(f"  Mean AUC-ROC: {class_metrics['auc_roc'].mean():.4f}")
            report.append(f"  N alleles:    {len(class_metrics)}")
        
        report.append("")
        report.append("="*70)
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = self.output_dir / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print("\n" + report_text)
        print(f"\nSaved evaluation report to {report_file}")


def main():
    """
    Main execution function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate donor-level HLA predictions from TCR-level probabilities (TensorFlow-accelerated)'
    )
    parser.add_argument(
        'tcr_predictions',
        help='Path to parquet file with TCR-level HLA predictions'
    )
    parser.add_argument(
        'ground_truth',
        help='Path to ground truth HLA typing (CSV/TSV/Parquet)'
    )
    parser.add_argument(
        'output_dir',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--aggregation-method',
        default='max',
        choices=['max', 'mean', 'top_k_mean', 'weighted_sum'],
        help='Method to aggregate TCR predictions to donor level'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of top TCRs to use for top_k_mean aggregation'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Probability threshold for binary classification'
    )
    parser.add_argument(
        '--compare-methods',
        action='store_true',
        help='Compare different aggregation methods'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100000,
        help='Chunk size for processing large parquet files'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration'
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = DonorLevelHLAEvaluator(args.output_dir, use_gpu=not args.no_gpu)
    
    # Load ground truth
    ground_truth = evaluator.load_ground_truth(args.ground_truth)
    
    if args.compare_methods:
        # Compare aggregation methods
        comparison = evaluator.compare_aggregation_methods(
            args.tcr_predictions,
            ground_truth
        )
        print("\n" + "="*60)
        print("AGGREGATION METHOD COMPARISON:")
        print(comparison.to_string(index=False))
        
    else:
        # Single evaluation with specified method
        print("\n" + "="*60)
        print("DONOR-LEVEL HLA PREDICTION EVALUATION (TensorFlow-ACCELERATED)")
        print("="*60)
        
        # Aggregate predictions
        predictions = evaluator.aggregate_tcr_to_donor(
            args.tcr_predictions,
            aggregation_method=args.aggregation_method,
            top_k=args.top_k if args.aggregation_method == 'top_k_mean' else None,
            chunk_size=args.chunk_size
        )
        
        # Save aggregated predictions
        pred_file = evaluator.predictions_dir / "donor_level_predictions.csv"
        predictions.to_csv(pred_file)
        print(f"Saved donor-level predictions to {pred_file}")
        
        # Align with ground truth
        pred_aligned, truth_aligned = evaluator.align_predictions_and_truth(
            predictions, ground_truth
        )
        
        # Evaluate
        metrics_df = evaluator.evaluate_all_alleles(
            pred_aligned, truth_aligned, threshold=args.threshold
        )
        
        # Generate visualizations
        evaluator.plot_roc_curves(pred_aligned, truth_aligned)
        evaluator.plot_precision_recall_curves(pred_aligned, truth_aligned)
        evaluator.plot_performance_summary(metrics_df)
        
        # Generate report
        evaluator.generate_report(metrics_df)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
