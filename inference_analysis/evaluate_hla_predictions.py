#!/usr/bin/env python3
"""
Donor-Level HLA Prediction Evaluation (GPU-Accelerated, TensorFlow-based)
Aggregates TCR-level HLA binding probabilities to donor-level predictions
and evaluates performance against ground truth HLA typing.

Optimized version with:
- Single data loading for all aggregation methods
- Vectorized pandas groupby operations
- Per-patient heatmap visualizations
- Proper caching and method separation
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, confusion_matrix
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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
    Optimized for efficiency with single data loading and vectorized operations.
    """
    
    def __init__(self, output_dir: str, use_gpu: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu and USE_GPU
        
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.predictions_dir = self.output_dir / "predictions"
        self.predictions_dir.mkdir(exist_ok=True)
        
        self.heatmaps_dir = self.output_dir / "heatmaps"
        self.heatmaps_dir.mkdir(exist_ok=True)
        
        # Cache for loaded data
        self._cached_tcr_data: Optional[pd.DataFrame] = None
        self._cached_hla_cols: Optional[List[str]] = None
        self._cached_file_path: Optional[str] = None
        
        print(f"GPU acceleration {'ENABLED' if self.use_gpu else 'DISABLED'}")
    
    def load_tcr_predictions(self, parquet_file: str, force_reload: bool = False) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load TCR predictions from parquet file. Caches result for reuse.
        
        Returns:
            Tuple of (DataFrame with predictions, list of HLA column names)
        """
        if not force_reload and self._cached_tcr_data is not None and self._cached_file_path == parquet_file:
            print("Using cached TCR predictions data")
            return self._cached_tcr_data, self._cached_hla_cols
        
        print(f"Loading TCR predictions from {parquet_file}...")
        
        # Load entire parquet file at once - more efficient than chunking for in-memory ops
        df = pd.read_parquet(parquet_file)
        
        # Identify HLA columns
        hla_cols = [col for col in df.columns if col not in ['donor_id', 'tcr_id']]
        print(f"Loaded {len(df):,} TCR predictions for {df['donor_id'].nunique():,} donors")
        print(f"Found {len(hla_cols)} HLA alleles")
        
        # Convert HLA columns to float32 for memory efficiency
        df[hla_cols] = df[hla_cols].astype(np.float32)
        
        # Cache the data
        self._cached_tcr_data = df
        self._cached_hla_cols = hla_cols
        self._cached_file_path = parquet_file
        
        return df, hla_cols
    
    def _aggregate_group_gpu(self, group_values: np.ndarray, method: str, top_k: int = None) -> np.ndarray:
        """GPU-accelerated aggregation for a single donor's TCR predictions."""
        t = tf.convert_to_tensor(group_values, dtype=tf.float32)
        
        if method == 'max':
            return tf.reduce_max(t, axis=0).numpy()
        elif method == 'mean':
            return tf.reduce_mean(t, axis=0).numpy()
        elif method == 'top_k_mean' and top_k:
            n_tcrs = t.shape[0]
            k = min(top_k, n_tcrs)
            # Top-k along axis 0 for each column
            top_vals, _ = tf.math.top_k(tf.transpose(t), k=k)
            return tf.reduce_mean(top_vals, axis=1).numpy()
        elif method == 'weighted_sum':
            weights = t / (tf.reduce_sum(t, axis=0, keepdims=True) + 1e-10)
            return tf.reduce_sum(t * weights, axis=0).numpy()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def _aggregate_group_cpu(self, group_values: np.ndarray, method: str, top_k: int = None) -> np.ndarray:
        """CPU aggregation for a single donor's TCR predictions."""
        if method == 'max':
            return np.max(group_values, axis=0)
        elif method == 'mean':
            return np.mean(group_values, axis=0)
        elif method == 'top_k_mean' and top_k:
            k = min(top_k, group_values.shape[0])
            # Partition is faster than full sort for top-k
            idx = np.argpartition(group_values, -k, axis=0)[-k:]
            top_vals = np.take_along_axis(group_values, idx, axis=0)
            return np.mean(top_vals, axis=0)
        elif method == 'weighted_sum':
            weights = group_values / (np.sum(group_values, axis=0, keepdims=True) + 1e-10)
            return np.sum(group_values * weights, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def aggregate_tcr_to_donor(
        self,
        tcr_data: Union[str, pd.DataFrame],
        aggregation_method: str = 'max',
        top_k: int = None,
        hla_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Aggregate TCR-level predictions to donor-level using vectorized operations.
        
        Parameters:
            tcr_data: Either path to parquet file or pre-loaded DataFrame
            aggregation_method: 'max', 'mean', 'top_k_mean', 'weighted_sum'
            top_k: Number of top TCRs for top_k_mean method
            hla_cols: List of HLA columns (required if tcr_data is DataFrame)
        """
        # Load data if path provided
        if isinstance(tcr_data, str):
            df, hla_cols = self.load_tcr_predictions(tcr_data)
        else:
            df = tcr_data
            if hla_cols is None:
                raise ValueError("hla_cols must be provided when tcr_data is DataFrame")
        
        print(f"Aggregating using '{aggregation_method}' method...")
        
        # Vectorized aggregation using pandas groupby
        grouped = df.groupby('donor_id')[hla_cols]
        
        if aggregation_method == 'max':
            donor_pred_df = grouped.max()
        elif aggregation_method == 'mean':
            donor_pred_df = grouped.mean()
        elif aggregation_method in ('top_k_mean', 'weighted_sum'):
            # These require custom aggregation
            agg_func = self._aggregate_group_gpu if self.use_gpu else self._aggregate_group_cpu
            
            results = {}
            donor_ids = df['donor_id'].unique()
            n_donors = len(donor_ids)
            
            for i, donor_id in enumerate(donor_ids):
                if i % 500 == 0:
                    print(f"  Processing donor {i+1}/{n_donors}...")
                
                donor_mask = df['donor_id'] == donor_id
                group_vals = df.loc[donor_mask, hla_cols].values
                results[donor_id] = agg_func(group_vals, aggregation_method, top_k)
            
            donor_pred_df = pd.DataFrame.from_dict(results, orient='index', columns=hla_cols)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        donor_pred_df.index.name = 'donor_id'
        print(f"Aggregated shape: {donor_pred_df.shape}")
        return donor_pred_df
    
    def load_ground_truth(self, ground_truth_file: str) -> pd.DataFrame:
        """Load ground truth HLA typing from CSV/TSV/Parquet."""
        print(f"Loading ground truth from {ground_truth_file}...")
        
        ext = Path(ground_truth_file).suffix.lower()
        if ext == '.csv':
            gt_df = pd.read_csv(ground_truth_file, index_col='donor_id')
        elif ext == '.tsv':
            gt_df = pd.read_csv(ground_truth_file, sep='\t', index_col='donor_id')
        elif ext == '.parquet':
            gt_df = pd.read_parquet(ground_truth_file).set_index('donor_id')
        else:
            raise ValueError("Ground truth must be CSV, TSV, or Parquet")
        
        print(f"Ground truth shape: {gt_df.shape}")
        return gt_df
    
    def align_predictions_and_truth(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align predictions and ground truth to common donors and HLA alleles."""
        common_donors = predictions.index.intersection(ground_truth.index)
        common_hlas = predictions.columns.intersection(ground_truth.columns)
        
        print(f"Common donors: {len(common_donors)}, Common HLAs: {len(common_hlas)}")
        
        return (
            predictions.loc[common_donors, common_hlas],
            ground_truth.loc[common_donors, common_hlas]
        )
    
    def evaluate_all_alleles(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame,
        threshold: float = 0.5,
        method_name: str = ""
    ) -> pd.DataFrame:
        """
        Evaluate predictions for all HLA alleles with vectorized operations.
        """
        print(f"Evaluating {len(predictions.columns)} HLA alleles...")
        
        y_true_all = ground_truth.values.astype(np.int32)
        y_scores_all = predictions.values.astype(np.float32)
        y_pred_all = (y_scores_all >= threshold).astype(np.int32)
        
        # Vectorized confusion matrix computation
        if self.use_gpu:
            t_true = tf.convert_to_tensor(y_true_all, dtype=tf.int32)
            t_pred = tf.convert_to_tensor(y_pred_all, dtype=tf.int32)
            
            t_true_pos = tf.equal(t_true, 1)
            t_true_neg = tf.equal(t_true, 0)
            t_pred_pos = tf.equal(t_pred, 1)
            t_pred_neg = tf.equal(t_pred, 0)
            
            tp = tf.reduce_sum(tf.cast(t_true_pos & t_pred_pos, tf.int32), axis=0).numpy()
            fp = tf.reduce_sum(tf.cast(t_true_neg & t_pred_pos, tf.int32), axis=0).numpy()
            tn = tf.reduce_sum(tf.cast(t_true_neg & t_pred_neg, tf.int32), axis=0).numpy()
            fn = tf.reduce_sum(tf.cast(t_true_pos & t_pred_neg, tf.int32), axis=0).numpy()
        else:
            tp = np.sum((y_true_all == 1) & (y_pred_all == 1), axis=0)
            fp = np.sum((y_true_all == 0) & (y_pred_all == 1), axis=0)
            tn = np.sum((y_true_all == 0) & (y_pred_all == 0), axis=0)
            fn = np.sum((y_true_all == 1) & (y_pred_all == 0), axis=0)
        
        # Vectorized metric computation
        n_samples = y_true_all.shape[0]
        accuracy = (tp + tn) / n_samples
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
        specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) > 0)
        prevalence = np.mean(y_true_all, axis=0)
        
        # Compute AUC metrics (requires loop due to sklearn)
        auc_roc = np.full(len(predictions.columns), np.nan)
        auc_pr = np.full(len(predictions.columns), np.nan)
        avg_precision = np.full(len(predictions.columns), np.nan)
        
        for idx in range(len(predictions.columns)):
            y_true = y_true_all[:, idx]
            y_scores = y_scores_all[:, idx]
            
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                auc_roc[idx] = auc(fpr, tpr)
                prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_scores)
                auc_pr[idx] = auc(rec_curve, prec_curve)
                avg_precision[idx] = average_precision_score(y_true, y_scores)
        
        # Build metrics DataFrame
        metrics_df = pd.DataFrame({
            'allele': predictions.columns,
            'n_positive': tp + fn,
            'n_negative': tn + fp,
            'prevalence': prevalence,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'sensitivity': recall,
            'specificity': specificity,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'avg_precision': avg_precision
        })
        
        # Save with method name if provided
        suffix = f"_{method_name}" if method_name else ""
        metrics_file = self.metrics_dir / f"per_allele_metrics{suffix}.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Saved metrics to {metrics_file}")
        
        return metrics_df
    
    def plot_per_patient_heatmaps(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame = None,
        n_patients: int = 50,
        n_alleles: int = 30,
        figsize: Tuple[int, int] = (16, 12),
        method_name: str = ""
    ):
        """
        Generate per-patient heatmaps showing aggregated HLA scores.
        
        Parameters:
            predictions: Donor-level predictions DataFrame
            ground_truth: Optional ground truth for comparison
            n_patients: Number of patients to show
            n_alleles: Number of top HLA alleles to show
            figsize: Figure size
            method_name: Name for file suffix
        """
        print("Generating per-patient heatmaps...")
        
        # Select top alleles by variance (most informative)
        allele_variance = predictions.var(axis=0).sort_values(ascending=False)
        top_alleles = allele_variance.head(n_alleles).index.tolist()
        
        # Select subset of patients
        patient_subset = predictions.index[:n_patients]
        pred_subset = predictions.loc[patient_subset, top_alleles]
        
        suffix = f"_{method_name}" if method_name else ""
        
        # Heatmap 1: Predictions only
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            pred_subset,
            cmap='YlOrRd',
            vmin=0, vmax=1,
            xticklabels=True,
            yticklabels=True,
            cbar_kws={'label': 'Aggregated Score'},
            ax=ax
        )
        ax.set_xlabel('HLA Allele')
        ax.set_ylabel('Patient ID')
        ax.set_title(f'Per-Patient HLA Prediction Scores{" (" + method_name + ")" if method_name else ""}')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(self.heatmaps_dir / f"patient_hla_scores{suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Heatmap 2: If ground truth provided, show side-by-side comparison
        if ground_truth is not None:
            common_patients = pred_subset.index.intersection(ground_truth.index)
            common_alleles = pred_subset.columns.intersection(ground_truth.columns)
            
            if len(common_patients) > 0 and len(common_alleles) > 0:
                pred_aligned = pred_subset.loc[common_patients, common_alleles]
                truth_aligned = ground_truth.loc[common_patients, common_alleles]
                
                fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 1.2, figsize[1]))
                
                # Predictions
                sns.heatmap(
                    pred_aligned,
                    cmap='YlOrRd', vmin=0, vmax=1,
                    xticklabels=True, yticklabels=True,
                    cbar_kws={'label': 'Pred Score'}, ax=axes[0]
                )
                axes[0].set_title('Predictions')
                axes[0].set_xlabel('HLA Allele')
                axes[0].set_ylabel('Patient ID')
                axes[0].tick_params(axis='x', rotation=45)
                
                # Ground truth
                sns.heatmap(
                    truth_aligned,
                    cmap='Blues', vmin=0, vmax=1,
                    xticklabels=True, yticklabels=True,
                    cbar_kws={'label': 'True Label'}, ax=axes[1]
                )
                axes[1].set_title('Ground Truth')
                axes[1].set_xlabel('HLA Allele')
                axes[1].set_ylabel('Patient ID')
                axes[1].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig(self.heatmaps_dir / f"patient_comparison{suffix}.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                # Heatmap 3: Difference (prediction - truth)
                diff = pred_aligned.values - truth_aligned.values
                diff_df = pd.DataFrame(diff, index=pred_aligned.index, columns=pred_aligned.columns)
                
                fig, ax = plt.subplots(figsize=figsize)
                sns.heatmap(
                    diff_df,
                    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    xticklabels=True, yticklabels=True,
                    cbar_kws={'label': 'Prediction - Truth'}, ax=ax
                )
                ax.set_title(f'Prediction Error (Red=FP, Blue=FN){" (" + method_name + ")" if method_name else ""}')
                ax.set_xlabel('HLA Allele')
                ax.set_ylabel('Patient ID')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.yticks(fontsize=8)
                plt.tight_layout()
                plt.savefig(self.heatmaps_dir / f"patient_error{suffix}.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        # Heatmap 4: Clustered heatmap
        fig = plt.figure(figsize=(figsize[0], figsize[1] + 2))
        g = sns.clustermap(
            pred_subset,
            cmap='YlOrRd', vmin=0, vmax=1,
            method='ward',
            figsize=figsize,
            cbar_kws={'label': 'Score'},
            xticklabels=True,
            yticklabels=True
        )
        g.ax_heatmap.set_xlabel('HLA Allele')
        g.ax_heatmap.set_ylabel('Patient ID')
        plt.suptitle(f'Clustered Patient-HLA Heatmap{" (" + method_name + ")" if method_name else ""}', y=1.02)
        plt.savefig(self.heatmaps_dir / f"patient_clustered{suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved heatmaps to {self.heatmaps_dir}")
    
    def plot_roc_curves(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame,
        selected_alleles: List[str] = None,
        n_alleles: int = 10,
        method_name: str = ""
    ):
        """Plot ROC curves for selected HLA alleles."""
        if selected_alleles is None:
            prevalences = ground_truth.sum(axis=0) / len(ground_truth)
            selected_alleles = prevalences.nlargest(n_alleles).index.tolist()
        
        n_plots = min(len(selected_alleles), 10)
        n_cols = min(5, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes).flatten()
        
        for idx, allele in enumerate(selected_alleles[:n_plots]):
            ax = axes[idx]
            y_true = ground_truth[allele].values
            y_scores = predictions[allele].values
            
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
                ax.plot([0, 1], [0, 1], 'k--', lw=1)
                ax.legend(loc='lower right')
            else:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.set_title(f'{allele}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused axes
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        suffix = f"_{method_name}" if method_name else ""
        plt.savefig(self.plots_dir / f"roc_curves{suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curves(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame,
        selected_alleles: List[str] = None,
        n_alleles: int = 10,
        method_name: str = ""
    ):
        """Plot Precision-Recall curves for selected HLA alleles."""
        if selected_alleles is None:
            prevalences = ground_truth.sum(axis=0) / len(ground_truth)
            selected_alleles = prevalences.nlargest(n_alleles).index.tolist()
        
        n_plots = min(len(selected_alleles), 10)
        n_cols = min(5, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes).flatten()
        
        for idx, allele in enumerate(selected_alleles[:n_plots]):
            ax = axes[idx]
            y_true = ground_truth[allele].values
            y_scores = predictions[allele].values
            
            if len(np.unique(y_true)) > 1:
                prec, rec, _ = precision_recall_curve(y_true, y_scores)
                ap = average_precision_score(y_true, y_scores)
                ax.plot(rec, prec, lw=2, label=f'AP = {ap:.3f}')
                ax.legend(loc='lower left')
            else:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'{allele}')
            ax.grid(True, alpha=0.3)
        
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        suffix = f"_{method_name}" if method_name else ""
        plt.savefig(self.plots_dir / f"pr_curves{suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_summary(self, metrics_df: pd.DataFrame, method_name: str = ""):
        """Create summary visualizations of performance across all alleles."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. AUC-ROC distribution
        ax = axes[0, 0]
        valid_aucs = metrics_df['auc_roc'].dropna()
        ax.hist(valid_aucs, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(valid_aucs.mean(), color='red', linestyle='--', label=f'Mean: {valid_aucs.mean():.3f}')
        ax.set_xlabel('AUC-ROC')
        ax.set_ylabel('Number of Alleles')
        ax.set_title('AUC-ROC Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Precision vs Prevalence
        ax = axes[0, 1]
        ax.scatter(metrics_df['prevalence'], metrics_df['precision'], alpha=0.6)
        ax.set_xlabel('Allele Prevalence')
        ax.set_ylabel('Precision')
        ax.set_title('Precision vs Prevalence')
        ax.grid(True, alpha=0.3)
        
        mask = metrics_df['prevalence'].notna() & metrics_df['precision'].notna()
        if mask.sum() > 2:
            corr, pval = stats.pearsonr(metrics_df.loc[mask, 'prevalence'], metrics_df.loc[mask, 'precision'])
            ax.text(0.05, 0.95, f'ρ = {corr:.3f}\np = {pval:.2e}', transform=ax.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Sensitivity vs Specificity
        ax = axes[0, 2]
        ax.scatter(metrics_df['specificity'], metrics_df['sensitivity'], alpha=0.6)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('Specificity')
        ax.set_ylabel('Sensitivity')
        ax.set_title('Sensitivity vs Specificity')
        ax.grid(True, alpha=0.3)
        
        # 4. Performance by HLA locus
        ax = axes[1, 0]
        metrics_df = metrics_df.copy()
        metrics_df['locus'] = metrics_df['allele'].str.split('*').str[0]
        locus_auc = metrics_df.groupby('locus')['auc_roc'].mean().sort_values()
        locus_auc.plot(kind='barh', ax=ax)
        ax.set_xlabel('Mean AUC-ROC')
        ax.set_title('Performance by HLA Locus')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 5. Sample size vs AUC
        ax = axes[1, 1]
        ax.scatter(metrics_df['n_positive'], metrics_df['auc_roc'], alpha=0.6)
        ax.set_xlabel('Number of Positive Samples')
        ax.set_ylabel('AUC-ROC')
        ax.set_title('AUC-ROC vs Sample Size')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # 6. Top/bottom alleles
        ax = axes[1, 2]
        top5 = metrics_df.nlargest(5, 'auc_roc')[['allele', 'auc_roc']]
        bottom5 = metrics_df.nsmallest(5, 'auc_roc')[['allele', 'auc_roc']]
        combined = pd.concat([top5, bottom5])
        colors = ['green'] * 5 + ['red'] * 5
        ax.barh(range(len(combined)), combined['auc_roc'].values, color=colors, alpha=0.6)
        ax.set_yticks(range(len(combined)))
        ax.set_yticklabels(combined['allele'].values)
        ax.set_xlabel('AUC-ROC')
        ax.set_title('Top 5 / Bottom 5 Alleles')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        suffix = f"_{method_name}" if method_name else ""
        plt.savefig(self.plots_dir / f"performance_summary{suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_aggregation_methods(
        self,
        parquet_file: str,
        ground_truth: pd.DataFrame,
        methods: List[str] = None,
        top_k_values: List[int] = None,
        threshold: float = 0.5,
        generate_plots: bool = True
    ) -> pd.DataFrame:
        """
        Compare different aggregation methods efficiently (single data load).
        """
        if methods is None:
            methods = ['max', 'mean', 'top_k_mean']
        if top_k_values is None:
            top_k_values = [5, 10, 20]
        
        print("\n" + "=" * 60)
        print("COMPARING AGGREGATION METHODS")
        print("=" * 60)
        
        # Load data once
        tcr_data, hla_cols = self.load_tcr_predictions(parquet_file)
        
        results = []
        all_predictions = {}
        all_metrics = {}
        
        for method in methods:
            if method == 'top_k_mean':
                for k in top_k_values:
                    method_name = f'{method}_k{k}'
                    print(f"\nEvaluating: {method_name}")
                    
                    predictions = self.aggregate_tcr_to_donor(
                        tcr_data, aggregation_method=method, top_k=k, hla_cols=hla_cols
                    )
                    pred_aligned, truth_aligned = self.align_predictions_and_truth(predictions, ground_truth)
                    metrics = self.evaluate_all_alleles(pred_aligned, truth_aligned, threshold, method_name)
                    
                    all_predictions[method_name] = pred_aligned
                    all_metrics[method_name] = metrics
                    
                    results.append({
                        'method': method_name,
                        'mean_auc_roc': metrics['auc_roc'].mean(),
                        'median_auc_roc': metrics['auc_roc'].median(),
                        'std_auc_roc': metrics['auc_roc'].std(),
                        'mean_precision': metrics['precision'].mean(),
                        'mean_recall': metrics['recall'].mean(),
                        'mean_accuracy': metrics['accuracy'].mean(),
                        'mean_specificity': metrics['specificity'].mean()
                    })
                    
                    if generate_plots:
                        self.plot_per_patient_heatmaps(pred_aligned, truth_aligned, method_name=method_name)
            else:
                print(f"\nEvaluating: {method}")
                
                predictions = self.aggregate_tcr_to_donor(
                    tcr_data, aggregation_method=method, hla_cols=hla_cols
                )
                pred_aligned, truth_aligned = self.align_predictions_and_truth(predictions, ground_truth)
                metrics = self.evaluate_all_alleles(pred_aligned, truth_aligned, threshold, method)
                
                all_predictions[method] = pred_aligned
                all_metrics[method] = metrics
                
                results.append({
                    'method': method,
                    'mean_auc_roc': metrics['auc_roc'].mean(),
                    'median_auc_roc': metrics['auc_roc'].median(),
                    'std_auc_roc': metrics['auc_roc'].std(),
                    'mean_precision': metrics['precision'].mean(),
                    'mean_recall': metrics['recall'].mean(),
                    'mean_accuracy': metrics['accuracy'].mean(),
                    'mean_specificity': metrics['specificity'].mean()
                })
                
                if generate_plots:
                    self.plot_per_patient_heatmaps(pred_aligned, truth_aligned, method_name=method)
        
        comparison_df = pd.DataFrame(results).sort_values('mean_auc_roc', ascending=False)
        comparison_df.to_csv(self.metrics_dir / "aggregation_comparison.csv", index=False)
        
        # Generate comparison visualization
        self._plot_method_comparison(comparison_df)
        
        print(f"\nSaved comparison to {self.metrics_dir / 'aggregation_comparison.csv'}")
        return comparison_df
    
    def _plot_method_comparison(self, comparison_df: pd.DataFrame):
        """Visualize comparison of aggregation methods."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # AUC-ROC comparison
        ax = axes[0]
        x = range(len(comparison_df))
        ax.bar(x, comparison_df['mean_auc_roc'], yerr=comparison_df['std_auc_roc'], capsize=3, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['method'], rotation=45, ha='right')
        ax.set_ylabel('Mean AUC-ROC')
        ax.set_title('AUC-ROC by Method')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Precision/Recall comparison
        ax = axes[1]
        width = 0.35
        ax.bar([i - width/2 for i in x], comparison_df['mean_precision'], width, label='Precision', alpha=0.7)
        ax.bar([i + width/2 for i in x], comparison_df['mean_recall'], width, label='Recall', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['method'], rotation=45, ha='right')
        ax.set_ylabel('Score')
        ax.set_title('Precision & Recall by Method')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Accuracy comparison
        ax = axes[2]
        ax.bar(x, comparison_df['mean_accuracy'], alpha=0.7, color='green')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['method'], rotation=45, ha='right')
        ax.set_ylabel('Mean Accuracy')
        ax.set_title('Accuracy by Method')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "method_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, metrics_df: pd.DataFrame, method_name: str = ""):
        """Generate a text report summarizing the evaluation."""
        lines = [
            "=" * 70,
            f"HLA PREDICTION EVALUATION REPORT{' (' + method_name + ')' if method_name else ''}",
            "=" * 70, "",
            f"Total HLA alleles evaluated: {len(metrics_df)}",
            f"Total donors: {int(metrics_df['n_positive'].iloc[0] + metrics_df['n_negative'].iloc[0])}",
            "",
            "OVERALL PERFORMANCE:",
            "-" * 70,
            f"Mean AUC-ROC:     {metrics_df['auc_roc'].mean():.4f} ± {metrics_df['auc_roc'].std():.4f}",
            f"Median AUC-ROC:   {metrics_df['auc_roc'].median():.4f}",
            f"Mean Precision:   {metrics_df['precision'].mean():.4f}",
            f"Mean Recall:      {metrics_df['recall'].mean():.4f}",
            f"Mean Accuracy:    {metrics_df['accuracy'].mean():.4f}",
            f"Mean Specificity: {metrics_df['specificity'].mean():.4f}",
            "",
            "TOP 10 ALLELES (by AUC-ROC):",
            "-" * 70
        ]
        
        for _, row in metrics_df.nlargest(10, 'auc_roc').iterrows():
            lines.append(f"  {row['allele']:20s}  AUC: {row['auc_roc']:.4f}  Prev: {row['prevalence']:.3f}  N+: {int(row['n_positive'])}")
        
        lines.extend(["", "BOTTOM 10 ALLELES (by AUC-ROC):", "-" * 70])
        for _, row in metrics_df.nsmallest(10, 'auc_roc').iterrows():
            lines.append(f"  {row['allele']:20s}  AUC: {row['auc_roc']:.4f}  Prev: {row['prevalence']:.3f}  N+: {int(row['n_positive'])}")
        
        # By HLA class
        metrics_df = metrics_df.copy()
        metrics_df['hla_class'] = metrics_df['allele'].apply(
            lambda x: 'Class I' if x.split('*')[0] in ['A', 'B', 'C'] else 'Class II'
        )
        
        lines.extend(["", "BY HLA CLASS:", "-" * 70])
        for hla_class in ['Class I', 'Class II']:
            subset = metrics_df[metrics_df['hla_class'] == hla_class]
            if len(subset) > 0:
                lines.append(f"{hla_class}: Mean AUC = {subset['auc_roc'].mean():.4f}, N alleles = {len(subset)}")
        
        lines.extend(["", "=" * 70])
        
        report_text = "\n".join(lines)
        suffix = f"_{method_name}" if method_name else ""
        report_file = self.output_dir / f"evaluation_report{suffix}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print("\n" + report_text)
        print(f"\nSaved report to {report_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate donor-level HLA predictions (optimized TensorFlow version)'
    )
    parser.add_argument('tcr_predictions', help='Path to parquet file with TCR predictions')
    parser.add_argument('ground_truth', help='Path to ground truth HLA typing')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--aggregation-method', default='max',
                        choices=['max', 'mean', 'top_k_mean', 'weighted_sum'])
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--compare-methods', action='store_true')
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--n-heatmap-patients', type=int, default=50)
    parser.add_argument('--n-heatmap-alleles', type=int, default=30)
    
    args = parser.parse_args()
    
    evaluator = DonorLevelHLAEvaluator(args.output_dir, use_gpu=not args.no_gpu)
    ground_truth = evaluator.load_ground_truth(args.ground_truth)
    
    if args.compare_methods:
        comparison = evaluator.compare_aggregation_methods(args.tcr_predictions, ground_truth)
        print("\n" + "=" * 60)
        print("AGGREGATION METHOD COMPARISON:")
        print(comparison.to_string(index=False))
    else:
        print("\n" + "=" * 60)
        print("DONOR-LEVEL HLA PREDICTION EVALUATION")
        print("=" * 60)
        
        # Load data once
        tcr_data, hla_cols = evaluator.load_tcr_predictions(args.tcr_predictions)
        
        # Aggregate
        predictions = evaluator.aggregate_tcr_to_donor(
            tcr_data,
            aggregation_method=args.aggregation_method,
            top_k=args.top_k if args.aggregation_method == 'top_k_mean' else None,
            hla_cols=hla_cols
        )
        
        # Save predictions
        predictions.to_csv(evaluator.predictions_dir / "donor_predictions.csv")
        
        # Align and evaluate
        pred_aligned, truth_aligned = evaluator.align_predictions_and_truth(predictions, ground_truth)
        metrics_df = evaluator.evaluate_all_alleles(pred_aligned, truth_aligned, args.threshold)
        
        # Generate all visualizations
        evaluator.plot_roc_curves(pred_aligned, truth_aligned)
        evaluator.plot_precision_recall_curves(pred_aligned, truth_aligned)
        evaluator.plot_performance_summary(metrics_df)
        evaluator.plot_per_patient_heatmaps(
            pred_aligned, truth_aligned,
            n_patients=args.n_heatmap_patients,
            n_alleles=args.n_heatmap_alleles
        )
        evaluator.generate_report(metrics_df)
    
    print("\n" + "=" * 60)
    print(f"EVALUATION COMPLETE! Results: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()