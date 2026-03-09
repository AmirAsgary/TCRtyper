#!/usr/bin/env python3
"""
Recompute PR/ROC performance curves from saved model weights.
Reads existing outputs (model.weights.h5, config.json) and original data,
then regenerates only the performance evaluation with corrected curve logic.

Usage:
    python recompute_performance_curves.py \
        --results_dir outputs/synthetic_25_2_25_withreg0001 \
        --donor_matrix data/autotcr/donor_hla_matrix.npz
"""
import os, sys, json, argparse, glob
import numpy as np
import tensorflow as tf
from pathlib import Path
# Import model and utilities
from utils import (
    SparseTCRModel, pad_list_to_array,
    PublicTcrHlaCsrReader, PublicTcrHlaCsrReaderChunk,
    evaluate_model_performance
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Recompute performance curves from saved weights')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Root directory containing all bX_nY result folders')
    parser.add_argument('--donor_matrix', type=str, required=True,
                        help='Path to donor HLA matrix (.npz)')
    parser.add_argument('--pad_token', type=float, default=-1.0,
                        help='Padding token value (default: -1.0)')
    parser.add_argument('--filter', type=str, default=None,
                        help='Optional glob pattern to filter runs (e.g. "b10_*")')
    return parser.parse_args()


def load_h5_sparse_indices(h5_path, pad_token=-1.0):
    """Load sparse indices from H5, trying new format first then legacy."""
    try:
        with PublicTcrHlaCsrReaderChunk(str(h5_path)) as reader:
            counts_set, max_all = reader.read_sparse_indices_of_counts()
    except KeyError:
        with PublicTcrHlaCsrReader(str(h5_path)) as reader:
            counts_set, max_all = reader.read_sparse_indices()
    return pad_list_to_array(counts_set, max_all, pad_token)


def reconstruct_model(config, binder_sets, donor_hla_matrix, pad_token=-1.0):
    """Reconstruct SparseTCRModel from config and data arrays without training."""
    num_tcrs = binder_sets.shape[0]
    max_hlas_per_tcr = binder_sets.shape[1]
    # Build model with same hyperparameters
    model = SparseTCRModel(
        num_tcrs=num_tcrs,
        max_hlas_per_tcr=max_hlas_per_tcr,
        donor_hla_matrix=donor_hla_matrix,
        binder_sets=binder_sets,
        beta=config.get('beta', 4.0),
        mode='continuous',
        pad_token=pad_token,
        l2_reg_lambda=config.get('l2_reg', 1e-5))
    # Build the model by calling it once with dummy data
    dummy_idx = tf.constant([0], dtype=tf.int32)
    dummy_donors = tf.constant([[0]], dtype=tf.int32)
    _ = model((dummy_idx, dummy_donors))
    return model


def process_single_run(run_dir, donor_hla_matrix, pad_token=-1.0):
    """Recompute performance curves for a single bX_nY run."""
    run_dir = Path(run_dir)
    run_name = run_dir.name
    # Check required files exist
    config_path = run_dir / 'config.json'
    weights_path = run_dir / 'model.weights.h5'
    if not config_path.exists() or not weights_path.exists():
        print(f"  SKIP {run_name}: missing config.json or model.weights.h5")
        return None
    # Load training config to find original data_dir
    with open(config_path, 'r') as f:
        config = json.load(f)
    data_dir = Path(config['data_dir'])
    if not data_dir.exists():
        print(f"  SKIP {run_name}: data_dir not found: {data_dir}")
        return None
    # Load data arrays
    h5_path = data_dir / 'synthetic_tcr_hla_counts.h5'
    true_hla_set = np.load(data_dir / 'synthetic_binder_sets.npy', mmap_mode='r')
    binder_sets = load_h5_sparse_indices(h5_path, pad_token)
    num_alleles = donor_hla_matrix.shape[1]
    # Reconstruct model and load saved weights
    model = reconstruct_model(config, binder_sets, donor_hla_matrix, pad_token)
    model.load_weights(str(weights_path))
    # Run performance evaluation with corrected curves
    figures_path = run_dir / 'figures'
    os.makedirs(figures_path, exist_ok=True)
    perf_metrics = evaluate_model_performance(
        model=model,
        binder_sets=binder_sets,
        true_hla_set=np.array(true_hla_set),
        num_total_alleles=num_alleles,
        output_path=str(figures_path),
        pad_token=pad_token)
    # Update final_metrics_summary.json with corrected values
    summary_path = run_dir / 'final_metrics_summary.json'
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        summary['auc_roc'] = float(perf_metrics['auc'])
        summary['average_precision'] = float(perf_metrics['ap'])
        summary['best_f1_score'] = float(perf_metrics['best_f1'])
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
    # Free memory
    del model, binder_sets, true_hla_set
    tf.keras.backend.clear_session()
    return perf_metrics


def main():
    """Iterate over all bX_nY runs and recompute performance curves."""
    args = parse_args()
    results_dir = Path(args.results_dir)
    # Load shared donor matrix once
    donor_hla_matrix = np.load(args.donor_matrix)['donor_hla_matrix']
    print(f"Donor matrix: {donor_hla_matrix.shape}")
    # Find all bX_nY subdirectories
    if args.filter:
        pattern = str(results_dir / args.filter)
    else:
        pattern = str(results_dir / 'b*_n*')
    run_dirs = sorted(glob.glob(pattern))
    run_dirs = [d for d in run_dirs if os.path.isdir(d)]
    print(f"Found {len(run_dirs)} runs to process\n")
    # Process each run
    all_metrics = {}
    for i, run_dir in enumerate(run_dirs):
        run_name = Path(run_dir).name
        print(f"[{i+1}/{len(run_dirs)}] Processing {run_name}...")
        try:
            metrics = process_single_run(run_dir, donor_hla_matrix, args.pad_token)
            if metrics:
                all_metrics[run_name] = metrics
                print(f"  AUC={metrics['auc']:.4f}  AP={metrics['ap']:.4f}  F1={metrics['best_f1']:.4f}\n")
        except Exception as e:
            print(f"  ERROR {run_name}: {e}\n")
            all_metrics[run_name] = {'error': str(e)}
    # Save combined summary
    summary_path = results_dir / 'recomputed_performance_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nDone! Summary saved to: {summary_path}")
    print(f"Successfully processed {sum(1 for v in all_metrics.values() if 'error' not in v)}/{len(run_dirs)} runs")


if __name__ == '__main__':
    main()