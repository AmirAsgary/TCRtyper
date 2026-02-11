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
    plot_precision_at_k_heatmap, plot_precision_at_k_curves, save_metrics_json
)


def parse_args():
    parser = argparse.ArgumentParser(description='TCR-HLA Binding Model Training Pipeline')
    # Data input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--data_dir', type=str, help='Path to single dataset directory')
    input_group.add_argument('--df', type=str, help='Path to CSV/TSV/JSON file with multiple dataset configs')
    parser.add_argument('--donor_matrix', type=str, help='Path to donor HLA matrix (.npz). Required if --data_dir is used')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size (default: 512)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--beta', type=float, default=4.0, help='Beta hyperparameter (default: 4.0)')
    parser.add_argument('--l2_reg', type=float, default=1e-5, help='L2 regularization lambda (default: 1e-5)')
    parser.add_argument('--pad_token', type=float, default=-1.0, help='Padding token value (default: -1.0)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold (default: 0.5)')
    # Analysis flags
    parser.add_argument('--analyze_all', action='store_true', help='Run all analysis modules')
    parser.add_argument('--analyze_donors', action='store_true', help='Run donor explanation analysis')
    parser.add_argument('--analyze_predictions', action='store_true', help='Run model predictions analysis')
    parser.add_argument('--analyze_performance', action='store_true', help='Run PR/ROC performance analysis')
    parser.add_argument('--analyze_precision_k', action='store_true', help='Run Precision@k analysis')
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
    """Load dataset and return all required arrays."""
    from dataset_processing.utils import PublicTcrHlaCsrReader
    data_dir = Path(data_dir)
    print(f"Loading data from {data_dir}...")
    # Load binder sets (ground truth)
    true_hla_set = np.load(data_dir / "synthetic_binder_sets.npy", mmap_mode="r")
    # Load donor indices
    donor_indices = np.load(data_dir / "synthetic_donor_indices.npy", mmap_mode="r")
    # Load counts from H5 file
    h5_path = data_dir / 'synthetic_tcr_hla_counts.h5'
    with PublicTcrHlaCsrReader(str(h5_path)) as reader:
        counts_set, max_all = reader.read_sparse_indices()
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
        beta=args.beta, mode='continuous', pad_token=args.pad_token, l2_reg_lambda=args.l2_reg)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer)
    # Train
    print("Starting training...")
    history = model.fit(train_dataset, epochs=args.epochs, verbose=args.verbose)
    # Save model and history
    model.save(os.path.join(output_path, 'model.keras'))
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
    # Save final metrics summary
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
