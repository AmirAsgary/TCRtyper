#!/usr/bin/env python3
"""
TCRtyper: High-Performance Vectorized MLE Pipeline (SciPy Edition)
Author: Gemini (Adapted for TCR-HLA Statistical Model)

This script performs deterministic Maximum Likelihood Estimation.
It treats each TCR i as an independent optimization problem.
"""

import os
import json
import time
import argparse
import shutil
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import gammaln
from joblib import Parallel, delayed
from tqdm import tqdm

# Import utility classes from your provided utils.py
from utils import (
    PublicTcrHlaCsrReaderChunk, 
    MleZprobsWriter, 
    NumpyEncoder
)

def parse_args():
    parser = argparse.ArgumentParser(description="Deterministic High-Speed SciPy MLE.")
    # Standard I/O
    parser.add_argument("--h5_data_path", type=str, required=True, help="Input H5 dataset")
    parser.add_argument("--donor_matrix_path", type=str, required=True, help="Donor NPZ matrix")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    # Statistical Hyperparameters
    parser.add_argument("--beta", type=float, default=4.0, help="Beta-Binomial prior parameter")
    parser.add_argument("--l2_reg", type=float, default=2e-7, help="L2 regularization strength")
    parser.add_argument("--pad_token", type=float, default=-1.0, help="Padding value for arrays")
    
    # Execution Control
    parser.add_argument("--chunk_size", type=int, default=1000, help="TCRs per chunk")
    parser.add_argument("--n_jobs", type=int, default=16, help="Parallel processes")
    parser.add_argument("--chunk_id", type=int, default=None, help="Index for job arrays")
    parser.add_argument("--resume", action="store_true", help="Skip existing chunks")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "merge"])
    
    # Compatibility placeholders
    parser.add_argument("--verbose", type=int, default=2)
    return parser.parse_args()

def vectorized_objective(gamma, x_matrix, pos_mask, neg_mask, n_pos, beta, l2_reg):
    """
    Core likelihood math (Equation 14).
    Optimized for BLAS/LAPACK execution.
    """
    # Sum of logs approach for p_ni stability
    # p_ni = 1 - exp(sum_a log(1 - x_na * gamma_a))
    log_1_minus_xg = np.log(np.maximum(1.0 - (x_matrix * gamma), 1e-12))
    sum_log_per_donor = np.sum(log_1_minus_xg, axis=1)
    
    p_ni = 1.0 - np.exp(sum_log_per_donor)
    p_ni = np.clip(p_ni, 1e-12, 1.0 - 1e-12)
    
    # Likelihood Components
    reward = np.sum(np.log(p_ni[pos_mask]))
    n_tilde = np.sum(p_ni[neg_mask])
    
    # Beta-Binomial Penalty
    penalty = gammaln(n_tilde + beta) - gammaln(n_pos + n_tilde + beta + 1.0)
    
    # Regularization
    reg_loss = l2_reg * np.sum(gamma**2)
    
    return -(reward + penalty - reg_loss)

def optimize_single_tcr(donor_indices, allele_indices, donor_matrix, beta, l2_reg):
    """Solves the MLE for a single TCR cluster."""
    num_donors = donor_matrix.shape[0]
    pos_mask = np.zeros(num_donors, dtype=bool)
    pos_mask[donor_indices] = True
    neg_mask = ~pos_mask
    n_pos = len(donor_indices)
    
    # Optimization-specific donor matrix slice
    relevant_x = donor_matrix[:, allele_indices]
    
    # Initial guess and bounds
    init_gamma = np.full(len(allele_indices), 0.05)
    bounds = [(0.0, 1.0)] * len(allele_indices)
    
    # L-BFGS-B with aggressive speed settings
    res = minimize(
        vectorized_objective, 
        init_gamma, 
        args=(relevant_x, pos_mask, neg_mask, n_pos, beta, l2_reg),
        method='L-BFGS-B', 
        bounds=bounds,
        options={'ftol': 1e-4, 'maxiter': 50} 
    )
    return res.x

def run_train(args):
    out_dir = Path(args.output_dir)
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    # Load donor matrix into float64
    donor_matrix = np.load(args.donor_matrix_path)["donor_hla_matrix"].astype(np.float32)
    
    with PublicTcrHlaCsrReaderChunk(args.h5_data_path, include_donors=True) as reader:
        total_clusters = reader.num_clusters
        start = args.chunk_id * args.chunk_size
        stop = min(start + args.chunk_size, total_clusters)
        
        out_file = chunks_dir / f"chunk_{args.chunk_id:06d}.npz"
        if args.resume and out_file.exists():
            print(f"Skipping chunk {args.chunk_id} (already exists).")
            return

        print(f"Loading data for chunk {args.chunk_id}...")
        chunk_iter = reader.iter_cluster_chunks(start=start, stop=stop, chunk_rows=args.chunk_size)
        chunk_data = next(chunk_iter)
        
        tasks = []
        for i in range(chunk_data.n_clusters):
            a_idx = np.flatnonzero(chunk_data.counts_dense[i])
            if len(a_idx) > 0:
                tasks.append((chunk_data.donor_ids[i], a_idx))
        
        print(f"Optimizing {len(tasks)} TCRs using {args.n_jobs} cores...")
        results = Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
            delayed(optimize_single_tcr)(
                d, a, donor_matrix, args.beta, args.l2_reg
            ) for d, a in tqdm(tasks, total=len(tasks))
        )
        
        # Prepare arrays for saving
        if not tasks: return
        max_hlas = max(len(t[1]) for t in tasks)
        b_sets = np.full((len(tasks), max_hlas), args.pad_token)
        z_p = np.zeros((len(tasks), max_hlas))
        
        for j, gamma_vals in enumerate(results):
            a_idx = tasks[j][1]
            b_sets[j, :len(a_idx)] = a_idx
            z_p[j, :len(a_idx)] = gamma_vals
            
        np.savez_compressed(
            out_file,
            cluster_start=np.array(start),
            cluster_end=np.array(stop),
            binder_sets=b_sets,
            z_probs=z_p,
            metadata_json=json.dumps({"chunk_id": args.chunk_id, "mode": "scipy_vectorized"})
        )

def run_merge(args):
    """Standard merge logic to reconstruct the H5 file."""
    output_h5 = Path(args.output_dir) / Path(args.h5_data_path).name
    if not output_h5.exists():
        print("Creating copy of H5 for results...")
        shutil.copy2(args.h5_data_path, output_h5)
    
    with PublicTcrHlaCsrReaderChunk(args.h5_data_path) as r:
        total = r.num_clusters
        
    with MleZprobsWriter(str(output_h5), num_clusters=total) as writer:
        for f in sorted((Path(args.output_dir) / "chunks").glob("chunk_*.npz")):
            ckpt = np.load(f)
            writer.write_chunk(
                cluster_start=int(ckpt["cluster_start"]),
                cluster_end=int(ckpt["cluster_end"]),
                binder_sets=ckpt["binder_sets"],
                z_probs=ckpt["z_probs"],
                pad_token=args.pad_token
            )
            print(f"Merged {f.name}")

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "merge":
        run_merge(args)
