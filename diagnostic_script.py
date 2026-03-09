import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

from src.utils import PublicTcrHlaCsrReaderChunk

def update_reservoir_vectorized(reservoir, current_total, new_items):
    """Highly optimized vectorized reservoir sampling."""
    B = new_items.shape[0]
    res_size = reservoir.shape[0]
    
    fill_count = max(0, min(B, res_size - current_total))
    if fill_count > 0:
        reservoir[current_total : current_total + fill_count] = new_items[:fill_count]
        
    rem_count = B - fill_count
    if rem_count > 0:
        start_idx = current_total + fill_count
        j_arr = np.random.randint(0, np.arange(start_idx, start_idx + rem_count) + 1)
        replace_mask = j_arr < res_size
        replace_indices = np.where(replace_mask)[0]
        for local_idx in replace_indices:
            reservoir[j_arr[local_idx]] = new_items[fill_count + local_idx]
            
    return current_total + B

def analyze_h5_streaming(file_path, output_path=None):
    print(f"Loading data from {file_path} using streaming CSR reader...")
    
    reader = PublicTcrHlaCsrReaderChunk(file_path, include_counts=True, include_donors=True)
    reader.open()
    
    total_tcrs = 0
    total_robust = 0
    
    sum_binary_all = None
    sum_binary_robust = None
    sum_enrich_active_all = None
    sum_enrich_active_robust = None
    
    # Track two separate reservoirs for the heatmaps
    reservoir_size = 100
    res_all = None
    res_robust = None
    
    for chunk in reader.iter_cluster_chunks():
        if chunk.counts_dense is None:
            continue
            
        counts = chunk.counts_dense.astype(np.float32)
        B, num_alleles = counts.shape
        hla_mask = (counts > 0).astype(np.float32)
        
        # Safe extraction of donors (preventing NaN / Inf)
        if hasattr(chunk, 'n_donors') and chunk.n_donors is not None:
            n_d = np.nan_to_num(chunk.n_donors.astype(np.float32), nan=1.0)
            robust_mask = n_d >= 10
        else:
            n_d = np.ones(B, dtype=np.float32)
            robust_mask = np.ones(B, dtype=bool)
            
        # Fix division by zero
        n_d_safe = np.maximum(n_d, 1.0)
        enrichment = counts / n_d_safe[:, None]
        enrichment = np.nan_to_num(enrichment, nan=0.0, posinf=0.0, neginf=0.0)
        
        if sum_binary_all is None:
            sum_binary_all = np.zeros(num_alleles, dtype=np.float64)
            sum_binary_robust = np.zeros(num_alleles, dtype=np.float64)
            sum_enrich_active_all = np.zeros(num_alleles, dtype=np.float64)
            sum_enrich_active_robust = np.zeros(num_alleles, dtype=np.float64)
            res_all = np.zeros((reservoir_size, num_alleles), dtype=np.float32)
            res_robust = np.zeros((reservoir_size, num_alleles), dtype=np.float32)
            
        # 1. Update All TCRs
        sum_binary_all += hla_mask.sum(axis=0)
        # Only add to enrichment sum if the allele is actually present
        sum_enrich_active_all += (enrichment * hla_mask).sum(axis=0)
        total_tcrs = update_reservoir_vectorized(res_all, total_tcrs, enrichment)
        
        # 2. Update Robust TCRs (>= 10 donors)
        robust_in_chunk = np.sum(robust_mask)
        if robust_in_chunk > 0:
            sum_binary_robust += hla_mask[robust_mask].sum(axis=0)
            sum_enrich_active_robust += (enrichment[robust_mask] * hla_mask[robust_mask]).sum(axis=0)
            total_robust = update_reservoir_vectorized(res_robust, total_robust, enrichment[robust_mask])

    reader.close()

    if total_tcrs == 0:
        print("No valid data found in the file.")
        return

    # Frequencies
    freq_all = sum_binary_all / total_tcrs
    freq_robust = sum_binary_robust / max(total_robust, 1)
    
    # Active Enrichment (Average only over TCRs where the allele is present)
    avg_enrich_all = sum_enrich_active_all / np.maximum(sum_binary_all, 1.0)
    avg_enrich_robust = sum_enrich_active_robust / np.maximum(sum_binary_robust, 1.0)
    
    print(f"\nProcessing Complete!")
    print(f"Total TCRs processed: {total_tcrs:,}")
    print(f"TCRs with >= 10 donors: {total_robust:,}")

    # --- Plotting Visualizations (2x2 Grid) ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    # Plot 1: Binary HLA Frequencies
    axes[0, 0].plot(freq_all, label='All TCRs', alpha=0.8, linewidth=2)
    axes[0, 0].plot(freq_robust, label='TCRs in >= 10 Donors', alpha=0.8, linewidth=2)
    axes[0, 0].set_title('Binary HLA Frequencies (Presence > 0)', fontsize=14)
    axes[0, 0].set_ylabel('Fraction of TCRs', fontsize=12)
    axes[0, 0].legend(fontsize=11)
    
    # Plot 2: Active Enrichment Score
    axes[0, 1].plot(avg_enrich_all, label='All TCRs (Active Only)', alpha=0.8, linewidth=2, color='purple')
    axes[0, 1].plot(avg_enrich_robust, label='>= 10 Donors (Active Only)', alpha=0.8, linewidth=2, color='orange')
    axes[0, 1].set_title('Mean Enrichment (counts / donors) Given Presence', fontsize=14)
    axes[0, 1].set_ylabel('Mean Active Enrichment', fontsize=12)
    axes[0, 1].legend(fontsize=11)
    
    # Plot 3: Heatmap (ALL TCRs)
    act_samples_all = min(reservoir_size, total_tcrs)
    sns.heatmap(res_all[:act_samples_all], ax=axes[1, 0], cmap='viridis', cbar=True)
    axes[1, 0].set_title(f'Enrichment Heatmap (Sampled {act_samples_all} from ALL TCRs)', fontsize=14)
    axes[1, 0].set_ylabel('TCR Instance', fontsize=12)
    
    # Plot 4: Heatmap (ROBUST TCRs)
    act_samples_rob = min(reservoir_size, total_robust)
    sns.heatmap(res_robust[:act_samples_rob], ax=axes[1, 1], cmap='viridis', cbar=True)
    axes[1, 1].set_title(f'Enrichment Heatmap (Sampled {act_samples_rob} from >= 10 DONORS)', fontsize=14)
    axes[1, 1].set_ylabel('TCR Instance', fontsize=12)
    
    for ax in axes.flatten():
        ax.set_xlabel('HLA Allele Index', fontsize=12)
        
    plt.tight_layout()
    
    if output_path:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir: os.makedirs(out_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plots successfully saved to: {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-efficient diagnostic script.")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to the .h5 dataset file.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output path (e.g., ./plots/hla.png).")
    args = parser.parse_args()
    analyze_h5_streaming(args.h5_path, args.output)