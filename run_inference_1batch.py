"""
TCRtyper Model Inference & Diagnostics Script

This script performs comprehensive visualization and statistical analysis
of the trained model outputs to diagnose issues like:
- Mode collapse in predictions
- Na=0 allele bug
- Gradient flow problems
- Distribution mismatches between true and predicted probabilities
- Reconstruction quality analysis

CHANGELOG:
    - Added EXACT_LIKELIHOOD flag to switch between Equation 8 (simplified) and 
      Equation 4 (exact) likelihood computation.
    - Added visualization for log_p_ni_all when using exact likelihood
"""

import keras
import tensorflow as tf
from keras import layers
from src.model_utils import LogSpaceLikelihood, LogSpaceExactLikelihood  # NEW: Import both
from src.utils import TCRFileManager
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import shutil
def tf_logit(p, epsilon=1e-7): # reverse sigmoid
    p = tf.clip_by_value(p, epsilon, 1.0 - epsilon)
    return tf.math.log(p / (1.0 - p))

def calculate_frequencies(mhc_donors): #(N,A)
    mhc_counts = tf.reduce_sum(mhc_donors, axis=0) #(A,)
    total_count = tf.reduce_sum(mhc_counts) #scalar
    mhc_counts = tf.cast(mhc_counts, tf.float32)
    total_count = tf.cast(total_count, tf.float32)
    freqs = mhc_counts / (total_count + 1e-9) #(A,)
    return freqs #(A,)

def normalize_gamma_logits(gamma_logits, freqs):
    freqs = tf.cast(freqs, dtype=gamma_logits.dtype)
    mask = tf.where(freqs == 0., 0., 1.)
    epsilon = 1e-7
    freqs_clipped = tf.clip_by_value(freqs, epsilon, 1.0 - epsilon)
    freq_logits = tf.math.log(freqs_clipped / (1.0 - freqs_clipped))
    freq_logits *= mask
    return gamma_logits - freq_logits

# =============================================================================
# CONFIGURATION
# =============================================================================
PAD_TOKEN = -2.
MASK_TOKEN = -1.
BATCH_SIZE = 500
MODEL_PATH = 'checkpoints/exactloss_q+gamma/model_epoch_10.keras'
OUTPUT_PATH = 'output/model_diagnostics_' + datetime.now().strftime('%Y%m%d_%H%M%S') + "_exactloss_q+gamma"
PATIENT_TO_HLA = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/processed/patient_to_hla.csv'
ATT_MODE = True  # Must match training configuration
NUM_BATCHES_TO_ANALYZE = 3  # Number of batches to analyze
ADJUST_LOGITS = False
CONFIG_PATH = os.path.join(os.path.dirname(MODEL_PATH), 'config.json')
# =============================================================================
# NEW: EXACT LIKELIHOOD FLAG - Must match training configuration!
# =============================================================================
# If True: Use Equation 4 (exact) - sums over ALL N donors for second term
# If False: Use Equation 8 (simplified) - approximates second term
EXACT_LIKELIHOOD = True

# Number of actual valid donors (not the expanded array size)
NUM_VALID_DONORS = 702.
# =============================================================================

# Amino acid mapping for reconstruction visualization
AA_VOCAB = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']  # 21 amino acids

# Data paths
#train_path = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/train_val_split/randompatientwise/train0.tfrecord'
train_path = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/train_val_split/datasetwise/public_train1.tfrecord'
val_path = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/train_val_split/randompatientwise/val0.tfrecord'
#val_path = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/train_val_split/datasetwise/public_valid1.tfrecord'
patient_id_path = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/patients_index_process.tsv'
assert os.path.exists(CONFIG_PATH)
os.makedirs(OUTPUT_PATH, exist_ok=True)
shutil.copy(CONFIG_PATH, os.path.join(OUTPUT_PATH, 'config.json'))
print(f"Output directory: {OUTPUT_PATH}")

# =============================================================================
# LOAD MODEL AND DATA
# =============================================================================
print("\n" + "="*80)
print("LOADING MODEL AND DATA")
print("="*80)

model = keras.saving.load_model(MODEL_PATH)
print(f"✓ Model loaded from: {MODEL_PATH}")
print(f"  Model outputs: {[o.name for o in model.outputs]}")

# Load donor MHC data
patient_tsv = pd.read_csv(patient_id_path, sep='\t')
max_num_patients = np.max(patient_tsv.sample_id.tolist())

mhc_file = np.load('data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/processed/donor_mhc.npz')
donor_mhc = mhc_file['array']
max_donor_mhc = np.zeros(shape=(max_num_patients + 1, donor_mhc.shape[1]))
patient_ids = np.array([int(i) for i in mhc_file['patient_id']])

for j, patient_id in enumerate(patient_ids):
    max_donor_mhc[patient_id] = donor_mhc[j]

donor_mhc = tf.constant(max_donor_mhc, dtype=tf.int32)
print(f"✓ Donor MHC shape: {donor_mhc.shape}")
print(f"✓ Number of actual valid donors: {len(patient_ids)}")
freqs = calculate_frequencies(donor_mhc)
# Load HLA names
hla_df = pd.read_csv(PATIENT_TO_HLA)
hla_names = [i for i in hla_df.columns if i != 'donor_id']
hla_with_counts = [f'{name}_{int(np.sum(hla_df[name]))}' for name in hla_names]

# Initialize managers
tcr_manager = TCRFileManager(
    tcr_path=train_path,
    batch_size=BATCH_SIZE,
    tcr_length=70,
    shuffle_buffer_size=100000,
    pad_token=PAD_TOKEN
)

# =============================================================================
# NEW: Conditional loss function selection based on EXACT_LIKELIHOOD flag
# =============================================================================
print("\n" + "-"*40)
if EXACT_LIKELIHOOD:
    print(">>> Using EXACT likelihood (Equation 4)")
    print(f"    - Computes exact sum over all {NUM_VALID_DONORS:.0f} donors for second term")
    print(f"    - No approximation assumptions")
    
    loss_func = LogSpaceExactLikelihood(
        donor_mhc=donor_mhc,
        pad_token=PAD_TOKEN,
        test_mode=True,  # Always True for inference/diagnostics
        fix_q=True,
        num_mhc=620,
        N=NUM_VALID_DONORS,
    )
else:
    print(">>> Using SIMPLIFIED likelihood (Equation 8)")
    print(f"    - Approximates second term as q * Σ(Na * γ)")
    
    loss_func = LogSpaceLikelihood(
        donor_mhc=donor_mhc,
        pad_token=PAD_TOKEN,
        test_mode=True,  # Always True for inference/diagnostics
        fix_q=True,
        num_mhc=620,
    )
print("-"*40)
# =============================================================================

# Initialize CCE loss function for reconstruction analysis
cce_loss_fn = keras.losses.CategoricalCrossentropy(reduction='none')

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_fig(fig, name, dpi=300):
    """Save figure and close."""
    path = os.path.join(OUTPUT_PATH, f'{name}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}.png")

def write_stats(filename, content):
    """Write statistics to a text file."""
    path = os.path.join(OUTPUT_PATH, filename)
    with open(path, 'w') as f:
        f.write(content)
    print(f"  Saved: {filename}")

def compute_reconstruction_loss(tcr_seqs, recon_output, pad_token):
    """
    Compute masked reconstruction loss and accuracy.
    
    Returns:
        loss: Scalar reconstruction loss
        accuracy: Per-position accuracy
        per_token_loss: Loss per token (B, S)
    """
    # Create mask for valid tokens
    mask = tf.cast(tcr_seqs != pad_token, tf.float32)
    
    # Prepare input tokens
    input_tokens_safe = tf.where(tcr_seqs == pad_token, 0.0, tcr_seqs)
    input_tokens_safe = tf.cast(input_tokens_safe, tf.int64)
    
    # One-hot encode targets
    one_hot_input = tf.one_hot(input_tokens_safe, depth=21)
    
    # Compute per-token loss
    loss_per_token = cce_loss_fn(one_hot_input, recon_output)
    
    # Apply mask
    masked_loss = loss_per_token * mask
    num_valid_tokens = tf.reduce_sum(mask)
    final_loss = tf.reduce_sum(masked_loss) / tf.maximum(num_valid_tokens, 1.0)
    
    # Compute accuracy
    pred_tokens = tf.argmax(recon_output, axis=-1)
    correct = tf.cast(pred_tokens == input_tokens_safe, tf.float32) * mask
    accuracy = tf.reduce_sum(correct) / tf.maximum(num_valid_tokens, 1.0)
    
    return final_loss, accuracy, masked_loss

def tokens_to_sequence(tokens, pad_token):
    """Convert token indices to amino acid sequence string."""
    seq = []
    for t in tokens:
        if t == pad_token:
            break
        if 0 <= int(t) < len(AA_VOCAB):
            seq.append(AA_VOCAB[int(t)])
        else:
            seq.append('?')
    return ''.join(seq)

# =============================================================================
# SECTION 1: ALLELE FREQUENCY ANALYSIS (Na=0 Bug Check)
# =============================================================================
print("\n" + "="*80)
print("SECTION 1: ALLELE FREQUENCY ANALYSIS")
print("="*80)

Na = loss_func.Na.numpy()
log_Na = loss_func.log_Na.numpy()
N = loss_func.N

# Find problematic alleles
zero_alleles = np.where(Na == 0)[0]
low_freq_alleles = np.where((Na > 0) & (Na < 5))[0]
high_freq_alleles = np.where(Na > 50)[0]

stats_content = f"""
ALLELE FREQUENCY ANALYSIS
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Likelihood Type: {'EXACT (Equation 4)' if EXACT_LIKELIHOOD else 'SIMPLIFIED (Equation 8)'}

SUMMARY STATISTICS
------------------
Total number of alleles: {len(Na)}
Total number of donors (N): {N}
Alleles with Na=0 (PROBLEMATIC): {len(zero_alleles)}
Alleles with Na < 5: {len(low_freq_alleles)}
Alleles with Na > 50: {len(high_freq_alleles)}

Na distribution:
  Min: {np.min(Na):.0f}
  Max: {np.max(Na):.0f}
  Mean: {np.mean(Na):.2f}
  Median: {np.median(Na):.0f}
  Std: {np.std(Na):.2f}

PROBLEMATIC ALLELES (Na=0)
--------------------------
These alleles have NO donors in the cohort and will cause
division-by-zero issues in enrichment calculations.

Indices: {zero_alleles.tolist()[:20]}{'...' if len(zero_alleles) > 20 else ''}

If using HLA names:
{[hla_names[i] for i in zero_alleles[:10]]}{'...' if len(zero_alleles) > 10 else ''}

INTERPRETATION
--------------
If Na=0 alleles appear in your top predictions, the valid_allele_mask
is NOT working correctly. These alleles should be masked out to have
near-zero probability after softmax normalization.

ACTION REQUIRED: Verify that valid_allele_mask is being applied in delta_loss()
"""

write_stats('01_allele_frequency_analysis.txt', stats_content)

# Plot Na distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram of Na
axes[0].hist(Na, bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(x=0.5, color='red', linestyle='--', label=f'Na=0 ({len(zero_alleles)} alleles)')
axes[0].set_xlabel('Number of donors (Na)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Distribution of Allele Frequencies', fontsize=14)
axes[0].legend()

# Bar plot of top 20 most frequent alleles
top_alleles = np.argsort(Na)[::-1][:20]
axes[1].barh(range(20), Na[top_alleles], color='steelblue')
axes[1].set_yticks(range(20))
axes[1].set_yticklabels([f'{i} ({hla_names[i][:15]})' for i in top_alleles], fontsize=8)
axes[1].set_xlabel('Number of donors (Na)', fontsize=12)
axes[1].set_title('Top 20 Most Frequent Alleles', fontsize=14)
axes[1].invert_yaxis()

# Bar plot of alleles with Na=0
if len(zero_alleles) > 0:
    display_zeros = zero_alleles[:20]
    axes[2].barh(range(len(display_zeros)), [0]*len(display_zeros), color='red')
    axes[2].set_yticks(range(len(display_zeros)))
    axes[2].set_yticklabels([f'{i} ({hla_names[i][:15]})' for i in display_zeros], fontsize=8)
    axes[2].set_xlabel('Na (all zero)', fontsize=12)
    axes[2].set_title(f'Alleles with Na=0 ({len(zero_alleles)} total)', fontsize=14)
    axes[2].invert_yaxis()
else:
    axes[2].text(0.5, 0.5, 'No alleles with Na=0', ha='center', va='center', fontsize=14)
    axes[2].set_title('Alleles with Na=0', fontsize=14)

plt.tight_layout()
save_fig(fig, '01_allele_frequency_distribution')

# =============================================================================
# SECTION 2: MODEL OUTPUT ANALYSIS (Per Batch)
# =============================================================================
print("\n" + "="*80)
print("SECTION 2: MODEL OUTPUT ANALYSIS")
print("="*80)

dataset = tcr_manager.get_dataset(shuffle=True)
all_stats = []

for batch_idx, data in enumerate(dataset, start=1):
    if batch_idx > NUM_BATCHES_TO_ANALYZE:
        break
    
    print(f"\n--- Analyzing Batch {batch_idx} ---")
    
    # Prepare data
    tcr_seqs, tcr_ids, tcr_donor_ids = data
    tcr_seqs = tf.cast(tcr_seqs, tf.float32)
    tcr_ids = tf.cast(tcr_ids, tf.int32)
    tcr_donor_ids = tf.cast(tcr_donor_ids, tf.int32)
    tcr_seq_mask = tf.where(tcr_seqs == PAD_TOKEN, PAD_TOKEN, 1.)
    tcr_seq_mask = tf.cast(tcr_seq_mask, tf.float32)
    
    # Forward pass
    if ATT_MODE:
        gamma_logits, q_logits, att_score, delta_logits = model([tcr_seqs, tcr_seq_mask])
    else:
        gamma_logits, q_logits, delta_logits = model([tcr_seqs, tcr_seq_mask])
        att_score = None
    
    # ==========================================================================
    # NEW: Handle different test_mode outputs based on EXACT_LIKELIHOOD
    # ==========================================================================
    if ADJUST_LOGITS: gamma_logits = normalize_gamma_logits(gamma_logits, freqs)
    loss_outputs = loss_func.call(gamma_logits, q_logits, tcr_donor_ids, delta_logits)
    
    if EXACT_LIKELIHOOD:
        # Exact likelihood returns additional log_p_ni_all tensor
        (Ni_size, Ni, gamma_donor_id_mask, log_p_ni, log_p_ni_all,
         log_qp, log_one_minus_qp, first_term, second_term, bce,
         log_gamma, log_one_minus_gamma) = loss_outputs
    else:
        # Simplified likelihood
        (Ni_size, Ni, gamma_donor_id_mask, log_p_ni,
         log_qp, log_one_minus_qp, first_term, second_term, bce,
         log_gamma, log_one_minus_gamma) = loss_outputs
        log_p_ni_all = None  # Not available in simplified version
    # ==========================================================================
    
    # Convert to numpy arrays
    gamma_probs = tf.nn.sigmoid(gamma_logits).numpy()
    q_probs = tf.nn.sigmoid(q_logits).numpy()
    
    # =========================================================================
    # 2.1: GAMMA VISUALIZATION
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Heatmap
    sns.heatmap(gamma_probs, ax=axes[0,0], cmap='viridis')
    axes[0,0].set_title(f'Gamma (HLA binding) - TCRs', fontsize=12)
    axes[0,0].set_xlabel('Allele')
    axes[0,0].set_ylabel('TCR')
    
    # Distribution histogram
    axes[0,1].hist(gamma_probs.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[0,1].set_xlabel('Gamma probability', fontsize=12)
    axes[0,1].set_ylabel('Count', fontsize=12)
    axes[0,1].set_title(f'Distribution of Gamma values\nMean={np.mean(gamma_probs):.4f}, Std={np.std(gamma_probs):.4f}')
    
    # Per-allele mean
    allele_means = np.mean(gamma_probs, axis=0)
    axes[1,0].bar(range(len(allele_means)), allele_means, width=1.0, alpha=0.7)
    axes[1,0].set_xlabel('Allele index', fontsize=12)
    axes[1,0].set_ylabel('Mean gamma', fontsize=12)
    axes[1,0].set_title('Mean Gamma per Allele (across all TCRs)')
    
    # Per-TCR max
    tcr_maxs = np.max(gamma_probs, axis=1)
    axes[1,1].hist(tcr_maxs, bins=50, edgecolor='black', alpha=0.7)
    axes[1,1].set_xlabel('Max gamma per TCR', fontsize=12)
    axes[1,1].set_ylabel('Count', fontsize=12)
    axes[1,1].set_title(f'Distribution of Max Gamma per TCR\nMean={np.mean(tcr_maxs):.4f}')
    
    plt.tight_layout()
    save_fig(fig, f'02_gamma_analysis_batch{batch_idx}')
    
    # =========================================================================
    # 2.2: Q PROBABILITY VISUALIZATION
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    q_flat = q_probs.flatten()
    axes[0].hist(q_flat, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0].set_xlabel('Q probability', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title(f'Distribution of Q (sampling probability)\nMean={np.mean(q_flat):.6f}, Max={np.max(q_flat):.6f}')
    
    axes[1].plot(range(len(q_flat)), np.sort(q_flat)[::-1], 'o-', markersize=1, alpha=0.5)
    axes[1].set_xlabel('TCR (sorted by Q)', fontsize=12)
    axes[1].set_ylabel('Q probability', fontsize=12)
    axes[1].set_title('Q values sorted descending')
    
    plt.tight_layout()
    save_fig(fig, f'02_q_analysis_batch{batch_idx}')
    
  
    # =========================================================================
    # 2.6: LOSS COMPONENTS
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # First term vs Second term
    first_np = first_term.numpy()
    second_np = second_term.numpy()
    
    axes[0,0].scatter(first_np, second_np, alpha=0.3, s=10)
    axes[0,0].set_xlabel('First Term (log-odds sum)', fontsize=12)
    axes[0,0].set_ylabel('Second Term (penalty)', fontsize=12)
    axes[0,0].set_title(f'Loss Components per TCR\n({"Exact" if EXACT_LIKELIHOOD else "Simplified"} Likelihood)')
    
    # BCE - handle both (B,) and (B, A) shapes
    bce_np = bce.numpy()
    
    if len(bce_np.shape) == 1:
        axes[0,1].hist(bce_np, bins=50, edgecolor='black', alpha=0.7, color='purple')
        axes[0,1].set_xlabel('BCE Loss per TCR', fontsize=12)
        axes[0,1].set_ylabel('Count', fontsize=12)
        axes[0,1].set_title(f'BCE Loss Distribution\nMean={np.mean(bce_np):.4f}')
        bce_per_sample = bce_np
        bce_per_allele = None
    else:
        sns.heatmap(bce_np, ax=axes[0,1], cmap='Reds', vmin=0)
        axes[0,1].set_xlabel('Allele', fontsize=12)
        axes[0,1].set_ylabel('TCR', fontsize=12)
        axes[0,1].set_title(f'BCE Loss per TCR-Allele (first 100 TCRs)\nMean={np.mean(bce_np):.4f}')
        bce_per_sample = np.sum(bce_np, axis=1)
        bce_per_allele = np.mean(bce_np, axis=0)
    
    # BCE per-sample histogram
    axes[0,2].hist(bce_per_sample, bins=50, edgecolor='black', alpha=0.7, color='purple')
    axes[0,2].set_xlabel('BCE Loss (per sample)', fontsize=12)
    axes[0,2].set_ylabel('Count', fontsize=12)
    axes[0,2].set_title(f'BCE Loss per TCR\nMean={np.mean(bce_per_sample):.4f}, Std={np.std(bce_per_sample):.4f}')
    
    # Likelihood
    ll_np = first_np + second_np if EXACT_LIKELIHOOD else first_np - second_np
    axes[1,0].hist(ll_np, bins=50, edgecolor='black', alpha=0.7, color='teal')
    axes[1,0].set_xlabel('Log-Likelihood', fontsize=12)
    axes[1,0].set_ylabel('Count', fontsize=12)
    axes[1,0].set_title(f'Log-Likelihood Distribution\nMean={np.mean(ll_np):.4f}')
    
    # Ni_size distribution
    ni_size_np = Ni_size.numpy()
    axes[1,1].hist(ni_size_np, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1,1].set_xlabel('Number of donors per TCR (Ni_size)', fontsize=12)
    axes[1,1].set_ylabel('Count', fontsize=12)
    axes[1,1].set_title(f'Donors per TCR\nMean={np.mean(ni_size_np):.1f}, Max={np.max(ni_size_np):.0f}')
    
    # BCE per-allele
    if bce_per_allele is not None:
        axes[1,2].bar(range(len(bce_per_allele)), bce_per_allele, width=1.0, alpha=0.7, color='purple')
        axes[1,2].set_xlabel('Allele index', fontsize=12)
        axes[1,2].set_ylabel('Mean BCE', fontsize=12)
        axes[1,2].set_title('Mean BCE Loss per Allele')
    else:
        axes[1,2].text(0.5, 0.5, 'BCE is per-sample only\n(not per-allele)', 
                       ha='center', va='center', fontsize=12)
        axes[1,2].set_title('BCE per Allele (N/A)')
    
    plt.tight_layout()
    save_fig(fig, f'06_loss_components_batch{batch_idx}')
    
    # =========================================================================
    # NEW 2.6b: EXACT LIKELIHOOD SPECIFIC ANALYSIS
    # =========================================================================
    if EXACT_LIKELIHOOD and log_p_ni_all is not None:
        print("  Generating exact likelihood specific plots...")
        
        log_p_ni_all_np = log_p_ni_all.numpy()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Heatmap of log_p_ni_all (first 50 TCRs, first 200 donors)
        display_tcrs = log_p_ni_all_np.shape[0]
        display_donors = log_p_ni_all_np.shape[1]
        sns.heatmap(log_p_ni_all_np[:display_tcrs, :display_donors], 
                    ax=axes[0,0], cmap='viridis', vmin=-10, vmax=0)
        axes[0,0].set_title(f'log(p_ni) for ALL donors\n(first {display_tcrs} TCRs, {display_donors} donors)', fontsize=12)
        axes[0,0].set_xlabel('Donor index')
        axes[0,0].set_ylabel('TCR index')
        
        # Distribution of log_p_ni_all
        valid_mask = loss_func.valid_donor_mask.numpy() > 0.5
        valid_log_pni = log_p_ni_all_np[:, valid_mask].flatten()
        axes[0,1].hist(valid_log_pni, bins=50, edgecolor='black', alpha=0.7, color='teal')
        axes[0,1].set_xlabel('log(p_ni)', fontsize=12)
        axes[0,1].set_ylabel('Count', fontsize=12)
        axes[0,1].set_title(f'Distribution of log(p_ni) over ALL valid donors\nMean={np.mean(valid_log_pni):.4f}')
        
        # Compare p_ni distributions: N_i donors vs all donors
        log_p_ni_Ni_np = log_p_ni.numpy()
        # Flatten and filter valid
        mask_Ni = gamma_donor_id_mask.numpy() > 0.5
        valid_log_pni_Ni = log_p_ni_Ni_np[mask_Ni].flatten()
        
        axes[0,2].hist(valid_log_pni_Ni, bins=50, alpha=0.5, label='Donors in N_i', color='blue')
        axes[0,2].hist(valid_log_pni, bins=50, alpha=0.5, label='ALL donors', color='red')
        axes[0,2].set_xlabel('log(p_ni)', fontsize=12)
        axes[0,2].set_ylabel('Count', fontsize=12)
        axes[0,2].set_title('Comparison: N_i donors vs ALL donors')
        axes[0,2].legend()
        
        # Per-donor mean log_p_ni
        per_donor_mean = np.mean(log_p_ni_all_np, axis=0)
        axes[1,0].bar(range(len(per_donor_mean)), per_donor_mean, width=1.0, alpha=0.7)
        axes[1,0].set_xlabel('Donor index', fontsize=12)
        axes[1,0].set_ylabel('Mean log(p_ni)', fontsize=12)
        axes[1,0].set_title('Mean log(p_ni) per Donor (across all TCRs)')
        
        # Second term breakdown: contribution from each donor
        # log_one_minus_qp_all = log(1 - q*p_ni) for all donors
        log_q_np = tf.math.log_sigmoid(q_logits).numpy()
        log_qp_all_np = log_q_np + log_p_ni_all_np
        
        # Approximate log(1 - exp(x)) for visualization
        qp_all = np.exp(np.clip(log_qp_all_np, -100, 0))
        one_minus_qp_all = np.clip(1 - qp_all, 1e-10, 1)
        log_one_minus_qp_all_np = np.log(one_minus_qp_all)
        
        # Mask invalid donors
        print('====================================================')
        print("log_p_ni_all_np:", log_p_ni_all_np.shape)
        print("log_one_minus_qp_all_np:", log_one_minus_qp_all_np.shape)
        print("valid_mask:", valid_mask.shape)
        print("q_logits: ", q_logits.shape)
        print("log_q_np: ", log_q_np.shape)
        print("log_qp_all_np: ", log_qp_all_np.shape)
        print('====================================================')

        log_one_minus_qp_all_np[:, ~valid_mask] = 0
        
        per_donor_contribution = np.mean(log_one_minus_qp_all_np, axis=0)
        axes[1,1].bar(range(len(per_donor_contribution)), per_donor_contribution, width=1.0, alpha=0.7, color='purple')
        axes[1,1].set_xlabel('Donor index', fontsize=12)
        axes[1,1].set_ylabel('Mean ln(1-q*p_ni)', fontsize=12)
        axes[1,1].set_title('Mean Second Term Contribution per Donor')
        
        # Scatter: first term vs exact second term
        axes[1,2].scatter(first_np, second_np, alpha=0.3, s=10, c='teal')
        axes[1,2].set_xlabel('First Term', fontsize=12)
        axes[1,2].set_ylabel('Second Term (Exact)', fontsize=12)
        axes[1,2].set_title(f'First vs Second Term (Exact)\nCorr={np.corrcoef(first_np, second_np)[0,1]:.3f}')
        
        plt.tight_layout()
        save_fig(fig, f'06b_exact_likelihood_analysis_batch{batch_idx}')

    # =========================================================================
    # 2.8: ATTENTION VISUALIZATION
    # =========================================================================
    if att_score is not None:
        att_np = att_score.numpy()
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        sample_idx = 0
        if len(att_np.shape) == 4:
            num_heads = min(8, att_np.shape[1])
            for h in range(num_heads):
                row, col = h // 4, h % 4
                sns.heatmap(att_np[sample_idx, h], ax=axes[row, col], cmap='viridis', 
                           square=True, cbar_kws={'shrink': 0.5})
                axes[row, col].set_title(f'Head {h+1}', fontsize=10)
        
        plt.suptitle(f'Attention Patterns for TCR Sample {sample_idx}', fontsize=14)
        plt.tight_layout()
        save_fig(fig, f'08_attention_patterns_batch{batch_idx}')
    
    # =========================================================================
    # COLLECT BATCH STATISTICS
    # =========================================================================
    batch_stats = {
        'batch': batch_idx,
        'exact_likelihood': EXACT_LIKELIHOOD,
        'gamma_mean': np.mean(gamma_probs),
        'gamma_std': np.std(gamma_probs),
        'gamma_max': np.max(gamma_probs),
        'q_mean': np.mean(q_probs),
        'q_max': np.max(q_probs),
        'bce_mean': np.mean(bce_np),
        'bce_per_sample_mean': np.mean(bce_per_sample),
        'll_mean': np.mean(ll_np),
        'first_term_mean': np.mean(first_np),
        'second_term_mean': np.mean(second_np),
        'per_allele_corr_mean': np.nanmean(allele_corrs),
    }
    
    # Add exact likelihood specific stats
    if EXACT_LIKELIHOOD and log_p_ni_all is not None:
        batch_stats['log_p_ni_all_mean'] = np.mean(valid_log_pni)
        batch_stats['log_p_ni_all_std'] = np.std(valid_log_pni)
    
    all_stats.append(batch_stats)

# =============================================================================
# SECTION 3: SUMMARY REPORT
# =============================================================================
print("\n" + "="*80)
print("SECTION 3: GENERATING SUMMARY REPORT")
print("="*80)

stats_df = pd.DataFrame(all_stats)
stats_df.to_csv(os.path.join(OUTPUT_PATH, 'batch_statistics.csv'), index=False)

summary_report = f"""
TCRtyper MODEL DIAGNOSTICS SUMMARY REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {MODEL_PATH}
Batches analyzed: {NUM_BATCHES_TO_ANALYZE}
Batch size: {BATCH_SIZE}
Likelihood Type: {'EXACT (Equation 4)' if EXACT_LIKELIHOOD else 'SIMPLIFIED (Equation 8)'}
Number of Valid Donors: {NUM_VALID_DONORS}

{'='*60}
1. ALLELE FREQUENCY ISSUE (Na=0 Bug)
{'='*60}
Total alleles with Na=0: {len(zero_alleles)}
Alleles with Na=0: {zero_alleles.tolist()[:20]}{'...' if len(zero_alleles) > 20 else ''}

Average Na=0 alleles appearing in TRUE top-5: {np.mean([s['na0_in_true_top5'] for s in all_stats]):.1f}
Average Na=0 alleles appearing in PRED top-5: {np.mean([s['na0_in_pred_top5'] for s in all_stats]):.1f}

INTERPRETATION:
- If Na=0 alleles appear frequently in top-5, the valid_allele_mask is NOT working
- These should be masked to have near-zero probability

STATUS: {'⚠️ BUG PRESENT - Na=0 alleles in predictions!' if np.mean([s['na0_in_true_top5'] for s in all_stats]) > 100 else '✓ Bug appears fixed or minimal'}

{'='*60}
2. MODE COLLAPSE ANALYSIS
{'='*60}
Average Top-5 Overlap (True vs Pred): {np.mean([s['top5_overlap_mean'] for s in all_stats]):.2f}/5
Average Top-10 Overlap (True vs Pred): {np.mean([s['top10_overlap_mean'] for s in all_stats]):.2f}/10

Expected by chance:
- Top-5 overlap: ~0.07/5 (for 358 classes)
- Top-10 overlap: ~0.28/10 (for 358 classes)

STATUS: {'✓ Good overlap - model learning' if np.mean([s['top5_overlap_mean'] for s in all_stats]) > 2 else '⚠️ Low overlap - check mode collapse'}

4. MODEL OUTPUT STATISTICS
{'='*60}
Gamma (HLA binding):
  Mean: {np.mean([s['gamma_mean'] for s in all_stats]):.6f}
  Max:  {np.mean([s['gamma_max'] for s in all_stats]):.6f}

Q (Sampling probability):
  Mean: {np.mean([s['q_mean'] for s in all_stats]):.8f}
  Max:  {np.mean([s['q_max'] for s in all_stats]):.8f}

{'='*60}
5. LOSS COMPONENTS
{'='*60}
BCE Loss (mean per element):    {np.mean([s['bce_mean'] for s in all_stats]):.4f}
BCE Loss (mean per sample):     {np.mean([s['bce_per_sample_mean'] for s in all_stats]):.4f}
Log-Likelihood (mean):          {np.mean([s['ll_mean'] for s in all_stats]):.4f}
First Term (mean):              {np.mean([s['first_term_mean'] for s in all_stats]):.4f}
Second Term (mean):             {np.mean([s['second_term_mean'] for s in all_stats]):.4f}
"""

# Add exact likelihood specific stats
if EXACT_LIKELIHOOD and 'log_p_ni_all_mean' in all_stats[0]:
    summary_report += f"""
{'='*60}
5b. EXACT LIKELIHOOD SPECIFIC STATISTICS
{'='*60}
log(p_ni) over ALL donors:
  Mean: {np.mean([s['log_p_ni_all_mean'] for s in all_stats]):.4f}
  Std:  {np.mean([s['log_p_ni_all_std'] for s in all_stats]):.4f}

INTERPRETATION:
- Exact likelihood computes Σ_n ln(1 - q*p_ni) over ALL N donors
- This avoids the approximation assumptions in Equation 8
- Second term should be more accurate for cross-reactive TCRs
"""

# Add recommendations
recommendations = []

if np.mean([s['na0_in_true_top5'] for s in all_stats]) > 100:
    recommendations.append("- ⚠️ FIX Na=0 BUG: Apply valid_allele_mask in delta_loss()")



if np.mean([s['per_allele_corr_mean'] for s in all_stats]) < 0.1:
    recommendations.append("- ⚠️ LOW CORRELATION: Model may not be learning TCR-specific patterns")

if EXACT_LIKELIHOOD:
    recommendations.append(f"- ℹ️ Using EXACT likelihood - compare with SIMPLIFIED to verify improvement")

if not recommendations:
    recommendations.append("- ✓ No critical issues detected")

summary_report += '\n'.join(recommendations)

summary_report += f"""

{'='*60}
FILES GENERATED
{'='*60}
"""
for f in sorted(os.listdir(OUTPUT_PATH)):
    summary_report += f"- {f}\n"

write_stats('00_SUMMARY_REPORT.txt', summary_report)

print(summary_report)

print("\n" + "="*80)
print(f"✓ DIAGNOSTICS COMPLETE")
print(f"✓ All outputs saved to: {OUTPUT_PATH}")
print("="*80)

# =============================================================================
# SECTION 4: DETAILED ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("SECTION 4: DETAILED ANALYSIS")
print("="*80)

# Which alleles have high gamma across all TCRs?
gamma_probs = tf.nn.sigmoid(gamma_logits).numpy()
allele_means = np.mean(gamma_probs, axis=0)

top_gamma_alleles = np.argsort(allele_means)[::-1][:10]
print("\nTop 10 alleles by mean gamma:", top_gamma_alleles)
print("Their Na values:", [loss_func.Na[a].numpy() for a in top_gamma_alleles])
print("Their mean gamma:", [allele_means[a] for a in top_gamma_alleles])

# Check first vs second term balance
first_np = first_term.numpy()
second_np = second_term.numpy()
print(f"\nFirst term: mean={np.mean(first_np):.2f}, std={np.std(first_np):.2f}")
print(f"Second term: mean={np.mean(second_np):.2f}, std={np.std(second_np):.2f}")
print(f"Likelihood type: {'EXACT' if EXACT_LIKELIHOOD else 'SIMPLIFIED'}")

# Check which alleles appear in most TCR donor pools
allele_presence = tf.reduce_sum(tf.cast(tf.reduce_sum(Ni, axis=1) > 0, tf.float32), axis=0).numpy()
print("\nAlleles appearing in most TCRs:")
top_present = np.argsort(allele_presence)[::-1][:10]
for a in top_present:
    print(f"  Allele {a}: present in {allele_presence[a]:.0f}/{Ni.shape[0]} TCRs, Na={loss_func.Na[a].numpy():.0f}")

# =============================================================================
# SECTION 5: EMBEDDING SPACE ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("SECTION 5: EMBEDDING SPACE ANALYSIS")
print("="*80)

try:
    pooled_model = keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer('pool').output
    )
    
    pooled_output = pooled_model([tcr_seqs, tcr_seq_mask]).numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].hist(pooled_output.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Pooled activation value', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title(f'Pooled Representation Distribution\nMean={np.mean(pooled_output):.4f}, Std={np.std(pooled_output):.4f}')
    
    dim_vars = np.var(pooled_output, axis=0)
    axes[1].bar(range(len(dim_vars)), dim_vars, width=1.0, alpha=0.7)
    axes[1].set_xlabel('Dimension', fontsize=12)
    axes[1].set_ylabel('Variance', fontsize=12)
    axes[1].set_title('Variance per Embedding Dimension')
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pooled_2d = pca.fit_transform(pooled_output[:500])
    
    axes[2].scatter(pooled_2d[:, 0], pooled_2d[:, 1], c=ni_size_np[:500], 
                    cmap='viridis', alpha=0.5, s=10)
    axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    axes[2].set_title('PCA of Pooled Representations\n(colored by donor count)')
    plt.colorbar(axes[2].collections[0], ax=axes[2], label='Ni_size')
    
    plt.tight_layout()
    save_fig(fig, '09_embedding_analysis')
    
    print("✓ Embedding analysis complete")
    
except Exception as e:
    print(f"⚠️ Could not perform embedding analysis: {e}")

print("\n" + "="*80)
print("ALL DIAGNOSTICS COMPLETE")
print("="*80)