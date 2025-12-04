"""
TCRtyper Model Training Script

This script trains the TCRtyper neural network to predict HLA allotype associations
from TCR sequences. It supports both standard training mode and TEST_MODE for
debugging and visualization of model internals.

Usage:
    python train_model.py

Configuration:
    Modify the constants at the top of the script to change training parameters,
    data paths, and operating mode (TEST_MODE vs. production training).
    
CHANGELOG:
    - Added EXACT_LIKELIHOOD flag to switch between Equation 8 (simplified) and 
      Equation 4 (exact) likelihood computation.
"""

import keras
import tensorflow as tf
import src
from keras import layers
from src.model_utils import (
    PositionalEncoding, 
    MaskedEmbedding, 
    AttentionLayer, 
    MaskedDense,
    LogSpaceLikelihood,
    LogSpaceExactLikelihood,  # NEW: Import exact likelihood
    SpatialTemporalDropout1D,
    MaskedEmbeddingOHE
)
from src.utils import TCRFileManager
from src.visualization_utils import TestModeVisualizer
import numpy as np
import os
import pandas as pd
import json
import pyarrow as pa
import pyarrow.parquet as pq
from keras.layers import Activation
from keras.utils import get_custom_objects

def prepend_test_data(tcr_seqs, tcr_ids, tcr_donor_ids, test_tcr_seq, test_donor_id, pad_token=-2.):
    """
    Prepend test TCR sequences and donor IDs to batch data with proper padding.
    
    Args:
        tcr_seqs: Batch TCR sequences [batch_size, seq_len]
        tcr_ids: Batch TCR IDs [batch_size]
        tcr_donor_ids: Batch donor IDs [batch_size, donor_len]
        test_tcr_seq: Test TCR sequences to prepend [num_test, test_seq_len]
        test_donor_id: Test donor IDs to prepend [num_test, test_donor_len]
        pad_token: Padding token value (default: -2.)
    
    Returns:
        Tuple of (tcr_seqs, tcr_ids, tcr_donor_ids) with test data prepended
    """
    # Get current shapes
    test_seq_len = tf.shape(test_tcr_seq)[1]
    batch_seq_len = tf.shape(tcr_seqs)[1]
    test_donor_len = tf.shape(test_donor_id)[1]
    batch_donor_len = tf.shape(tcr_donor_ids)[1]
    num_test_samples = tf.shape(test_tcr_seq)[0]

    # Compute max lengths
    max_seq_len = tf.maximum(test_seq_len, batch_seq_len)
    max_donor_len = tf.maximum(test_donor_len, batch_donor_len)

    # Pad sequences to max length (post-padding)
    tcr_seqs_padded = tf.pad(tcr_seqs, [[0, 0], [0, max_seq_len - batch_seq_len]], 
                              constant_values=pad_token)
    test_tcr_seq_padded = tf.pad(test_tcr_seq, [[0, 0], [0, max_seq_len - test_seq_len]], 
                                  constant_values=pad_token)

    # Pad donor IDs to max length (post-padding)
    pad_token_int = tf.cast(pad_token, tf.int32)
    tcr_donor_ids_padded = tf.pad(tcr_donor_ids, [[0, 0], [0, max_donor_len - batch_donor_len]], 
                                   constant_values=pad_token_int)
    test_donor_id_padded = tf.pad(test_donor_id, [[0, 0], [0, max_donor_len - test_donor_len]], 
                                   constant_values=pad_token_int)

    # Concatenate test data at the beginning
    tcr_seqs = tf.concat([test_tcr_seq_padded, tcr_seqs_padded], axis=0)
    tcr_donor_ids = tf.concat([test_donor_id_padded, tcr_donor_ids_padded], axis=0)
    
    # Add dummy IDs for test samples
    dummy_tcr_ids = tf.zeros([num_test_samples], dtype=tf.int32)
    tcr_ids = tf.concat([dummy_tcr_ids, tcr_ids], axis=0)

    return tcr_seqs, tcr_ids, tcr_donor_ids


# Custom bounded activation for q logits
def bounded_activation_fn(x):
    return -42.0 + 37.0 * tf.nn.sigmoid(x)

get_custom_objects().update({
    "bounded_activation_fn": Activation(bounded_activation_fn)
})

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
# CONFIGURATION CONSTANTS
# =============================================================================
MHC_NUM = 620
PAD_TOKEN = -2.
MASK_TOKEN = -1.
BATCH_SIZE = 10
EPOCH = 1
TEST_MODE = False
ATT_MODE = True
TEST_MODE_OUTPUT_DIR = 'output/test_mode_Nov27_3'
GRAD_CLIP = True
DECOUPLE_TRAINING = False
CONTINUE_TRAINING = False
MODEL_PATH = 'checkpoints/exactloss_q+gamma+recon+reg+L1REGonQ+REGONGAMMA+public+RegOnDiversingGamma+FixQ/model_epoch_10.keras'
q_ampl = 1.0
gamma_ampl = 1.0
CHECKPOINT_PATH = 'checkpoints/exactloss_q+gamma+newdata'
LL_WEIGHTS = 1.
REG_WEIGHT = 0.
DROPOUT_RATE = 0.0
RECON_WEIGHT = 0.0
FIX_Q = False
EXACT_LIKELIHOOD = True
NUM_VALID_DONORS = 705.
PARQUET_DIR = os.path.join(CHECKPOINT_PATH, 'logs')
REG_ON_GAMMA = False
L1_REG_LAMBDA = 0.
ADJUST_LOGITS = False
L1_REG_Q =0.
REG_ON_Q = False
LAMBDA_REG_GAMMA_DIVERSE = 0.
log_dir = os.path.join(CHECKPOINT_PATH, 'gradients')
grad_log = True
# Data paths
writer = tf.summary.create_file_writer(log_dir)
train_path = 'data/processed_data/all_03December/more_than_1/processed_train/train.tfrecord'
val_path = 'data/processed_data/all_03December/more_than_1/processed_test/test.tfrecord'
patient_id_path = 'data/processed_data/all_03December/more_than_1/processed_train/patients_index_process.tsv'
mhc_path = 'data/processed_data/all_03December/more_than_1/processed_train/processed/donor_mhc.npz'
################# Write configs
config = {
    "MHC_NUM": MHC_NUM,
    "PAD_TOKEN": PAD_TOKEN,
    "MASK_TOKEN": MASK_TOKEN,
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCH": EPOCH,
    "TEST_MODE": TEST_MODE,
    "ATT_MODE": ATT_MODE,
    "TEST_MODE_OUTPUT_DIR": TEST_MODE_OUTPUT_DIR,
    "PARQUET_DIR": PARQUET_DIR,
    "GRAD_CLIP": GRAD_CLIP,
    "DECOUPLE_TRAINING": DECOUPLE_TRAINING,
    "CONTINUE_TRAINING": CONTINUE_TRAINING,
    "MODEL_PATH": MODEL_PATH,
    "q_ampl": q_ampl,
    "gamma_ampl": gamma_ampl,
    "CHECKPOINT_PATH": CHECKPOINT_PATH,
    "LL_WEIGHTS": LL_WEIGHTS,
    "REG_WEIGHT": REG_WEIGHT,
    "DROPOUT_RATE": DROPOUT_RATE,
    "RECON_WEIGHT": RECON_WEIGHT,
    "FIX_Q": FIX_Q,
    "EXACT_LIKELIHOOD": EXACT_LIKELIHOOD,
    "NUM_VALID_DONORS": NUM_VALID_DONORS,
    "REG_ON_GAMMA":REG_ON_GAMMA,
    "L1_REG_LAMBDA": L1_REG_LAMBDA,
    "ADJUST_LOGITS": ADJUST_LOGITS,
    "L1_REG_Q": L1_REG_Q,
    "REG_ON_Q": REG_ON_Q,
    "train_path":train_path,
    "val_path":val_path,
    "patient_id_path":patient_id_path,
    "log_dir":log_dir,
    "log_dir":log_dir


}

# Create checkpoint directory if missing
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Save JSON
config_path = os.path.join(CHECKPOINT_PATH, "config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=4)

print("Config saved to:", config_path)
# =============================================================================
# NEW: EXACT LIKELIHOOD FLAG
# =============================================================================
# If True: Use Equation 4 (exact) - sums over ALL N donors for second term
#          Avoids approximation assumptions but is more computationally expensive
#          O(B * N * A) for second term
#
# If False: Use Equation 8 (simplified) - approximates second term as q * Σ(Na * γ)
#           Faster but assumes: (1) q*p_ni << 1, (2) single-allele restriction
#           O(B * A) for second term


# Number of actual valid donors (not the expanded array size)
# This is used by LogSpaceExactLikelihood

# =============================================================================



# =============================================================================
# STARTUP BANNER
# =============================================================================
os.makedirs(PARQUET_DIR, exist_ok=True)
assert isinstance(GRAD_CLIP, (float, int, bool))
if GRAD_CLIP == True: 
    GRAD_CLIP = 7.0
    print(f'GRAD_CLIP Active with {GRAD_CLIP}')
print("=" * 80)
print("TCRtyper Model Training")
print("=" * 80)

# Auto-print all config entries
print("CONFIGURATION:")
for k, v in config.items():
    print(f"{k}: {v}")

print("=" * 80)

# =============================================================================
# LOAD DONOR MHC DATA
# =============================================================================
print("\nLoading donor MHC data...")
patient_tsv = pd.read_csv(patient_id_path, sep='\t')
max_num_patients = np.max(patient_tsv.sample_id.tolist())

mhc_file = np.load(mhc_path)
donor_mhc = mhc_file['array']
print(f"Donor MHC shape before adding removed patients: {donor_mhc.shape}")

# Expand donor_mhc to account for all patient IDs (including removed ones)
max_donor_mhc = np.zeros(shape=(max_num_patients + 1, donor_mhc.shape[1]))
patient_ids = np.array([int(i) for i in mhc_file['patient_id']])

for j, patient_id in enumerate(patient_ids):
    max_donor_mhc[patient_id] = donor_mhc[j]

donor_mhc = tf.constant(max_donor_mhc, dtype=tf.int32)
print(f"Donor MHC shape after adding removed patients: {donor_mhc.shape}")
print(f"Number of actual valid donors: {len(patient_ids)}")  # NEW: Verify donor count
freqs = calculate_frequencies(donor_mhc)

# =============================================================================
# MODEL DEFINITION
# =============================================================================
def define_model(
    mhc_num: int = MHC_NUM,
    embedding_dim: int = 11,
    vocab_size: int = 21,
    rope: bool = True,
    gate: bool = True,
    pad_token: float = PAD_TOKEN,
    mask_token: float = MASK_TOKEN,
    return_att_weights: bool = ATT_MODE,
    heads: int = 8,
    test_mode: bool = TEST_MODE
) -> keras.Model:
    """
    Define the TCRtyper model architecture.
    """
    # Force attention weights in test mode for visualization
    if test_mode:
        return_att_weights = True
    
    # Custom initialization for output layers
    gamma_init_weights = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
    gamma_init_bias = keras.initializers.Constant(-3.5)
    q_init_weights = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
    q_init_bias = keras.initializers.Constant(-10)

    # Input layers
    input_layer = layers.Input(shape=(None,), name='tcr_input')
    mask_input = layers.Input(shape=(None,), name='mask_input')
    
    # Embedding: (B, S) --> (B, S, D)
    me = MaskedEmbeddingOHE(
        vocab_size=vocab_size,
        mask_token=mask_token,
        pad_token=pad_token,
        name='masked_embedding_ohe'
    )(input_layer, mask_input)
    
    me = layers.Dropout(DROPOUT_RATE, name='dropout1')(me)
    me = layers.Dense(embedding_dim, 'relu', use_bias=False, name='mlp_embedding')(me)
    
    # Add positional encoding
    pe = PositionalEncoding(
        embed_dim=embedding_dim,
        pos_range=200,
        mask_token=mask_token,
        pad_token=pad_token,
        name='positional_encoding'
    )(me, mask_input)
    
    # Self-attention layer
    att_output = AttentionLayer(
        query_dim=embedding_dim,
        context_dim=embedding_dim,
        output_dim=embedding_dim,
        type='self',
        heads=heads,
        resnet=True,
        return_att_weights=return_att_weights,
        name='attention1',
        epsilon=1e-8,
        gate=gate,
        mask_token=mask_token,
        pad_token=pad_token,
        use_rope=rope,
        rope_max_seq_len=70,
        rope_base=10000.0
    )(pe, mask_input)
    
    # Unpack attention outputs
    if return_att_weights:
        att1, att_score = att_output
    else:
        att1 = att_output
        att_score = None

    att1 = layers.Dropout(DROPOUT_RATE, name='dropout2')(att1)
    
    # MLP transformation with relu
    mlp1 = MaskedDense(
        units=embedding_dim * 2,
        pad_token=pad_token,
        use_bias=True,
        name='mlp1'
    )([att1, mask_input])
    
    # Global pooling to sequence-level representation
    pooled = layers.GlobalAveragePooling1D(name='pool')(mlp1)
    pooled = layers.Dropout(rate=DROPOUT_RATE, name='dropout3')(pooled)

    # Output heads - both output LOGITS (no activation)
    gamma_logits = layers.Dense(
        mhc_num, 
        activation=None,
        name='gamma_logits', 
        #kernel_initializer=gamma_init_weights, 
        #bias_initializer=gamma_init_bias,
        kernel_regularizer=keras.regularizers.L2(l2=1e-3), 
        bias_regularizer=keras.regularizers.L2(l2=1e-3)
    )(pooled)
    
    q_logits = layers.Dense(
        1, 
        activation=None,
        name='q_logits', 
        #kernel_initializer=q_init_weights, 
        #bias_initializer=q_init_bias,
        kernel_regularizer=keras.regularizers.L2(l2=1e-2), 
        bias_regularizer=keras.regularizers.L2(l2=1e-2)
    )(pooled)
    
    
    # Build model with appropriate outputs
    if test_mode:
        model = keras.Model(
            inputs=[input_layer, mask_input],
            outputs=[gamma_logits, q_logits, att_score, me, pe, mlp1, pooled],
            name='TCRtyper_test'
        )
    elif return_att_weights:
        model = keras.Model(
            inputs=[input_layer, mask_input],
            outputs=[gamma_logits, q_logits, att_score],
            name='TCRtyper_att'
        )
    else:
        model = keras.Model(
            inputs=[input_layer, mask_input],
            outputs=[gamma_logits, q_logits],
            name='TCRtyper'
        )
    
    return model


# =============================================================================
# HELPER FUNCTION FOR RECONSTRUCTION LOSS
# =============================================================================
def compute_reconstruction_loss(tcr_seqs, recon_output, pad_token, cce_loss_fn):
    """
    Compute masked reconstruction loss.
    """
    mask = tf.cast(tcr_seqs != pad_token, tf.float32)
    input_tokens_safe = tf.where(tcr_seqs == pad_token, 0.0, tcr_seqs)
    input_tokens_safe = tf.cast(input_tokens_safe, tf.int32)
    one_hot_input = tf.one_hot(input_tokens_safe, depth=21)
    loss_per_token = cce_loss_fn(one_hot_input, recon_output)
    masked_loss = loss_per_token * mask
    num_valid_tokens = tf.reduce_sum(mask)
    final_loss = tf.reduce_sum(masked_loss) / tf.maximum(num_valid_tokens, 1.0)
    return final_loss


# =============================================================================
# BUILD AND COMPILE MODEL
# =============================================================================
print("\nBuilding model...")
model = define_model(
    mhc_num=MHC_NUM,
    embedding_dim=64,
    vocab_size=21,
    rope=True,
    gate=True,
    pad_token=PAD_TOKEN,
    mask_token=MASK_TOKEN,
    return_att_weights=ATT_MODE,
    heads=8,
    test_mode=TEST_MODE
)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=1,
    decay_rate=0.999,
    staircase=False
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer)

if CONTINUE_TRAINING:
    with tf.device("/GPU:0"):
        model = keras.saving.load_model(MODEL_PATH)
    print(model.optimizer)

print("\nModel Summary:")
print(model.summary())

# =============================================================================
# INITIALIZE LOSS FUNCTION AND DATA MANAGERS
# =============================================================================
print("\nInitializing loss function...")

# =============================================================================
# NEW: Conditional loss function selection based on EXACT_LIKELIHOOD flag
# =============================================================================
if EXACT_LIKELIHOOD:
    print(">>> Using EXACT likelihood (Equation 4)")
    print(f"    - Computes exact sum over all {NUM_VALID_DONORS:.0f} donors for second term")
    print(f"    - No approximation assumptions")
    print(f"    - Higher computational cost: O(B * N * A)")
    
    loss_func = LogSpaceExactLikelihood(
        donor_mhc=donor_mhc,
        pad_token=PAD_TOKEN,
        test_mode=TEST_MODE,
        fix_q=FIX_Q,  # Use formula for q_i
        num_mhc=MHC_NUM,
        N=NUM_VALID_DONORS,  # Pass actual number of valid donors
    )
else:
    print(">>> Using SIMPLIFIED likelihood (Equation 8)")
    print(f"    - Approximates second term as q * Σ(Na * γ)")
    print(f"    - Assumes: (1) q*p_ni << 1, (2) single-allele restriction")
    print(f"    - Lower computational cost: O(B * A)")
    
    loss_func = LogSpaceLikelihood(
        donor_mhc=donor_mhc,
        pad_token=PAD_TOKEN,
        test_mode=TEST_MODE,
        fix_q=FIX_Q,
        num_mhc=MHC_NUM,
    )
# =============================================================================

# Initialize CCE loss function once (outside the training loop)
cce_loss_fn = keras.losses.CategoricalCrossentropy(reduction='none')

print("\nSetting up training data manager...")
tcr_manager = TCRFileManager(
    tcr_path=train_path,
    batch_size=BATCH_SIZE,
    tcr_length=70,
    shuffle_buffer_size=200000,
    pad_token=PAD_TOKEN
)
print(f"Training data path: {train_path}")

print("Setting up validation data manager...")


# Create checkpoint directory
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
print("\nCheckpoint directory created/verified:", CHECKPOINT_PATH)

# =============================================================================
# TRAINING LOOP
# =============================================================================
print("\n" + "=" * 80)
print("Starting Training...")
print("=" * 80)

train_hist = []
val_hist = []

l1_reg_loss = tf.constant(0., dtype=tf.float32) # will be replaced later if True
l1_reg_q_loss = tf.constant(0., dtype=tf.float32)
test_tcr_seq = tf.constant([[12,  2,  8, 20, 20, 20, 20, 20, 20, 20,  5, 18, 21, 15, 12, 2,20, 20, 20, 20, 19,  5, 19, 21, 11, 20,  5, 11,  1,  2, 21,  4, 0, 15, 15, 14, 16,  7,  7, 17,  7,  5,  6, 18, 13, -2],
            [12,  2,  8, 20, 20, 20, 20, 20, 20, 20,  2, 15, 21, 15,  0, 15, 20, 20, 20, 20,  5,  7, 16, 21, 10, 20,  2, 11,  1,  5, 21,  4, 0, 15, 15,  5, 17, 15,  7, 15,  7,  2, 16,  9, 18, 13]], dtype=tf.float32)
test_donor_id = tf.constant([[   0, 1147, 1215, 1297,  367],[   0,   -2,   -2,   -2,   -2]], dtype=tf.int32)
for epoch in range(EPOCH):
    print(f"\n{'=' * 80}")
    print(f"EPOCH {epoch + 1}/{EPOCH}")
    print(f"{'=' * 80}")
    
    # Get training dataset for this epoch
    dataset = tcr_manager.get_dataset(shuffle=True)
    epoch_losses = []
    ll_loss_hist = []
    step = 0
    
    for step, data in enumerate(dataset, start=1):
        # Unpack and cast data
        tcr_seqs, tcr_ids, tcr_donor_ids = data
        tcr_seqs = tf.cast(tcr_seqs, tf.float32)
        tcr_ids = tf.cast(tcr_ids, tf.int32)
        tcr_donor_ids = tf.cast(tcr_donor_ids, tf.int32)
        
        ##### ADDED AS TEST
        tcr_seqs, tcr_ids, tcr_donor_ids = prepend_test_data(
            tcr_seqs, tcr_ids, tcr_donor_ids, 
            test_tcr_seq, test_donor_id, 
            pad_token=PAD_TOKEN)
        ##### END OF TEST

        # Create sequence mask
        tcr_seq_mask = tf.where(tcr_seqs == PAD_TOKEN, PAD_TOKEN, 1.)
        tcr_seq_mask = tf.cast(tcr_seq_mask, tf.float32)
        
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass
            if TEST_MODE:
                gamma_logits, q_logits, att_score, me, pe, mlp1, pooled = model([tcr_seqs, tcr_seq_mask])
                if ADJUST_LOGITS: gamma_logits = normalize_gamma_logits(gamma_logits, freqs)
                loss_components = loss_func.call(gamma_logits, q_logits, tcr_donor_ids)
                # Note: TEST_MODE returns different outputs depending on EXACT_LIKELIHOOD
                # Both return similar structure, but exact has additional log_p_ni_all
                if EXACT_LIKELIHOOD:
                    (Ni_size, Ni, gamma_donor_id_mask, log_p_ni, log_p_ni_all,
                     log_qp, log_one_minus_qp, first_term, second_term, bce,
                     log_gamma, log_one_minus_gamma) = loss_components
                else:
                    (Ni_size, Ni, gamma_donor_id_mask, log_p_ni,
                     log_qp, log_one_minus_qp, first_term, second_term, bce,
                     log_gamma, log_one_minus_gamma) = loss_components
            elif ATT_MODE:
                gamma_logits, q_logits, att_score = model([tcr_seqs, tcr_seq_mask])
                if ADJUST_LOGITS: gamma_logits = normalize_gamma_logits(gamma_logits, freqs)
                ll_loss, bce_loss, reg_term2 = loss_func.call(gamma_logits, q_logits, tcr_donor_ids)
            else:
                gamma_logits, q_logits = model([tcr_seqs, tcr_seq_mask])
                if ADJUST_LOGITS: gamma_logits = normalize_gamma_logits(gamma_logits, freqs)
                ll_loss, bce_loss, reg_term2 = loss_func.call(gamma_logits, q_logits, tcr_donor_ids)
            
            # L1 regularization on loss
            if REG_ON_GAMMA:
                l1_reg_loss = tf.reduce_sum(tf.reduce_sum(tf.nn.sigmoid(gamma_logits), axis=-1))
            if REG_ON_Q:
                l1_reg_q_loss = tf.reduce_sum(tf.nn.sigmoid(q_logits))
            # Regularization losses from model layers
            reg_loss = tf.reduce_sum(model.losses) if model.losses else 0.0
            reg_loss2 = tf.reduce_sum(reg_term2)
            # Total loss
            bce_loss_r = tf.reduce_mean(bce_loss)
            ll_loss = tf.reduce_mean(ll_loss)
            total_loss = (LL_WEIGHTS * ll_loss + 
                         REG_WEIGHT * reg_loss + 
                         L1_REG_LAMBDA * l1_reg_loss +
                         L1_REG_Q * l1_reg_q_loss +
                         LAMBDA_REG_GAMMA_DIVERSE * reg_loss2)
        
        # In TEST_MODE, we only process one batch for visualization
        if TEST_MODE:
            break
        
        # Compute gradients
        grads = tape.gradient(total_loss, model.trainable_variables)
        
        # Optionally clip gradients
        if GRAD_CLIP:
            grads, global_norm = tf.clip_by_global_norm(grads, GRAD_CLIP)
        else:
            global_norm = tf.linalg.global_norm(grads)
        
        if grad_log:
            grads_LL_gamma = tape.gradient(ll_loss, gamma_logits)
            g0 = grads_LL_gamma[0]
            g1 = grads_LL_gamma[1]
            g0_flat = tf.reshape(g0, [-1])
            g1_flat = tf.reshape(g1, [-1]) 
            with writer.as_default():
                tf.summary.scalar('loss/total', total_loss, step=step)
                tf.summary.scalar('loss/ll', ll_loss, step=step)
                # Gradient stats for parameters (your existing loop)
                for var, grad in zip(model.trainable_variables, grads):
                    if grad is not None:
                        tf.summary.histogram(f'gradients/{var.name}', grad, step=step)
                        tf.summary.scalar(f'gradient_norm/{var.name}', tf.norm(grad), step=step)
                tf.summary.histogram('grads_LL_gamma', grads_LL_gamma, step=step)
                tf.summary.scalar('grads_LL_gamma_norm', tf.norm(grads_LL_gamma), step=step)
                tf.summary.histogram('grads_LL_gamma_idx0', g0_flat, step=step)
                tf.summary.scalar('grads_LL_gamma_idx0_norm', tf.norm(g0_flat), step=step)
                tf.summary.histogram('grads_LL_gamma_idx1', g1_flat, step=step)
                tf.summary.scalar('grads_LL_gamma_idx1_norm', tf.norm(g1_flat), step=step)
            
            


        # Optionally scale gradients for decoupled learning rates
        if DECOUPLE_TRAINING:
            scaled_gradients = []
            for grad, var in zip(grads, model.trainable_variables):
                if grad is None:
                    scaled_gradients.append(None)
                elif 'gamma_logits' in var.name:
                    scaled_gradients.append(grad * gamma_ampl)
                elif 'q_logits' in var.name:
                    scaled_gradients.append(grad * q_ampl)
                else:
                    scaled_gradients.append(grad)
            model.optimizer.apply_gradients(zip(scaled_gradients, model.trainable_variables))
        else:
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        epoch_losses.append(total_loss.numpy())
        ll_loss_hist.append(ll_loss.numpy())
        
        # Compute gamma and q probabilities for logging
        gamma_probs = tf.nn.sigmoid(gamma_logits)
        q_probs = tf.nn.sigmoid(q_logits)
        
        # Build the logging statement
        statement = (
            f"  [Step {step:4d}] "
            f"Loss: {total_loss.numpy():.4f} | "
            f"LL: {ll_loss.numpy():.4f} | "
            f"Reg: {reg_loss.numpy():.4f} | "
            f"GradNorm: {global_norm.numpy():.4f} | "
            f"AvgGamma: {tf.reduce_mean(gamma_probs).numpy():.6f} | "
            f"AvgQ: {tf.reduce_mean(q_probs).numpy():.6f} | "
            f"MaxGamma: {tf.reduce_max(gamma_probs).numpy():.6f} | "
            f"MinGamma: {tf.reduce_min(gamma_probs).numpy():.6f} | "
            f"MaxQ: {tf.reduce_max(q_probs).numpy():.6f} | "
            f"MinQ: {tf.reduce_min(q_probs).numpy():.6f} | "
            f"L1 RegGAMMA: {l1_reg_loss.numpy():.6f} | "
            f"L1 RegQ: {l1_reg_q_loss.numpy():.6f} | "
            f"GammaDiverseREG: {reg_loss2.numpy():.6f}"
        )
        

        
        print(statement)
        
        # Log to parquet - NEW: Add likelihood type to logs
        log_row = {
            "epoch": epoch + 1,
            "step": step,
            "exact_likelihood": EXACT_LIKELIHOOD,  # NEW: Log which likelihood is used
            "total_loss": float(total_loss.numpy()),
            "ll_loss": float(ll_loss.numpy()),
            "reg_loss": float(reg_loss.numpy()),
            "grad_norm": float(global_norm.numpy()),
            "avg_gamma": float(tf.reduce_mean(gamma_probs).numpy()),
            "max_gamma": float(tf.reduce_max(gamma_probs).numpy()),
            "min_gamma": float(tf.reduce_min(gamma_probs).numpy()),
            "std_gamma": float(tf.math.reduce_std(gamma_probs).numpy()),
            "avg_q": float(tf.reduce_mean(q_probs).numpy()),
            "max_q": float(tf.reduce_max(q_probs).numpy()),
            "min_q": float(tf.reduce_min(q_probs).numpy()),
        }
        table = pa.Table.from_pandas(pd.DataFrame([log_row]))
        pq.write_to_dataset(table, root_path=PARQUET_DIR)

    # Exit training loop in TEST_MODE
    if TEST_MODE:
        break
    
    # Epoch statistics
    avg_loss = np.mean(epoch_losses)
    min_loss = np.min(epoch_losses)
    max_loss = np.max(epoch_losses)
    train_hist.append(avg_loss)
    
    print(f"\n  Training Summary:")
    print(f"    Average Loss: {avg_loss:.4f}")
    print(f"    Min Loss:     {min_loss:.4f}")
    print(f"    Max Loss:     {max_loss:.4f}")
    print(f"    LL Loss:      {np.mean(ll_loss_hist):.4f}")
    print(f"    Total Steps:  {step}")


# =============================================================================
# TEST MODE VISUALIZATIONS
# =============================================================================
if TEST_MODE:
    print("\n" + "=" * 80)
    print("TEST MODE: Generating Visualizations")
    print("=" * 80)
    
    # Initialize visualizer
    visualizer = TestModeVisualizer(output_dir=TEST_MODE_OUTPUT_DIR)
    
    # Determine sample indices to visualize
    batch_size_actual = gamma_logits.shape[0]
    sample_indices = list(range(min(3, batch_size_actual)))
    print(f"Visualizing samples: {sample_indices}")
    
    # Compute probabilities from logits for visualization
    gamma_probs = tf.nn.sigmoid(gamma_logits)
    q_probs = tf.nn.sigmoid(q_logits)
    
    # Compute p_ni from log_p_ni for visualization
    p_ni = tf.exp(log_p_ni)
    
    # Print tensor shapes for reference
    tensor_dict = {
        'tcr_seqs': tcr_seqs,
        'tcr_ids': tcr_ids,
        'tcr_donor_ids': tcr_donor_ids,
        'gamma_logits': gamma_logits,
        'gamma_probs': gamma_probs,
        'q_logits': q_logits,
        'q_probs': q_probs,
        'att_score': att_score,
        'me (embedding)': me,
        'pe (pos encoding)': pe,
        'mlp1': mlp1,
        'pooled': pooled,
        'Ni_size': Ni_size,
        'Ni': Ni,
        'gamma_donor_id_mask': gamma_donor_id_mask,
        'log_p_ni': log_p_ni,
        'log_qp': log_qp,
        'log_one_minus_qp': log_one_minus_qp,
        'first_term': first_term,
        'second_term': second_term,
        'bce': bce,
    }
    
    # NEW: Add log_p_ni_all if using exact likelihood
    if EXACT_LIKELIHOOD:
        tensor_dict['log_p_ni_all'] = log_p_ni_all
    
    visualizer.print_tensor_shapes(tensor_dict)
    
    print("\n--- Generating Plots ---")
    
    visualizer.plot_input_sequences(tcr_seqs, sample_indices)
    visualizer.plot_donor_ids(tcr_donor_ids, sample_indices)
    visualizer.plot_masked_embedding(me, sample_indices)
    visualizer.plot_positional_encoding(pe, sample_indices)
    visualizer.plot_attention_heads(att_score, sample_indices)
    visualizer.plot_attention_averaged(att_score, sample_indices)
    visualizer.plot_mlp_output(mlp1, sample_indices)
    visualizer.plot_pooled_representation(pooled, sample_indices)
    visualizer.plot_gamma(gamma_probs, sample_indices)
    visualizer.plot_q_probability(q_probs, sample_indices)
    visualizer.plot_ni_size(Ni_size, sample_indices)
    visualizer.plot_ni_matrix(Ni, gamma_donor_id_mask, sample_indices)
    visualizer.plot_donor_id_mask(gamma_donor_id_mask, sample_indices)
    visualizer.plot_pn(p_ni, sample_indices)
    visualizer.plot_numerator(log_qp, sample_indices)
    visualizer.plot_denominator(log_one_minus_qp, sample_indices)
    visualizer.plot_likelihood_terms(first_term, second_term, sample_indices)
    visualizer.plot_all_scalars(first_term, second_term, Ni_size, q_probs, sample_indices)
    visualizer.plot_gamma_histogram(gamma_probs, sample_indices)
    visualizer.save_summary_statistics(
        tcr_seqs=tcr_seqs,
        gamma=gamma_probs,
        q=q_probs,
        att_score=att_score,
        Ni=Ni,
        pn=p_ni,
        Ni_size=Ni_size,
        first_term=first_term,
        second_term=second_term,
        sample_indices=sample_indices
    )
    
    print("\n" + "=" * 80)
    print(f"✓ TEST MODE visualizations complete!")
    print(f"✓ All outputs saved to: {TEST_MODE_OUTPUT_DIR}")
    print("=" * 80)


# =============================================================================
# SAVE FINAL MODEL (Production Mode Only)
# =============================================================================
if not TEST_MODE:
    print("\n" + "=" * 80)
    
    # Save final model
    final_model_path = os.path.join(CHECKPOINT_PATH, 'final_model.keras')
    model.save(final_model_path)
    
    # Save training history - NEW: Include likelihood type
    history_dict = {
        'train': [float(loss) for loss in train_hist],
        'val': [float(loss) for loss in val_hist],
        'exact_likelihood': EXACT_LIKELIHOOD,  # NEW: Record which likelihood was used
        'num_valid_donors': NUM_VALID_DONORS,
    }
    with open('train_hist.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"✓ Training completed successfully!")
    print(f"✓ Final model saved: {final_model_path}")
    print(f"✓ Training history saved: train_hist.json")
    print(f"✓ Likelihood type used: {'EXACT (Eq. 4)' if EXACT_LIKELIHOOD else 'SIMPLIFIED (Eq. 8)'}")
    print("=" * 80)