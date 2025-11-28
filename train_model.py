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
    LogSpaceLikelihood,  # Added this import
    log1mexp,  # Added this import if needed
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

# Custom bounded activation for q logits
# This bounds the logits to a reasonable range for numerical stability
def bounded_activation_fn(x):
    return -42.0 + 37.0 * tf.nn.sigmoid(x)

get_custom_objects().update({
    "bounded_activation_fn": Activation(bounded_activation_fn)
})

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
PAD_TOKEN = -2.
MASK_TOKEN = -1.
BATCH_SIZE = 5000
EPOCH = 15
TEST_MODE = False
ATT_MODE = True
TEST_MODE_OUTPUT_DIR = 'output/test_mode_Nov27_3'
PARQUET_DIR = "training_logs2"
GRAD_CLIP = False
DECOUPLE_TRAINING = True
CONTINUE_TRAINING = False
MODEL_PATH = 'checkpoints/model_epoch_2.keras'
q_ampl = 0.05
gamma_ampl = 2.
CHECKPOINT_PATH = 'checkpoints/larger_buffer'
SOFTMAX_LOSS = True
SOFTMAX_WEIGHTS = 0.5
SOFTMAX_TEMP = 1.

# Data paths
train_path = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/train_val_split/datasetwise/train1.tfrecord'
val_path = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/train_val_split/datasetwise/val1.tfrecord'
patient_id_path = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/patients_index_process.tsv'

# =============================================================================
# STARTUP BANNER
# =============================================================================
os.makedirs(PARQUET_DIR, exist_ok=True)
assert isinstance(GRAD_CLIP, (float, int, bool))
if GRAD_CLIP == True: 
    GRAD_CLIP = 5.0
    print(f'GRAD_CLIP Active with {GRAD_CLIP}')
print("=" * 80)
print("TCRtyper Model Training")
print("=" * 80)
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCH}")
print(f"Pad Token: {PAD_TOKEN}")
print(f"Mask Token: {MASK_TOKEN}")
print(f"TEST_MODE: {TEST_MODE}")
print(f"ATT_MODE: {ATT_MODE}")
print("=" * 80)

# =============================================================================
# LOAD DONOR MHC DATA
# =============================================================================
print("\nLoading donor MHC data...")
patient_tsv = pd.read_csv(patient_id_path, sep='\t')
max_num_patients = np.max(patient_tsv.sample_id.tolist())

mhc_file = np.load('data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/processed/donor_mhc.npz')
donor_mhc = mhc_file['array']
print(f"Donor MHC shape before adding removed patients: {donor_mhc.shape}")

# Expand donor_mhc to account for all patient IDs (including removed ones)
max_donor_mhc = np.zeros(shape=(max_num_patients + 1, donor_mhc.shape[1]))
patient_ids = np.array([int(i) for i in mhc_file['patient_id']])

for j, patient_id in enumerate(patient_ids):
    max_donor_mhc[patient_id] = donor_mhc[j]

donor_mhc = tf.constant(max_donor_mhc, dtype=tf.int32)
print(f"Donor MHC shape after adding removed patients: {donor_mhc.shape}")


# =============================================================================
# MODEL DEFINITION
# =============================================================================
def define_model(
    mhc_num: int = 358,
    embedding_dim: int = 64,
    vocab_size: int = 21,
    rope: bool = True,
    gate: bool = True,
    pad_token: float = PAD_TOKEN,
    mask_token: float = MASK_TOKEN,
    return_att_weights: bool = ATT_MODE,
    heads: int = 4,
    test_mode: bool = TEST_MODE
) -> keras.Model:
    """
    Define the TCRtyper model architecture.
    
    The model consists of:
    1. Masked embedding layer for amino acid tokens
    2. Sinusoidal positional encoding
    3. Self-attention layer with optional RoPE and gating
    4. MLP layer for feature transformation
    5. Global average pooling
    6. Output heads for gamma (HLA binding) and q (sampling probability)
    
    Args:
        mhc_num: Number of HLA alleles to predict (output dimension).
        embedding_dim: Dimension of token embeddings.
        vocab_size: Size of amino acid vocabulary.
        rope: Whether to use Rotary Position Embedding.
        gate: Whether to use gating mechanism in attention.
        pad_token: Value used for padding.
        mask_token: Value used for masking.
        return_att_weights: Whether to return attention weights.
        heads: Number of attention heads.
        test_mode: If True, return all intermediate tensors.
    
    Returns:
        Compiled Keras model with appropriate outputs based on mode.
    """
    # Force attention weights in test mode for visualization
    if test_mode:
        return_att_weights = True
    
    # Input layers
    input_layer = layers.Input(shape=(None,), name='tcr_input')
    mask_input = layers.Input(shape=(None,), name='mask_input')
    
    # Embedding: (B, S) --> (B, S, D)
    me = MaskedEmbedding(
        vocab_size=vocab_size,
        embed_dim=embedding_dim,
        mask_token=mask_token,
        pad_token=pad_token,
        name='masked_embedding'
    )(input_layer, mask_input)
    
    # Add positional encoding
    pe = PositionalEncoding(
        embed_dim=embedding_dim,
        pos_range=200,
        mask_token=mask_token,
        pad_token=pad_token,
        name='positional_encoding'
    )(me, mask_input)
    
    pe = layers.Dropout(rate=0.01, name='dropout1')(pe)
    
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
        rope_max_seq_len=100,
        rope_base=10000.0
    )(pe, mask_input)
    
    # Unpack attention outputs
    if return_att_weights:
        att1, att_score = att_output
    else:
        att1 = att_output
        att_score = None
    
    # MLP transformation
    mlp1 = MaskedDense(
        units=128,
        pad_token=pad_token,
        use_bias=True,
        name='mlp1'
    )([att1, mask_input])
    
    # Global pooling to sequence-level representation
    pooled = layers.GlobalAveragePooling1D(name='pool')(mlp1)
    pooled = layers.Dropout(rate=0.01, name='dropout2')(pooled)
    
    # Custom initialization for output layers
    gamma_init_weights = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
    gamma_init_bias = keras.initializers.Constant(-3.5)  # sigmoid(-3.5) ≈ 0.03
    q_init_weights = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
    q_init_bias = keras.initializers.Constant(-10)  # sigmoid(-5) ≈ 0.007

    # Output heads - both output LOGITS (no activation)
    # The LogSpaceLikelihood will compute log-probabilities from these
    gamma_logits = layers.Dense(
        mhc_num, 
        activation=None,  # Output raw logits
        name='gamma', 
        kernel_initializer=gamma_init_weights, 
        bias_initializer=gamma_init_bias,
        kernel_regularizer=keras.regularizers.L2(1e-4), 
        bias_regularizer=keras.regularizers.L2(1e-2)
    )(pooled)
    
    q_logits = layers.Dense(
        1, 
        activation=None,  # Output raw logits (changed from bounded_activation_fn for consistency)
        name='q_prob', 
        kernel_initializer=q_init_weights, 
        bias_initializer=q_init_bias,
        kernel_regularizer=keras.regularizers.L2(1e-2), 
        bias_regularizer=keras.regularizers.L2(1e-2)
    )(pooled)
    
    # Build model with appropriate outputs
    if test_mode:
        # Return all intermediate tensors for visualization
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
# BUILD AND COMPILE MODEL
# =============================================================================
print("\nBuilding model...")
model = define_model(
    mhc_num=358,
    embedding_dim=64,
    vocab_size=21,
    rope=True,
    gate=True,
    pad_token=PAD_TOKEN,
    mask_token=MASK_TOKEN,
    return_att_weights=ATT_MODE,
    heads=4,
    test_mode=TEST_MODE
)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=1,          # decay every iteration
    decay_rate=0.999,       # reduce a little each step
    staircase=False         # smooth decay
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
loss_func = LogSpaceLikelihood(
    donor_mhc, 
    pad_token=PAD_TOKEN, 
    test_mode=TEST_MODE,                 
    use_softmax_loss=SOFTMAX_LOSS,
    softmax_loss_weight=SOFTMAX_WEIGHTS,
    softmax_temperature=SOFTMAX_TEMP,
)

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
val_manager = TCRFileManager(
    tcr_path=val_path,
    batch_size=BATCH_SIZE,
    tcr_length=70,
    shuffle_buffer_size=1000,
    pad_token=PAD_TOKEN
)
print(f"Validation data path: {val_path}")

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

for epoch in range(EPOCH):
    print(f"\n{'=' * 80}")
    print(f"EPOCH {epoch + 1}/{EPOCH}")
    print(f"{'=' * 80}")
    
    # Get training dataset for this epoch
    dataset = tcr_manager.get_dataset(shuffle=True)
    epoch_losses = []
    
    for step, data in enumerate(dataset, start=1):
        # Unpack and cast data
        tcr_seqs, tcr_ids, tcr_donor_ids = data
        tcr_seqs = tf.cast(tcr_seqs, tf.float32)
        tcr_ids = tf.cast(tcr_ids, tf.int32)
        tcr_donor_ids = tf.cast(tcr_donor_ids, tf.int32)
        
        # Create sequence mask
        tcr_seq_mask = tf.where(tcr_seqs == PAD_TOKEN, PAD_TOKEN, 1.)
        tcr_seq_mask = tf.cast(tcr_seq_mask, tf.float32)
        
        with tf.GradientTape() as tape:
            # Forward pass - outputs depend on mode
            # Model outputs LOGITS, not probabilities
            if TEST_MODE:
                gamma_logits, q_logits, att_score, me, pe, mlp1, pooled = model([tcr_seqs, tcr_seq_mask])
                loss_components = loss_func.call(gamma_logits, q_logits, tcr_donor_ids)
                (Ni_size, Ni, gamma_donor_id_mask, log_p_ni,
                 log_qp, log_one_minus_qp, first_term, second_term, cce,
                 log_gamma, log_one_minus_gamma) = loss_components
            elif ATT_MODE:
                gamma_logits, q_logits, att_score = model([tcr_seqs, tcr_seq_mask])
                LL, cce_loss = loss_func.call(gamma_logits, q_logits, tcr_donor_ids)
            else:
                gamma_logits, q_logits = model([tcr_seqs, tcr_seq_mask])
                LL, cce_loss = loss_func.call(gamma_logits, q_logits, tcr_donor_ids)
            
            # Add regularization losses from model layers
            reg_loss = tf.reduce_sum(model.losses) if model.losses else 0.0
            total_loss = LL + cce_loss + reg_loss
        
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
        
        # Optionally scale gradients for decoupled learning rates
        if DECOUPLE_TRAINING:
            scaled_gradients = []
            for grad, var in zip(grads, model.trainable_variables):
                if grad is None:
                    scaled_gradients.append(None)
                elif 'gamma' in var.name:
                    scaled_gradients.append(grad * gamma_ampl)  # Amplify gamma gradients
                elif 'q_prob' in var.name:
                    scaled_gradients.append(grad * q_ampl)  # Dampen q gradients
                else:
                    scaled_gradients.append(grad)
            # FIXED: Apply scaled_gradients instead of grads
            model.optimizer.apply_gradients(zip(scaled_gradients, model.trainable_variables))
        else:
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        epoch_losses.append(total_loss.numpy())
        
        # Compute gamma and q probabilities for logging
        gamma_probs = tf.nn.sigmoid(gamma_logits)
        q_probs = tf.nn.sigmoid(q_logits)
        statement = (f"  [Step {step:4d}] Loss: {total_loss.numpy():.4f} | "
              f"LL Loss: {LL.numpy():.4f} | "
              f"reg_loss: {reg_loss.numpy():.4f} | "
              f"Grad Norm: {global_norm.numpy():.4f} | "
              f"Avg Gamma: {tf.reduce_mean(gamma_probs).numpy():.6f} | "
              f"Avg q: {tf.reduce_mean(q_probs).numpy():.6f} | "
              f"Max Gamma: {tf.reduce_max(gamma_probs).numpy():.6f} | "
              f"Min Gamma: {tf.reduce_min(gamma_probs).numpy():.6f} | "
              f"Max q: {tf.reduce_max(q_probs).numpy():.6f} | "
              f"Min q: {tf.reduce_min(q_probs).numpy():.6f}")
        if SOFTMAX_LOSS: statement = f'| CCE loss: {cce_loss.numpy():.6f}'
        print(statement)
        
        # Log to parquet
        log_row = {
            "step": step,
            "loss": float(total_loss.numpy()),
            "grad_norm": float(global_norm.numpy()),
            "avg_gamma": float(tf.reduce_mean(gamma_probs).numpy()),
            "avg_q": float(tf.reduce_mean(q_probs).numpy())
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
    print(f"    Total Steps:  {step}")
    
    # Validation loop
    print(f"\n  Running validation...")
    val_losses = []
    val_dataset = val_manager.get_dataset(shuffle=False)
    
    for val_step, val_data in enumerate(val_dataset, start=1):
        tcr_seqs, tcr_ids, tcr_donor_ids = val_data
        tcr_seqs = tf.cast(tcr_seqs, tf.float32)
        tcr_ids = tf.cast(tcr_ids, tf.int32)
        tcr_donor_ids = tf.cast(tcr_donor_ids, tf.int32)
        tcr_seq_mask = tf.where(tcr_seqs == PAD_TOKEN, PAD_TOKEN, 1.)
        
        if ATT_MODE:
            gamma_logits, q_logits, _ = model([tcr_seqs, tcr_seq_mask], training=False)
        else:
            gamma_logits, q_logits = model([tcr_seqs, tcr_seq_mask], training=False)
        
        # FIXED: loss_func.call returns a single scalar in non-test mode
        val_loss = loss_func.call(gamma_logits, q_logits, tcr_donor_ids)
        val_losses.append(val_loss.numpy())
    
    avg_val_loss = np.mean(val_losses)
    min_val_loss = np.min(val_losses)
    max_val_loss = np.max(val_losses)
    val_hist.append(avg_val_loss)
    
    print(f"  Validation Summary:")
    print(f"    Average Loss: {avg_val_loss:.4f}")
    print(f"    Min Loss:     {min_val_loss:.4f}")
    print(f"    Max Loss:     {max_val_loss:.4f}")
    print(f"    Total Steps:  {val_step}")
    
    # Save checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_PATH, f'model_epoch_{epoch + 1}.keras')
    model.save(checkpoint_path)
    print(f"\n  ✓ Model checkpoint saved: {checkpoint_path}")


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
    # UPDATED: Using correct variable names from LogSpaceLikelihood
    visualizer.print_tensor_shapes({
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
        'cce': cce,
    })
    
    print("\n--- Generating Plots ---")
    
    # Step 1: Input TCR sequences
    visualizer.plot_input_sequences(tcr_seqs, sample_indices)
    
    # Step 2: Donor IDs per TCR
    visualizer.plot_donor_ids(tcr_donor_ids, sample_indices)
    
    # Step 3: Masked embedding output
    visualizer.plot_masked_embedding(me, sample_indices)
    
    # Step 4: Positional encoding output
    visualizer.plot_positional_encoding(pe, sample_indices)
    
    # Step 5: Attention heads (per-head visualization)
    visualizer.plot_attention_heads(att_score, sample_indices)
    
    # Step 6: Attention averaged over heads
    visualizer.plot_attention_averaged(att_score, sample_indices)
    
    # Step 7: MLP output
    visualizer.plot_mlp_output(mlp1, sample_indices)
    
    # Step 8: Pooled representation
    visualizer.plot_pooled_representation(pooled, sample_indices)
    
    # Step 9: Gamma predictions (HLA binding probabilities)
    # Use gamma_probs (sigmoid of logits) for visualization
    visualizer.plot_gamma(gamma_probs, sample_indices)
    
    # Step 10: Q predictions (sampling probability)
    visualizer.plot_q_probability(q_probs, sample_indices)
    
    # Step 11: Ni_size (number of donors per TCR)
    visualizer.plot_ni_size(Ni_size, sample_indices)
    
    # Step 12: Ni matrix (donor MHC profiles)
    visualizer.plot_ni_matrix(Ni, gamma_donor_id_mask, sample_indices)
    
    # Step 13: Donor ID mask
    visualizer.plot_donor_id_mask(gamma_donor_id_mask, sample_indices)
    
    # Step 14: p_ni values (TCR-donor binding probability)
    # UPDATED: Convert from log-space for visualization
    visualizer.plot_pn(p_ni, sample_indices)
    
    # Step 15: log(q * p_ni) - in log space
    visualizer.plot_numerator(log_qp, sample_indices)
    
    # Step 16: log(1 - q * p_ni) - in log space
    visualizer.plot_denominator(log_one_minus_qp, sample_indices)
    
    # Step 17: First vs Second term comparison
    visualizer.plot_likelihood_terms(first_term, second_term, sample_indices)
    
    # Step 18: All scalar metrics comparison
    visualizer.plot_all_scalars(first_term, second_term, Ni_size, q_probs, sample_indices)
    
    # Step 19: Gamma distribution histogram
    visualizer.plot_gamma_histogram(gamma_probs, sample_indices)
    
    # Step 20: Summary statistics text file
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
    # FIXED: typo ow.path.join -> os.path.join
    final_model_path = os.path.join(CHECKPOINT_PATH, 'final_model.keras')
    model.save(final_model_path)
    
    # Save training history
    history_dict = {
        'train': [float(loss) for loss in train_hist],
        'val': [float(loss) for loss in val_hist]
    }
    with open('train_hist.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"✓ Training completed successfully!")
    print(f"✓ Final model saved: {final_model_path}")
    print(f"✓ Training history saved: train_hist.json")
    print("=" * 80)