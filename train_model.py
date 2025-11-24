import keras
import tensorflow as tf
import src
from keras import layers
from src.model_utils import (PositionalEncoding, MaskedEmbedding, AttentionLayer, Likelihood, MaskedDense)
from src.utils import TCRFileManager
import numpy as np
import os
import pandas as pd

PAD_TOKEN = -2.
MASK_TOKEN = -1.
BATCH_SIZE = 1000
EPOCH = 10

train_path = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/train_val_split/datasetwise/train1.tfrecord'
val_path = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/train_val_split/datasetwise/val1.tfrecord'
patient_id_path = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/patients_index_process.tsv'

print("="*80)
print("TCRtyper Model Training")
print("="*80)
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCH}")
print(f"Pad Token: {PAD_TOKEN}")
print(f"Mask Token: {MASK_TOKEN}")
print("="*80)

#### Load donor MHC data
print("\nLoading donor MHC data...")
patient_tsv = pd.read_csv(patient_id_path, sep='\t')
max_num_patients = np.max(patient_tsv.sample_id.tolist())
mhc_file = np.load('data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/processed/donor_mhc.npz')
donor_mhc = mhc_file['array']
print(f"Donor shape MHC Before adding removed patients: {donor_mhc.shape}")
max_donor_mhc = np.zeros(shape=(max_num_patients+1, donor_mhc.shape[1]))
patient_ids = np.array([int(i) for i in mhc_file['patient_id']])
j = 0
for i in patient_ids:
    max_donor_mhc[i] = donor_mhc[j]
    j += 1
donor_mhc = tf.constant(max_donor_mhc, dtype=tf.int32)
print(f"Donor shape MHC After adding removed patients: {donor_mhc.shape}")
##### Utility Functions
def define_model(mhc_num=358, embedding_dim: int = 64, vocab_size: int = 21, rope: bool = True, gate: bool = True,
                 pad_token: float = PAD_TOKEN, mask_token: float = MASK_TOKEN,
                 return_att_weights: bool = False, heads: int = 4):
    input = layers.Input(shape=(None,))
    mask_input = layers.Input(shape=(None,))
    
    # (B,S) --> (B,S,D)
    me = MaskedEmbedding(vocab_size=vocab_size, embed_dim=embedding_dim, mask_token=mask_token, pad_token=pad_token, name='masked_embedding')(input, mask_input)
    pe = PositionalEncoding(embed_dim=embedding_dim, pos_range=100, mask_token=mask_token, pad_token=pad_token, name='positional_encoding')(me, mask_input)
    pe = layers.Dropout(rate=0.3, name='dropout1')(pe)
    
    # Fixed: Added closing parenthesis and conditional handling
    att_output = AttentionLayer(query_dim=embedding_dim,
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
                                 rope_base=10000.0)(pe, mask_input)
    
    if return_att_weights:
        att1, att_score = att_output
    else:
        att1 = att_output
        att_score = None
    
    mlp1 = MaskedDense(units=128, pad_token=pad_token, use_bias=True, name='mlp1')([att1, mask_input])
    pooled = layers.GlobalAveragePooling1D(name='pool')(mlp1)
    pooled = layers.Dropout(rate=0.3)(pooled)
    
    gamma = layers.Dense(mhc_num, activation='sigmoid', name='gamma')(pooled)  #(B,A)
    q = layers.Dense(1, 'sigmoid', name='q_prob')(pooled)  #(B,1)
    
    # Fixed: Only include att_score if return_att_weights is True
    if return_att_weights:
        model = keras.Model([input, mask_input], [gamma, q, att_score], name='TCRtyper')
    else:
        model = keras.Model([input, mask_input], [gamma, q], name='TCRtyper')
    
    return model

#### Start training
print("\nBuilding model...")
model = define_model(mhc_num=358, embedding_dim=64, vocab_size=21, rope=True, gate=True,
                     pad_token=PAD_TOKEN, mask_token=MASK_TOKEN,
                     return_att_weights=False, heads=4)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6))
print("\nModel Summary:")
print(model.summary())

print("\nInitializing loss function...")
loss_func = Likelihood(donor_mhc, pad_token=PAD_TOKEN)

# Training data manager
print("\nSetting up training data manager...")
tcr_manager = TCRFileManager(
    tcr_path=train_path,
    batch_size=BATCH_SIZE,
    tcr_length=70,
    shuffle_buffer_size=15000,
    pad_token=PAD_TOKEN
)
print(f"Training data path: {train_path}")

# Validation data manager
print("Setting up validation data manager...")
val_manager = TCRFileManager(
    tcr_path=val_path,
    batch_size=BATCH_SIZE,
    tcr_length=70,
    shuffle_buffer_size=1000,
    pad_token=PAD_TOKEN
)
print(f"Validation data path: {val_path}")

# Create checkpoint directory if it doesn't exist
os.makedirs('checkpoints', exist_ok=True)
print("\nCheckpoint directory created/verified: ./checkpoints/")

# Training loop
print("\n" + "="*80)
print("Starting Training...")
print("="*80)

for epoch in range(EPOCH):
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch+1}/{EPOCH}")
    print(f"{'='*80}")
    
    # Get training dataset
    dataset = tcr_manager.get_dataset(shuffle=True)
    
    # Track epoch-level statistics
    epoch_losses = []
    
    for step, data in enumerate(dataset, start=1):
        tcr_seqs, tcr_ids, tcr_donor_ids = data
        tcr_seqs = tf.cast(tcr_seqs, tf.float32)
        tcr_ids = tf.cast(tcr_ids, tf.int32)
        tcr_donor_ids = tf.cast(tcr_donor_ids, tf.int32)
        
        # Fixed: Use tcr_seqs instead of tcr_donor_ids for sequence mask
        tcr_seq_mask = tf.where(tcr_seqs == PAD_TOKEN, PAD_TOKEN, 1.)
        tcr_seq_mask = tf.cast(tcr_seq_mask, tf.int32)
        
        with tf.GradientTape() as tape:
            # Model returns [gamma, q] since return_att_weights=False
            gamma, q = model([tcr_seqs, tcr_seq_mask])
            LL = loss_func.call(gamma, q, tcr_donor_ids)
        
        grads = tape.gradient(LL, model.trainable_variables)
        
        # Fixed: Add gradient clipping for stability
        grads, global_norm = tf.clip_by_global_norm(grads, 5.0)
        
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        epoch_losses.append(LL.numpy())
        
        # Print every 10 steps
        print(f"  [Step {step:4d}] Loss: {LL.numpy():.4f} | Grad Norm: {global_norm.numpy():.4f}")
    
    # Print epoch-level statistics
    avg_loss = np.mean(epoch_losses)
    min_loss = np.min(epoch_losses)
    max_loss = np.max(epoch_losses)
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
        
        gamma, q = model([tcr_seqs, tcr_seq_mask], training=False)
        LL = loss_func(gamma, q, tcr_donor_ids)
        val_losses.append(LL.numpy())
    
    avg_val_loss = np.mean(val_losses)
    min_val_loss = np.min(val_losses)
    max_val_loss = np.max(val_losses)
    print(f"  Validation Summary:")
    print(f"    Average Loss: {avg_val_loss:.4f}")
    print(f"    Min Loss:     {min_val_loss:.4f}")
    print(f"    Max Loss:     {max_val_loss:.4f}")
    print(f"    Total Steps:  {val_step}")
    
    # Save model checkpoint
    checkpoint_path = f'checkpoints/model_epoch_{epoch+1}.keras'
    model.save(checkpoint_path)
    print(f"\n  ✓ Model checkpoint saved: {checkpoint_path}")

# Save final model
print("\n" + "="*80)
final_model_path = 'checkpoints/final_model.keras'
model.save(final_model_path)
print(f"✓ Training completed successfully!")
print(f"✓ Final model saved: {final_model_path}")
print("="*80)
