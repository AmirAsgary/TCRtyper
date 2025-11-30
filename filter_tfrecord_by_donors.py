"""
Filter TFRecord files to keep only TCRs with >= MIN_DONORS valid donors.
Uses batched reading and vectorized GPU operations.
"""
import keras
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import sys
import src
from src.utils import TCRFileManager

# =============================================================================
# CONFIGURATION
# =============================================================================
PAD_TOKEN = -2
MIN_DONORS = 10
BATCH_SIZE = 20000

TRAIN_PATH = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/train_val_split/datasetwise/train1.tfrecord'
VAL_PATH = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/train_val_split/datasetwise/val1.tfrecord'

OUTPUT_DIR = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/train_val_split/datasetwise/'
OUTPUT_TRAIN = os.path.join(OUTPUT_DIR, 'public_train1.tfrecord')
OUTPUT_VAL = os.path.join(OUTPUT_DIR, 'public_valid1.tfrecord')


def filter_tfrecord(input_path: str, output_path: str, min_donors: int = 10, 
                    batch_size: int = 20000, pad_token: int = -2):
    """
    Filter TFRecord keeping only TCRs with >= min_donors valid donors.
    Uses batched reading and GPU-accelerated operations.
    """
    print(f"\n{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Filter: >= {min_donors} donors | Batch: {batch_size}")
    print('='*60)

    manager = TCRFileManager(
        tcr_path=input_path,
        batch_size=batch_size,
        shuffle_buffer_size=1,
        pad_token=pad_token
    )
    
    dataset = manager.get_dataset(shuffle=False, drop_remainder=False)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    total = 0
    kept = 0

    with tf.io.TFRecordWriter(output_path) as writer:
        for tcr_seqs, tcr_ids, tcr_donor_ids in tqdm(dataset, desc="Processing"):
            # Vectorized on GPU: count valid donors per TCR
            donor_counts = tf.reduce_sum(tf.cast(tcr_donor_ids != pad_token, tf.int32), axis=1)
            total += donor_counts.shape[0]
            
            # Vectorized filter
            keep_mask = donor_counts >= min_donors
            keep_indices = tf.where(keep_mask)[:, 0]
            
            if keep_indices.shape[0] == 0:
                continue
            
            # Gather filtered records (still on GPU)
            filtered_seqs = tf.gather(tcr_seqs, keep_indices)
            filtered_ids = tf.gather(tcr_ids, keep_indices)
            filtered_donors = tf.gather(tcr_donor_ids, keep_indices)
            
            # Single transfer to CPU for writing
            seqs_np = filtered_seqs.numpy()
            ids_np = filtered_ids.numpy()
            donors_np = filtered_donors.numpy()
            
            num_kept = seqs_np.shape[0]
            kept += num_kept
            
            # Write filtered records
            for i in range(num_kept):
                seq = seqs_np[i]
                seq = seq[seq != pad_token]
                
                donors = donors_np[i]
                donors = donors[donors != pad_token]
                
                serialized = TCRFileManager._tcr_serialize(seq, int(ids_np[i]), donors)
                writer.write(serialized)

    print(f"\nTotal: {total:,} | Kept: {kept:,} | Retention: {100*kept/total:.2f}%")
    return total, kept


if __name__ == "__main__":
    print("="*70)
    print(f"FILTERING TCRs WITH >= {MIN_DONORS} DONORS")
    print("="*70)

    train_total, train_kept = filter_tfrecord(TRAIN_PATH, OUTPUT_TRAIN, MIN_DONORS, BATCH_SIZE, PAD_TOKEN)
    val_total, val_kept = filter_tfrecord(VAL_PATH, OUTPUT_VAL, MIN_DONORS, BATCH_SIZE, PAD_TOKEN)

    print(f"\n{'='*70}")
    print(f"Train: {train_kept:,}/{train_total:,} ({100*train_kept/train_total:.1f}%)")
    print(f"Val:   {val_kept:,}/{val_total:,} ({100*val_kept/val_total:.1f}%)")
    print("✓ Done")