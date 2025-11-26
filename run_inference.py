import keras
import tensorflow as tf
import src
from keras import layers
from src.model_utils import (PositionalEncoding, MaskedEmbedding, AttentionLayer, Likelihood, MaskedDense)
from src.utils import TCRFileManager
import numpy as np
import os
import pandas as pd
import json

PAD_TOKEN = -2.
MASK_TOKEN = -1.
BATCH_SIZE = 2000
EPOCH = 15

val_path = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/train_val_split/datasetwise/val1.tfrecord'
patient_id_path = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/patients_index_process.tsv'
model_path = 'checkpoints/model_epoch_15.keras'
mhc_path = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/processed/donor_mhc.npz'
output_file = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/results/val1.csv'
hla_to_id = 'data/dataproj_old/hla_to_id.json'
#### Load donor MHC data
print("\nLoading donor MHC data...")
patient_tsv = pd.read_csv(patient_id_path, sep='\t')
max_num_patients = np.max(patient_tsv.sample_id.tolist())
mhc_file = np.load(mhc_path)
donor_mhc = mhc_file['array']
print(f"Donor shape MHC Before adding removed patients: {donor_mhc.shape}")
max_donor_mhc = np.zeros(shape=(max_num_patients+1, donor_mhc.shape[1]))
patient_ids = np.array([int(i) for i in mhc_file['patient_id']])
j = 0
for i in patient_ids:
    max_donor_mhc[i] = donor_mhc[j]
    j += 1
#load hla_to_id
with open(hla_to_id, 'r') as f:
    hlatoid = json.load(f)
hlatoid = [i for i in hlatoid.keys()]

import pyarrow as pa
import pyarrow.parquet as pq

# Load model
model = keras.saving.load_model(
    model_path, safe_mode=True,
    custom_objects={'MaskedEmbedding': MaskedEmbedding}
)

# Dataset
val_manager = TCRFileManager(
    tcr_path=val_path,
    batch_size=BATCH_SIZE,
    tcr_length=70,
    shuffle_buffer_size=1000,
    pad_token=PAD_TOKEN
)
val_dataset = val_manager.get_dataset(shuffle=False)

A = len(hlatoid)
columns = hlatoid + ["donor_id", "tcr_id"]

output_parquet = output_file.replace(".csv", ".parquet")
parquet_writer = None

for val_step, val_data in enumerate(val_dataset, start=1):
    tcr_seqs, tcr_ids, tcr_donor_ids = val_data

    # Cast to float32/int32 for GPU
    tcr_seqs = tf.cast(tcr_seqs, tf.float32)
    tcr_ids = tf.cast(tcr_ids, tf.int32)
    tcr_donor_ids = tf.cast(tcr_donor_ids, tf.int32)

    # Mask padded TCR positions
    tcr_seq_mask = tf.where(tcr_seqs == PAD_TOKEN, PAD_TOKEN, 1.)

    # Model forward pass (on GPU)
    gamma, q = model([tcr_seqs, tcr_seq_mask], training=False)  # gamma: (B, A)

    # -------------------------------
    # Vectorized donor handling on GPU
    # -------------------------------

    valid_mask = tcr_donor_ids != tf.constant(int(PAD_TOKEN), dtype=tf.int32)  # (B, D)
    counts = tf.reduce_sum(tf.cast(valid_mask, tf.int32), axis=1)  # (B,)

    # Repeat gamma rows according to valid donors
    repeat_idx = tf.repeat(tf.range(tf.shape(gamma)[0]), counts)
    gamma_expanded = tf.gather(gamma, repeat_idx)  # (total_valid, A)

    # Flatten donor and tcr_id, select valid ones
    flat_donor_ids = tf.reshape(tcr_donor_ids, [-1])
    flat_tcr_ids = tf.repeat(tcr_ids, valid_mask.shape[1])
    flat_mask = tf.reshape(valid_mask, [-1])

    valid_donors = tf.boolean_mask(flat_donor_ids, flat_mask)
    valid_tcr_ids = tf.boolean_mask(flat_tcr_ids, flat_mask)

    # Concatenate final array: (total_valid, A + 2)
    final_tensor = tf.concat(
        [
            gamma_expanded,
            tf.cast(valid_donors[:, None], tf.float32),
            tf.cast(valid_tcr_ids[:, None], tf.float32)
        ],
        axis=1
    )

    # Move to CPU once per batch
    np_data = final_tensor.numpy()

    # -------------------------------
    # Parquet batch write
    # -------------------------------
    table = pa.Table.from_arrays(
        [np_data[:, i] for i in range(np_data.shape[1])],
        names=columns
    )

    if parquet_writer is None:
        parquet_writer = pq.ParquetWriter(output_parquet, table.schema)

    parquet_writer.write_table(table)

    
    print(f"Processed {val_step} batches")


# Close Parquet writer
if parquet_writer:
    parquet_writer.close()

print(f"Saved Parquet results to {output_parquet}")



#gamma, q, att_score
# /user/a.hajialiasgarynaj01/u14286/.project/dir.project/autotcr/science_new_paper_3_datasets/export_v2_no_hla/export_train_dataset