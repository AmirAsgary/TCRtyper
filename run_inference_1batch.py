import keras
import tensorflow as tf
import src
from keras import layers
from src.model_utils import (
    PositionalEncoding, 
    MaskedEmbedding, 
    AttentionLayer, 
    Likelihood, 
    MaskedDense,
    QDense
)
from src.utils import TCRFileManager
from src.visualization_utils import TestModeVisualizer
import numpy as np
import os
import pandas as pd
import json
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
import matplotlib.pyplot as plt



PAD_TOKEN = -2.
MASK_TOKEN = -1.
BATCH_SIZE = 100
TEST_MODE_OUTPUT_DIR = 'output/test_mode_Nov27_3'
MODEL_PATH = 'checkpoints/model_epoch_1.keras'
OUTPUT_PATH = 'output/model_train_Nov28_gammaandq_initializedsmall_Ni_removed_2'
PATIENT_TO_HLA = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/processed/patient_to_hla.csv'
# Data paths
train_path = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/train_val_split/datasetwise/train1.tfrecord'
val_path = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/train_val_split/datasetwise/val1.tfrecord'
patient_id_path = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022_Nov20/patients_index_process.tsv'
STEP_JUMP = [10]
os.makedirs(OUTPUT_PATH, exist_ok=True)

model = keras.saving.load_model(MODEL_PATH)

tcr_manager = TCRFileManager(
    tcr_path=train_path,
    batch_size=BATCH_SIZE,
    tcr_length=70,
    shuffle_buffer_size=100000,
    pad_token=PAD_TOKEN
)

val_manager = TCRFileManager(
    tcr_path=val_path,
    batch_size=BATCH_SIZE,
    tcr_length=70,
    shuffle_buffer_size=1000,
    pad_token=PAD_TOKEN
)
hla_df = pd.read_csv(PATIENT_TO_HLA)
hla = [i for i in hla_df.columns if i != 'donor_id']
hla = [f'{i}_{int(np.sum(hla_df[i]))}' for i in hla]

dataset = tcr_manager.get_dataset(shuffle=True)
for step, data in enumerate(dataset, start=1):
    if step in STEP_JUMP:
        # Unpack and cast data
        tcr_seqs, tcr_ids, tcr_donor_ids = data
        tcr_seqs = tf.cast(tcr_seqs, tf.float32)
        tcr_ids = tf.cast(tcr_ids, tf.int32)
        tcr_donor_ids = tf.cast(tcr_donor_ids, tf.int32)
        
        # Create sequence mask
        tcr_seq_mask = tf.where(tcr_seqs == PAD_TOKEN, PAD_TOKEN, 1.)
        tcr_seq_mask = tf.cast(tcr_seq_mask, tf.float32)
        gamma, q, att_score = model([tcr_seqs, tcr_seq_mask])

 
        plt.figure(figsize=(30,20))
        sns.heatmap(gamma.numpy())
        plt.xlabel('HLAs', size=24)
        plt.ylabel('TCRs', size=24)
        plt.xticks(range(len(hla)), hla, size=3, rotation=45, ha='right')
        plt.savefig(os.path.join(OUTPUT_PATH, f'tcr_hla_heatmap_{step}.png'), dpi=600)
        plt.close()

        plt.figure(figsize=(10,7))
        sns.heatmap(q.numpy())
        plt.xlabel('Probs', size=24)
        plt.ylabel('TCRs', size=24)
        plt.savefig(os.path.join(OUTPUT_PATH, f'tcr_q_{step}.png'), dpi=600)
        plt.close()

