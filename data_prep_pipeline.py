import pandas as pd
import numpy as np
import os
import src
from src import utils
import json

output_dir = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022'
relpath_input = 'data/dataproj/export_train_dataset_2025_11_17/'
mhc_maximum_num = 358
mhc_num_allele_thr = (10, 20)
tcr_tsv_path_col = 'relpath_tcr'
mhc_arr_path_col = 'relpath_mask'
cdr1_col = 'cdr1aa_gapped'
cdr2_col = 'cdr2aa_gapped'
cdr2_5_col = 'cdr2.5aa_gapped'
cdr3_col = 'cdr3aa'
sample_id = 'sample_id'
add_rel_path = True
assign_numeric_ids = True
assign_cdr_columns = True


######################################
os.makedirs(output_dir, exist_ok=True)

patient_df_path = os.path.join(relpath_input, 'patients_index.tsv')
patient_df = pd.read_csv(patient_df_path, sep='\t')

if add_rel_path:
    cols = [tcr_tsv_path_col, mhc_arr_path_col]
    patient_df[cols] = patient_df[cols].astype(str).radd(relpath_input)

if assign_numeric_ids:
    patient_df['sample_id_str'] = patient_df[sample_id]
    patient_df[sample_id] = pd.factorize(patient_df['sample_id_str'])[0]
    

if assign_cdr_columns:
    patient_df[cdr1_col] = len(patient_df)*[cdr1_col]
    patient_df[cdr2_col] = len(patient_df)*[cdr2_col]
    patient_df[cdr2_5_col] = len(patient_df)*[cdr2_5_col]
    patient_df[cdr3_col] = len(patient_df)*[cdr3_col]

patient_df.to_csv(os.path.join(output_dir, 'patients_index_process.tsv'), sep='\t', index=False)

# Use subset for testing (first 3 rows)
patient_df = patient_df.iloc[:, :]

# Initialize ReadAndPreprocess class
rap = utils.ReadAndPreprocess(
    df=patient_df,
    csv_output_path=os.path.join(output_dir, 'processed'),
    pad_token=-1,
    tcr_tsv_path=tcr_tsv_path_col,
    mhc_arr_path=mhc_arr_path_col,
    cdr3_col=cdr3_col,
    cdr1_col=cdr1_col,
    cdr2_col=cdr2_col,
    cdr2_5_col=cdr2_5_col,
    id_col=sample_id,
    mhc_maximum_num=mhc_maximum_num,
    mhc_num_allele_thr=mhc_num_allele_thr
)

# Run processing pipeline with sequence mapping for all CDR regions
rap.call(sequence_mapping=True, cdr2_sequence=True, cdr2_5_sequence=True, cdr1_sequence=True)