import src
from src.utils import TCRFileManager
import pandas as pd
import numpy as np
import sys
import os
# 20 == gap token, 21 == spacing token

def prepare_lists(input_csv_path):
    '''
    input csv looks like this after being read:
    cdr1,cdr2,cdr2.5,cdr3,tcr_id,sample_id
    ...
    Where ',' is the separator of columns.
    '''
    print(f"Reading CSV from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    print("Initial DataFrame shape:", df.shape)
    print("Columns present:", df.columns.tolist())
    print("First 2 rows:\n", df.head(2).to_string(index=False))  # Print first 2 rows for inspection

    expected_cols = {'cdr1','cdr2','cdr2.5','cdr3','tcr_id','sample_id'}
    assert set(df.columns) == expected_cols, f"Unexpected columns: {df.columns}"
    print("Column names match expected.")

    # Combine cdr columns
    print("Combining CDR columns with ';21;' separator.")
    sequences = df[['cdr1','cdr2','cdr2.5','cdr3']].astype(str).agg(";21;".join, axis=1).tolist()
    print("First combined (string) sequence:", sequences[0])

    # Split combined string into int lists
    sequences = [list(map(int, i.split(';'))) for i in sequences]
    print("First combined sequence as int list:", sequences[0])

    tcr_id = df.tcr_id.astype(int).tolist()
    print(f"First TCR ID: {tcr_id[0]}")

    sample_id = df.sample_id.tolist()
    print(f"First raw sample_id: {sample_id[0]}")

    sample_id = [list(map(int, str(i).split(';'))) for i in sample_id]
    print("First sample_id as int list:", sample_id[0])

    assert len(sequences) == len(tcr_id) == len(sample_id), f'sequences: {len(sequences)} | tcrid: {len(tcr_id)} | sampleid: {len(sample_id)}'
    print("Lengths match: sequences, tcr_id, sample_id")

    return sequences, tcr_id, sample_id

if __name__ == "__main__":
    input_csv_path = sys.argv[1]
    outpath = sys.argv[2]

    print("-----Starting Preparation-----")
    sequences, tcr_id, sample_id = prepare_lists(input_csv_path)
    print("-----Preparation Finished-----")

    print("Instantiating TCRFileManager and writing output...")
    tfr = TCRFileManager(
            tcr_path = outpath, 
            tcr_length = 30,
            batch_size = 32, 
            shuffle_buffer_size = 1000,
            pad_token = -1
        )

    tfr.write_tcr_samples(sequences, tcr_id, sample_id)
    print("Data writing complete.")
