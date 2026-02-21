from __future__ import annotations
import keras
import tensorflow as tf
import numpy as np
from typing import Dict, Iterator, Optional, Sequence
from src.constants import AMINO_ACID_IDX, AMINO_ACIDS, PHYSICHE_STOCKHOLM, PHYSICHE_STOCKHOLM_AA_to_IDX, PHYSICHE_STOCKHOLM_IDX_to_ENCODE
import pandas as pd
import os
import subprocess
from pathlib import Path
import shutil
import re
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence
from dataclasses import dataclass, field
import json
from tensorflow.keras import layers




aa_order = list("ARNDCEQGHILKMFPSTWYVX")
AMINO_ACIDS = ''.join(aa_order)
AMINO_ACID_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
AMINO_ACID_IDX['.'] = AMINO_ACID_IDX['X']
AMINO_ACID_IDX['-'] = AMINO_ACID_IDX['X']
encode_table = str.maketrans({aa: chr(i) for aa, i in AMINO_ACID_IDX.items()})
decode_table = str.maketrans({chr(i): aa for aa, i in AMINO_ACID_IDX.items()})

def convert_protein_sequences_fast(data_list, convert_to_aa=True, convert_to_ind=False):
    """
    Extremely fast conversion between amino acid sequences and index-encoded strings.
    Works for both directions.
    """
    converted = []
    for item in data_list:
        item = item.strip()
        if convert_to_aa:
            if all(ch.isdigit() or ch == ';' for ch in item):
                # Index → sequence
                indices = map(int, item.split(';'))
                seq = ''.join(aa_order[i] for i in indices)
                converted.append(seq)
            else:
                converted.append(item)
        elif convert_to_ind:
            # Sequence → index form using translation
            encoded = item.translate(encode_table)
            indices = [str(ord(c)) for c in encoded]
            converted.append(';'.join(indices))
            
    return converted

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """
    Helper function to create an Int64List feature.
    
    Args:
        value: Can be a single int, list of ints, or numpy array of ints.
    
    Returns:
        tf.train.Feature: Feature containing Int64List.
    """
    if isinstance(value, (int, np.integer)):
        value = [value]
    elif isinstance(value, np.ndarray):
        value = value.flatten().tolist()
    elif isinstance(value, list):
        pass
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    




class ReadAndPreprocess:
    """
    Read and preprocess TCR (T-cell receptor) and MHC (Major Histocompatibility Complex) data.
    
    This class handles loading TCR sequences and MHC arrays from multiple patients,
    validates the data, maps sequences to numeric representations, and aggregates
    results for downstream analysis.
    """
    
    def __init__(self, df: Union[str, pd.DataFrame],
                 csv_output_path: str,
                 pad_token: int = -2,
                 tcr_tsv_path: str = 'tcr_tsv_path',
                 mhc_arr_path: str = 'mhc_arr_path',
                 cdr3_col: str = 'cdr3_col',
                 cdr1_col: str = 'cdr1_col',
                 cdr2_col: str = 'cdr2_col',
                 cdr2_5_col: str = 'cdr2.5_col',
                 id_col: str = 'id',
                 mhc_maximum_num: int = 200,
                 mhc_num_allele_thr: Union[List, Tuple] = (10, 12)):
        """
        Initialize the preprocessor with data paths and configuration.
        
        Args:
            df: Path to CSV/TSV file or loaded DataFrame containing patient data.
                Must have columns specified by tcr_tsv_path and mhc_arr_path parameters.
            csv_output_path: Directory path where output files will be saved.
            pad_token: Integer token used for padding sequences to the same length (default: -1).
            tcr_tsv_path: Column name in df containing paths to TCR data files (default: 'tcr_tsv_path').
            mhc_arr_path: Column name in df containing paths to MHC array files (default: 'mhc_arr_path').
            cdr3_col: Column name in df specifying the CDR3 column name in TCR files (default: 'cdr3_col').
            cdr1_col: Column name in df specifying the CDR1 column name in TCR files (default: 'cdr1_col').
            cdr2_col: Column name in df specifying the CDR2 column name in TCR files (default: 'cdr2_col').
            cdr2_5_col: Column name in df specifying the CDR2.5 column name in TCR files (default: 'cdr2.5_col').
            id_col: Column name for patient IDs in df (default: 'id'). If not present, IDs are auto-assigned.
            mhc_maximum_num: Expected number of MHC alleles in arrays (default: 200).
            mhc_num_allele_thr: Tuple of (min, max) number of alleles that should be present (default: (10, 12)).
        """
        # Load dataframe if path is provided
        if isinstance(df, str):
            try:
                df = pd.read_csv(df)
            except:
                df = pd.read_csv(df, sep='\t')
        # Validate required columns
        assert tcr_tsv_path in df.columns and mhc_arr_path in df.columns, (
            f"DataFrame must have {[tcr_tsv_path, mhc_arr_path]} columns. Found: {df.columns.tolist()}"
        )
        # Handle ID column
        if id_col in df.columns:
            assert len(df[id_col].dropna()) == len(df), (
                "ID column provided but contains missing values."
            )
            # Ensure IDs are strings for consistency
            df[id_col] = df[id_col].astype(str)
        else:
            df[id_col] = [str(i) for i in range(len(df))]
        # Handle CDR column name specifications
        for cdr_num, cdr_col in [(1, cdr1_col), (2, cdr2_col), (2.5, cdr2_5_col), (3, cdr3_col)]:
            if cdr_col in df.columns:
                assert len(df[cdr_col].dropna()) == len(df), (
                    f"{cdr_col} column provided but contains missing values."
                )
            else:
                # Default CDR column names
                df[cdr_col] = f'cdr{cdr_num}'
        self.df = df
        self.csv_output_path = csv_output_path
        self.pad_token = pad_token
        self.tcr_tsv_path = tcr_tsv_path
        self.mhc_arr_path = mhc_arr_path
        self.cdr3_col = cdr3_col
        self.cdr1_col = cdr1_col
        self.cdr2_col = cdr2_col
        self.cdr2_5_col = cdr2_5_col
        self.id_col = id_col
        self.mhc_maximum_num = mhc_maximum_num
        self.mhc_num_allele_thr = mhc_num_allele_thr
        gap_value = AMINO_ACID_IDX.get('-')
        self.get_idx = lambda aa: AMINO_ACID_IDX.get(aa, gap_value)
    
    def read_tcr_df(self, path: str, cdr3_col: str = 'cdr3', 
                    cdr2_col: str = 'cdr2', cdr2_5_col: str = 'cdr2.5', 
                    cdr1_col: str = 'cdr1') -> pd.DataFrame:
        """
        Read TCR (T-cell receptor) data from a CSV or TSV file.
        
        Args:
            path: Path to the TCR data file.
            cdr3_col: Name of the CDR3 sequence column (default: 'cdr3').
            cdr2_col: Name of the CDR2 sequence column (default: 'cdr2').
            cdr2_5_col: Name of the CDR2.5 sequence column (default: 'cdr2.5').
            cdr1_col: Name of the CDR1 sequence column (default: 'cdr1').
            
        Returns:
            DataFrame containing TCR sequences with rows containing missing CDR3 values removed.
            
        Raises:
            AssertionError: If required columns are missing or dataframe is empty.
        """
        tcr_df = pd.read_csv(path)
        if not all(col in tcr_df.columns for col in [cdr3_col, cdr2_col, cdr2_5_col, cdr1_col]):
            tcr_df = pd.read_csv(path, sep='\t')
        
        # Validate required columns exist
        assert all(col in tcr_df.columns for col in [cdr3_col, cdr2_col, cdr2_5_col, cdr1_col]), (
            f"Required columns {[cdr3_col, cdr2_col, cdr2_5_col, cdr1_col]} not found. "
            f"Available columns: {tcr_df.columns.tolist()}"
        )
        
        assert len(tcr_df) > 0, f"Empty dataframe loaded from {path}"
        
        # Filter out rows with missing CDR3 values (CDR3 is mandatory)
        tcr_df = tcr_df[~tcr_df[cdr3_col].isna()]
        
        return tcr_df
    
    def read_mhc_arr(self, path: str, easy_mode: bool = True) -> np.ndarray:
        """
        Read and validate MHC (Major Histocompatibility Complex) array from .npy file.
        Args:
            path: Path to the .npy file containing MHC array.
            easy_mode: If True, excludes patients where their HLAs are less thatn defined limit by self.mhc_num_allele_thr.
        Returns:
            Validated MHC array as numpy ndarray.
        Raises:
            AssertionError: If array size, values, or allele count don't meet requirements.
        """
        arr = np.load(path)
        
        # Validate array shape
        assert arr.shape[0] == self.mhc_maximum_num, (
            f"Expected array size of {self.mhc_maximum_num}, "
            f"but received array of shape {arr.shape}"
            f"Path: {path}"
        )
        # Validate array contains only binary values
        assert np.isin(arr, [0, 1]).all(), (
            f"Expected all elements to be [0, 1], found unique values: {np.unique(arr)}"
        )
        # Validate number of present alleles (1s)
        allele_sum = np.sum(arr)
        if easy_mode:
            if not self.mhc_num_allele_thr[0] <= allele_sum <= self.mhc_num_allele_thr[1]:
                return arr, False
        else:
            assert self.mhc_num_allele_thr[0] <= allele_sum <= self.mhc_num_allele_thr[1], (
                f"Expected between {self.mhc_num_allele_thr[0]} and {self.mhc_num_allele_thr[1]} "
                f"alleles to be present, found {allele_sum}"
                f"File {path}"
            )
        return arr, True
    
    def assign_patient_ids_to_tcrs(self, tcr_data: pd.DataFrame, 
                                   separator: str = ';', 
                                   cdr_columns: List[str] = None) -> pd.DataFrame:
        """
        Group TCR sequences by CDR regions and aggregate patient IDs.
        For identical TCR sequences from different patients, this combines their IDs
        into a single semicolon-separated string.
        Args:
            tcr_data: DataFrame containing TCR sequences and patient IDs.
            separator: String to join multiple patient IDs (default: ';').
            cdr_columns: List of CDR column names to group by (default: ['cdr1', 'cdr2', 'cdr2.5', 'cdr3']).
            
        Returns:
            DataFrame with unique TCR sequences and aggregated patient IDs.
        """
        if cdr_columns is None:
            cdr_columns = ['cdr1', 'cdr2', 'cdr2.5', 'cdr3']
        
        result = tcr_data.groupby(cdr_columns, as_index=False, sort=False).agg({
            self.id_col: lambda x: separator.join(sorted(set(map(str, x))))
        })
        return result
    
    def map_df_to_num(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Map amino acid sequences to numeric representations.
        Converts amino acid sequences to semicolon-separated numeric indices
        based on AMINO_ACID_IDX mapping. Unknown amino acids are mapped to '?'.
        Args:
            df: DataFrame containing sequence columns to map.
            columns: List of column names containing amino acid sequences.
        Returns:
            DataFrame with new 'mapped_{column}' columns added.
        """
        df = df.dropna()
        for col in columns:
            df[col] = [
                ';'.join(str(self.get_idx(aa)) for aa in seq) 
                for seq in df[str(col)].values
            ]
        return df
    
    def call(self, sequence_mapping: bool = False, 
             cdr2_sequence: bool = False, 
             cdr2_5_sequence: bool = False,
             cdr1_sequence: bool = False) -> None:
        """
        Main processing pipeline: read, validate, map, and save TCR and MHC data.
        
        Iterates through all patients in the input dataframe, loads their TCR and MHC data,
        optionally maps sequences to numeric representations, and saves three output files:
        1. tcr_seq.csv: All TCR sequences with patient IDs
        2. tcr_donor_ids.csv: Unique TCR sequences with aggregated patient IDs
        3. donor_mhc.npz: MHC arrays and patient IDs
        
        Args:
            sequence_mapping: If True, map CDR3 sequences to numeric representation (default: False).
            cdr2_sequence: If True, also map CDR2 sequences (requires sequence_mapping=True).
            cdr2_5_sequence: If True, also map CDR2.5 sequences (requires sequence_mapping=True).
            cdr1_sequence: If True, also map CDR1 sequences (requires sequence_mapping=True).
        """
        # Create output directory
        os.makedirs(self.csv_output_path, exist_ok=True)
        
        # Define output paths
        tcr_seq_path = os.path.join(self.csv_output_path, 'tcr_seq.csv')
        tcr_donor_ids_path = os.path.join(self.csv_output_path, 'tcr_donor_ids.csv')
        donor_mhc_arr_path = os.path.join(self.csv_output_path, 'donor_mhc.npz')
        patients_processed_path = os.path.join(self.csv_output_path, 'patients_remained_index.csv')
        petients_faulty_path = os.path.join(self.csv_output_path, 'petients_faulty_removed.csv')
        
        # Collect all data
        mhc_arrays = []
        patient_ids = []
        all_tcr_dfs = []
        #TODO add the below lists
        patients_processed = [] # updates self.df and keeps only patients that are passing all the tests.
        patients_faulty = [] 
        # Use itertuples for better performance than iterrows
        for row in self.df.itertuples(index=False):
            patient_id = getattr(row, self.id_col)
            cdr3_col = getattr(row, self.cdr3_col, self.cdr3_col)
            cdr2_col = getattr(row, self.cdr2_col, self.cdr2_col)
            cdr2_5_col = getattr(row, self.cdr2_5_col, self.cdr2_5_col)
            cdr1_col = getattr(row, self.cdr1_col, self.cdr1_col)
            tcr_path = getattr(row, self.tcr_tsv_path)
            mhc_path = getattr(row, self.mhc_arr_path)
            
            try:
                # Read TCR data
                tcr_df = self.read_tcr_df(tcr_path, cdr3_col, cdr2_col, cdr2_5_col, cdr1_col)
                
                # Read MHC array
                mhc_arr, mhc_state = self.read_mhc_arr(mhc_path)
                if mhc_state == False:
                    patients_faulty.append({**row._asdict(), 'error': 'mhc_state False', 'error_path': mhc_path})
                    continue
                    
                # Add patient ID column and rename CDR columns to standard names
                tcr_df[self.id_col] = patient_id
                tcr_df = tcr_df.rename(columns={
                    cdr3_col: 'cdr3', 
                    cdr2_col: 'cdr2',
                    cdr2_5_col: 'cdr2.5',
                    cdr1_col: 'cdr1'
                })
                # sequence mapping
                if sequence_mapping:
                    cols_to_map = ['cdr3']
                    if cdr1_sequence:
                        cols_to_map.append('cdr1')
                    if cdr2_sequence:
                        cols_to_map.append('cdr2')
                    if cdr2_5_sequence:
                        cols_to_map.append('cdr2.5')
                    
                    self.map_df_to_num(tcr_df, cols_to_map)
                
                patients_processed.append(row._asdict())
                mhc_arrays.append(mhc_arr)
                patient_ids.append(patient_id)
                all_tcr_dfs.append(tcr_df)
                
            except Exception as e:
                patients_faulty.append({
                    **row._asdict(), 
                    'error': str(e), 
                    'error_type': type(e).__name__,
                    'tcr_path': tcr_path,
                    'mhc_path': mhc_path
                })
                continue
        
        # Concatenate all TCR dataframes and save (more efficient than appending)
        combined_tcr_df = pd.concat(all_tcr_dfs, ignore_index=True)
        #combined_tcr_df.to_csv(tcr_seq_path, index=False)
        
        # Stack MHC arrays and save with patient IDs
        mhc_array_stacked = np.stack(mhc_arrays, axis=0)
        patient_ids_array = np.array(patient_ids)
        np.savez(donor_mhc_arr_path, array=mhc_array_stacked, patient_id=patient_ids_array)
        
        # Aggregate TCR sequences by patient IDs
        tcr_aggregated = self.assign_patient_ids_to_tcrs(combined_tcr_df)
        tcr_aggregated['tcr_id'] = range(len(tcr_aggregated))
        tcr_aggregated.to_csv(tcr_donor_ids_path, index=False)
        
        # merge and assign tcr ids from tcr_aggregated to combined tcr df.
        # the first would have one row for each tcr and all patient ids are in one column
        # the second would have for each patient id, all tcrs that it has, so they are just compressed and uncompressed forms of each other
        combined_tcr_df = combined_tcr_df.merge(tcr_aggregated[['cdr1', 'cdr2', 'cdr3', 'cdr2.5', 'tcr_id']], on=['cdr1', 'cdr2', 'cdr3', 'cdr2.5'], how='left')
        combined_tcr_df.to_csv(tcr_seq_path, index=False)

        if len(patients_faulty) > 0:
            patients_faulty_df = pd.DataFrame(patients_faulty)
            patients_faulty_df.to_csv(petients_faulty_path, index=False)  
        if len(patients_processed) > 0:
            patients_processed_df = pd.DataFrame(patients_processed)
            patients_processed_df.to_csv(patients_processed_path, index=False)  #
        else: raise ValueError(f'expected remained patients, found none')
        print(f"Processing complete!")
        print(f"  - TCR sequences saved to: {tcr_seq_path}")
        print(f"  - Aggregated TCR-patient IDs saved to: {tcr_donor_ids_path}")
        print(f"  - MHC arrays saved to: {donor_mhc_arr_path}")


        



class TCRFileManager:
    """
    A class to manage TCR (T-cell receptor) datasets stored as TFRecords with variable-length sequences.
    This class supports:
        - Writing variable-length TCR sequences, fixed-length IDs, and variable-length donor IDs to TFRecord files
        - Reading TFRecord files into tf.data.Dataset pipelines with dynamic batch padding
        - Parallel reading, shuffling, batching, and prefetching
        - Distributed dataset support for multi-GPU or multi-worker training
        - Dynamic padding to maximum length within each batch for both TCR sequences and donor IDs
    Attributes:
        tcr_path (Union[str, List[str]]): Path(s) to TFRecord file(s).
        tcr_length (int): Maximum expected length of TCR sequences (for validation).
        batch_size (int): Number of sequences per batch in the dataset.
        shuffle_buffer_size (int): Buffer size for shuffling the dataset.
        pad_token (int): Token value used for padding sequences (default: -1).
    """
    
    def __init__(
        self, 
        tcr_path: Union[str, List[str]], 
        tcr_length: int = 25,
        batch_size: int = 32, 
        shuffle_buffer_size: int = 1000,
        pad_token: int = -2.
    ):
        """
        Initialize the TCRFileManager.
        Args:
            tcr_path: Path to TFRecord file (str) or list of paths for reading.
            tcr_length: Maximum expected length of TCR sequences.
            batch_size: Number of sequences per batch.
            shuffle_buffer_size: Buffer size for dataset shuffling.
            pad_token: Integer token used for padding variable-length sequences.
        """
        self.tcr_path = tcr_path
        self.tcr_length = tcr_length
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pad_token = pad_token
    
    @staticmethod
    def _tcr_serialize(
        tcr_seq: np.ndarray | list, 
        tcr_ids: int, 
        tcr_donor_ids: np.ndarray | list
    ) -> bytes:
        """
        Serialize a single TCR sequence, its ID, and donor IDs into a TFRecord Example.
        This method handles variable-length sequences by storing them as lists without fixed dimensions.
        Args:
            tcr_seq: TCR sequence as 1D array of integers, shape (seq_len,). 
                     Each integer represents an amino acid. Can be variable length.
            tcr_ids: Single integer ID corresponding to the TCR sequence.
            tcr_donor_ids: 1D array of integers representing donor IDs, shape (num_donors,).
                          Can be variable length.
        Returns:
            bytes: Serialized TFRecord Example.
        """
        # Create feature dictionary with variable-length sequences
        tcr_data = {
            "tcr_seq": _int64_feature(tcr_seq),          # Variable-length sequence
            "tcr_ids": _int64_feature(tcr_ids),          # Single integer ID
            "tcr_donor_ids": _int64_feature(tcr_donor_ids)  # Variable-length donor IDs
        }
        # Serialize the example
        tcr_example = tf.train.Example(
            features=tf.train.Features(feature=tcr_data)
        ).SerializeToString()
        return tcr_example

    def write_tcr_samples(
        self, 
        tcr_seq: np.ndarray | list, 
        tcr_ids: np.ndarray | list, 
        tcr_donor_ids: np.ndarray | list
    ):
        """
        Write multiple TCR sequences, IDs, and donor IDs to a TFRecord file.
        Each TCR sequence and its corresponding donor IDs can have variable lengths.
        The method removes padding tokens before writing to save space.
        Args:
            tcr_seq: Array of TCR sequences, shape (num_seqs, seq).
                    Each sequence is padded with pad_token to max_seq_len.
                    If lists provided, no padding required.
            tcr_ids: Array of integer IDs, shape (num_seqs,).
            tcr_donor_ids: Array or list of variable-length donor ID arrays.
                          Can be a 2D padded array (num_seqs, donors) or list of arrays.
        Raises:
            AssertionError: If input shapes are inconsistent.
            ValueError: If tcr_path is not a single file path string.
        """
        # Validate that tcr_path is a string for writing
        if not isinstance(self.tcr_path, str):
            raise ValueError(
                f'To write a file, tcr_path must be a file path string, not a list. '
                f'Found: {type(self.tcr_path)}'
            )
        
        # Validate input shapes
        num_seqs = len(tcr_seq)
        if isinstance(tcr_seq, np.ndarray):
            assert tcr_seq.ndim == 2, f"tcr_seq must be 2D array, got shape {tcr_seq.shape}"
        elif isinstance(tcr_seq, list):
            assert isinstance(tcr_seq[0], list), f"tcr_seq must be a list of lists, got tcr_seq[0]--> {tcr_seq[0]}"
        assert len(tcr_ids) == num_seqs, (
            f"Shape mismatch: tcr_seq has {num_seqs} sequences "
            f"but tcr_ids has {len(tcr_ids)} IDs"
        )
        
        # Handle tcr_donor_ids as either 2D array or list of arrays
        if isinstance(tcr_donor_ids, np.ndarray):
            if tcr_donor_ids.ndim == 1:
                # Single sequence case - reshape
                tcr_donor_ids = tcr_donor_ids.reshape(1, -1)
            assert tcr_donor_ids.shape[0] == num_seqs, (
                f"Shape mismatch: tcr_seq has {num_seqs} sequences "
                f"but tcr_donor_ids has {tcr_donor_ids.shape[0]} entries"
            )
        else:
            assert len(tcr_donor_ids) == num_seqs, (
                f"Shape mismatch: tcr_seq has {num_seqs} sequences "
                f"but tcr_donor_ids has {len(tcr_donor_ids)} entries"
            )
        
        # Write TFRecord file
        print(f"Writing {num_seqs} TCR samples to {self.tcr_path}...")
        
        with tf.io.TFRecordWriter(self.tcr_path) as writer:
            for i in tqdm(range(num_seqs), desc="Writing TCR samples", unit="seq"):
                # Remove padding from tcr_seq to store variable-length sequence
                #### REMOVED
                #seq = tcr_seq[i]
                #seq = np.array(seq)
                # Find actual sequence length (before padding)
                #actual_seq = seq[seq != self.pad_token]
                ####
                actual_seq = tcr_seq[i]
                actual_donor_ids = tcr_donor_ids[i]
                # Serialize and write
                serialized = self._tcr_serialize(
                    actual_seq, 
                    tcr_ids[i], 
                    actual_donor_ids
                )
                writer.write(serialized)
        
        print(f"Successfully wrote {num_seqs} samples to {self.tcr_path}")
    
    def _tcr_parse(self, example_proto: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Parse a single TFRecord Example into TCR sequence, ID, and donor IDs.

        This method handles variable-length features using VarLenFeature, which allows
        each sequence and donor ID list to have different lengths.
        
        Args:
            example_proto: Serialized TFRecord Example.

        Returns:
            tuple: (tcr_seq, tcr_ids, tcr_donor_ids)
                - tcr_seq: 1D tensor of variable length containing amino acid integers
                - tcr_ids: Scalar tensor containing the TCR ID
                - tcr_donor_ids: 1D tensor of variable length containing donor IDs
        """
        # Define feature description with variable-length features
        feature_description = {
            "tcr_seq": tf.io.VarLenFeature(tf.int64),       # Variable-length sequence
            "tcr_ids": tf.io.FixedLenFeature([], tf.int64), # Single integer
            "tcr_donor_ids": tf.io.VarLenFeature(tf.int64), # Variable-length donor IDs
        }
        
        # Parse the example
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        
        # Convert VarLenFeature (SparseTensor) to dense tensor
        tcr_seq = tf.sparse.to_dense(parsed["tcr_seq"])
        tcr_donor_ids = tf.sparse.to_dense(parsed["tcr_donor_ids"])
        
        # Cast to appropriate types
        tcr_seq = tf.cast(tcr_seq, tf.int32)
        tcr_ids = tf.cast(parsed["tcr_ids"], tf.int32)
        tcr_donor_ids = tf.cast(tcr_donor_ids, tf.int32)
        
        return tcr_seq, tcr_ids, tcr_donor_ids
    
    def _pad_batch(
        self, 
        tcr_seq: tf.Tensor, 
        tcr_ids: tf.Tensor, 
        tcr_donor_ids: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Pad variable-length sequences in a batch to the maximum length in that batch.
        
        This function is applied after batching to ensure all sequences in a batch
        have the same length for efficient tensor operations.
        
        Args:
            tcr_seq: Ragged tensor of shape (batch_size, None) - variable-length sequences
            tcr_ids: Tensor of shape (batch_size,) - fixed-length IDs
            tcr_donor_ids: Ragged tensor of shape (batch_size, None) - variable-length donor IDs
        
        Returns:
            tuple: (padded_tcr_seq, tcr_ids, padded_tcr_donor_ids)
                - padded_tcr_seq: Tensor of shape (batch_size, max_seq_len_in_batch)
                - tcr_ids: Unchanged tensor of shape (batch_size,)
                - padded_tcr_donor_ids: Tensor of shape (batch_size, max_donors_in_batch)
        """
        # Pad tcr_seq to maximum length in the batch
        padded_tcr_seq = tcr_seq.to_tensor(default_value=int(self.pad_token))
        
        # Pad tcr_donor_ids to maximum length in the batch
        padded_tcr_donor_ids = tcr_donor_ids.to_tensor(default_value=int(self.pad_token))
        
        return padded_tcr_seq, tcr_ids, padded_tcr_donor_ids
    
    def get_dataset(
        self, 
        shuffle: bool = True, 
        num_parallel_reads: int = tf.data.AUTOTUNE,
        prefetch_size: int = tf.data.AUTOTUNE,
        drop_remainder: bool = False
    ) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset from TFRecord file(s) with dynamic batch padding.
        
        This method creates an efficient data pipeline that:
        1. Reads TFRecord files in parallel
        2. Parses variable-length features
        3. Optionally shuffles the data
        4. Batches examples into ragged batches
        5. Pads each batch to its maximum sequence length
        6. Prefetches batches for optimal performance
        
        Args:
            shuffle: Whether to shuffle the dataset.
            num_parallel_reads: Number of files to read in parallel (default: AUTOTUNE).
            prefetch_size: Number of batches to prefetch (default: AUTOTUNE).
            drop_remainder: Whether to drop the last incomplete batch.
        
        Returns:
            tf.data.Dataset: Dataset yielding batches of (tcr_seq, tcr_ids, tcr_donor_ids)
                - tcr_seq: shape (batch_size, max_seq_len_in_batch)
                - tcr_ids: shape (batch_size,)
                - tcr_donor_ids: shape (batch_size, max_donors_in_batch)
        """
        # Handle single file or list of files
        if isinstance(self.tcr_path, str):
            file_paths = [self.tcr_path]
        else:
            file_paths = self.tcr_path
        
        # Create dataset from TFRecord files
        dataset = tf.data.TFRecordDataset(
            file_paths, 
            num_parallel_reads=num_parallel_reads
        )
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)

        # Parse examples
        dataset = dataset.map(
            self._tcr_parse, 
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Batch with ragged tensors (variable-length sequences)
        # ragged_batch creates batches where sequences can have different lengths
        dataset = dataset.ragged_batch(
            self.batch_size, 
            drop_remainder=drop_remainder
        )
        
        # Pad each batch to its maximum length
        dataset = dataset.map(
            self._pad_batch,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Prefetch for performance
        dataset = dataset.prefetch(prefetch_size)
        
        return dataset
    
    def get_distributed_dataset(
        self,
        strategy: tf.distribute.Strategy,
        shuffle: bool = True,
        num_parallel_reads: int = tf.data.AUTOTUNE,
        prefetch_size: int = tf.data.AUTOTUNE,
        drop_remainder: bool = True
    ) -> tf.distribute.DistributedDataset:
        """
        Create a distributed dataset for multi-GPU or multi-worker training.
        
        This method creates a dataset that is sharded across multiple devices/workers
        according to the provided distribution strategy.
        
        Args:
            strategy: TensorFlow distribution strategy (e.g., MirroredStrategy, TPUStrategy).
            shuffle: Whether to shuffle the dataset.
            num_parallel_reads: Number of files to read in parallel.
            prefetch_size: Number of batches to prefetch.
            drop_remainder: Whether to drop the last incomplete batch (recommended for distributed training).
        
        Returns:
            tf.distribute.DistributedDataset: Distributed dataset for training.
        """
        # Get regular dataset
        dataset = self.get_dataset(
            shuffle=shuffle,
            num_parallel_reads=num_parallel_reads,
            prefetch_size=prefetch_size,
            drop_remainder=drop_remainder
        )
        
        # Distribute dataset across devices/workers
        distributed_dataset = strategy.experimental_distribute_dataset(dataset)
        
        return distributed_dataset

class CrossValGen():
    """
    Generates cross-validation splits for TCR data, ensuring patient-level separation.
    Handles both dataset-wise and random patient-wise split strategies.
    """
    
    def __init__(self, patient_index_path, tcr_donor_id_path, output_path, 
                 datasetwise=True, random_patient_wise=True, K=5):
        """
        Args:
            patient_index_path: Path to patient metadata (sample_id, dataset)
            tcr_donor_id_path: Path to TCR data (cdr1-3, cdr2.5, sample_id, tcr_id)
            output_path: Directory for saving train/test splits
            datasetwise: Generate dataset-wise splits (leave-one-dataset-out)
            random_patient_wise: Generate random K-fold patient-wise splits
            K: Number of folds for random patient-wise CV
        """
        self.patient_index_path = patient_index_path
        self.tcr_donor_id_path = tcr_donor_id_path
        self.output_path = output_path
        self.datasetwise = datasetwise
        self.random_patient_wise = random_patient_wise
        self.K = K
    
    def call(self):
        """Execute cross-validation split generation."""
        print(f"Loading patient index from: {self.patient_index_path}")
        patient_index = pd.read_csv(self.patient_index_path) if self.patient_index_path.endswith('.csv') else pd.read_csv(self.patient_index_path, sep='\t')
        print(f"  → Loaded {len(patient_index)} patients")
        
        print(f"Loading TCR data from: {self.tcr_donor_id_path}")
        tcr_donor_id = pd.read_csv(self.tcr_donor_id_path) if self.tcr_donor_id_path.endswith('.csv') else pd.read_csv(self.tcr_donor_id_path, sep='\t')
        print(f"  → Loaded {len(tcr_donor_id)} TCR sequences\n")
        
        output = []
        if self.datasetwise:
            print("=" * 60)
            print("DATASET-WISE CROSS-VALIDATION")
            print("=" * 60)
            datasetwise_dict = self.DatasetWise(patient_index, tcr_donor_id)
            output.append(datasetwise_dict)
            
        if self.random_patient_wise:
            print("\n" + "=" * 60)
            print("RANDOM PATIENT-WISE CROSS-VALIDATION")
            print("=" * 60)
            patientwise_dict = self.RandomPatientWise(patient_index, tcr_donor_id)
            output.append(patientwise_dict)
            
        assert len(output) != 0
        print("\n" + "=" * 60)
        print("✓ Cross-validation generation completed successfully")
        print("=" * 60)
        return output
            
    def DatasetWise(self, patient_index, tcr_donor_id):
        """Leave-one-dataset-out cross-validation."""
        datasets = np.unique(patient_index.dataset.tolist())
        datasets = [str(d) for d in datasets]
        print(f"Found {len(datasets)} datasets: {datasets}\n")
        
        DICT = {}
        outdir = os.path.join(self.output_path, 'datasetwise')
        os.makedirs(outdir, exist_ok=True)
        
        for number, dataset in enumerate(datasets):
            print(f"Fold {number+1}/{len(datasets)}: Validation on dataset '{dataset}'")
            pid = patient_index[patient_index['dataset'] == dataset]['sample_id'].tolist()
            print(f"  → {len(pid)} validation patients")
            
            train_tcr_donor, validation_df, included, excluded = self.iterate_and_create(pid, tcr_donor_id)
            
            print(f"  → Train: {len(train_tcr_donor)} TCRs | Test: {len(validation_df)} TCRs")
            print(f"  → Public: {len(included)} TCRs | Private: {len(excluded)} TCRs\n")
            
            DICT[f'{number}'] = {
                'patient_ids': pid, 
                'included_tcr_ids': included,
                'excluded_tcr_ids': excluded,
                'validation_set': [str(dataset)], 
                'training_set': [i for i in datasets if i != dataset]
            }
            
            assert len(train_tcr_donor) != 0
            assert len(validation_df) != 0
            
            train_tcr_donor.to_csv(os.path.join(outdir, f'train{number}.csv'), index=False)
            validation_df.to_csv(os.path.join(outdir, f'test{number}.csv'), index=False)
            
        with open(os.path.join(outdir, 'info.json'), 'w') as f:
            json.dump(DICT, f)
        print(f"✓ Saved dataset-wise splits to: {outdir}")
        return DICT

    def RandomPatientWise(self, patient_index, tcr_donor_id):
        """Random K-fold patient-wise cross-validation."""
        sampleids = np.unique(patient_index.sample_id.tolist())
        np.random.shuffle(sampleids)
        valsize = len(sampleids) // self.K
        print(f"Generating {self.K}-fold CV with ~{valsize} patients per fold\n")
        
        outdir = os.path.join(self.output_path, 'randompatientwise')
        os.makedirs(outdir, exist_ok=True)
        
        DICT = {}
        sampleids = [int(i) for i in sampleids]
        
        for k in range(self.K):
            print(f"Fold {k+1}/{self.K}")
            pid = sampleids[int(valsize*k):int(valsize*(k+1))]
            print(f"  → {len(pid)} validation patients")
            
            train_tcr_donor, validation_df, included, excluded = self.iterate_and_create(pid, tcr_donor_id)
            
            print(f"  → Train: {len(train_tcr_donor)} TCRs | Test: {len(validation_df)} TCRs")
            print(f"  → Public: {len(included)} TCRs | Private: {len(excluded)} TCRs\n")
            
            DICT[f'{k}'] = {
                'patient_ids': pid, 
                'included_tcr_ids': included, 
                'excluded_tcr_ids': excluded
            }
            
            assert len(train_tcr_donor) != 0
            train_tcr_donor.to_csv(os.path.join(outdir, f'train{k}.csv'), index=False)
            validation_df.to_csv(os.path.join(outdir, f'test{k}.csv'), index=False)
            
        with open(os.path.join(outdir, 'info.json'), 'w') as f:
            json.dump(DICT, f)
        print(f"✓ Saved patient-wise splits to: {outdir}")
        return DICT

    def iterate_and_create(self, sample_ids, tcr_donor_id):
        """
        Remove validation patients from training data (VECTORIZED VERSION).
        Returns:
            train_df: Training data with validation patients removed
            validation_df: Validation data
            included: TCR IDs that remain in training (public/shared)
            excluded: TCR IDs removed from training (private/unique to validation patients)
        """
        import time
        start = time.time()
        print(f"    Processing {len(sample_ids)} validation patients from {len(tcr_donor_id):,} TCRs...")
        
        val_sample_ids = set(str(sid) for sid in sample_ids)
        
        # Create single regex pattern
        escaped_ids = [re.escape(str(sid)) for sid in sample_ids]
        pattern = '(?:^|;)(?:' + '|'.join(escaped_ids) + ')(?:;|$)'
        
        print(f"    → Identifying affected rows...")
        affected_mask = tcr_donor_id['sample_id'].astype(str).str.contains(pattern, regex=True, na=False)
        
        num_affected = affected_mask.sum()
        print(f"    → Found {num_affected:,} affected TCRs ({num_affected/len(tcr_donor_id)*100:.2f}%)")
        
        if num_affected == 0:
            return (tcr_donor_id.copy(), pd.DataFrame(), [], [])
        
        affected = tcr_donor_id[affected_mask].copy()
        unaffected = tcr_donor_id[~affected_mask]
        
        print(f"    → Processing affected rows...")
        affected['sample_id_list'] = affected['sample_id'].str.split(';')
        
        # Vectorized filtering
        def filter_samples(sample_list):
            remaining = [sid for sid in sample_list if sid not in val_sample_ids]
            val_only = [sid for sid in sample_list if sid in val_sample_ids]
            return pd.Series({'remaining': remaining, 'val_samples': val_only})
        
        filtered = affected['sample_id_list'].apply(filter_samples)
        affected['remaining'] = filtered['remaining']
        affected['val_samples'] = filtered['val_samples']
        
        # Determine included vs excluded
        affected['has_remaining'] = affected['remaining'].apply(len) > 0
        included_mask = affected['has_remaining']
        
        included = affected[included_mask].copy()
        excluded = affected[~included_mask].copy()
        
        print(f"    → Public TCRs: {len(included):,} | Private TCRs: {len(excluded):,}")
        
        # Update sample_id for included rows
        included['sample_id'] = included['remaining'].apply(lambda x: ';'.join(x))
        included = included.drop(columns=['sample_id_list', 'remaining', 'val_samples', 'has_remaining'])
        
        # Build training dataframe
        print(f"    → Building training set...")
        train_df = pd.concat([unaffected, included], ignore_index=True)
        
        # Build validation dataframe - EFFICIENT VERSION using explode()
        print(f"    → Building validation set...")
        
        # Keep only necessary columns and explode val_samples
        val_df = affected[['cdr1', 'cdr2', 'cdr3', 'cdr2.5', 'val_samples']].copy()
        
        # Use pandas explode (vectorized, much faster than iterrows)
        val_df = val_df.explode('val_samples').rename(columns={'val_samples': 'sample_id'})
        
        # Drop any None/NaN values
        val_df = val_df[val_df['sample_id'].notna()]
        
        # Drop duplicates
        val_df.drop_duplicates(inplace=True)
        
        # Aggregate sample_ids
        validation_df = (
            val_df.groupby(['cdr1', 'cdr2', 'cdr3', 'cdr2.5'])['sample_id']
            .agg(lambda x: ';'.join(x))
            .reset_index()
        )
        
        # Collect TCR IDs
        INCLUDED = included['tcr_id'].tolist() if len(included) > 0 else []
        EXCLUDED = excluded['tcr_id'].tolist() if len(excluded) > 0 else []
        
        elapsed = time.time() - start
        print(f"    ✓ Completed in {elapsed:.2f}s ({len(tcr_donor_id)/elapsed:,.0f} rows/sec)")
        
        return (train_df, validation_df, INCLUDED, EXCLUDED)
    






class CrossValGen2():
    """
    Generates cross-validation splits for TCR data, ensuring patient-level separation.
    Handles both dataset-wise and random patient-wise split strategies.
    """
    
    def __init__(self, patient_index_path, tcr_seq_path, output_path, 
                 datasetwise=True, random_patient_wise=True, K=5, id_col='sample_id'):
        """
        Args:
            patient_index_path: Path to patient metadata (sample_id, dataset)
            tcr_seq_path: Path to TCR data (cdr1-3, cdr2.5, sample_id, tcr_id)
            output_path: Directory for saving train/test splits
            datasetwise: Generate dataset-wise splits (leave-one-dataset-out)
            random_patient_wise: Generate random K-fold patient-wise splits
            K: Number of folds for random patient-wise CV
            id_col: Column name to aggregate in TCR data (default: 'sample_id')
        """
        self.patient_index_path = patient_index_path
        self.tcr_seq_path = tcr_seq_path
        self.output_path = output_path
        self.datasetwise = datasetwise
        self.random_patient_wise = random_patient_wise
        self.K = K
        self.id_col = id_col  # FIX: Added missing attribute
    
    def call(self):
        """Execute cross-validation split generation."""
        print(f"Loading patient index from: {self.patient_index_path}")
        patient_index = pd.read_csv(self.patient_index_path) if self.patient_index_path.endswith('.csv') else pd.read_csv(self.patient_index_path, sep='\t')
        print(f"  → Loaded {len(patient_index)} patients")
        
        print(f"Loading TCR data from: {self.tcr_seq_path}")
        tcr_donor_id = pd.read_csv(self.tcr_seq_path) if self.tcr_seq_path.endswith('.csv') else pd.read_csv(self.tcr_seq_path, sep='\t')
        print(f"  → Loaded {len(tcr_donor_id)} TCR sequences\n")
        
        output = []
        if self.datasetwise:
            print("=" * 60)
            print("DATASET-WISE CROSS-VALIDATION")
            print("=" * 60)
            datasetwise_dict = self.DatasetWise(patient_index, tcr_donor_id)
            output.append(datasetwise_dict)
            
        if self.random_patient_wise:
            print("\n" + "=" * 60)
            print("RANDOM PATIENT-WISE CROSS-VALIDATION")
            print("=" * 60)
            patientwise_dict = self.RandomPatientWise(patient_index, tcr_donor_id)
            output.append(patientwise_dict)
            
        assert len(output) != 0, "No cross-validation strategy selected"  # FIX: Added error message
        print("\n" + "=" * 60)
        print("✓ Cross-validation generation completed successfully")
        print("=" * 60)
        return output
            
    def DatasetWise(self, patient_index, tcr_donor_id):
        """Leave-one-dataset-out cross-validation."""
        datasets = np.unique(patient_index.dataset.tolist())
        datasets = [str(d) for d in datasets]
        print(f"Found {len(datasets)} datasets: {datasets}\n")
        
        DICT = {}
        outdir = os.path.join(self.output_path, 'datasetwise')
        os.makedirs(outdir, exist_ok=True)
        
        for number, dataset in enumerate(datasets):
            print(f"Fold {number+1}/{len(datasets)}: Validation on dataset '{dataset}'")
            pid = patient_index[patient_index['dataset'] == dataset]['sample_id'].tolist()
            print(f"  → {len(pid)} validation patients")
            
            train_tcr_donor, validation_df = self.iterate_and_create(pid, tcr_donor_id)
            
            print(f"  → Train: {len(train_tcr_donor)} TCRs | Test: {len(validation_df)} TCRs")
            
            DICT[f'{number}'] = {
                'patient_ids': pid, 
                'validation_set': [str(dataset)], 
                'training_set': [i for i in datasets if i != dataset]
            }
            
            assert len(train_tcr_donor) != 0, f"Empty training set for fold {number}"  # FIX: Added error message
            assert len(validation_df) != 0, f"Empty validation set for fold {number}"  # FIX: Added error message
            
            train_tcr_donor.to_csv(os.path.join(outdir, f'train{number}.csv'), index=False)
            validation_df.to_csv(os.path.join(outdir, f'val{number}.csv'), index=False)
            
        with open(os.path.join(outdir, 'info.json'), 'w') as f:
            json.dump(DICT, f, indent=2)  # FIX: Added indent for readability
        print(f"✓ Saved dataset-wise splits to: {outdir}")
        return DICT

    def RandomPatientWise(self, patient_index, tcr_donor_id):
        """Random K-fold patient-wise cross-validation."""
        sampleids = np.unique(patient_index.sample_id.tolist())
        np.random.shuffle(sampleids)
        valsize = len(sampleids) // self.K
        print(f"Generating {self.K}-fold CV with ~{valsize} patients per fold\n")
        
        outdir = os.path.join(self.output_path, 'randompatientwise')
        os.makedirs(outdir, exist_ok=True)
        
        DICT = {}
        sampleids = [int(i) for i in sampleids]
        
        for k in range(self.K):
            print(f"Fold {k+1}/{self.K}")
            # FIX: Handle last fold properly to include remaining samples
            if k == self.K - 1:
                pid = sampleids[int(valsize*k):]  # Include all remaining samples in last fold
            else:
                pid = sampleids[int(valsize*k):int(valsize*(k+1))]
            print(f"  → {len(pid)} validation patients")
            
            train_tcr_donor, validation_df = self.iterate_and_create(pid, tcr_donor_id)
            
            print(f"  → Train: {len(train_tcr_donor)} TCRs | Test: {len(validation_df)} TCRs")
            
            DICT[f'{k}'] = {
                'patient_ids': pid, 
            }
            
            assert len(train_tcr_donor) != 0, f"Empty training set for fold {k}"  # FIX: Added error message
            assert len(validation_df) != 0, f"Empty validation set for fold {k}"  # FIX: Added error message
            train_tcr_donor.to_csv(os.path.join(outdir, f'train{k}.csv'), index=False)
            validation_df.to_csv(os.path.join(outdir, f'val{k}.csv'), index=False)
            
        with open(os.path.join(outdir, 'info.json'), 'w') as f:
            json.dump(DICT, f, indent=2)  # FIX: Added indent for readability
        print(f"✓ Saved patient-wise splits to: {outdir}")
        return DICT

    def iterate_and_create(self, sample_ids, tcr_donor_id):
        """
        Remove validation patients from training data (VECTORIZED VERSION).
        Returns:
            train_df: Training data with validation patients removed
            validation_df: Validation data
        """
        valid_df = tcr_donor_id[tcr_donor_id['sample_id'].isin(sample_ids)].copy()  # FIX: Added .copy() to avoid SettingWithCopyWarning
        train_df = tcr_donor_id[~tcr_donor_id['sample_id'].isin(sample_ids)].copy()  # FIX: Added .copy()
        # self.id_col is 'sample_id'
        share_columns = ['cdr1', 'cdr2', 'cdr2.5', 'cdr3', 'tcr_id']
        
        valid_df = valid_df.groupby(share_columns, as_index=False, sort=False).agg({
            self.id_col: lambda x: ';'.join(sorted(set(map(str, x))))
        })
        train_df = train_df.groupby(share_columns, as_index=False, sort=False).agg({
            self.id_col: lambda x: ';'.join(sorted(set(map(str, x))))
        })
        
        return train_df, valid_df
    




def pad_and_mask_tcr(tcr_list, pad_token=-1.0, mask_token=-2.0, masking_rate=0.05):
    """
    Pads a list of variable-length TCR frequency inputs and generates a tracking mask.
    Accepts tf.Tensor, np.ndarray, or Python lists.
    """
    # 1. BULLETPROOFING: Convert all inputs to float32 tf.Tensors immediately.
    # This safely handles NumPy (float64) or Python lists and standardizes them.
    tcr_tensors = [tf.convert_to_tensor(t, dtype=tf.float32) for t in tcr_list]
    # 2. Get their lengths safely
    seq_lengths = tf.stack([tf.shape(t)[0] for t in tcr_tensors])
    # 3. Flatten and reconstruct as RaggedTensor
    flat_tcr = tf.concat(tcr_tensors, axis=0)
    ragged_tcr = tf.RaggedTensor.from_row_lengths(flat_tcr, row_lengths=seq_lengths)
    # 4. Convert Ragged to Dense (Padding)
    padded_tcr = ragged_tcr.to_tensor(default_value=0.0)
    # Get dynamic shapes
    B = tf.shape(padded_tcr)[0]
    max_L = tf.shape(padded_tcr)[1]
    # 5. Create the Base Masking Tensor
    valid_mask = tf.sequence_mask(seq_lengths, maxlen=max_L)
    masking_tensor = tf.where(
        valid_mask, 
        tf.ones((B, max_L), dtype=tf.float32), 
        tf.constant(pad_token, dtype=tf.float32)
    )
    # 6. Apply Random Masking (if requested)
    if masking_rate > 0.0:
        rand_probs = tf.random.uniform((B, max_L), minval=0.0, maxval=1.0, dtype=tf.float32)
        is_masked = tf.logical_and(rand_probs < masking_rate, valid_mask)
        masking_tensor = tf.where(
            is_masked, 
            tf.constant(mask_token, dtype=tf.float32), 
            masking_tensor
        )
        is_masked_expanded = tf.expand_dims(is_masked, axis=-1)
        padded_tcr = tf.where(
            is_masked_expanded, 
            tf.zeros_like(padded_tcr), 
            padded_tcr
        )
    return padded_tcr, masking_tensor



def concatenate_cdrs_with_separator(
    features_list, 
    masks_list, 
    sep_feature_vector, 
    sep_mask_value=2.0
):
    """
    Concatenates a list of padded CDR tensors and their masks, 
    inserting a separator token between each region.
    
    Args:
        features_list: List of feature tensors, e.g., [cdr1, cdr2, cdr25, cdr3]
        masks_list: List of mask tensors, e.g., [cdr1_mask, cdr2_mask, cdr25_mask, cdr3_mask]
        sep_feature_vector: A 1D tensor/list of length 21 representing the separator's features.
        sep_mask_value: A float value to represent the separator in the tracking mask.
        
    Returns:
        combined_features: Tensor of shape (B, total_L, 21)
        combined_masks: Tensor of shape (B, total_L)
    """
    # 1. Get the batch size from the first tensor
    B = tf.shape(features_list[0])[0]
    
    # 2. Build the separator blocks
    # Ensure the feature vector is the right shape and dtype
    sep_feat_vec = tf.cast(tf.reshape(sep_feature_vector, (1, 1, 21)), tf.float32)
    
    # Tile it to match the batch size: (B, 1, 21)
    sep_feat_batch = tf.tile(sep_feat_vec, [B, 1, 1])
    
    # Create the mask separator: (B, 1)
    sep_mask_batch = tf.ones((B, 1), dtype=tf.float32) * sep_mask_value
    
    # 3. Interleave the separators into our lists
    combined_features_list = []
    combined_masks_list = []
    
    for i in range(len(features_list)):
        combined_features_list.append(features_list[i])
        combined_masks_list.append(masks_list[i])
        
        # Add a separator after every CDR EXCEPT the last one
        if i < len(features_list) - 1:
            combined_features_list.append(sep_feat_batch)
            combined_masks_list.append(sep_mask_batch)
            
    # 4. Concatenate everything along the sequence axis (axis=1)
    final_features = tf.concat(combined_features_list, axis=1)
    final_masks = tf.concat(combined_masks_list, axis=1)
    
    return final_features, final_masks


@dataclass
class PublicTcrHlaRow:
    row: int
    cluster_id: int
    cdr3aa: str
    cdr2aa_gapped: str
    cdr1aa_gapped: str
    cdr2_5aa_gapped: str
    n_donors: int
    n_identical_sequences: int
    v_gene_id: Optional[int]
    counts: np.ndarray


import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Iterator
from scipy import sparse

# Assuming PublicTcrHlaRow is defined elsewhere
# from your_module import PublicTcrHlaRow 

class PublicTcrHlaCsrReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._h5 = None

    def __enter__(self) -> "PublicTcrHlaCsrReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._h5 is not None:
            return
        # Increase cache size for faster sequential reads (4MB cache)
        rdcc_nbytes = 4 * 1024 * 1024 
        self._h5 = h5py.File(self.path, "r", rdcc_nbytes=rdcc_nbytes)

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    @property
    def num_rows(self) -> int:
        self.open()
        return int(self._h5["cluster_id"].shape[0])

    @property
    def num_alleles(self) -> int:
        self.open()
        return int(self._h5.attrs["num_alleles"])

    def get_counts_only(self, start: int = 0, stop: Optional[int] = None) -> np.ndarray:
        """
        FASTEST METHOD: Extracts ONLY the dense counts matrix as a single numpy array.
        Skips all other metadata processing.
        """
        self.open()
        counts_grp = self._h5["y_counts"]
        indptr = counts_grp["indptr"]
        indices = counts_grp["indices"]
        data = counts_grp["data"]
        
        n_rows = self.num_rows
        if stop is None or stop > n_rows:
            stop = n_rows
        
        # 1. Read sparse structure for the entire range at once
        # Note: indptr has n_rows + 1 elements
        subset_indptr = indptr[start : stop + 1]
        
        # Calculate the range of data indices we need
        data_start = subset_indptr[0]
        data_end = subset_indptr[-1]
        
        subset_indices = indices[data_start:data_end]
        subset_data = data[data_start:data_end]
        
        # 2. Adjust indptr to start at 0 for the new matrix
        subset_indptr = subset_indptr - data_start
        
        # 3. Create CSR matrix and convert to dense in one highly optimized step
        n_cols = self.num_alleles
        matrix = sparse.csr_matrix(
            (subset_data, subset_indices, subset_indptr),
            shape=(stop - start, n_cols)
        )
        
        return matrix.toarray()

    def iter_rows(
        self,
        *,
        start: int = 0,
        stop: Optional[int] = None,
        batch_size: int = 4096  # Process 4k rows at a time
    ) -> Iterator["PublicTcrHlaRow"]:
        self.open()
        h5 = self._h5
        
        # Cache dataset handles
        loops = h5["loops"]
        ds_cdr3 = loops["cdr3aa"].asstr()
        ds_cdr2 = loops["cdr2aa_gapped"].asstr()
        ds_cdr1 = loops["cdr1aa_gapped"].asstr()
        ds_cdr25 = loops["cdr2_5aa_gapped"].asstr()
        
        ds_cluster = h5["cluster_id"]
        ds_donors = h5["n_donors"]
        ds_identical = h5["n_identical_sequences"]
        ds_vgenes = h5.get("v_gene_ids")
        
        counts_grp = h5["y_counts"]
        ds_indptr = counts_grp["indptr"]
        ds_indices = counts_grp["indices"]
        ds_data = counts_grp["data"]

        n_rows = int(ds_cluster.shape[0])
        n_cols = self.num_alleles
        if stop is None or stop > n_rows:
            stop = n_rows

        # --- BATCHED LOOP ---
        for b_start in range(start, stop, batch_size):
            b_stop = min(stop, b_start + batch_size)
            batch_len = b_stop - b_start
            
            # 1. Bulk Read Metadata (Vectorized read)
            # Reading a slice is 100x faster than reading 1 item 100 times
            b_cluster = ds_cluster[b_start:b_stop]
            b_donors = ds_donors[b_start:b_stop]
            b_identical = ds_identical[b_start:b_stop]
            b_vgenes = ds_vgenes[b_start:b_stop] if ds_vgenes else None
            
            b_cdr3 = ds_cdr3[b_start:b_stop]
            b_cdr2 = ds_cdr2[b_start:b_stop]
            b_cdr1 = ds_cdr1[b_start:b_stop]
            b_cdr25 = ds_cdr25[b_start:b_stop]

            # 2. Bulk Read Sparse Data
            b_indptr = ds_indptr[b_start : b_stop + 1]
            data_start = b_indptr[0]
            data_end = b_indptr[-1]
            
            b_indices = ds_indices[data_start:data_end]
            b_data = ds_data[data_start:data_end]
            
            # 3. Fast Densification (Batch Level)
            # Adjust indptr to be relative to this batch
            b_indptr_rel = b_indptr - data_start
            
            # Use Scipy to construct the whole batch matrix at once
            # This is significantly faster than looping manually
            batch_csr = sparse.csr_matrix(
                (b_data, b_indices, b_indptr_rel), 
                shape=(batch_len, n_cols)
            )
            # Convert whole batch to dense numpy array
            batch_dense = batch_csr.toarray()

            # 4. In-Memory Iteration (Fast Python loop)
            # Now we just iterate over RAM, which is instant
            for i in range(batch_len):
                row_obj = PublicTcrHlaRow(
                    row=b_start + i,
                    cluster_id=int(b_cluster[i]),
                    cdr3aa=b_cdr3[i],
                    cdr2aa_gapped=b_cdr2[i],
                    cdr1aa_gapped=b_cdr1[i],
                    cdr2_5aa_gapped=b_cdr25[i],
                    n_donors=int(b_donors[i]),
                    n_identical_sequences=int(b_identical[i]),
                    v_gene_id=int(b_vgenes[i]) if b_vgenes is not None else None,
                    counts=batch_dense[i], # Pre-computed dense row
                )
                yield row_obj
    def read_sparse_indices(self):
            """
            Extracts nonzero indices directly from HDF5 CSR structure.
            Skips dense conversion entirely for maximum speed.
            """
            self.open()
            counts_grp = self._h5["y_counts"]
            
            # 1. Bulk read the structure arrays (Very fast, just integers)
            # indptr: points to the start/end of each row
            # indices: contains the column index of every nonzero element
            indptr = counts_grp["indptr"][:]
            indices = counts_grp["indices"][:]
            
            # 2. Vectorized calculation of row lengths
            # (The number of nonzero elements is just the difference in pointers)
            row_lengths = indptr[1:] - indptr[:-1]
            
            # Get max_all instantly without a loop
            max_all = int(row_lengths.max()) if row_lengths.size > 0 else 0
            
            # 3. Slice the indices array into your list of lists
            # We use a list comprehension which is faster than a robust for-loop
            n_rows = len(row_lengths)
            counts_set = [indices[indptr[i] : indptr[i+1]] for i in range(n_rows)]
            
            return counts_set, max_all

class PublicTcrHlaCsrWriter:
    """
    CSR-backed HDF5 writer for public_tcr_hla_counts.h5 using dense inputs.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        num_alleles: int,
        counts_dtype: np.dtype,
        indices_dtype: np.dtype,
        chunk_rows: int,
        chunk_nnz: int,
        flush_rows: int,
        compression: Optional[dict] = None,
        include_v_genes: bool = True,
        attrs: Optional[Dict[str, object]] = None,
    ) -> None:
        self.path = Path(path)
        self.num_alleles = int(num_alleles)
        self.counts_dtype = np.dtype(counts_dtype)
        self.indices_dtype = np.dtype(indices_dtype)
        self.chunk_rows = int(chunk_rows)
        self.chunk_nnz = int(chunk_nnz)
        self.flush_rows = int(flush_rows)
        self.include_v_genes = bool(include_v_genes)
        self.attrs = attrs or {}
        self.comp = compression or {}

        self._h5 = None
        self._rows_written = 0
        self._nnz_total = 0

        self._buffer_loops: list[tuple[str, str, str, str]] = []
        self._buffer_n_donors: list[int] = []
        self._buffer_cluster_ids: list[int] = []
        self._buffer_n_identical: list[int] = []
        self._buffer_v_gene_ids: list[int] = []
        self._buffer_indices: list[np.ndarray] = []
        self._buffer_data: list[np.ndarray] = []
        self._buffer_nnz: list[int] = []

    def __enter__(self) -> "PublicTcrHlaCsrWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def rows_written(self) -> int:
        return self._rows_written

    def open(self) -> None:
        if self._h5 is not None:
            return
        import h5py

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(self.path, "w")

        self._h5.attrs["num_alleles"] = int(self.num_alleles)
        self._h5.attrs["counts_dtype"] = str(self.counts_dtype)
        self._h5.attrs["indices_dtype"] = str(self.indices_dtype)
        self._h5.attrs["string_encoding"] = "ascii"
        for k, v in self.attrs.items():
            self._h5.attrs[k] = v

        loops_grp = self._h5.create_group("loops")
        str_dtype = h5py.string_dtype(encoding="ascii")
        loops_grp.create_dataset(
            "cdr3aa",
            shape=(0,),
            maxshape=(None,),
            dtype=str_dtype,
            chunks=(self.chunk_rows,),
            **self.comp,
        )
        loops_grp.create_dataset(
            "cdr2aa_gapped",
            shape=(0,),
            maxshape=(None,),
            dtype=str_dtype,
            chunks=(self.chunk_rows,),
            **self.comp,
        )
        loops_grp.create_dataset(
            "cdr1aa_gapped",
            shape=(0,),
            maxshape=(None,),
            dtype=str_dtype,
            chunks=(self.chunk_rows,),
            **self.comp,
        )
        loops_grp.create_dataset(
            "cdr2_5aa_gapped",
            shape=(0,),
            maxshape=(None,),
            dtype=str_dtype,
            chunks=(self.chunk_rows,),
            **self.comp,
        )

        self._h5.create_dataset(
            "n_donors",
            shape=(0,),
            maxshape=(None,),
            dtype=self.counts_dtype,
            chunks=(self.chunk_rows,),
            **self.comp,
        )
        self._h5.create_dataset(
            "cluster_id",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=(self.chunk_rows,),
            **self.comp,
        )
        self._h5.create_dataset(
            "n_identical_sequences",
            shape=(0,),
            maxshape=(None,),
            dtype=self.counts_dtype,
            chunks=(self.chunk_rows,),
            **self.comp,
        )

        if self.include_v_genes:
            self._h5.create_dataset(
                "v_gene_ids",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int32,
                chunks=(self.chunk_rows,),
                **self.comp,
            )

        counts_grp = self._h5.create_group("y_counts")
        counts_grp.create_dataset(
            "indptr",
            shape=(1,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=(self.chunk_rows + 1,),
            **self.comp,
        )
        counts_grp["indptr"][0] = 0
        counts_grp.create_dataset(
            "indices",
            shape=(0,),
            maxshape=(None,),
            dtype=self.indices_dtype,
            chunks=(self.chunk_nnz,),
            **self.comp,
        )
        counts_grp.create_dataset(
            "data",
            shape=(0,),
            maxshape=(None,),
            dtype=self.counts_dtype,
            chunks=(self.chunk_nnz,),
            **self.comp,
        )

    def close(self) -> None:
        if self._h5 is None:
            return
        self.flush()
        self._h5.close()
        self._h5 = None

    def _validate_ascii(self, value: str) -> str:
        s = "" if value is None else str(value)
        try:
            s.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(f"Non-ASCII string in loops dataset: {s!r}") from exc
        return s

    def add_row(
        self,
        *,
        loops: Sequence[str],
        n_donors: int,
        cluster_id: int,
        n_identical: int,
        counts: np.ndarray,
        v_gene_id: Optional[int] = None,
    ) -> None:
        counts = np.asarray(counts, dtype=self.counts_dtype)
        nz = np.flatnonzero(counts)
        if nz.size == 0:
            return
        self._add_sparse_row(
            loops=loops,
            n_donors=n_donors,
            cluster_id=cluster_id,
            n_identical=n_identical,
            indices=nz.astype(self.indices_dtype, copy=False),
            data=counts[nz],
            v_gene_id=v_gene_id,
        )

    def _add_sparse_row(
        self,
        *,
        loops: Sequence[str],
        n_donors: int,
        cluster_id: int,
        n_identical: int,
        indices: np.ndarray,
        data: np.ndarray,
        v_gene_id: Optional[int] = None,
    ) -> None:
        if len(loops) != 4:
            raise ValueError("loops must be a 4-tuple of strings")
        self._buffer_loops.append(tuple(self._validate_ascii(x) for x in loops))
        self._buffer_n_donors.append(int(n_donors))
        self._buffer_cluster_ids.append(int(cluster_id))
        self._buffer_n_identical.append(int(n_identical))
        if self.include_v_genes:
            self._buffer_v_gene_ids.append(
                int(v_gene_id) if v_gene_id is not None else -1
            )
        self._buffer_indices.append(np.asarray(indices, dtype=self.indices_dtype))
        self._buffer_data.append(np.asarray(data, dtype=self.counts_dtype))
        self._buffer_nnz.append(int(len(indices)))

        if len(self._buffer_loops) >= self.flush_rows:
            self.flush()

    def add_rows(self, rows: Sequence[dict]) -> None:
        for row in rows:
            self.add_row(**row)

    def flush(self) -> None:
        if self._h5 is None:
            self.open()
        if not self._buffer_loops:
            return

        h5 = self._h5
        n_new = len(self._buffer_loops)
        loops_grp = h5["loops"]

        cdr3_vals = [v[0] for v in self._buffer_loops]
        cdr2_vals = [v[1] for v in self._buffer_loops]
        cdr1_vals = [v[2] for v in self._buffer_loops]
        cdr25_vals = [v[3] for v in self._buffer_loops]

        loops_grp["cdr3aa"].resize((self._rows_written + n_new,))
        loops_grp["cdr2aa_gapped"].resize((self._rows_written + n_new,))
        loops_grp["cdr1aa_gapped"].resize((self._rows_written + n_new,))
        loops_grp["cdr2_5aa_gapped"].resize((self._rows_written + n_new,))
        loops_grp["cdr3aa"][self._rows_written:self._rows_written + n_new] = cdr3_vals
        loops_grp["cdr2aa_gapped"][self._rows_written:self._rows_written + n_new] = cdr2_vals
        loops_grp["cdr1aa_gapped"][self._rows_written:self._rows_written + n_new] = cdr1_vals
        loops_grp["cdr2_5aa_gapped"][self._rows_written:self._rows_written + n_new] = cdr25_vals

        h5["n_donors"].resize((self._rows_written + n_new,))
        h5["n_donors"][self._rows_written:self._rows_written + n_new] = np.asarray(
            self._buffer_n_donors, dtype=self.counts_dtype
        )
        h5["cluster_id"].resize((self._rows_written + n_new,))
        h5["cluster_id"][self._rows_written:self._rows_written + n_new] = np.asarray(
            self._buffer_cluster_ids, dtype=np.int64
        )
        h5["n_identical_sequences"].resize((self._rows_written + n_new,))
        h5["n_identical_sequences"][self._rows_written:self._rows_written + n_new] = np.asarray(
            self._buffer_n_identical, dtype=self.counts_dtype
        )

        if self.include_v_genes:
            h5["v_gene_ids"].resize((self._rows_written + n_new,))
            h5["v_gene_ids"][self._rows_written:self._rows_written + n_new] = np.asarray(
                self._buffer_v_gene_ids, dtype=np.int32
            )

        total_nnz_new = sum(self._buffer_nnz)
        indptr_vals = []
        cursor = self._nnz_total
        for nnz in self._buffer_nnz:
            cursor += nnz
            indptr_vals.append(cursor)

        counts_grp = h5["y_counts"]
        counts_grp["indptr"].resize((self._rows_written + n_new + 1,))
        counts_grp["indptr"][self._rows_written + 1:self._rows_written + n_new + 1] = np.asarray(
            indptr_vals, dtype=np.int64
        )

        if total_nnz_new:
            indices_concat = np.concatenate(self._buffer_indices)
            data_concat = np.concatenate(self._buffer_data)
            counts_grp["indices"].resize((self._nnz_total + total_nnz_new,))
            counts_grp["data"].resize((self._nnz_total + total_nnz_new,))
            counts_grp["indices"][self._nnz_total:self._nnz_total + total_nnz_new] = indices_concat
            counts_grp["data"][self._nnz_total:self._nnz_total + total_nnz_new] = data_concat
            self._nnz_total += total_nnz_new

        self._rows_written += n_new
        self._buffer_loops.clear()
        self._buffer_n_donors.clear()
        self._buffer_cluster_ids.clear()
        self._buffer_n_identical.clear()
        self._buffer_v_gene_ids.clear()
        self._buffer_indices.clear()
        self._buffer_data.clear()
        self._buffer_nnz.clear()




def create_dataset(donor_indices, batch_size, shuffle=False):
    """Create TF dataset from donor indices."""
    tcr_ids = np.arange(donor_indices.shape[0], dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((tcr_ids, donor_indices))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def pad_list_to_array(counts_set, max_all, pad_token=-1.):
    """Pad variable-length lists to fixed array."""
    n_samples = len(counts_set)
    result = np.full((n_samples, max_all), pad_token)
    for i, row in enumerate(counts_set):
        result[i, :len(row)] = row
    return result


def pad_list_to_array_without_max(counts_set, pad_token=-1.):
    """Pad variable-length arrays to fixed array using vectorized ops."""
    lengths = np.array([len(row) for row in counts_set])
    max_all = int(lengths.max()) if lengths.size > 0 else 0
    n_samples = len(counts_set)
    # Concatenate all indices into one flat array
    flat = np.concatenate(counts_set) if max_all > 0 else np.array([], dtype=np.int64)
    # Build the padded array using advanced indexing
    result = np.full((n_samples, max_all), pad_token)
    # Create row indices for each element in flat
    row_idx = np.repeat(np.arange(n_samples), lengths)
    # Create column indices (position within each row)
    col_idx = np.concatenate([np.arange(l) for l in lengths]) if max_all > 0 else np.array([], dtype=np.int64)
    result[row_idx, col_idx] = flat
    return result, max_all

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)



@dataclass
class PublicTcrHlaClusterChunk:
    cluster_start: int
    cluster_end: int
    cluster_id: np.ndarray
    n_donors: np.ndarray
    raw_csr_tcr_indptr: np.ndarray
    raw_csr_donor_indptr: Optional[np.ndarray]
    raw_csr_donor_indices: Optional[np.ndarray]
    raw_csr_donor_data: Optional[np.ndarray]
    counts_dense: Optional[np.ndarray]
    pvals_dense: Optional[np.ndarray]
    z_probs_dense: Optional[np.ndarray]
    raw_csr_tcr_loops: np.ndarray
    raw_csr_tcr_int_fields: np.ndarray
    raw_csr_tcr_int_field_names: tuple[str, ...]
    _tcr_loops_accessor: RaggedClusterAccessor = field(init=False, repr=False)
    _n_identical_accessor: RaggedClusterAccessor = field(init=False, repr=False)
    _v_gene_id_accessor: RaggedClusterAccessor | None = field(init=False, repr=False)
    _donor_ids_accessor: RaggedClusterAccessor | None = field(init=False, repr=False)
    _tcr_counts: np.ndarray = field(init=False, repr=False)

    @property
    def n_clusters(self) -> int:
        return int(self.cluster_end - self.cluster_start)

    def __init__(
        self,
        *,
        cluster_start: int,
        cluster_end: int,
        cluster_id: np.ndarray,
        n_donors: np.ndarray,
        raw_csr_tcr_indptr: np.ndarray,
        raw_csr_donor_indptr: Optional[np.ndarray],
        raw_csr_donor_indices: Optional[np.ndarray],
        raw_csr_donor_data: Optional[np.ndarray],
        counts_dense: Optional[np.ndarray],
        pvals_dense: Optional[np.ndarray],
        z_probs_dense: Optional[np.ndarray] = None,
        raw_csr_tcr_loops: np.ndarray,
        raw_csr_tcr_int_fields: np.ndarray,
        raw_csr_tcr_int_field_names: tuple[str, ...],
        cdr_freq: Optional[Dict[str, "RaggedClusterAccessor"]] = None,
    ) -> None:
        self.cluster_start = int(cluster_start)
        self.cluster_end = int(cluster_end)
        self.cluster_id = cluster_id
        self.n_donors = n_donors
        self.raw_csr_tcr_indptr = raw_csr_tcr_indptr
        self.raw_csr_donor_indptr = raw_csr_donor_indptr
        self.raw_csr_donor_indices = raw_csr_donor_indices
        self.raw_csr_donor_data = raw_csr_donor_data
        self.counts_dense = counts_dense
        self.pvals_dense = pvals_dense
        self.z_probs_dense = z_probs_dense
        self.raw_csr_tcr_loops = raw_csr_tcr_loops
        self.raw_csr_tcr_int_fields = raw_csr_tcr_int_fields
        self.raw_csr_tcr_int_field_names = raw_csr_tcr_int_field_names
        self.cdr_freq = cdr_freq

        self._tcr_loops_accessor = RaggedClusterAccessor(
            self.raw_csr_tcr_loops, self.raw_csr_tcr_indptr
        )
        idx = _index_int_field(
            self.raw_csr_tcr_int_field_names, "n_identical_sequences"
        )
        self._n_identical_accessor = RaggedClusterAccessor(
            self.raw_csr_tcr_int_fields[:, idx], self.raw_csr_tcr_indptr
        )
        if "v_gene_id" in self.raw_csr_tcr_int_field_names:
            idx = _index_int_field(self.raw_csr_tcr_int_field_names, "v_gene_id")
            self._v_gene_id_accessor = RaggedClusterAccessor(
                self.raw_csr_tcr_int_fields[:, idx], self.raw_csr_tcr_indptr
            )
        else:
            self._v_gene_id_accessor = None
        if raw_csr_donor_indptr is not None and raw_csr_donor_indices is not None:
            self._donor_ids_accessor = RaggedClusterAccessor(
                raw_csr_donor_indices, raw_csr_donor_indptr
            )
        else:
            self._donor_ids_accessor = None
        self._tcr_counts = np.diff(self.raw_csr_tcr_indptr).astype(np.int64, copy=False)

    @property
    def tcr_int_field_names(self) -> tuple[str, ...]:
        return self.raw_csr_tcr_int_field_names

    @property
    def tcr_loops(self) -> "RaggedClusterAccessor":
        return self._tcr_loops_accessor

    @property
    def n_identical_sequences(self) -> "RaggedClusterAccessor":
        return self._n_identical_accessor

    @property
    def v_gene_id(self) -> Optional["RaggedClusterAccessor"]:
        return self._v_gene_id_accessor

    @property
    def donor_ids(self) -> Optional["RaggedClusterAccessor"]:
        return self._donor_ids_accessor

    @property
    def tcr_counts(self) -> np.ndarray:
        return self._tcr_counts


class RaggedClusterAccessor:
    __slots__ = ("_data", "_indptr")

    def __init__(self, data: np.ndarray, indptr: np.ndarray) -> None:
        self._data = data
        self._indptr = indptr

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._indptr.shape[0] - 1)
            return [self[i] for i in range(start, stop, step)]
        if isinstance(idx, (list, tuple, np.ndarray)):
            return [self[i] for i in idx]
        i = int(idx)
        t0 = int(self._indptr[i])
        t1 = int(self._indptr[i + 1])
        return self._data[t0:t1]


def _index_int_field(names: Sequence[str], name: str) -> int:
    try:
        return names.index(name)
    except ValueError as exc:
        raise ValueError(f"Missing {name} in tcr_int_field_names.") from exc


class PublicTcrHlaCsrReaderChunk:
    """
    Cluster-level reader for public_tcr_hla_counts.h5 with ragged TCR arrays.

    Set include_counts/include_pvals to control which sparse arrays are loaded.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        include_counts: bool = True,
        include_pvals: bool = False,
        include_donors: bool = False,
        include_z_probs: bool = False,
        include_cdr_freq: bool = False,
    ):
        self.path = Path(path)
        self.include_counts = bool(include_counts)
        self.include_pvals = bool(include_pvals)
        self.include_donors = bool(include_donors)
        self.include_z_probs = bool(include_z_probs)
        self._h5 = None
        self.include_cdr_freq = bool(include_cdr_freq)

    def __enter__(self) -> "PublicTcrHlaCsrReaderChunk":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._h5 is not None:
            return
        import h5py

        self._h5 = h5py.File(self.path, "r")

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    @property
    def num_clusters(self) -> int:
        self.open()
        return int(self._h5["clusters"]["cluster_id"].shape[0])

    @property
    def num_rows(self) -> int:
        return self.num_clusters

    @property
    def num_alleles(self) -> int:
        self.open()
        num = self._h5.attrs.get("num_alleles")
        if num is None:
            raise KeyError("HDF5 missing 'num_alleles' attribute.")
        return int(num)

    @property
    def num_tcrs(self) -> int:
        self.open()
        return int(self._h5["tcrs"]["n_identical_sequences"].shape[0])

    def iter_cluster_chunks(
        self,
        *,
        chunk_rows: int = 100_000,
        include_v_genes: bool = True,
        include_donors: Optional[bool] = None,
        start: int = 0,
        stop: Optional[int] = None,
    ) -> Iterator[PublicTcrHlaClusterChunk]:
        self.open()
        h5 = self._h5
        if include_donors is None:
            include_donors = self.include_donors
        clusters_grp = h5["clusters"]
        cluster_id = clusters_grp["cluster_id"]
        n_donors = clusters_grp["n_donors"]
        tcr_indptr = clusters_grp["tcr_indptr"]
        counts_grp = clusters_grp.get("counts") if self.include_counts else None
        pvals_grp = clusters_grp.get("pvals") if self.include_pvals else None
        donors_grp = clusters_grp.get("donors") if include_donors else None
        z_probs_grp = clusters_grp.get("z_probs") if self.include_z_probs else None
        cdr_freq_grp = clusters_grp.get("cdr_freq") if self.include_cdr_freq else None
        if self.include_counts and counts_grp is None:
            raise KeyError("HDF5 missing clusters/counts (include_counts=True).")
        if self.include_pvals and pvals_grp is None:
            raise KeyError("HDF5 missing clusters/pvals (include_pvals=True).")
        if include_donors and donors_grp is None:
            raise KeyError("HDF5 missing clusters/donors (include_donors=True).")
        if self.include_z_probs and z_probs_grp is None:
            raise KeyError("HDF5 missing clusters/z_probs (include_z_probs=True).")
        counts_indptr = counts_grp["indptr"] if counts_grp is not None else None
        counts_indices = counts_grp["indices"] if counts_grp is not None else None
        counts_data = counts_grp["data"] if counts_grp is not None else None
        pvals_indptr = pvals_grp["indptr"] if pvals_grp is not None else None
        pvals_indices = pvals_grp["indices"] if pvals_grp is not None else None
        pvals_data = pvals_grp["data"] if pvals_grp is not None else None
        donors_indptr = donors_grp["indptr"] if donors_grp is not None else None
        donors_indices = donors_grp["indices"] if donors_grp is not None else None
        donors_data = donors_grp["data"] if donors_grp is not None else None
        zp_indptr = z_probs_grp["indptr"] if z_probs_grp is not None else None
        zp_indices = z_probs_grp["indices"] if z_probs_grp is not None else None
        zp_data = z_probs_grp["data"] if z_probs_grp is not None else None

        tcrs_grp = h5["tcrs"]
        loops_grp = tcrs_grp["loops"]
        loops_cdr3 = loops_grp["cdr3aa"].asstr()
        loops_cdr2 = loops_grp["cdr2aa_gapped"].asstr()
        loops_cdr1 = loops_grp["cdr1aa_gapped"].asstr()
        loops_cdr25 = loops_grp["cdr2_5aa_gapped"].asstr()
        n_identical = tcrs_grp["n_identical_sequences"]
        v_gene_ids = tcrs_grp.get("v_gene_id") if include_v_genes else None

        n_clusters = int(cluster_id.shape[0])
        if stop is None or stop > n_clusters:
            stop = n_clusters
        if start < 0 or start > stop:
            raise ValueError(f"Invalid start/stop for chunking: {start}..{stop}")

        step = max(1, int(chunk_rows))
        for cluster_start in range(start, stop, step):
            cluster_end = min(cluster_start + step, stop)
            if cluster_end <= cluster_start:
                continue

            cluster_chunk = np.asarray(cluster_id[cluster_start:cluster_end])
            donors_chunk = np.asarray(n_donors[cluster_start:cluster_end])

            tcr_indptr_chunk = np.asarray(tcr_indptr[cluster_start : cluster_end + 1])
            tcr_start = int(tcr_indptr_chunk[0])
            tcr_end = int(tcr_indptr_chunk[-1])
            tcr_indptr_chunk = tcr_indptr_chunk - tcr_start

            if counts_indptr is not None:
                counts_indptr_chunk = np.asarray(
                    counts_indptr[cluster_start : cluster_end + 1]
                )
                counts_start = int(counts_indptr_chunk[0])
                counts_end = int(counts_indptr_chunk[-1])
                counts_indptr_chunk = counts_indptr_chunk - counts_start
            else:
                counts_indptr_chunk = None
                counts_start = counts_end = 0

            if pvals_indptr is not None:
                pvals_indptr_chunk = np.asarray(
                    pvals_indptr[cluster_start : cluster_end + 1]
                )
                pvals_start = int(pvals_indptr_chunk[0])
                pvals_end = int(pvals_indptr_chunk[-1])
                pvals_indptr_chunk = pvals_indptr_chunk - pvals_start
            else:
                pvals_indptr_chunk = None
                pvals_start = pvals_end = 0

            if donors_indptr is not None:
                donors_indptr_chunk = np.asarray(
                    donors_indptr[cluster_start : cluster_end + 1]
                )
                donors_start = int(donors_indptr_chunk[0])
                donors_end = int(donors_indptr_chunk[-1])
                donors_indptr_chunk = donors_indptr_chunk - donors_start
            else:
                donors_indptr_chunk = None
                donors_start = donors_end = 0

            if tcr_end > tcr_start:
                cdr3_vals = np.asarray(loops_cdr3[tcr_start:tcr_end])
                cdr2_vals = np.asarray(loops_cdr2[tcr_start:tcr_end])
                cdr1_vals = np.asarray(loops_cdr1[tcr_start:tcr_end])
                cdr25_vals = np.asarray(loops_cdr25[tcr_start:tcr_end])
                tcr_loops = np.column_stack([cdr3_vals, cdr2_vals, cdr1_vals, cdr25_vals])
                ident_vals = np.asarray(n_identical[tcr_start:tcr_end])
                int_fields = [ident_vals.astype(np.int64, copy=False)]
                int_field_names = ["n_identical_sequences"]
                if v_gene_ids is not None:
                    v_vals = np.asarray(v_gene_ids[tcr_start:tcr_end])
                    int_fields.append(v_vals.astype(np.int64, copy=False))
                    int_field_names.append("v_gene_id")
                tcr_int_fields = np.column_stack(int_fields)
            else:
                tcr_loops = np.empty((0, 4), dtype=object)
                tcr_int_fields = np.empty((0, 1), dtype=np.int64)
                int_field_names = ["n_identical_sequences"]
                if v_gene_ids is not None:
                    tcr_int_fields = np.empty((0, 2), dtype=np.int64)
                    int_field_names.append("v_gene_id")

            n_clusters = int(cluster_end - cluster_start)
            counts_dense = None
            pvals_dense = None
            if counts_indptr_chunk is not None and counts_indices is not None:
                indices_chunk = np.asarray(counts_indices[counts_start:counts_end])
                data_chunk = np.asarray(counts_data[counts_start:counts_end])
                counts_dense = np.zeros(
                    (n_clusters, self.num_alleles), dtype=data_chunk.dtype
                )
                for i in range(n_clusters):
                    lo = int(counts_indptr_chunk[i])
                    hi = int(counts_indptr_chunk[i + 1])
                    if hi > lo:
                        idx = indices_chunk[lo:hi].astype(np.int64, copy=False)
                        counts_dense[i, idx] = data_chunk[lo:hi]

            if pvals_indptr_chunk is not None and pvals_indices is not None:
                pvals_indices_chunk = np.asarray(pvals_indices[pvals_start:pvals_end])
                pvals_data_chunk = np.asarray(pvals_data[pvals_start:pvals_end])
                pvals_dense = np.zeros(
                    (n_clusters, self.num_alleles), dtype=pvals_data_chunk.dtype
                )
                for i in range(n_clusters):
                    lo = int(pvals_indptr_chunk[i])
                    hi = int(pvals_indptr_chunk[i + 1])
                    if hi > lo:
                        idx = pvals_indices_chunk[lo:hi].astype(np.int64, copy=False)
                        pvals_dense[i, idx] = pvals_data_chunk[lo:hi]
            if donors_indptr_chunk is not None and donors_indices is not None:
                donor_indices_chunk = np.asarray(
                    donors_indices[donors_start:donors_end]
                )
                donor_data_chunk = (
                    np.asarray(donors_data[donors_start:donors_end])
                    if donors_data is not None
                    else None
                )
            else:
                donor_indices_chunk = None
                donor_data_chunk = None

            # --- z_probs loading (sparse CSR -> dense) ---
            z_probs_dense = None
            if zp_indptr is not None and zp_indices is not None:
                zp_indptr_chunk = np.asarray(
                    zp_indptr[cluster_start : cluster_end + 1]
                )
                zp_start = int(zp_indptr_chunk[0])
                zp_end = int(zp_indptr_chunk[-1])
                zp_indptr_chunk = zp_indptr_chunk - zp_start
                if zp_end > zp_start:
                    zp_indices_chunk = np.asarray(zp_indices[zp_start:zp_end])
                    zp_data_chunk = np.asarray(zp_data[zp_start:zp_end])
                    z_probs_dense = np.zeros(
                        (n_clusters, self.num_alleles), dtype=zp_data_chunk.dtype
                    )
                    for i in range(n_clusters):
                        lo = int(zp_indptr_chunk[i])
                        hi = int(zp_indptr_chunk[i + 1])
                        if hi > lo:
                            idx = zp_indices_chunk[lo:hi].astype(np.int64, copy=False)
                            z_probs_dense[i, idx] = zp_data_chunk[lo:hi]
                else:
                    z_probs_dense = np.zeros(
                        (n_clusters, self.num_alleles), dtype=np.float32
                    )

            cdr_freq = None
            if cdr_freq_grp is not None:
                cdr_freq = {}
                for nm in ("cdr3", "cdr1", "cdr2", "cdr25"):
                    ip_ds = cdr_freq_grp.get(f"{nm}_indptr")
                    fr_ds = cdr_freq_grp.get(f"{nm}_freq")
                    if ip_ds is not None and fr_ds is not None:
                        ip_chunk = np.asarray(
                            ip_ds[cluster_start : cluster_end + 1]
                        )
                        fr_start = int(ip_chunk[0])
                        fr_end = int(ip_chunk[-1])
                        ip_local = ip_chunk - fr_start
                        if fr_end > fr_start:
                            fr_data = np.asarray(fr_ds[fr_start:fr_end])
                        else:
                            fr_data = np.empty((0, 21), dtype=np.float32)
                        cdr_freq[nm] = RaggedClusterAccessor(fr_data, ip_local)

            yield PublicTcrHlaClusterChunk(
                cluster_start=cluster_start,
                cluster_end=cluster_end,
                cluster_id=cluster_chunk,
                n_donors=donors_chunk,
                raw_csr_tcr_indptr=tcr_indptr_chunk,
                raw_csr_donor_indptr=donors_indptr_chunk,
                raw_csr_donor_indices=donor_indices_chunk,
                raw_csr_donor_data=donor_data_chunk,
                counts_dense=counts_dense,
                pvals_dense=pvals_dense,
                z_probs_dense=z_probs_dense,
                raw_csr_tcr_loops=tcr_loops,
                raw_csr_tcr_int_fields=tcr_int_fields,
                raw_csr_tcr_int_field_names=tuple(int_field_names),
                cdr_freq=cdr_freq,
            )
    def read_sparse_indices_of_counts(self):
        """
        Extracts nonzero indices directly from HDF5 CSR structure.
        Skips dense conversion entirely for maximum speed.
        """
        self.open()
        counts_grp = self._h5["clusters"]["counts"]

        indptr = counts_grp["indptr"][:]
        indices = counts_grp["indices"][:]

        row_lengths = indptr[1:] - indptr[:-1]
        max_all = int(row_lengths.max()) if row_lengths.size > 0 else 0

        n_rows = len(row_lengths)
        counts_set = [indices[indptr[i] : indptr[i+1]] for i in range(n_rows)]

        return counts_set, max_all


class PublicTcrHlaCsrWriterChunk:
    """
    Cluster-level HDF5 writer for public_tcr_hla_counts.h5 with ragged TCR arrays.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        num_alleles: int,
        counts_dtype: np.dtype,
        indices_dtype: np.dtype,
        chunk_rows: int,
        chunk_nnz: int,
        flush_rows: int,
        compression: Optional[dict] = None,
        include_v_genes: bool = True,
        include_donors: bool = False,
        donor_indices_dtype: Optional[np.dtype] = None,
        attrs: Optional[Dict[str, object]] = None,
    ) -> None:
        self.path = Path(path)
        self.num_alleles = int(num_alleles)
        self.counts_dtype = np.dtype(counts_dtype)
        self.indices_dtype = np.dtype(indices_dtype)
        self.chunk_rows = int(chunk_rows)
        self.chunk_nnz = int(chunk_nnz)
        self.flush_rows = int(flush_rows)
        self.include_v_genes = bool(include_v_genes)
        self.include_donors = bool(include_donors)
        self.donor_indices_dtype = (
            np.dtype(donor_indices_dtype)
            if donor_indices_dtype is not None
            else np.dtype(np.int64)
        )
        self.attrs = attrs or {}
        self.comp = compression or {}

        self._h5 = None
        self._clusters_written = 0
        self._tcrs_written = 0
        self._counts_nnz_total = 0
        self._donors_nnz_total = 0

        self._buffer_cluster_ids: list[int] = []
        self._buffer_n_donors: list[int] = []
        self._buffer_tcr_counts: list[int] = []
        self._buffer_tcr_loops: list[tuple[str, str, str, str]] = []
        self._buffer_tcr_n_identical: list[int] = []
        self._buffer_tcr_v_gene_ids: list[int] = []
        self._buffer_counts_indices: list[np.ndarray] = []
        self._buffer_counts_data: list[np.ndarray] = []
        self._buffer_counts_nnz: list[int] = []
        self._buffer_donor_indices: list[np.ndarray] = []
        self._buffer_donor_data: list[np.ndarray] = []
        self._buffer_donor_nnz: list[int] = []

    def __enter__(self) -> "PublicTcrHlaCsrWriterChunk":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def rows_written(self) -> int:
        return self._clusters_written

    @property
    def clusters_written(self) -> int:
        return self._clusters_written

    @property
    def tcrs_written(self) -> int:
        return self._tcrs_written

    def open(self) -> None:
        if self._h5 is not None:
            return
        import h5py

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(self.path, "w")

        self._h5.attrs["num_alleles"] = int(self.num_alleles)
        self._h5.attrs["counts_dtype"] = str(self.counts_dtype)
        self._h5.attrs["indices_dtype"] = str(self.indices_dtype)
        self._h5.attrs["string_encoding"] = "ascii"
        if self.include_donors:
            self._h5.attrs["donor_indices_dtype"] = str(self.donor_indices_dtype)
        for k, v in self.attrs.items():
            self._h5.attrs[k] = v

        clusters_grp = self._h5.create_group("clusters")
        clusters_grp.create_dataset(
            "cluster_id", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(self.chunk_rows,), **self.comp,)
        clusters_grp.create_dataset( "n_donors", shape=(0,), maxshape=(None,), dtype=self.counts_dtype, chunks=(self.chunk_rows,), **self.comp,)
        clusters_grp.create_dataset( "tcr_indptr", shape=(1,), maxshape=(None,), dtype=np.int64, chunks=(self.chunk_rows + 1,), **self.comp,)
        clusters_grp["tcr_indptr"][0] = 0

        counts_grp = clusters_grp.create_group("counts")
        counts_grp.create_dataset(
            "indptr", shape=(1,), maxshape=(None,), dtype=np.int64, chunks=(self.chunk_rows + 1,), **self.comp,)
        counts_grp["indptr"][0] = 0
        counts_grp.create_dataset(
            "indices", shape=(0,), maxshape=(None,), dtype=self.indices_dtype, chunks=(self.chunk_nnz,), **self.comp,)
        counts_grp.create_dataset(
            "data", shape=(0,), maxshape=(None,), dtype=self.counts_dtype, chunks=(self.chunk_nnz,), **self.comp,)

        if self.include_donors:
            donors_grp = clusters_grp.create_group("donors")
            donors_grp.create_dataset(
                "indptr",
                shape=(1,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=(self.chunk_rows + 1,),
                **self.comp,
            )
            donors_grp["indptr"][0] = 0
            donors_grp.create_dataset(
                "indices",
                shape=(0,),
                maxshape=(None,),
                dtype=self.donor_indices_dtype,
                chunks=(self.chunk_nnz,),
                **self.comp,
            )
            donors_grp.create_dataset(
                "data",
                shape=(0,),
                maxshape=(None,),
                dtype=self.counts_dtype,
                chunks=(self.chunk_nnz,),
                **self.comp,
            )

        tcrs_grp = self._h5.create_group("tcrs")
        loops_grp = tcrs_grp.create_group("loops")
        str_dtype = h5py.string_dtype(encoding="ascii")
        loops_grp.create_dataset(
            "cdr3aa", shape=(0,), maxshape=(None,), dtype=str_dtype, chunks=(self.chunk_rows,), **self.comp,)
        loops_grp.create_dataset(
            "cdr2aa_gapped", shape=(0,), maxshape=(None,), dtype=str_dtype, chunks=(self.chunk_rows,), **self.comp,)
        loops_grp.create_dataset(
            "cdr1aa_gapped", shape=(0,), maxshape=(None,), dtype=str_dtype, chunks=(self.chunk_rows,), **self.comp,)
        loops_grp.create_dataset(
            "cdr2_5aa_gapped", shape=(0,), maxshape=(None,), dtype=str_dtype, chunks=(self.chunk_rows,), **self.comp,)
        tcrs_grp.create_dataset(
            "n_identical_sequences", shape=(0,), maxshape=(None,), dtype=self.counts_dtype, chunks=(self.chunk_rows,), **self.comp,)
        if self.include_v_genes:
            tcrs_grp.create_dataset(
                "v_gene_id", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(self.chunk_rows,), **self.comp,)

    def close(self) -> None:
        if self._h5 is None:
            return
        self.flush()
        self._h5.close()
        self._h5 = None

    def _validate_ascii(self, value: str) -> str:
        s = "" if value is None else str(value)
        try:
            s.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(f"Non-ASCII string in loops dataset: {s!r}") from exc
        return s

    def add_cluster(
        self,
        *,
        cluster_id: int,
        n_donors: int,
        counts: np.ndarray,
        donor_ids: Optional[Sequence[int]] = None,
        tcr_loops: Sequence[Sequence[str]],
        tcr_n_identical: Sequence[int],
        tcr_v_gene_ids: Optional[Sequence[Optional[int]]] = None,
    ) -> None:
        loops_arr = np.asarray(tcr_loops, dtype=object)
        if loops_arr.ndim != 2 or loops_arr.shape[1] != 4:
            raise ValueError("tcr_loops must be an array of shape (n_tcr, 4)")
        n_tcr = int(loops_arr.shape[0])
        if n_tcr < 1:
            return

        n_identical = np.asarray(tcr_n_identical, dtype=np.int64)
        if n_identical.shape[0] != n_tcr:
            raise ValueError("tcr_n_identical length must match tcr_loops")

        if self.include_v_genes:
            if tcr_v_gene_ids is None:
                v_gene_vals = np.full(n_tcr, -1, dtype=np.int64)
            else:
                v_gene_vals = np.asarray(tcr_v_gene_ids)
                if v_gene_vals.shape[0] != n_tcr:
                    raise ValueError("tcr_v_gene_ids length must match tcr_loops")
                if v_gene_vals.dtype == object:
                    v_gene_vals = np.asarray(
                        [(-1 if v is None else int(v)) for v in v_gene_vals],
                        dtype=np.int64,
                    )
                else:
                    v_gene_vals = v_gene_vals.astype(np.int64, copy=False)
        else:
            v_gene_vals = None

        counts = np.asarray(counts, dtype=self.counts_dtype)
        nz = np.flatnonzero(counts)
        if nz.size == 0:
            return
        indices = nz.astype(self.indices_dtype, copy=False)
        data = counts[nz]

        if self.include_donors:
            if donor_ids is None:
                donor_ids_arr = np.zeros(0, dtype=self.donor_indices_dtype)
            else:
                donor_ids_arr = np.asarray(donor_ids, dtype=self.donor_indices_dtype)
                if donor_ids_arr.ndim != 1:
                    donor_ids_arr = donor_ids_arr.reshape(-1)
            donor_data_arr = np.ones(
                donor_ids_arr.shape[0], dtype=self.counts_dtype
            )

        self._buffer_cluster_ids.append(int(cluster_id))
        self._buffer_n_donors.append(int(n_donors))
        self._buffer_tcr_counts.append(n_tcr)

        for loops in loops_arr:
            if len(loops) != 4:
                raise ValueError("tcr_loops must be an array of shape (n_tcr, 4)")
            self._buffer_tcr_loops.append(
                tuple(self._validate_ascii(x) for x in loops)
            )

        self._buffer_tcr_n_identical.extend(n_identical.astype(int).tolist())
        if self.include_v_genes and v_gene_vals is not None:
            self._buffer_tcr_v_gene_ids.extend(v_gene_vals.astype(int).tolist())

        self._buffer_counts_indices.append(np.asarray(indices, dtype=self.indices_dtype))
        self._buffer_counts_data.append(np.asarray(data, dtype=self.counts_dtype))
        self._buffer_counts_nnz.append(int(len(indices)))
        if self.include_donors:
            self._buffer_donor_indices.append(donor_ids_arr)
            self._buffer_donor_data.append(donor_data_arr)
            self._buffer_donor_nnz.append(int(donor_ids_arr.shape[0]))

        if len(self._buffer_cluster_ids) >= self.flush_rows:
            self.flush()

    def add_row(
        self,
        *,
        loops: Sequence[str],
        n_donors: int,
        cluster_id: int,
        n_identical: int,
        counts: np.ndarray,
        v_gene_id: Optional[int] = None,
        donor_ids: Optional[Sequence[int]] = None,
    ) -> None:
        tcr_v_gene_ids = [v_gene_id] if self.include_v_genes else None
        self.add_cluster(
            cluster_id=cluster_id,
            n_donors=n_donors,
            counts=counts,
            donor_ids=donor_ids,
            tcr_loops=[loops],
            tcr_n_identical=[n_identical],
            tcr_v_gene_ids=tcr_v_gene_ids,
        )

    def add_clusters(self, clusters: Sequence[dict]) -> None:
        for cluster in clusters:
            self.add_cluster(**cluster)

    def flush(self) -> None:
        if self._h5 is None:
            self.open()
        if not self._buffer_cluster_ids:
            return

        h5 = self._h5
        n_new_clusters = len(self._buffer_cluster_ids)
        n_new_tcrs = int(sum(self._buffer_tcr_counts))

        clusters_grp = h5["clusters"]
        clusters_grp["cluster_id"].resize((self._clusters_written + n_new_clusters,))
        clusters_grp["cluster_id"][
            self._clusters_written:self._clusters_written + n_new_clusters
        ] = np.asarray(self._buffer_cluster_ids, dtype=np.int64)

        clusters_grp["n_donors"].resize((self._clusters_written + n_new_clusters,))
        clusters_grp["n_donors"][
            self._clusters_written:self._clusters_written + n_new_clusters
        ] = np.asarray(self._buffer_n_donors, dtype=self.counts_dtype)

        tcr_indptr_vals = self._tcrs_written + np.cumsum(
            np.asarray(self._buffer_tcr_counts, dtype=np.int64)
        )
        clusters_grp["tcr_indptr"].resize(
            (self._clusters_written + n_new_clusters + 1,)
        )
        clusters_grp["tcr_indptr"][
            self._clusters_written + 1:self._clusters_written + n_new_clusters + 1
        ] = tcr_indptr_vals

        loops_grp = h5["tcrs"]["loops"]
        cdr3_vals = [v[0] for v in self._buffer_tcr_loops]
        cdr2_vals = [v[1] for v in self._buffer_tcr_loops]
        cdr1_vals = [v[2] for v in self._buffer_tcr_loops]
        cdr25_vals = [v[3] for v in self._buffer_tcr_loops]

        loops_grp["cdr3aa"].resize((self._tcrs_written + n_new_tcrs,))
        loops_grp["cdr2aa_gapped"].resize((self._tcrs_written + n_new_tcrs,))
        loops_grp["cdr1aa_gapped"].resize((self._tcrs_written + n_new_tcrs,))
        loops_grp["cdr2_5aa_gapped"].resize((self._tcrs_written + n_new_tcrs,))
        loops_grp["cdr3aa"][
            self._tcrs_written:self._tcrs_written + n_new_tcrs
        ] = cdr3_vals
        loops_grp["cdr2aa_gapped"][
            self._tcrs_written:self._tcrs_written + n_new_tcrs
        ] = cdr2_vals
        loops_grp["cdr1aa_gapped"][
            self._tcrs_written:self._tcrs_written + n_new_tcrs
        ] = cdr1_vals
        loops_grp["cdr2_5aa_gapped"][
            self._tcrs_written:self._tcrs_written + n_new_tcrs
        ] = cdr25_vals

        tcrs_grp = h5["tcrs"]
        tcrs_grp["n_identical_sequences"].resize(
            (self._tcrs_written + n_new_tcrs,)
        )
        tcrs_grp["n_identical_sequences"][
            self._tcrs_written:self._tcrs_written + n_new_tcrs
        ] = np.asarray(self._buffer_tcr_n_identical, dtype=self.counts_dtype)

        if self.include_v_genes:
            tcrs_grp["v_gene_id"].resize((self._tcrs_written + n_new_tcrs,))
            tcrs_grp["v_gene_id"][
                self._tcrs_written:self._tcrs_written + n_new_tcrs
            ] = np.asarray(self._buffer_tcr_v_gene_ids, dtype=np.int32)

        counts_grp = clusters_grp["counts"]
        total_nnz_new = int(sum(self._buffer_counts_nnz))
        counts_indptr_vals = self._counts_nnz_total + np.cumsum(
            np.asarray(self._buffer_counts_nnz, dtype=np.int64)
        )
        counts_grp["indptr"].resize((self._clusters_written + n_new_clusters + 1,))
        counts_grp["indptr"][
            self._clusters_written + 1:self._clusters_written + n_new_clusters + 1
        ] = counts_indptr_vals

        if total_nnz_new:
            indices_concat = np.concatenate(self._buffer_counts_indices)
            data_concat = np.concatenate(self._buffer_counts_data)
            counts_grp["indices"].resize((self._counts_nnz_total + total_nnz_new,))
            counts_grp["data"].resize((self._counts_nnz_total + total_nnz_new,))
            counts_grp["indices"][
                self._counts_nnz_total:self._counts_nnz_total + total_nnz_new
            ] = indices_concat
            counts_grp["data"][
                self._counts_nnz_total:self._counts_nnz_total + total_nnz_new
            ] = data_concat
            self._counts_nnz_total += total_nnz_new

        if self.include_donors:
            donors_grp = clusters_grp["donors"]
            total_donor_nnz_new = int(sum(self._buffer_donor_nnz))
            donor_indptr_vals = self._donors_nnz_total + np.cumsum(
                np.asarray(self._buffer_donor_nnz, dtype=np.int64)
            )
            donors_grp["indptr"].resize(
                (self._clusters_written + n_new_clusters + 1,)
            )
            donors_grp["indptr"][
                self._clusters_written + 1:self._clusters_written + n_new_clusters + 1
            ] = donor_indptr_vals

            if total_donor_nnz_new:
                donor_indices_concat = (
                    np.concatenate(self._buffer_donor_indices)
                    if self._buffer_donor_indices
                    else np.zeros(0, dtype=self.donor_indices_dtype)
                )
                donor_data_concat = (
                    np.concatenate(self._buffer_donor_data)
                    if self._buffer_donor_data
                    else np.zeros(0, dtype=self.counts_dtype)
                )
                donors_grp["indices"].resize(
                    (self._donors_nnz_total + total_donor_nnz_new,)
                )
                donors_grp["data"].resize(
                    (self._donors_nnz_total + total_donor_nnz_new,)
                )
                donors_grp["indices"][
                    self._donors_nnz_total:self._donors_nnz_total + total_donor_nnz_new
                ] = donor_indices_concat
                donors_grp["data"][
                    self._donors_nnz_total:self._donors_nnz_total + total_donor_nnz_new
                ] = donor_data_concat
                self._donors_nnz_total += total_donor_nnz_new

        self._clusters_written += n_new_clusters
        self._tcrs_written += n_new_tcrs

        self._buffer_cluster_ids.clear()
        self._buffer_n_donors.clear()
        self._buffer_tcr_counts.clear()
        self._buffer_tcr_loops.clear()
        self._buffer_tcr_n_identical.clear()
        self._buffer_tcr_v_gene_ids.clear()
        self._buffer_counts_indices.clear()
        self._buffer_counts_data.clear()
        self._buffer_counts_nnz.clear()
        self._buffer_donor_indices.clear()
        self._buffer_donor_data.clear()
        self._buffer_donor_nnz.clear()


class MleZprobsWriter:
    """
    Appends MLE z-probability results into an HDF5 file as a sparse CSR group.

    Creates `clusters/z_probs` group with CSR arrays (indptr, indices, data).
    The z_probs are stored per-cluster: indices are allele column IDs, data are
    float32 probability values from sigmoid(z_logits).

    Typical workflow:
        1. Copy the original dataset h5 to output path (preserves all original data).
        2. Open this writer on the copy in append mode.
        3. For each training chunk, call write_chunk() with the z_probs.
        4. Close the writer.

    Parameters
    ----------
    path : str | Path
        Path to the HDF5 file to append z_probs into (must already exist).
    num_clusters : int
        Total number of clusters that will be written.
    chunk_nnz : int
        HDF5 chunk size for the indices/data arrays.
    chunk_rows : int
        HDF5 chunk size for the indptr array.
    compression : dict | None
        HDF5 compression kwargs (e.g. {"compression": "gzip", "compression_opts": 4}).
    z_probs_dtype : np.dtype
        Data type for z_prob values (default: float32).
    indices_dtype : np.dtype
        Data type for allele indices (default: uint32).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        num_clusters: int,
        chunk_nnz: int = 100_000,
        chunk_rows: int = 10_000,
        compression: Optional[dict] = None,
        z_probs_dtype: np.dtype = np.float32,
        indices_dtype: np.dtype = np.uint32,
    ) -> None:
        self.path = Path(path)
        self.num_clusters = int(num_clusters)
        self.chunk_nnz = int(chunk_nnz)
        self.chunk_rows = int(chunk_rows)
        self.comp = compression or {}
        self.z_probs_dtype = np.dtype(z_probs_dtype)
        self.indices_dtype = np.dtype(indices_dtype)
        self._h5 = None
        self._nnz_written = 0

    def __enter__(self) -> "MleZprobsWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._h5 is not None:
            return
        import h5py

        self._h5 = h5py.File(self.path, "a")
        clusters_grp = self._h5["clusters"]

        if "z_probs" not in clusters_grp:
            zp_grp = clusters_grp.create_group("z_probs")
            zp_grp.create_dataset(
                "indptr",
                shape=(self.num_clusters + 1,),
                dtype=np.int64,
                chunks=(min(self.chunk_rows, self.num_clusters + 1),),
                **self.comp,
            )
            zp_grp["indptr"][:] = 0
            zp_grp.create_dataset(
                "indices",
                shape=(0,),
                maxshape=(None,),
                dtype=self.indices_dtype,
                chunks=(self.chunk_nnz,),
                **self.comp,
            )
            zp_grp.create_dataset(
                "data",
                shape=(0,),
                maxshape=(None,),
                dtype=self.z_probs_dtype,
                chunks=(self.chunk_nnz,),
                **self.comp,
            )

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def write_chunk(
        self,
        cluster_start: int,
        cluster_end: int,
        binder_sets: np.ndarray,
        z_probs: np.ndarray,
        pad_token: float = -1.0,
    ) -> None:
        """
        Write z_probs for a chunk of clusters as sparse CSR.

        Parameters
        ----------
        cluster_start : int
            Global index of the first cluster in this chunk.
        cluster_end : int
            Global index one past the last cluster in this chunk.
        binder_sets : np.ndarray, shape (n_chunk, max_hlas_per_tcr)
            Padded allele indices per cluster (pad_token marks invalid).
        z_probs : np.ndarray, shape (n_chunk, max_hlas_per_tcr)
            Sigmoid z-probabilities aligned with binder_sets.
        pad_token : float
            Padding value used in binder_sets.
        """
        if self._h5 is None:
            self.open()

        zp_grp = self._h5["clusters"]["z_probs"]
        n_chunk = cluster_end - cluster_start

        # --- Vectorized sparse extraction ---
        valid_mask = binder_sets != pad_token  # (n_chunk, max_hlas)
        nnz_per_row = valid_mask.sum(axis=1).astype(np.int64)
        total_nnz_new = int(nnz_per_row.sum())

        # Update indptr
        chunk_indptr = self._nnz_written + np.cumsum(nnz_per_row)
        zp_grp["indptr"][cluster_start + 1 : cluster_end + 1] = chunk_indptr

        if total_nnz_new > 0:
            # Extract valid entries in row-major order
            indices_concat = binder_sets[valid_mask].astype(self.indices_dtype)
            data_concat = z_probs[valid_mask].astype(self.z_probs_dtype)

            old_len = zp_grp["indices"].shape[0]
            new_len = old_len + total_nnz_new
            zp_grp["indices"].resize((new_len,))
            zp_grp["data"].resize((new_len,))
            zp_grp["indices"][old_len:new_len] = indices_concat
            zp_grp["data"][old_len:new_len] = data_concat

        self._nnz_written += total_nnz_new




# ═══════════════════════════════════════════════════════════════════
# 1.  CdrFreqWriter — writes CDR freq profiles into existing H5
# ═══════════════════════════════════════════════════════════════════
CDR_NAMES = ("cdr3", "cdr1", "cdr2", "cdr25")
N_SYM = 21  # 20 amino acids + gap

class CdrFreqWriter:
    """
    Append-mode HDF5 writer that adds CDR amino-acid frequency profiles
    into an existing dataset_pval.h5 under ``clusters/cdr_freq/``.

    Storage layout (ragged — each cluster has variable alignment length):
    ::
        clusters/cdr_freq/
            cdr3_freq     (total_rows, 21) float32   — concatenated freq matrices
            cdr3_indptr   (num_clusters+1,)  int64    — row boundaries per cluster
            cdr1_freq     ...
            cdr1_indptr   ...
            cdr2_freq     ...
            cdr2_indptr   ...
            cdr25_freq    ...
            cdr25_indptr  ...

    Follows the same append pattern as ``MleZprobsWriter``.

    Parameters
    ----------
    path : str | Path
        Path to the HDF5 file to append into (must already exist).
    num_clusters : int
        Total number of clusters in the file.
    chunk_nnz : int
        HDF5 chunk size (rows) for the freq data arrays.
    chunk_rows : int
        HDF5 chunk size for the indptr arrays.
    compression : dict | None
        HDF5 compression kwargs.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        num_clusters: int,
        chunk_nnz: int = 50_000,
        chunk_rows: int = 10_000,
        compression: Optional[dict] = None,
    ) -> None:
        self.path = Path(path)
        self.num_clusters = int(num_clusters)
        self.chunk_nnz = int(chunk_nnz)
        self.chunk_rows = int(chunk_rows)
        self.comp = compression or {}
        self._h5: Optional[h5py.File] = None
        # track how many freq rows have been written per CDR
        self._nnz_written: Dict[str, int] = {nm: 0 for nm in CDR_NAMES}

    def __enter__(self) -> "CdrFreqWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        """Open the H5 in append mode and create datasets if missing."""
        if self._h5 is not None:
            return
        self._h5 = h5py.File(self.path, "a")
        clusters_grp = self._h5["clusters"]
        # create cdr_freq group if it doesn't exist
        if "cdr_freq" not in clusters_grp:
            grp = clusters_grp.create_group("cdr_freq")
            # store metadata
            grp.attrs["n_symbols"] = N_SYM
            grp.attrs["aa_alphabet"] = "ACDEFGHIKLMNPQRSTVWY"
            grp.attrs["gap_index"] = 20
            for nm in CDR_NAMES:
                # indptr — fixed size, pre-filled with zeros
                grp.create_dataset(
                    f"{nm}_indptr",
                    shape=(self.num_clusters + 1,),
                    dtype=np.int64,
                    chunks=(min(self.chunk_rows, self.num_clusters + 1),),
                    **self.comp,
                )
                grp[f"{nm}_indptr"][:] = 0
                # freq data — resizable (rows grow, cols = N_SYM)
                grp.create_dataset(
                    f"{nm}_freq",
                    shape=(0, N_SYM),
                    maxshape=(None, N_SYM),
                    dtype=np.float32,
                    chunks=(min(self.chunk_nnz, 10_000), N_SYM),
                    **self.comp,
                )

    def close(self) -> None:
        """Flush and close the H5 file handle."""
        if self._h5 is not None:
            self._h5.flush()
            self._h5.close()
            self._h5 = None

    def write_chunk(
        self,
        cluster_start: int,
        cluster_end: int,
        freq_data: Dict[str, List[np.ndarray]],
    ) -> None:
        """
        Write frequency profiles for a contiguous block of clusters.

        Parameters
        ----------
        cluster_start : int
            Global index of the first cluster in this chunk.
        cluster_end : int
            Global index one past the last cluster (exclusive).
        freq_data : dict[str, list[np.ndarray]]
            Keys are CDR names ('cdr3','cdr1','cdr2','cdr25').
            Values are lists (length = cluster_end - cluster_start)
            of float32 arrays with shape (alignment_len_i, 21).
        """
        if self._h5 is None:
            self.open()
        n_cl = cluster_end - cluster_start
        grp = self._h5["clusters"]["cdr_freq"]
        for nm in CDR_NAMES:
            blocks = freq_data[nm]
            assert len(blocks) == n_cl, (
                f"freq_data['{nm}'] has {len(blocks)} entries, expected {n_cl}"
            )
            # compute new row counts per cluster
            row_counts = np.array([b.shape[0] for b in blocks], dtype=np.int64)
            total_new = int(row_counts.sum())
            # concatenate all freq blocks for this CDR
            if total_new > 0:
                new_data = np.vstack(blocks).astype(np.float32)
            else:
                new_data = np.empty((0, N_SYM), dtype=np.float32)
            # append freq data
            ds = grp[f"{nm}_freq"]
            old_rows = ds.shape[0]
            ds.resize(old_rows + total_new, axis=0)
            if total_new > 0:
                ds[old_rows:] = new_data
            # update indptr (cumulative offsets)
            indptr_ds = grp[f"{nm}_indptr"]
            base = self._nnz_written[nm]
            cumsum = base + np.cumsum(row_counts)
            indptr_ds[cluster_start + 1 : cluster_end + 1] = cumsum
            self._nnz_written[nm] = int(cumsum[-1]) if cumsum.size > 0 else base
        # flush periodically
        self._h5.flush()


class TCRLikelihoodLoss(tf.keras.layers.Layer):
    """
    Computes Negative Log-Likelihood and L2 regularization for dense TCR binding.
    Optimized via matrix multiplication for large-scale datasets.
    """
    def __init__(self, donor_hla_matrix, beta=4.0, pad_token=-1., 
                 l2_reg_lambda=1e-5, reduction='sum', 
                 poisson_approx_untyped_hlas=False,
                 hla_bias_init=None, invariant_lambda=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.pad_token = pad_token
        self.l2_reg_lambda = l2_reg_lambda
        self.reduction = reduction
        self.poisson_approx_untyped_hlas = poisson_approx_untyped_hlas
        self.invariant_lambda = invariant_lambda
        
        # self.X_T shape: (A, N) 
        # A: Total Alleles, N: Total Donors
        self.X_T = tf.constant(donor_hla_matrix.T, dtype=tf.float32)
        # Store the prior as a constant tensor
        if hla_bias_init is not None:
            self.prior_logits = tf.constant(hla_bias_init, dtype=tf.float32)
        else:
            self.prior_logits = 0.0

    def logit_drift_penalty(self, z_logits):
        """Prevents sigmoid saturation by pulling logits gently back to the empirical prior."""
        if self.l2_reg_lambda > 0:
            # Penalize the squared distance from the prior log-odds, NOT from zero!
            return self.l2_reg_lambda * tf.reduce_mean(tf.square(z_logits - self.prior_logits))
        return 0.0

    def l2_reg(self, z_logits, mask):
        """L2 regularization normalized by valid positions."""
        if self.l2_reg_lambda:
            n_valid = tf.reduce_sum(mask)
            n_valid = tf.maximum(n_valid, 1.0)  # avoid division by zero
            return self.l2_reg_lambda * tf.reduce_sum(tf.pow(z_logits, 2) * mask) / n_valid
        return 0.
    
    def scale_invariant_sparsity(self, z_prob, active_mask):
        """
        Implements lambda * (||gamma_i||_1 / ||gamma_i||_2) from the manuscript.
        Applied only to the active (co-occurring) alleles to force the model
        to pick a single winner from the haplotype block without shrinking the max prob.
        """
        if self.l2_reg_lambda > 0:
            # Mask the probabilities so we only penalize spreading among active alleles
            masked_probs = z_prob * active_mask
            # L1 norm (sum of probabilities)
            l1 = tf.reduce_sum(masked_probs, axis=-1)
            # L2 norm (sqrt of sum of squares, with epsilon to prevent NaN gradients)
            l2 = tf.sqrt(tf.reduce_sum(tf.square(masked_probs), axis=-1) + 1e-8)
            # Calculate ratio per TCR, average over batch
            sparsity_penalty = l1 / l2
            return self.l2_reg_lambda * tf.reduce_mean(sparsity_penalty)
        return 0.0

    def call(self, z_logits, binder_dense_set, pos_donor_indices):
        """
        Parameters
        ----------
        z_logits : tf.Tensor
            Logits predicted by Transformer over ALL alleles. Shape: (B, A)
        binder_dense_set : tf.Tensor
            Binary mask of TCR-HLA co-occurrences. Shape: (B, A)
        pos_donor_indices : tf.Tensor
            Indices of positive donors (padded with pad_token). Shape: (B, P)
        """
        # 1. Masking and Probabilities
        mask = tf.cast(binder_dense_set, tf.float32)
        
        # z_prob (gamma_ia) shape: (B, A) 
        z_prob = tf.sigmoid(z_logits) #* mask
        # 2. Regularization (Anchor to the prior)
        regularization_term = self.logit_drift_penalty(z_logits)
        # 2. Regularization this one pushes probs to 0.5
        #regularization_term = self.l2_reg(z_logits, mask)
        
        # 3. Likelihood Calculation (p_ni)
        if self.poisson_approx_untyped_hlas:
            # MODE: Continuous x_na (Poisson Approximation)
            # p_ni = 1 - exp(-sum(x_na * gamma_ia))
            sum_x_gamma = tf.matmul(z_prob, self.X_T) # (B, A) @ (A, N) -> (B, N)
            p_ni = 1.0 - tf.exp(-sum_x_gamma)
        else:
            # MODE: Binary x_na (Exact Calculation)
            # p_ni = 1 - exp(sum(x_na * log(1 - gamma_ia)))
            z_prob_safe = tf.minimum(z_prob, 1.0 - 1e-7)
            log_1_minus_z = tf.math.log(1.0 - z_prob_safe)
            log_prod = tf.matmul(log_1_minus_z, self.X_T) # (B, A) @ (A, N) -> (B, N)
            p_ni = 1.0 - tf.exp(log_prod)
            
        p_ni_safe = p_ni + 1e-7 #tf.maximum(p_ni, 1e-7)
        
        # 4. Positive Donors (Reward)
        safe_pos_indices = tf.maximum(pos_donor_indices, 0)
        pos_mask = tf.cast(tf.not_equal(pos_donor_indices, tf.cast(self.pad_token, tf.int32)), tf.float32)
        
        # p_pos shape: (B, P)
        p_pos = tf.gather(p_ni_safe, safe_pos_indices, batch_dims=1)
        reward = tf.reduce_sum(tf.math.log(p_pos) * pos_mask, axis=1)
        
        # 5. Negative Donors (Penalty via Beta-Binomial)
        n_i = tf.reduce_sum(pos_mask, axis=1)
        sum_p_all = tf.reduce_sum(p_ni_safe, axis=1)
        sum_p_pos = tf.reduce_sum(p_pos * pos_mask, axis=1)
        n_tilde = sum_p_all - sum_p_pos
        
        penalty = tf.math.lgamma(n_tilde + self.beta) - tf.math.lgamma(n_i + n_tilde + self.beta + 1.0)
        log_likelihood = reward + penalty
        
        # 6. Reduction
        if self.reduction == 'sum':
            nll = -tf.reduce_sum(log_likelihood)
        else:
            nll = -tf.reduce_mean(log_likelihood)
        return nll, regularization_term


import re
import numpy as np
from collections import defaultdict

def match_hla_alleles(hla_embed, idx_to_hla, gene_defaults=None):
    """
    Match HLA alleles from a target dataset to allele embeddings from IMGT.
    
    Parameters
    ----------
    hla_embed : dict
        Dictionary of {full_resolution_allele: embedding} from IMGT (e.g., 38k alleles).
    idx_to_hla : dict
        Dictionary of {idx: allele_name} for target alleles (e.g., 440 alleles at 2-field resolution).
        Order is preserved in all outputs.
    gene_defaults : dict, optional
        Fallback alleles per gene for completely unmatched cases.
    
    Returns
    -------
    hla_matched : dict
        {allele: {'embed': embedding, 'matched': matched_allele, 'match_level': str}}
        Ordered according to idx_to_hla.
    unmatched : list
        List of alleles that could not be matched at any level.
    embed_matrix : np.ndarray
        Array of shape (N_alleles, embed_dim), rows ordered by idx_to_hla keys (0, 1, 2, ...).
        Unmatched alleles get a row of zeros.
    """
    
    if gene_defaults is None:
        gene_defaults = {
            'HLA-DQA1': 'HLA-DQA1*01:01',
            'HLA-DRB1': 'HLA-DRB1*01:01',
            'HLA-DPA1': 'HLA-DPA1*01:03',
        }
    
    # --- Helper functions ---
    def truncate_to_two_field(allele):
        match = re.match(r'(HLA-\w+\*\d+:\d+)', allele)
        return match.group(1) if match else allele

    def clean_allele(allele):
        return re.sub(r'[NLSCAQ]$', '', allele)

    def get_first_field(allele):
        match = re.match(r'(HLA-\w+\*\d+)', allele)
        return match.group(1) if match else None

    # --- Build lookup indices ---
    truncated_map = defaultdict(list)
    for full_key in hla_embed:
        short = truncate_to_two_field(full_key)
        truncated_map[short].append(full_key)

    first_field_map = defaultdict(list)
    for short_key in truncated_map:
        ff = get_first_field(short_key)
        if ff:
            first_field_map[ff].append(short_key)

    def pick_best_truncated(candidates, prefix):
        canonical = [c for c in candidates if c.startswith(prefix + ":01")]
        return canonical[0] if canonical else candidates[0]

    def pick_best_first_field(ff):
        candidates = sorted(first_field_map[ff])
        best_short = next((c for c in candidates if c.endswith(":01")), candidates[0])
        return best_short, truncated_map[best_short][0]

    # --- Match alleles in idx_to_hla order ---
    sorted_indices = sorted(idx_to_hla.keys(), key=lambda x: int(x))
    hla_matched = {}
    unmatched = []

    for idx in sorted_indices:
        allele = idx_to_hla[idx]
        cleaned = clean_allele(allele)

        if allele in hla_embed:
            hla_matched[allele] = {
                'embed': hla_embed[allele],
                'matched': allele,
                'match_level': 'exact'
            }
        elif cleaned in hla_embed:
            hla_matched[allele] = {
                'embed': hla_embed[cleaned],
                'matched': cleaned,
                'match_level': 'suffix_stripped'
            }
        elif cleaned in truncated_map:
            best = pick_best_truncated(truncated_map[cleaned], cleaned)
            hla_matched[allele] = {
                'embed': hla_embed[best],
                'matched': best,
                'match_level': 'two_field'
            }
        elif get_first_field(cleaned) and get_first_field(cleaned) in first_field_map:
            ff = get_first_field(cleaned)
            best_short, full_key = pick_best_first_field(ff)
            hla_matched[allele] = {
                'embed': hla_embed[full_key],
                'matched': best_short,
                'match_level': 'first_field'
            }
        elif allele.split('*')[0] in gene_defaults:
            gene = allele.split('*')[0]
            fallback = gene_defaults[gene]
            if fallback in truncated_map:
                full_key = truncated_map[fallback][0]
                hla_matched[allele] = {
                    'embed': hla_embed[full_key],
                    'matched': fallback,
                    'match_level': 'gene_fallback'
                }
            else:
                unmatched.append(allele)
        else:
            unmatched.append(allele)

    # --- Build embedding matrix: rows ordered by sorted idx_to_hla keys ---
    # Infer embedding dimension from first entry
    sample_embed = next(iter(hla_embed.values()))
    embed_dim = len(sample_embed) if hasattr(sample_embed, '__len__') else sample_embed.shape[0]

    embed_matrix = np.zeros((len(sorted_indices), embed_dim))
    for i, idx in enumerate(sorted_indices):
        allele = idx_to_hla[idx]
        if allele in hla_matched:
            embed_matrix[i] = hla_matched[allele]['embed']

    # --- Summary ---
    levels = defaultdict(int)
    for v in hla_matched.values():
        levels[v['match_level']] += 1

    print(f"Matched: {len(hla_matched)} / {len(sorted_indices)}")
    print(f"Unmatched: {len(unmatched)}")
    print("Match levels:")
    for level in ['exact', 'suffix_stripped', 'two_field', 'first_field', 'gene_fallback']:
        if levels[level]:
            print(f"  {level}: {levels[level]}")
    if unmatched:
        print(f"Unmatched alleles: {unmatched}")
    print(f"\nEmbed matrix shape: {embed_matrix.shape}")

    return hla_matched, unmatched, embed_matrix


blosum62_array = [
    [ 4,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2,  0],
    [ 0,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2, -2],
    [-2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3, -1],
    [-1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2, -1],
    [-2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3, -1],
    [ 0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3, -1],
    [-2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2, -1],
    [-1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1, -1],
    [-1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2, -1],
    [-1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1, -1],
    [-1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1, -1],
    [-2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2, -1],
    [-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3, -2],
    [-1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1, -1],
    [-1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2, -1],
    [ 1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2,  0],
    [ 0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2,  0],
    [ 0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1, -1],
    [-3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2, -2],
    [-2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7, -1],
    [ 0, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1, -1,  0,  0, -1, -2, -1, -1],
]
BLOSUM_TENSOR = tf.constant(blosum62_array, dtype=tf.float32)


DEFAULT_PAD_TOKEN = -1
DEFAULT_MASK_TOKEN = -2
DEFAULT_SEP_TOKEN = -3
DEFAULT_NORMAL_TOKEN = 1

@tf.function(jit_compile=True)
def encode_msa_frequencies(freq_batch):
    """BLOSUM62-weighted averaging on MSA frequency tensors.
    Args:  freq_batch  (B, L, 21) float32, rows should sum to 1.
    Returns: (B, L, 21) expected BLOSUM scores.
    """
    freq_batch = tf.cast(freq_batch, tf.float32)
    return tf.matmul(freq_batch, BLOSUM_TENSOR)


@tf.function(jit_compile=True)
def encode_msa_frequencies(freq_batch):
    """BLOSUM62-weighted averaging on MSA frequency tensors.
    Args:  freq_batch  (B, L, 21) float32, rows should sum to 1.
    Returns: (B, L, 21) expected BLOSUM scores.
    """
    freq_batch = tf.cast(freq_batch, tf.float32)
    return tf.matmul(freq_batch, BLOSUM_TENSOR)


# ======================================================================
# 1.  Sequence Encoder Layer
# ======================================================================
class SequenceEncoderLayer(layers.Layer):
    """Projects a sequence encoding into a dense embedding space,
    with optional BLOSUM62 pre-projection.

    encoding_mode
    -------------
    - "raw"    : feed input features directly into the dense projection.
    - "blosum" : apply BLOSUM62 weighted averaging first, then project.
                 Use this when your input is one-hot (B,L,21) or MSA
                 frequency profiles (B,L,21).  The BLOSUM62 transform
                 maps raw/frequency features to physicochemically
                 meaningful representations before the learned projection.

    Inputs
    ------
    seq_input : (B, L, D)   — one-hot, MSA freq profiles, or any encoding
    mask      : (B, L)      — int tensor with PAD / MASK / SEP / NORMAL tokens

    Output
    ------
    (B, L, embed_dim)  — embedded sequence with PAD positions zeroed out.
    """

    VALID_MODES = ("raw", "blosum")

    def __init__(
        self,
        embed_dim: int = 32,
        max_len: int = 2048,
        dropout_rate: float = 0.1,
        encoding_mode: str = "blosum",
        pad_token: int = DEFAULT_PAD_TOKEN,
        mask_token: int = DEFAULT_MASK_TOKEN,
        sep_token: int = DEFAULT_SEP_TOKEN,
        normal_token: int = DEFAULT_NORMAL_TOKEN,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert encoding_mode in self.VALID_MODES, (
            f"encoding_mode must be one of {self.VALID_MODES}, got '{encoding_mode}'"
        )
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.dropout_rate = dropout_rate
        self.encoding_mode = encoding_mode
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.sep_token = sep_token
        self.normal_token = normal_token

    # ---- build -----------------------------------------------------
    def build(self, input_shape):
        # input_shape = [(B, L, D), (B, L)]
        if isinstance(input_shape, (list, tuple)):
            seq_shape = input_shape[0]
        else:
            seq_shape = input_shape
        input_dim = seq_shape[-1]

        # linear projection  D -> embed_dim
        self.input_projection = layers.Dense(
            self.embed_dim, use_bias=True, name="input_proj"
        )

        # learnable special-token embeddings
        self.mask_embedding = self.add_weight(
            name="mask_embedding",
            shape=(1, 1, self.embed_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.sep_embedding = self.add_weight(
            name="sep_embedding",
            shape=(1, 1, self.embed_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        # learnable positional encoding
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, self.max_len, self.embed_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.layer_norm = layers.LayerNormalization(name="enc_ln")
        self.dropout = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    # ---- forward ---------------------------------------------------
    def call(self, inputs, training=None):
        seq_input, mask = inputs  # (B,L,D), (B,L)
        mask = tf.cast(mask, tf.int32)

        # — identify non-normal positions (PAD, MASK, SEP) -----------
        is_special = tf.cast(
            tf.not_equal(mask, self.normal_token), tf.float32
        )[:, :, tf.newaxis]  # (B, L, 1)

        # zero out special positions *before* BLOSUM so garbage
        # amino-acid features at PAD/MASK/SEP don't pollute the
        # BLOSUM weighted average
        x = seq_input * (1.0 - is_special)

        # — optional BLOSUM62 pre-projection -------------------------
        if self.encoding_mode == "blosum":
            x = encode_msa_frequencies(x)  # (B, L, 21) → (B, L, 21)

        # learned dense projection  D → embed_dim
        x = self.input_projection(x)  # (B, L, E)

        L = tf.shape(x)[1]

        # — replace MASK positions with learnable embedding -----------
        mask_flag = tf.cast(
            tf.equal(mask, self.mask_token), tf.float32
        )[:, :, tf.newaxis]                                   # (B,L,1)
        x = x * (1.0 - mask_flag) + self.mask_embedding * mask_flag

        # — replace SEP positions with learnable embedding -----------
        sep_flag = tf.cast(
            tf.equal(mask, self.sep_token), tf.float32
        )[:, :, tf.newaxis]                                    # (B,L,1)
        x = x * (1.0 - sep_flag) + self.sep_embedding * sep_flag

        # — add positional encoding ----------------------------------
        x = x + self.pos_embedding[:, :L, :]

        # norm + dropout
        x = self.layer_norm(x)
        x = self.dropout(x, training=training)

        # — zero-out PAD positions -----------------------------------
        pad_keep = tf.cast(
            tf.not_equal(mask, self.pad_token), tf.float32
        )[:, :, tf.newaxis]                                    # (B,L,1)
        x = x * pad_keep

        return x  # (B, L, embed_dim)

    # ---- config (serialisation) ------------------------------------
    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "embed_dim": self.embed_dim,
                "max_len": self.max_len,
                "dropout_rate": self.dropout_rate,
                "encoding_mode": self.encoding_mode,
                "pad_token": self.pad_token,
                "mask_token": self.mask_token,
                "sep_token": self.sep_token,
                "normal_token": self.normal_token,
            }
        )
        return cfg


# ======================================================================
# 2.  Gated Transformer Layer
# ======================================================================
def _rope_frequencies(head_dim, max_len=4096, theta=10000.0):
    """Precompute sin/cos tables for Rotary Position Embeddings.
    Returns (max_len, head_dim) sin and cos tensors.
    """
    dim_pairs = head_dim // 2
    freq_seq = tf.cast(tf.range(0, dim_pairs), tf.float32)
    inv_freq = 1.0 / (theta ** (freq_seq / tf.cast(dim_pairs, tf.float32)))
    positions = tf.cast(tf.range(0, max_len), tf.float32)
    angles = tf.einsum("i,j->ij", positions, inv_freq)  # (max_len, dim_pairs)
    sin_table = tf.sin(angles)  # (max_len, dim_pairs)
    cos_table = tf.cos(angles)  # (max_len, dim_pairs)
    return sin_table, cos_table


def _apply_rope(x, sin_table, cos_table):
    """Apply RoPE to a tensor of shape (B, H, L, d_h).
    Pairs adjacent dimensions: (x0,x1), (x2,x3), ... and rotates them.
    """
    L = tf.shape(x)[2]
    sin = sin_table[:L, :]  # (L, d_h//2)
    cos = cos_table[:L, :]  # (L, d_h//2)
    # broadcast to (1, 1, L, d_h//2)
    sin = sin[tf.newaxis, tf.newaxis, :, :]
    cos = cos[tf.newaxis, tf.newaxis, :, :]

    # split into even / odd pairs along last axis
    x1 = x[..., 0::2]  # (B, H, L, d_h//2)
    x2 = x[..., 1::2]  # (B, H, L, d_h//2)

    # rotation:  x1' = x1*cos - x2*sin,  x2' = x1*sin + x2*cos
    r1 = x1 * cos - x2 * sin
    r2 = x1 * sin + x2 * cos

    # interleave back
    out = tf.stack([r1, r2], axis=-1)  # (B, H, L, d_h//2, 2)
    out = tf.reshape(out, tf.shape(x))
    return out


class GatedTransformerLayer(layers.Layer):
    """Transformer block with **Rotary Position Embeddings (RoPE)** and
    gated head combination.

    Gate mechanism
    --------------
    A gate matrix  G = sigmoid(X @ W_g)  of shape (B, L, num_heads)
    is element-wise multiplied with the per-head attention outputs,
    then the heads are mean-pooled to a single representation.

    Inputs
    ------
    x    : (B, L, embed_dim)
    mask : (B, L)   — int tensor with PAD / MASK / SEP / NORMAL tokens

    Output
    ------
    (B, L, embed_dim)  — with PAD positions zeroed out.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        num_heads: int = 1,
        ff_dim: int | None = None,
        resnet: bool = True,
        dropout_rate: float = 0.1,
        rope_max_len: int = 4096,
        rope_theta: float = 10000.0,
        pad_token: int = DEFAULT_PAD_TOKEN,
        mask_token: int = DEFAULT_MASK_TOKEN,
        sep_token: int = DEFAULT_SEP_TOKEN,
        normal_token: int = DEFAULT_NORMAL_TOKEN,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        assert (embed_dim // num_heads) % 2 == 0, (
            f"head_dim ({embed_dim // num_heads}) must be even for RoPE"
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ff_dim = ff_dim if ff_dim is not None else embed_dim * 4
        self.resnet = resnet
        self.dropout_rate = dropout_rate
        self.rope_max_len = rope_max_len
        self.rope_theta = rope_theta
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.sep_token = sep_token
        self.normal_token = normal_token

    # ---- build -----------------------------------------------------
    def build(self, input_shape):
        # Q / K / V projections  (all project to full embed_dim then reshape)
        self.wq = layers.Dense(self.embed_dim, use_bias=False, name="W_q")
        self.wk = layers.Dense(self.embed_dim, use_bias=False, name="W_k")
        self.wv = layers.Dense(self.embed_dim, use_bias=False, name="W_v")

        # gate projection  →  one scalar per head
        self.wg = layers.Dense(self.num_heads, use_bias=True, name="W_gate")

        # output projection  (head_dim → embed_dim after gated mean-pool)
        self.out_proj = layers.Dense(self.embed_dim, use_bias=True, name="out_proj")

        # feed-forward network
        self.ffn_dense1 = layers.Dense(self.ff_dim, activation="gelu", name="ffn_1")
        self.ffn_dense2 = layers.Dense(self.embed_dim, name="ffn_2")
        self.ffn_drop1 = layers.Dropout(self.dropout_rate)
        self.ffn_drop2 = layers.Dropout(self.dropout_rate)

        # layer norms  (pre-norm style)
        self.ln_attn = layers.LayerNormalization(name="ln_attn")
        self.ln_ffn = layers.LayerNormalization(name="ln_ffn")

        # attention dropout
        self.attn_dropout = layers.Dropout(self.dropout_rate)

        # precompute RoPE sin/cos tables (not trainable)
        sin_t, cos_t = _rope_frequencies(
            self.head_dim, self.rope_max_len, self.rope_theta
        )
        self.rope_sin = tf.Variable(sin_t, trainable=False, name="rope_sin")
        self.rope_cos = tf.Variable(cos_t, trainable=False, name="rope_cos")

        super().build(input_shape)

    # ---- helpers ---------------------------------------------------
    def _split_heads(self, x):
        """(B, L, embed_dim) → (B, num_heads, L, head_dim)"""
        B = tf.shape(x)[0]
        L = tf.shape(x)[1]
        x = tf.reshape(x, [B, L, self.num_heads, self.head_dim])
        return tf.transpose(x, [0, 2, 1, 3])

    # ---- forward ---------------------------------------------------
    def call(self, inputs, training=None):
        x, mask = inputs  # (B, L, E), (B, L)
        mask = tf.cast(mask, tf.int32)

        residual = x

        # ---------- self-attention with gated head combination ------
        x_norm = self.ln_attn(x)

        Q = self._split_heads(self.wq(x_norm))  # (B, H, L, d_h)
        K = self._split_heads(self.wk(x_norm))
        V = self._split_heads(self.wv(x_norm))

        # apply Rotary Position Embeddings to Q and K
        Q = _apply_rope(Q, self.rope_sin, self.rope_cos)
        K = _apply_rope(K, self.rope_sin, self.rope_cos)

        # scaled dot-product attention
        scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        scores = tf.matmul(Q, K, transpose_b=True) / scale  # (B, H, L, L)

        # mask out PAD keys  →  -1e9 so softmax ≈ 0
        pad_is_key = tf.equal(mask, self.pad_token)            # (B, L) bool
        key_mask = tf.cast(
            pad_is_key[:, tf.newaxis, tf.newaxis, :], tf.float32
        )                                                      # (B,1,1,L)
        scores = scores + key_mask * -1e9

        # also prevent PAD queries from producing non-zero output
        pad_is_query = tf.cast(
            pad_is_key[:, tf.newaxis, :, tf.newaxis], tf.float32
        )                                                      # (B,1,L,1)
        scores = scores + pad_is_query * -1e9

        attn_weights = tf.nn.softmax(scores, axis=-1)          # (B, H, L, L)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        head_outputs = tf.matmul(attn_weights, V)              # (B, H, L, d_h)

        # ---------- gating ------------------------------------------
        gate = tf.nn.sigmoid(self.wg(x_norm))                  # (B, L, H)
        gate = tf.transpose(gate, [0, 2, 1])                   # (B, H, L)
        gate = gate[:, :, :, tf.newaxis]                       # (B, H, L, 1)

        gated = head_outputs * gate                            # (B, H, L, d_h)
        pooled = tf.reduce_mean(gated, axis=1)                 # (B, L, d_h)

        attn_out = self.out_proj(pooled)                       # (B, L, E)

        # ---------- residual 1 --------------------------------------
        if self.resnet:
            x = residual + attn_out
        else:
            x = attn_out

        # ---------- feed-forward with pre-norm ----------------------
        ff_residual = x
        x_norm2 = self.ln_ffn(x)
        ff = self.ffn_dense1(x_norm2)
        ff = self.ffn_drop1(ff, training=training)
        ff = self.ffn_dense2(ff)
        ff = self.ffn_drop2(ff, training=training)

        if self.resnet:
            x = ff_residual + ff
        else:
            x = ff

        # ---------- zero-out PAD positions --------------------------
        pad_keep = tf.cast(
            tf.not_equal(mask, self.pad_token), tf.float32
        )[:, :, tf.newaxis]
        x = x * pad_keep

        return x  # (B, L, embed_dim)

    # ---- config (serialisation) ------------------------------------
    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "resnet": self.resnet,
                "dropout_rate": self.dropout_rate,
                "rope_max_len": self.rope_max_len,
                "rope_theta": self.rope_theta,
                "pad_token": self.pad_token,
                "mask_token": self.mask_token,
                "sep_token": self.sep_token,
                "normal_token": self.normal_token,
            }
        )
        return cfg