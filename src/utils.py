import keras
import tensorflow as tf
import numpy as np
from typing import List, Tuple, Union
from src.constants import AMINO_ACID_IDX, AMINO_ACIDS, PHYSICHE_STOCKHOLM, PHYSICHE_STOCKHOLM_AA_to_IDX, PHYSICHE_STOCKHOLM_IDX_to_ENCODE
import pandas as pd
import os
import subprocess
from pathlib import Path
import shutil
import re
import json



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
                 pad_token: int = -1,
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
            
            # Read TCR data
            tcr_df = self.read_tcr_df(tcr_path, cdr3_col, cdr2_col, cdr2_5_col, cdr1_col)
            
            # Read MHC array
            mhc_arr, mhc_state = self.read_mhc_arr(mhc_path)
            if mhc_state == False:
                patients_faulty.append(row._asdict())
                continue
            patients_processed.append(row._asdict())
            mhc_arrays.append(mhc_arr)
            patient_ids.append(patient_id)
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
            
            all_tcr_dfs.append(tcr_df)
        
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
        pad_token: int = -1
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
            for i in range(num_seqs):
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
        padded_tcr_seq = tcr_seq.to_tensor(default_value=self.pad_token)
        
        # Pad tcr_donor_ids to maximum length in the batch
        padded_tcr_donor_ids = tcr_donor_ids.to_tensor(default_value=self.pad_token)
        
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
        
        # Parse examples
        dataset = dataset.map(
            self._tcr_parse, 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        
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



class Likelihood(keras.losses.Loss):
    def __init__(self, donor_mhc: tf.Tensor, 
                 pad_token = -1, **kwargs):
        super().__init__()
        self.donor_mhc = tf.constant(donor_mhc) #(N, A)
        self.pad_token = pad_token
        assert len(tf.shape(self.donor_mhc)) == 2, f'the shape of donor_mhc should be (donor, mhc_allele), found {tf.shape(donor_mhc)}'
        self.Na = tf.reduce_sum(self.donor_mhc, axis=0) #(A,)

    def call(self, gamma, q, gamma_donor_id):
        '''
        gamma: output of model. dimension (batch, mhc_allele) or (B,A)
        q: output of model. dimension (batch,)
        gamma_donor_id: padded integers of donor ids per each tcr. dimension (batch, padded_donor_id) or (B, D_i). map tcr to donors.
        '''
        #### Calculate The second Term ####
        # |Ni_size| * Sum^A( Na * gamma_ia )
        Ni_size, Ni, gamma_donor_id_mask = self.calculate_Ni_Nisize(gamma_donor_id) #(B,) and (B, N_i, A) and (B, N_i)
        second_term = q * Ni_size * tf.reduce_sum(self.Na[tf.newaxis, :] * gamma, axis=-1) #(B,) * (B,) * Sum^A ( (B, A) * (B, A)) --> (B,)

        #### Calculate The first Term ####
        # Sum^Ni (ln( qi*pni / 1 - qi*pni))
        ## calculate pni: 1 - Prod( 1 - gamma_ia) ** x_na
        pn = self.calculate_pni(Ni, gamma, gamma_donor_id_mask) # (B, N_i)
        numerator = tf.multiply(q[:, tf.newaxis], pn) # (B,1) * (B, N_i)
        denominator = 1. - numerator
        first_term = tf.math.log(numerator + 1e-10) - tf.math.log(denominator + 1e-10)
        # apply mask, because padded ones are now log(0) == 1 
        first_term = first_term * gamma_donor_id_mask
        first_term = tf.reduce_sum(first_term, axis=-1) #(B, N_i) --> (B,)
        
        LL_batch = first_term - second_term
        return -tf.reduce_sum(LL_batch) 

    def calculate_Ni_Nisize(self, gamma_donor_id): #(B, max)
        # calculate count of donors per each tcr
        gamma_donor_id_count = tf.where(gamma_donor_id == self.pad_token, 0., 1.) #(B, pad_donor_id)
        Ni_size = tf.reduce_sum(gamma_donor_id_count, axis=-1) #(B,) each has the number of Ni 
        ## extract (Ni, A) from (N, A)
        # First, we need to create a masking for padded tokens
        gamma_donor_id_converted = tf.where(gamma_donor_id == self.pad_token, 0, gamma_donor_id) # just to make sure tf.gather does not raise error
        gamma_donor_id_converted = tf.cast(gamma_donor_id_converted, dtype=tf.int32)
        gamma_donor_id_mask = tf.where(gamma_donor_id == self.pad_token, 0., 1) #(B,D_i) or (B,N_i)
        Ni = tf.gather(self.donor_mhc, gamma_donor_id_converted, axis=0) # (N, A), (B,D_i) --> (B, N_i, A) N_i are simply gathered D_is , D_i is index and is padded. len(N_i) == len(D_i)
        Ni = tf.multiply(Ni, gamma_donor_id_mask[:, :, tf.newaxis]) #(B,N_i,A) masked out
        return Ni_size, Ni, gamma_donor_id_mask

    def calculate_pni(self, Ni, gamma, gamma_donor_id_mask): #(B,N_i,A) and (B,A)
        # pni = 1 - Prod (1 - gamma_ia) ^ xna
        # Ni has only 0 and 1 now. Also, the N_i dim is padded to maximum number of donors for a tcr.
        # gamma should be expanded from (B,A) to (B, N_i, A)
        gamma_expanded = tf.expand_dims(gamma, axis=1) #(B, 1, A)
        gamma_expanded = tf.broadcast_to(gamma_expanded, shape=tf.shape(Ni)) #(B, N_i, A)
        # output = (1 - gamma)^ xna
        output = tf.pow(1. - gamma_expanded, Ni) #(B, N_i, A)
        # 1. - Prod(output)
        #pni = 1. - tf.reduce_prod(output, axis=-1) #(B, N_i)
        log_prod = tf.reduce_sum(tf.math.log(output + 1e-10), axis=-1)
        pni = 1. - tf.exp(log_prod)
        # apply mask
        pni = pni * gamma_donor_id_mask #(B, N_i)
        return pni
    



class TCRAlignmentPipeline:
    """Pipeline for aligning TCR sequences using Profile HMM"""
    
    def __init__(self, output_dir: str = "tcr_alignment_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir = None
        
    def check_dependencies(self) -> bool:
        """Check if required tools are installed"""
        required_tools = {
            'mmseqs': 'MMseqs2',
            'mafft': 'MAFFT',
            'hmmbuild': 'HMMER (hmmbuild)',
            'hmmalign': 'HMMER (hmmalign)'
        }
        
        missing = []
        for tool, name in required_tools.items():
            if shutil.which(tool) is None:
                missing.append(name)
        
        if missing:
            print("ERROR: Missing required tools:")
            for tool in missing:
                print(f"  - {tool}")
            print("\nInstall with: conda install -c bioconda hmmer mafft mmseqs2")
            return False
        
        print("✓ All required tools found")
        return True
    
    def read_sequences_from_csv(self, csv_path: str, column_name: str) -> pd.DataFrame:
        """Read sequences from CSV file"""
        print(f"\n[1/6] Reading sequences from {csv_path}...")
        
        df = pd.read_csv(csv_path)
        
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found. Available columns: {list(df.columns)}")
        
        # Remove duplicates and empty sequences
        df = df[df[column_name].notna()].copy()
        df[column_name] = df[column_name].str.strip()
        df = df[df[column_name] != '']
        
        # Add sequence ID if not present
        if 'seq_id' not in df.columns:
            df['seq_id'] = [f"seq_{i}" for i in range(len(df))]
        
        print(f"  Found {len(df)} valid sequences")
        df[column_name] = convert_protein_sequences_fast(df[column_name].tolist())
        return df
    
    def write_fasta(self, df: pd.DataFrame, column_name: str, output_path: Path, id_column: str = 'tcr_id') -> None:
        """Write sequences to FASTA format"""
        with open(output_path, 'w') as f:
            for idx, row in df.iterrows():
                seq_id = 'seq_' + str(row[id_column])
                sequence = row[column_name]
                f.write(f">{seq_id}\n{sequence}\n")
    
    def cluster_sequences(self, input_fasta: Path, min_seq_id: float = 0.5, min_seq_cov=0.8, 
                        sensitivity=7.5, gap_open=10, gap_extend=1, 
                        alphabet_size=21, cluster_mode=0) -> Path:
        """Cluster sequences using MMseqs2 to get redundancy-reduced set"""
        print(f"\n[2/6] Clustering sequences (min identity: {min_seq_id}), min coverage {min_seq_cov}")
        cluster_prefix = self.output_dir / "cluster"
        tmp_dir = self.output_dir / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        
        cmd = [
            'mmseqs', 'easy-cluster',
            str(input_fasta),
            str(cluster_prefix),
            str(tmp_dir),
            '--min-seq-id', str(min_seq_id),
            '-c', str(min_seq_cov),
            '--cov-mode', '1',  # CHANGED: coverage of target (more lenient for short seqs)
            '-s', str(sensitivity),  # Max sensitivity for short sequences
            '--gap-open', str(gap_open),  # Lower gap penalties
            '--gap-extend', str(gap_extend),
            '--alph-size', str(alphabet_size),
            '--cluster-mode', str(cluster_mode),  # 0=SetCover (greedy), 2=Connected component
            '--max-seqs', '1000',  # Consider more sequences per query
            '-e', '100',  # IMPORTANT: Very permissive E-value for short sequences
            '--alignment-mode', '3',  # 3=ungapped alignment can work better for very short seqs
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  Clustering complete")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: MMseqs2 clustering failed")
            print(e.stderr)
            raise
        
        rep_seq_file = self.output_dir / "cluster_rep_seq.fasta"
        if not rep_seq_file.exists():
            raise FileNotFoundError(f"Expected output {rep_seq_file} not found")
        
        with open(rep_seq_file) as f:
            n_reps = sum(1 for line in f if line.startswith('>'))
        
        total_seqs = sum(1 for line in open(input_fasta) if line.startswith('>'))
        print(f"  {n_reps} representative sequences from {total_seqs} total ({100*n_reps/total_seqs:.1f}% retained)")
        
        return rep_seq_file
    
    def create_msa(self, input_fasta: Path) -> Path:
        """Create Multiple Sequence Alignment using MAFFT"""
        print("\n[3/6] Creating Multiple Sequence Alignment...")
        
        output_msa = self.output_dir / "msa_alignment.afa"
        
        cmd = [
            'mafft',
            '--retree', '1',
            '--quiet',
            '--op', '11.0',
            '--ep', '10.0',
            '--thread', '-1',
            str(input_fasta)]
        
        try:
            with open(output_msa, 'w') as f:
                result = subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.PIPE, text=True)
            print(f"  MSA created: {output_msa}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: MAFFT alignment failed")
            print(e.stderr)
            raise
        
        return output_msa
    
    def build_profile_hmm(self, msa_file: Path) -> Path:
        """Build Profile HMM from MSA using HMMER"""
        print("\n[4/6] Building Profile HMM...")
        
        hmm_file = self.output_dir / "tcr_profile.hmm"
        
        cmd = ['hmmbuild', '--amino', str(hmm_file), str(msa_file)]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  Profile HMM built: {hmm_file}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: hmmbuild failed")
            print(e.stderr)
            raise
        
        return hmm_file
    
    def align_to_profile(self, hmm_file: Path, sequences_fasta: Path) -> Path:
        """Align all sequences to the profile HMM"""
        print("\n[5/6] Aligning all sequences to Profile HMM...")
        
        aligned_file = self.output_dir / "aligned_sequences.sto"
        
        cmd = ['hmmalign', '-o', str(aligned_file), str(hmm_file), str(sequences_fasta)]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  Alignment complete: {aligned_file}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: hmmalign failed")
            print(e.stderr)
            raise
        
        return aligned_file
    
    def parse_stockholm_alignment(self, stockholm_file: Path) -> Tuple[List[str], List[str]]:
        """Parse Stockholm format alignment file"""
        print("\n[6/6] Parsing alignment...")
        
        sequences = {}
        
        with open(stockholm_file) as f:
            for line in f:
                line = line.strip()
                # Skip comments and markup
                if line.startswith('#') or line.startswith('//') or not line:
                    continue
                
                parts = line.split()
                if len(parts) == 2:
                    seq_id, seq = parts
                    if seq_id in sequences:
                        sequences[seq_id] += seq
                    else:
                        sequences[seq_id] = seq
        
        seq_ids = list(sequences.keys())
        aligned_seqs = list(sequences.values())
        
        print(f"  Parsed {len(seq_ids)} aligned sequences")
        if aligned_seqs:
            print(f"  Alignment length: {len(aligned_seqs[0])} positions")
        
        return seq_ids, aligned_seqs
    
    def save_aligned_sequences(self, seq_ids: List[str], aligned_seqs: List[str]) -> Path:
        """Save aligned sequences in FASTA format"""
        output_file = self.output_dir / "aligned_sequences.fasta"
        
        with open(output_file, 'w') as f:
            for seq_id, seq in zip(seq_ids, aligned_seqs):
                f.write(f">{seq_id}\n{seq}\n")
        
        print(f"\n✓ Aligned sequences saved: {output_file}")
        return output_file
    
    def prepare_data_from_aln(self, seq_ids, aligned_seqs, tcr_df_donor):
        """
        Process aligned sequences and donor information, then save the results as JSON.
        Outputs of parse_stockholm_alignment and read_sequences_from_csv.
        """
        CDRS = []
        TCR_ID = []
        DONOR_IDS = []
        sep = PHYSICHE_STOCKHOLM_AA_to_IDX[';']
        for seq_id, seq in zip(seq_ids, aligned_seqs):
            cdr3 = []
            tcr_id = int(seq_id.split('_')[-1])
            for s in seq:
                numeric = PHYSICHE_STOCKHOLM_AA_to_IDX[s]
                cdr3.append(numeric)  # (S,)
            row = tcr_df_donor[tcr_df_donor['tcr_id'] == tcr_id]
            if row.empty:
                print(f"Warning: No data found for tcr_id {tcr_id}")
                continue
            cdr1_str = str(row['cdr1'].iloc[0])
            cdr2_str = str(row['cdr2'].iloc[0])
            donor_id_val = str(row['id'].iloc[0])
            donor_id_val = [int(i) for i in donor_id_val.split(';')]
            if isinstance(cdr1_str, str):
                cdr1 = [int(x) for x in cdr1_str.split(';') if x.strip()]
            else:
                cdr1 = []
            if isinstance(cdr2_str, str):
                cdr2 = [int(x) for x in cdr2_str.split(';') if x.strip()]
            else:
                cdr2 = []
            # Concatenate CDR1 + separator + CDR2 + separator + CDR3
            cdrs = cdr1 + [sep] + cdr2 + [sep] + cdr3
            CDRS.append(cdrs)
            TCR_ID.append(tcr_id)
            DONOR_IDS.append(donor_id_val)
        # --- Save outputs as JSON ---
        output_path = Path(self.output_dir) / "aligned_tcr_data.json"
        output_data = {
            "CDRS": CDRS,
            "TCR_ID": TCR_ID,
            "DONOR_IDS": DONOR_IDS
        }
        # Convert NumPy types (if any) to Python native types before dumping
        def to_native(obj):
            if hasattr(obj, "tolist"):
                return obj.tolist()
            return obj
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4, default=to_native)
        print(f"Saved aligned data to {output_path}")
        return CDRS, TCR_ID, DONOR_IDS

    def run_pipeline(self, csv_path: str, column_name: str, min_seq_id: float = 0.9):
        """Run the complete alignment pipeline"""
        print("="*70)
        print("TCR Sequence Alignment Pipeline")
        print("="*70)
        # Check dependencies
        if not self.check_dependencies():
            return None
        try:
            # Step 1: Read sequences
            df = self.read_sequences_from_csv(csv_path, column_name)
            
            # Write to FASTA
            input_fasta = self.output_dir / "input_sequences.fasta"
            self.write_fasta(df, column_name, input_fasta)
            
            # Step 2: Cluster sequences
            rep_fasta = self.cluster_sequences(input_fasta, min_seq_id)
            
            # Step 3: Create MSA
            msa_file = self.create_msa(rep_fasta)
            
            # Step 4: Build Profile HMM
            hmm_file = self.build_profile_hmm(msa_file)
            
            # Step 5: Align all sequences
            aligned_stockholm = self.align_to_profile(hmm_file, input_fasta)
            
            # Step 6: Parse and save
            seq_ids, aligned_seqs = self.parse_stockholm_alignment(aligned_stockholm)
            output_fasta = self.save_aligned_sequences(seq_ids, aligned_seqs)
            
            print("\n" + "="*70)
            print("Pipeline complete!")
            print("="*70)
            print(f"\nOutput files in: {self.output_dir}/")
            print(f"  - Profile HMM: {hmm_file.name}")
            print(f"  - Aligned sequences (Stockholm): {aligned_stockholm.name}")
            print(f"  - Aligned sequences (FASTA): {output_fasta.name}")
            
            return {
                'hmm_file': hmm_file,
                'aligned_fasta': output_fasta,
                'aligned_stockholm': aligned_stockholm,
                'seq_ids': seq_ids,
                'aligned_seqs': aligned_seqs
            }
            
        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None


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
