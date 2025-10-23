import keras
import tensorflow as tf
import numpy as np
from typing import List, Tuple, Union


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
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class DonorFileManager:
    def __init__(self, 
                 donor_seqid_path: None | Union[str, List[str]], # (n_donor, padded_num_seq_ids_indexes) e.g (1000, 200k)
                 donor_mhc_path: None | Union[str, List[str]], # (n_donor, n_mhc) binary)
    ):
        self.donor_seqid_path = donor_seqid_path
        self.donor_mhc_path = donor_mhc_path


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
        tcr_seq: np.ndarray, 
        tcr_ids: Union[np.ndarray, int], 
        tcr_donor_ids: np.ndarray
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
        # Ensure tcr_ids is an integer
        if isinstance(tcr_ids, np.ndarray):
            tcr_ids = int(tcr_ids.item())
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
        tcr_seq: np.ndarray, 
        tcr_ids: np.ndarray, 
        tcr_donor_ids: np.ndarray
    ):
        """
        Write multiple TCR sequences, IDs, and donor IDs to a TFRecord file.
        Each TCR sequence and its corresponding donor IDs can have variable lengths.
        The method removes padding tokens before writing to save space.
        Args:
            tcr_seq: Array of TCR sequences, shape (num_seqs, max_seq_len).
                    Each sequence is padded with pad_token to max_seq_len.
            tcr_ids: Array of integer IDs, shape (num_seqs,).
            tcr_donor_ids: Array or list of variable-length donor ID arrays.
                          Can be a 2D padded array (num_seqs, max_donors) or list of arrays.
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
        num_seqs = tcr_seq.shape[0]
        assert tcr_seq.ndim == 2, f"tcr_seq must be 2D array, got shape {tcr_seq.shape}"
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
                seq = tcr_seq[i]
                # Find actual sequence length (before padding)
                actual_seq = seq[seq != self.pad_token]
                
                # Get donor IDs and remove padding
                if isinstance(tcr_donor_ids, np.ndarray):
                    donor_ids = tcr_donor_ids[i]
                    # Remove padding from donor_ids
                    actual_donor_ids = donor_ids[donor_ids != self.pad_token]
                else:
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
    def calculateLL(self, gamma, q, gamma_donor_id):
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