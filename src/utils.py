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
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value if isinstance(value, (list, np.ndarray)) else [value]))


class DonorFileManager:
    def __init__(self, 
                 donor_seqid_path: None | Union[str, List[str]], # (n_donor, padded_num_seq_ids_indexes) e.g (1000, 200k)
                 donor_mhc_path: None | Union[str, List[str]], # (n_donor, n_mhc) binary)
    ):
        self.donor_seqid_path = donor_seqid_path
        self.donor_mhc_path = donor_mhc_path
    


class TCRFileManager:
    """
    A class to manage TCR (T-cell receptor) feature datasets stored as TFRecords.

    This class supports:
        - Writing padded TCR features and IDs to TFRecord files
        - Reading TFRecord files into tf.data.Dataset pipelines
        - Parallel reading, shuffling, batching, and prefetching
        - Distributed dataset support for multi-GPU or multi-worker training

    Attributes:
        tcr_path (Union[str, List[str]]): Path(s) to TFRecord file(s).
        tcr_length (int): Length of TCR sequences (all sequences are padded to this length).
        feature_dim (int): Dimensionality of TCR features per residue.
        batch_size (int): Number of sequences per batch in the dataset.
        shuffle_buffer_size (int): Buffer size for shuffling the dataset.
    """
    
    def __init__(self, tcr_path: Union[str, List[str]], #(num_seqs, length, dim), (num_seqs, id)
                 tcr_length: int = 25,  # sequences are already padded to this number
                 feature_dim: int = 20, batch_size: int = 32, shuffle_buffer_size: int = 1000):
        self.tcr_path = tcr_path
        self.tcr_length = tcr_length
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
    
    @staticmethod
    def _tcr_serialize(tcr_features: np.ndarray, tcr_ids: np.ndarray) -> bytes:
        """
        Serialize a single TCR feature matrix and its ID into a TFRecord Example.

        Args:
            tcr_features (np.ndarray): TCR feature matrix of shape (tcr_length, feature_dim).
            tcr_ids (np.ndarray or int): Single integer ID corresponding to the TCR sequence.

        Returns:
            bytes: Serialized TFRecord Example.
        """
        tcr_features_flat = tcr_features.astype(np.float32).flatten() #(length, dim) -> (length*dim)
        
        tcr_data = {
            "tcr_features": _float_feature(tcr_features_flat), #(length*dim)
            "tcr_ids": _int64_feature(tcr_ids), #[int]
        }
        tcr_example = tf.train.Example(features=tf.train.Features(feature=tcr_data)).SerializeToString()
        return tcr_example

    def write_tcr_samples(self, tcr_features: np.ndarray, tcr_ids: np.ndarray):
        """
        Write multiple TCR feature matrices and IDs to a TFRecord file.

        Args:
            tcr_features (np.ndarray): Array of shape (num_seqs, tcr_length, feature_dim).
            tcr_ids (np.ndarray): Array of integer IDs of shape (num_seqs,).

        Raises:
            AssertionError: If input shapes do not match expected dimensions.
        """
        assert tcr_features.shape[-1] == self.feature_dim, f"tcr_feature dimension is not correct, expected (N, {self.tcr_length}, {self.feature_dim}), received {tcr_features.shape}"
        assert tcr_features.shape[1] == self.tcr_length, f"tcr_feature dimension is not correct, expected (N, {self.tcr_length}, {self.feature_dim}), received {tcr_features.shape}"
        assert tcr_features.shape[0] == tcr_ids.shape[0], f"feature missmatch: tcr_features {tcr_features.shape[0]} != tcr_ids {tcr_ids.shape[0]}"
        assert isinstance(self.tcr_path, str), f'to write a file, you need tcr_path to be a file path, not a list. found: {self.tcr_path}'

        T = tcr_features.shape[0]
        with tf.io.TFRecordWriter(self.tcr_path) as writer:
            for i in range(T):
                writer.write(self._tcr_serialize(tcr_features[i], tcr_ids[i]))
    
    def _tcr_parse(self, example_proto: tf.Tensor) -> tuple:
        """
        Parse a single TFRecord Example into a TCR feature matrix and ID.

        Args:
            example_proto (tf.Tensor): Serialized TFRecord Example.

        Returns:
            tuple:
                tcr_features (tf.Tensor): Tensor of shape (tcr_length, feature_dim).
                tcr_ids (tf.Tensor): Scalar tensor containing the TCR ID.
        """
        # Flattened length * dim
        total_len = self.feature_dim * self.tcr_length  # fixed-length flattened array
        desc = {
            "tcr_features": tf.io.FixedLenFeature([total_len], tf.float32),  # fixed-length flattened array
            "tcr_ids": tf.io.FixedLenFeature([], tf.int64),                  # single int
        }
        parsed = tf.io.parse_single_example(example_proto, desc)
        # Reshape to (S, D)
        tcr_features = tf.reshape(parsed["tcr_features"], [self.tcr_length, self.feature_dim])
        tcr_ids = tf.cast(parsed["tcr_ids"], tf.int32)
        return tcr_features, tcr_ids
    
    @staticmethod
    @tf.autograph.experimental.do_not_convert
    def parse_file(f):
        """
        Parse a TFRecord file into a TFRecordDataset.

        Args:
            f (str): Path to a TFRecord file.

        Returns:
            tf.data.TFRecordDataset
        """
        return tf.data.TFRecordDataset(f, buffer_size=16*1024*1024)

    def read_dataset(self, num_parallel_reads: int = tf.data.AUTOTUNE, shuffle : bool = True) -> tf.data.Dataset:
        """
        Build a tf.data.Dataset that:
            - Supports single or multiple TFRecord files
            - Shuffles files & records
            - Reads files in parallel
            - Batches & prefetches

        Args:
            num_parallel_reads (int): Number of files to read in parallel.
            shuffle (bool): Whether to shuffle files and records.

        Returns:
            tf.data.Dataset: Yields (tcr_features, tcr_ids) tuples.
        """
        paths = [self.tcr_path] if isinstance(self.tcr_path, str) else self.tcr_path
        files_ds = tf.data.Dataset.list_files(paths, shuffle=shuffle)
        ds = files_ds.interleave(self.parse_file,
                                 cycle_length=num_parallel_reads,
                                 num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(self.shuffle_buffer_size)
        ds = ds.map(self._tcr_parse, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def get_distributed_dataset(self, strategy: tf.distribute.Strategy) -> tf.data.Dataset:
        """
        Wrap the dataset with a tf.distribute.Strategy for multi-worker or multi-GPU training.

        Args:
            strategy (tf.distribute.Strategy): TensorFlow distribution strategy.

        Returns:
            tf.data.Dataset: Distributed dataset.
        """
        ds = self.read_dataset()
        return strategy.experimental_distribute_dataset(ds)



class Likelihood(keras.losses.Loss):
    def __init__(self, donor_mhc: tf.Tensor, memory_efficient: bool = False, 
                 pad_token = -1, **kwargs):
        super().__init__()
        self.donor_mhc = tf.Variable(donor_mhc) #(N, A)
        self.memory_efficient = memory_efficient
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
        second_term = Ni_size * tf.reduce_sum(self.Na[tf.newaxis, :] * gamma, axis=-1) #(B,) * Sum^A ( (B, A) * (B, A)) --> (B,)

        #### Calculate The first Term ####
        # Sum^Ni (ln( qi*pni / 1 - qi*pni))
        ## calculate pni: 1 - Prod( 1 - gamma_ia) ** x_na
        pn = self.calculate_pni(Ni, gamma, gamma_donor_id_mask) # (B, N_i)
        numerator = tf.multiply(q[:, tf.newaxis], pn) # (B,1) * (B, N_i)
        denominator = 1. - numerator
        first_term = tf.math.log(numerator / denominator)
        first_term = tf.reduce_sum(first_term, axis=-1) #(B, N_i) --> (B,)
        
        LL_batch = first_term - second_term
        return tf.reduce_sum(LL_batch)

    def calculate_Ni_Nisize(self, gamma_donor_id): #(B, max)
        # calculate count of donors per each tcr
        gamma_donor_id_count = tf.where(gamma_donor_id == self.pad_token, 0., 1.) #(B, pad_donor_id)
        Ni_size = tf.reduce_sum(gamma_donor_id_count, axis=-1) #(B,) each has the number of Ni 
        ## extract (Ni, A) from (N, A)
        # First, we need to create a masking for padded tokens
        gamma_donor_id_converted = tf.where(gamma_donor_id == self.pad_token, 0., gamma_donor_id) # just to make sure tf.gather does not raise error
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
        pni = 1. - tf.reduce_prod(output, axis=-1) #(B, N_i)
        # apply mask
        pni = pni * gamma_donor_id_mask #(B, N_i)
        return pni