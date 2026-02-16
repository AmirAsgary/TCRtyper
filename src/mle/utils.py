"""
Utility functions and model definitions for TCR-HLA binding analysis.

Requirements:
    - tensorflow>=2.10
    - tensorflow_probability
    - numpy
    - matplotlib
    - seaborn
    - pandas (for precision@k heatmaps)

Author: TCR-HLA Binding Analysis Pipeline
"""
from __future__ import annotations
import os, json
import numpy as np
import tensorflow as tf
#import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence
from dataclasses import dataclass, field


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




# --- MODEL DEFINITION ---
class SparseTCRModel(tf.keras.Model):
    """TCR Binding Model using sparse representation with likelihood maximization.
    
    Parameters
    ----------
    accumulation_steps : int
        Number of mini-batches to accumulate gradients over before applying.
        Effective batch = batch_size * accumulation_steps.
        Set to 1 for no accumulation (default).
    reduction : str
        'sum' or 'mean'. Controls how per-TCR losses are aggregated.
        'sum' (recommended): gradient per z_i is independent of batch_size.
        'mean': gradient per z_i scales as 1/batch_size (original behavior).
    """
    def __init__(self, num_tcrs, max_hlas_per_tcr, donor_hla_matrix, binder_sets, 
                 beta=4.0, mode='continuous', pad_token=-1., l2_reg_lambda=1e-5,
                 accumulation_steps=1, reduction='sum'):
        super().__init__()
        self.beta = beta
        self.mode = mode
        self.pad_token = pad_token
        self.l2_reg_lambda = l2_reg_lambda
        self.accumulation_steps = accumulation_steps
        self.reduction = reduction
        # Store donor matrix transposed: (NumAlleles, NumDonors)
        self.X_T = tf.constant(donor_hla_matrix.T, dtype=tf.float32)
        # Store binder_sets with pad replaced by 0 for gathering
        self.binder_sets = tf.constant(np.maximum(binder_sets, 0), dtype=tf.int32)
        self.binder_mask = tf.constant(binder_sets != pad_token, dtype=tf.float32)
        # Embedding layer for z parameters
        self.z_embedding = tf.keras.layers.Embedding(
            input_dim=num_tcrs, output_dim=max_hlas_per_tcr,
            embeddings_initializer=tf.keras.initializers.RandomNormal(mean=-1.25, stddev=0.75),
            name="z_values")
        # Metric trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.final_loss_tracker = tf.keras.metrics.Mean(name="final_loss")
        self.reg_tracker = tf.keras.metrics.Mean(name="reg_term")
        # Gradient accumulation state (built lazily on first train_step)
        self._grad_accumulators = None
        self._accum_step = None

    def l2_reg(self, z_logits, mask):
        """L2 regularization normalized by number of valid positions."""
        if self.l2_reg_lambda:
            # Normalize by valid entries so reg strength is independent of
            # batch_size, chunk_size, and max_hlas padding width
            n_valid = tf.reduce_sum(mask)
            n_valid = tf.maximum(n_valid, 1.0)  # avoid division by zero
            return self.l2_reg_lambda * tf.reduce_sum(tf.pow(z_logits, 2) * mask) / n_valid
        return 0.

    def get_z_probabilities(self, z_logits, mask):
        """Toggle between continuous relaxation and binary sampling."""
        return tf.sigmoid(z_logits) * mask

    def call(self, inputs):
        """Compute negative log-likelihood loss for a batch of TCRs."""
        tcr_idx, pos_donor_indices = inputs
        batch_binder_indices = tf.gather(self.binder_sets, tcr_idx)
        batch_mask = tf.gather(self.binder_mask, tcr_idx)
        z_logits = self.z_embedding(tcr_idx)
        # Compute p_ni (Prob TCR i binds in Donor n)
        relevant_x = tf.gather(self.X_T, batch_binder_indices)
        z_prob = self.get_z_probabilities(z_logits, batch_mask)
        z_prob_expanded = tf.expand_dims(z_prob, axis=-1)
        regularization_term = self.l2_reg(z_logits, batch_mask)
        term_raw = 1.0 - (relevant_x * z_prob_expanded)
        term_safe = tf.maximum(term_raw, 1e-7)
        log_prod = tf.reduce_sum(tf.math.log(term_safe), axis=1)
        p_ni = 1.0 - tf.exp(log_prod)
        p_ni_safe = tf.maximum(p_ni, 1e-7)
        # Positive donors (Reward)
        safe_pos_indices = tf.maximum(pos_donor_indices, 0)
        pos_mask = tf.cast(tf.not_equal(pos_donor_indices, tf.cast(self.pad_token, tf.int32)), tf.float32)
        p_pos = tf.gather(p_ni_safe, safe_pos_indices, batch_dims=1)
        reward = tf.reduce_sum(tf.math.log(p_pos) * pos_mask, axis=1)
        # Negative donors (Penalty via Beta-Binomial)
        n_i = tf.reduce_sum(pos_mask, axis=1)
        sum_p_all = tf.reduce_sum(p_ni_safe, axis=1)
        sum_p_pos = tf.reduce_sum(p_pos * pos_mask, axis=1)
        n_tilde = sum_p_all - sum_p_pos
        penalty = tf.math.lgamma(n_tilde + self.beta) - tf.math.lgamma(n_i + n_tilde + self.beta + 1.0)
        log_likelihood = reward + penalty
        # Reduction: sum makes gradient per z_i independent of batch_size
        if self.reduction == 'sum':
            nll = -tf.reduce_sum(log_likelihood)
        else:
            nll = -tf.reduce_mean(log_likelihood)
        return nll, regularization_term

    @property
    def metrics(self):
        return [self.loss_tracker, self.final_loss_tracker, self.reg_tracker]

    def _ensure_accumulators(self):
            """Create gradient accumulator variables on the same device as the model."""
            if self._grad_accumulators is None:
                gpus = tf.config.list_logical_devices("GPU")
                device = gpus[0].name if gpus else "/CPU:0"
                with tf.device(device):
                    self._grad_accumulators = [
                        tf.Variable(tf.zeros_like(v), trainable=False, name=f"grad_acc_{i}")
                        for i, v in enumerate(self.trainable_variables)
                    ]
                    self._accum_step = tf.Variable(0, trainable=False, dtype=tf.int32)

    def _apply_and_reset(self):
        """Apply accumulated gradients and reset accumulators."""
        self.optimizer.apply_gradients(
            zip(self._grad_accumulators, self.trainable_variables)
        )
        for acc in self._grad_accumulators:
            acc.assign(tf.zeros_like(acc))
        return tf.constant(0.0)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss, reg_term = self(data, training=True)
            final_loss = loss + reg_term

        gradients = tape.gradient(final_loss, self.trainable_variables)

        if self.accumulation_steps <= 1:
            # Standard: apply gradients directly
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        else:
            # Gradient accumulation: accumulate N mini-batches, apply once.
            # Divide each gradient by N so the accumulated sum = average.
            self._ensure_accumulators()
            accum_f = tf.cast(self.accumulation_steps, tf.float32)
            for acc, grad in zip(self._grad_accumulators, gradients):
                if grad is not None:
                    acc.assign_add(grad / accum_f)
            self._accum_step.assign_add(1)
            tf.cond(
                tf.equal(self._accum_step % self.accumulation_steps, 0),
                true_fn=self._apply_and_reset,
                false_fn=lambda: tf.constant(0.0),
            )

        # Report per-TCR loss for readability (always mean for metrics)
        batch_size = tf.cast(tf.shape(data[0])[0], tf.float32)
        if self.reduction == 'sum':
            display_loss = loss / batch_size
            display_final = final_loss / batch_size
        else:
            display_loss = loss
            display_final = final_loss
        self.loss_tracker.update_state(display_loss)
        self.final_loss_tracker.update_state(display_final)
        self.reg_tracker.update_state(reg_term)
        return {m.name: m.result() for m in self.metrics}


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


def assess_explanation_for_donors(model, donor_indices, donor_hla_matrix, batch_size=1024, 
                                   output_path=None, pad_token=-1.):
    """Check if we can explain TCR presence in donors at various strictness levels."""
    if output_path: os.makedirs(output_path, exist_ok=True)
    report_lines = []
    def log(msg):
        print(msg)
        if output_path: report_lines.append(msg)
    
    log(f"\n{'='*60}\nASSESSING DONOR EXPLANATION\n{'='*60}")
    num_tcrs = donor_indices.shape[0]
    donor_hla_tensor = tf.constant(donor_hla_matrix, dtype=tf.float32)
    all_donor_scores = []
    
    log(f"Processing {num_tcrs} TCRs in batches of {batch_size}...")
    num_batches = int(np.ceil(num_tcrs / batch_size))
    
    for i in range(num_batches):
        start, end = i * batch_size, min((i + 1) * batch_size, num_tcrs)
        tcr_indices = tf.range(start, end, dtype=tf.int32)
        z_logits = model.z_embedding(tcr_indices)
        probs = tf.sigmoid(z_logits) * tf.gather(model.binder_mask, tcr_indices)
        candidates = tf.gather(model.binder_sets, tcr_indices)
        batch_donors = donor_indices[start:end]
        valid_donor_mask = tf.not_equal(batch_donors, tf.cast(pad_token, tf.int32))
        safe_donor_ids = tf.maximum(batch_donors, 0)
        batch_donor_hlas = tf.gather(donor_hla_tensor, safe_donor_ids)
        candidates_tiled = tf.tile(tf.expand_dims(candidates, 1), [1, batch_donor_hlas.shape[1], 1])
        donor_has_candidate = tf.gather(batch_donor_hlas, candidates_tiled, batch_dims=2)
        probs_expanded = tf.expand_dims(probs, 1)
        explanation_scores = probs_expanded * donor_has_candidate
        max_score_per_donor = tf.reduce_max(explanation_scores, axis=2)
        scores_masked = tf.where(valid_donor_mask, max_score_per_donor, pad_token)
        all_donor_scores.append(scores_masked.numpy())
        if i % 10 == 0: print(f"  Batch {i}/{num_batches} done...", end='\r')
    
    donor_scores_matrix = np.concatenate(all_donor_scores)
    log("\n\nAnalysis Complete. Generating Report...")
    
    thresholds = np.linspace(0.01, 0.99, 100)
    curves = {level: [] for level in range(100, 9, -10)}
    curves[1] = []
    total_donors_per_tcr = np.maximum(np.sum(donor_scores_matrix != pad_token, axis=1), 1)
    
    for t in thresholds:
        is_explained = (donor_scores_matrix > t)
        num_explained = np.sum(is_explained, axis=1)
        fraction_explained = num_explained / total_donors_per_tcr
        for level in curves.keys():
            if level == 100: perc = np.mean(fraction_explained == 1.0) * 100
            elif level == 1: perc = np.mean(num_explained >= 1) * 100
            else: perc = np.mean(fraction_explained >= (level / 100.0)) * 100
            curves[level].append(perc)
    
    # Visualization
    fig = plt.figure(figsize=(20, 6))
    plt.subplot(1, 3, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, 11))
    for i, level in enumerate(range(100, 9, -10)):
        label_text = "100% Donors" if level == 100 else f"≥ {level}% Donors"
        plt.plot(thresholds, curves[level], color=colors[i], linewidth=2, label=label_text)
    plt.plot(thresholds, curves[1], color='grey', linewidth=2, linestyle='--', label="≥ 1 Donor")
    plt.title("Explanation Robustness Spectrum")
    plt.xlabel("Binarization Threshold")
    plt.ylabel("% TCRs satisfying condition")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    masked_scores = np.ma.masked_equal(donor_scores_matrix, pad_token)
    min_scores = np.min(masked_scores, axis=1).filled(0.0)
    plt.subplot(1, 3, 2)
    plt.hist(min_scores, bins=50, color='#d65f5f', edgecolor='white', range=(0,1))
    plt.title("Critical Score Distribution")
    plt.xlabel("Probability")
    plt.ylabel("Count of TCRs")
    
    check_t = 0.5
    is_explained_check = (donor_scores_matrix > check_t)
    fracs = np.sum(is_explained_check, axis=1) / total_donors_per_tcr
    plt.subplot(1, 3, 3)
    plt.hist(fracs * 100, bins=20, color='#4c72b0', edgecolor='white', range=(0,100))
    plt.title(f"Fraction of Donors Explained (T={check_t})")
    plt.xlabel("% of Donors Explained")
    plt.ylabel("Count of TCRs")
    plt.tight_layout()
    
    if output_path:
        fig.savefig(os.path.join(output_path, "donor_explanation_plots.png"), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(output_path, "donor_explanation_plots.pdf"), bbox_inches='tight')
        plt.close(fig)
        # Save report and data
        columns = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 1]
        with open(os.path.join(output_path, "donor_explanation_report.txt"), 'w') as f:
            f.write('\n'.join(report_lines))
        curve_data = np.column_stack([thresholds] + [curves[level] for level in columns])
        np.savetxt(os.path.join(output_path, "explanation_curves.csv"), curve_data, delimiter=',',
                   header='threshold,' + ','.join([f'pct_{c}_donors' for c in columns]), comments='')
        np.savez_compressed(os.path.join(output_path, "donor_scores_matrix.npz"),
                           donor_scores=donor_scores_matrix, thresholds=thresholds,
                           total_donors_per_tcr=total_donors_per_tcr)
    else:
        plt.show()
    
    # Summary statistics
    perfect_count = np.sum(fracs == 1.0)
    summary_stats = {
        'num_tcrs': num_tcrs, 'perfect_100pct': int(perfect_count),
        'mean_fraction_explained_t005': float(np.mean(fracs)),
        'median_fraction_explained_t005': float(np.median(fracs)),
    }
    return donor_scores_matrix, summary_stats


def analyze_model_predictions(model, binder_sets, num_total_alleles, threshold=0.5, 
                               output_path=None, pad_token=-1.):
    """Full analysis pipeline with visualizations."""
    if output_path: os.makedirs(output_path, exist_ok=True)
    report_lines = []
    def log(msg):
        print(msg)
        if output_path: report_lines.append(msg)
    
    log(f"\n{'='*50}\nSTARTING MODEL ANALYSIS\n{'='*50}")
    trained_logits = model.z_embedding.get_weights()[0]
    trained_probs = tf.sigmoid(trained_logits).numpy()
    valid_mask = (binder_sets != pad_token)
    viz_probs = trained_probs.copy()
    viz_probs[~valid_mask] = pad_token
    analysis_probs = trained_probs.copy()
    analysis_probs[~valid_mask] = 0.0
    
    # Threshold optimization
    log("\n--- Threshold Optimization Analysis ---")
    threshold_range = np.linspace(0.01, 0.999, 1000)
    coverages, avg_counts = [], []
    idx_99, idx_95, best_tradeoff_idx = -1, -1, -1
    max_tradeoff_score = -float('inf')
    
    for i, t in enumerate(threshold_range):
        matches = np.sum(analysis_probs > t, axis=1)
        cov = np.mean(matches > 0) * 100
        avg = np.mean(matches)
        coverages.append(cov)
        avg_counts.append(avg)
        if cov >= 99.0: idx_99 = i
        if cov >= 95.0: idx_95 = i
        score = cov - (5.0 * avg)
        if score > max_tradeoff_score:
            max_tradeoff_score, best_tradeoff_idx = score, i
    
    def print_stat(name, idx):
        if idx >= 0:
            log(f"Strategy: {name:<25} | Threshold: {threshold_range[idx]:.3f} | Coverage: {coverages[idx]:.2f}%")
    print_stat("'Strict' (99% Coverage)", idx_99)
    print_stat("'Relaxed' (95% Coverage)", idx_95)
    print_stat("'Balanced' (Elbow Point)", best_tradeoff_idx)
    
    # Current threshold stats
    log(f"\n--- Statistics for Threshold ({threshold}) ---")
    final_decisions = (analysis_probs > threshold)
    matches_per_tcr = np.sum(final_decisions, axis=1)
    current_coverage = np.mean(matches_per_tcr > 0) * 100
    current_avg = np.mean(matches_per_tcr)
    current_median = np.median(matches_per_tcr)
    current_max = np.max(matches_per_tcr)
    zero_matches = np.sum(matches_per_tcr == 0)
    log(f"Coverage: {current_coverage:.2f}% | Avg HLAs/TCR: {current_avg:.2f} | Zero matches: {zero_matches}")
    
    # Visualizations
    fig = plt.figure(figsize=(18, 12))
    plt.subplot(2, 2, 1)
    max_probs = np.max(viz_probs, axis=1)
    valid_plot_data = max_probs[max_probs >= 0.0]
    if len(valid_plot_data) > 0:
        plt.hist(valid_plot_data, bins=50, range=(0, 1), color='#4c72b0', edgecolor='white')
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'T={threshold}')
    plt.title("Distribution of Model Confidence")
    plt.legend()
    
    plt.subplot(2, 2, 2)
    ax1 = plt.gca()
    ax1.plot(threshold_range, coverages, 'b-', linewidth=2, label='Coverage %')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Coverage (%)', color='b')
    ax2 = ax1.twinx()
    ax2.plot(threshold_range, avg_counts, 'r--', linewidth=2, label='Avg HLAs/TCR')
    ax2.set_ylabel('Avg Count', color='r')
    plt.title("Optimization Curve")
    
    plt.subplot(2, 2, 3)
    chosen_allele_ids = binder_sets[final_decisions]
    if len(chosen_allele_ids) > 0:
        unique_ids, counts = np.unique(chosen_allele_ids, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        top_n = min(30, len(sorted_indices))
        plt.bar(range(top_n), counts[sorted_indices[:top_n]], color='#55a868')
        plt.title(f"Top {top_n} Predicted Alleles")
    
    plt.subplot(2, 2, 4)
    sample_size = min(20, len(viz_probs))
    random_indices = np.random.choice(len(viz_probs), sample_size, replace=False)
    full_heatmap_matrix = np.full((sample_size, num_total_alleles), 0.0)
    for i, idx in enumerate(random_indices):
        valid_idx = binder_sets[idx] != pad_token
        tcr_allele_ids = binder_sets[idx][valid_idx].astype(int)
        full_heatmap_matrix[i, tcr_allele_ids] = viz_probs[idx][valid_idx]
    plt.imshow(full_heatmap_matrix, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)
    plt.colorbar(label="Probability")
    plt.title(f"Binding Probabilities ({num_total_alleles} Alleles)")
    plt.tight_layout()
    
    if output_path:
        fig.savefig(os.path.join(output_path, "analysis_plots.png"), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(output_path, "analysis_plots.pdf"), bbox_inches='tight')
        plt.close(fig)
        with open(os.path.join(output_path, "analysis_report.txt"), 'w') as f:
            f.write('\n'.join(report_lines))
        np.savetxt(os.path.join(output_path, "threshold_optimization.csv"),
                   np.column_stack([threshold_range, coverages, avg_counts]), delimiter=',',
                   header='threshold,coverage_percent,avg_hlas_per_tcr', comments='')
        np.savez(os.path.join(output_path, "analysis_arrays.npz"), trained_probs=trained_probs,
                 analysis_probs=analysis_probs, final_decisions=final_decisions,
                 matches_per_tcr=matches_per_tcr, threshold_range=threshold_range,
                 coverages=np.array(coverages), avg_counts=np.array(avg_counts))
    else:
        plt.show()
    
    return {
        'coverage': current_coverage, 'avg_hlas_per_tcr': current_avg,
        'median_hlas_per_tcr': current_median, 'max_hlas_per_tcr': current_max,
        'tcrs_with_zero_hlas': zero_matches,
        'threshold_95_coverage': threshold_range[idx_95] if idx_95 >= 0 else None,
        'threshold_99_coverage': threshold_range[idx_99] if idx_99 >= 0 else None,
    }


def evaluate_model_performance(model, binder_sets, true_hla_set, num_total_alleles=440, 
                                output_path=None, pad_token=-1.):
    """Calculate PR Curve, ROC Curve, and statistical metrics (optimized sparse version)."""
    if output_path: os.makedirs(output_path, exist_ok=True)
    print(f"\n--- Performance Evaluation (PR & ROC) ---")
    
    z_logits = model.z_embedding.get_weights()[0]
    candidate_probs = tf.sigmoid(z_logits).numpy()
    num_tcrs = binder_sets.shape[0]
    
    # Build sparse representation
    valid_mask = (binder_sets != pad_token)
    pred_probs_sparse = candidate_probs[valid_mask]
    pred_allele_ids = binder_sets[valid_mask].astype(int)
    pred_tcr_ids = np.repeat(np.arange(num_tcrs), binder_sets.shape[1])[valid_mask.flatten()]
    
    # True allele lookup
    true_allele_sets = []
    for i in range(num_tcrs):
        valid_true = true_hla_set[i] >= 0
        true_allele_sets.append(set(true_hla_set[i][valid_true].astype(int)))
    
    # Create binary labels
    y_true_sparse = np.array([
        1 if pred_allele_ids[j] in true_allele_sets[pred_tcr_ids[j]] else 0
        for j in range(len(pred_probs_sparse))
    ], dtype=np.int32)
    
    total_true_positives = sum(len(s) for s in true_allele_sets)
    true_positives_in_candidates = np.sum(y_true_sparse)
    fn_from_non_candidates = total_true_positives - true_positives_in_candidates
    total_negatives = num_tcrs * num_total_alleles - total_true_positives
    
    # Sort by probability descending
    sorted_indices = np.argsort(-pred_probs_sparse)
    y_true_sorted = y_true_sparse[sorted_indices]
    y_pred_sorted = pred_probs_sparse[sorted_indices]
    
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    
    # Compute curves
    precision = tps / (tps + fps + 1e-7)
    recall = tps / (total_true_positives + 1e-7)
    fpr = fps / (total_negatives + 1e-7)
    tpr = recall
    
    # Metrics
    roc_auc = np.trapz(tpr, fpr)
    average_precision = np.sum(np.diff(np.concatenate([[0], recall])) * precision)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = y_pred_sorted[best_f1_idx] if best_f1_idx < len(y_pred_sorted) else 0.5
    best_f1 = f1_scores[best_f1_idx]
    
    print(f"AUC ROC: {roc_auc:.5f} | AP: {average_precision:.5f} | Best F1: {best_f1:.5f}")
    
    if output_path:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        axes[0].plot(recall, precision, color='#2ca02c', lw=2, label=f'AP = {average_precision:.3f}')
        axes[0].set_title('Precision-Recall Curve')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].legend()
        axes[1].plot(fpr, tpr, color='#1f77b4', lw=2, label=f'AUC = {roc_auc:.3f}')
        axes[1].plot([0, 1], [0, 1], 'gray', linestyle='--')
        axes[1].set_title('ROC Curve')
        axes[1].set_xlabel('FPR')
        axes[1].set_ylabel('TPR')
        axes[1].legend()
        plt.tight_layout()
        fig.savefig(os.path.join(output_path, "performance_curves.png"), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(output_path, "performance_curves.pdf"), bbox_inches='tight')
        plt.close(fig)
        np.savez(os.path.join(output_path, "curve_data.npz"), precision=precision, recall=recall,
                 fpr=fpr, tpr=tpr, y_pred_sorted=y_pred_sorted)
    
    return {'auc': roc_auc, 'ap': average_precision, 'best_f1': best_f1, 'best_threshold': best_threshold}


def compute_precision_at_k(output_dir, data_dir, max_k=20, pad_token=-1.):
    """Compute Precision@k and Recall@k for k=1,2,...,max_k."""
    output_dir, data_dir = Path(output_dir), Path(data_dir)
    
    arrays_path = output_dir / "figures" / "analysis_arrays.npz"
    arrays = np.load(arrays_path)
    trained_probs = arrays['trained_probs']
    
    h5_path = data_dir / 'synthetic_tcr_hla_counts.h5'
    with PublicTcrHlaCsrReader(str(h5_path)) as reader:
        counts_set, max_all = reader.read_sparse_indices()
    
    num_tcrs = len(counts_set)
    binder_sets = np.full((num_tcrs, max_all), pad_token)
    for i, row in enumerate(counts_set):
        binder_sets[i, :len(row)] = row
    
    true_hla_set = np.load(data_dir / "synthetic_binder_sets.npy")
    true_allele_sets = []
    for i in range(num_tcrs):
        valid_mask = true_hla_set[i] >= 0
        true_allele_sets.append(set(true_hla_set[i][valid_mask].astype(int)))
    
    print(f"Computing Precision@k for k=1 to {max_k}...")
    precision_at_k = {k: [] for k in range(1, max_k + 1)}
    recall_at_k = {k: [] for k in range(1, max_k + 1)}
    
    for i in range(num_tcrs):
        valid_mask = binder_sets[i] != pad_token
        candidate_ids = binder_sets[i][valid_mask].astype(int)
        candidate_probs = trained_probs[i][valid_mask]
        sorted_indices = np.argsort(-candidate_probs)
        sorted_candidate_ids = candidate_ids[sorted_indices]
        true_set = true_allele_sets[i]
        num_true = len(true_set)
        
        hits_so_far = 0
        for k in range(1, min(max_k + 1, len(sorted_candidate_ids) + 1)):
            if sorted_candidate_ids[k-1] in true_set:
                hits_so_far += 1
            precision_at_k[k].append(hits_so_far / k)
            recall_at_k[k].append(hits_so_far / num_true if num_true > 0 else 0.0)
        for k in range(len(sorted_candidate_ids) + 1, max_k + 1):
            precision_at_k[k].append(hits_so_far / k)
            recall_at_k[k].append(hits_so_far / num_true if num_true > 0 else 0.0)
    
    results = {
        'mean_precision_at_k': {k: np.mean(v) for k, v in precision_at_k.items()},
        'std_precision_at_k': {k: np.std(v) for k, v in precision_at_k.items()},
        'mean_recall_at_k': {k: np.mean(v) for k, v in recall_at_k.items()},
        'num_tcrs': num_tcrs,
    }
    return results


def plot_precision_at_k_heatmap(all_results, k_values=[1, 2, 3, 5, 10], output_path=None):
    """Create heatmaps of Precision@k across configurations."""
    import pandas as pd
    b_vals = sorted(set(int(k.split('_')[0].replace('b', '')) for k in all_results.keys()))
    n_vals = sorted(set(int(k.split('_')[1].replace('n', '')) for k in all_results.keys()))
    columns = [f'b{b}' for b in b_vals]
    indexes = [f'n{n}' for n in n_vals]
    
    fig, axes = plt.subplots(1, len(k_values), figsize=(5*len(k_values), 4))
    if len(k_values) == 1: axes = [axes]
    
    for ax, k in zip(axes, k_values):
        df = pd.DataFrame(columns=columns, index=indexes, dtype=float)
        for config_name, results in all_results.items():
            n, b = config_name.split('_')[1], config_name.split('_')[0]
            df.loc[n, b] = results['mean_precision_at_k'][k]
        df = df.astype(float)
        sns.heatmap(df, annot=True, fmt='.3f', cmap='viridis', ax=ax, vmin=0, vmax=1)
        ax.set_title(f'Precision@{k}')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_precision_at_k_curves(all_results, configs_to_plot=None, max_k=20, output_path=None):
    """Plot Precision@k curves for configurations."""
    if configs_to_plot is None:
        configs_to_plot = list(all_results.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(configs_to_plot)))
    
    for config_name, color in zip(sorted(configs_to_plot), colors):
        if config_name not in all_results: continue
        results = all_results[config_name]
        k_vals = list(range(1, max_k + 1))
        precisions = [results['mean_precision_at_k'][k] for k in k_vals]
        ax.plot(k_vals, precisions, marker='o', markersize=3, label=config_name, color=color)
    
    ax.set_xlabel('k')
    ax.set_ylabel('Precision@k')
    ax.set_title('Precision@k Curves')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def save_metrics_json(output_path, perf_metrics, analysis_results, donor_stats, threshold=0.5):
    """Save final metrics summary to JSON."""
    simple_dict = {
        "auc_roc": float(perf_metrics['auc']),
        "average_precision": float(perf_metrics['ap']),
        "best_f1_score": float(perf_metrics['best_f1']),
        "tcr_coverage_pct": float(analysis_results['coverage']),
        "avg_alleles_per_tcr": float(analysis_results['avg_hlas_per_tcr']),
        "donor_explanation_mean": float(donor_stats['mean_fraction_explained_t005']),
    }
    json_path = os.path.join(output_path, "final_metrics_summary.json")
    with open(json_path, 'w') as f:
        json.dump(simple_dict, f, indent=4, cls=NumpyEncoder)
    print(f"Metrics saved to: {json_path}")
    return simple_dict



##################################3
###################################

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
    ):
        self.path = Path(path)
        self.include_counts = bool(include_counts)
        self.include_pvals = bool(include_pvals)
        self.include_donors = bool(include_donors)
        self.include_z_probs = bool(include_z_probs)
        self._h5 = None

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
