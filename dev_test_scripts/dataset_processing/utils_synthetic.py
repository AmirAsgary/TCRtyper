#!/usr/bin/env python3
"""
Shared data-processing utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence

import numpy as np


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
