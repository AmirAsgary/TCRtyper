#!/usr/bin/env python3
"""
Shared data-processing utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence

import numpy as np


@dataclass
class PublicTcrHlaClusterChunk:
    cluster_start: int
    cluster_end: int
    cluster_id: np.ndarray
    n_donors: np.ndarray
    raw_csr_tcr_indptr: np.ndarray
    counts_dense: Optional[np.ndarray]
    pvals_dense: Optional[np.ndarray]
    raw_csr_tcr_loops: np.ndarray
    raw_csr_tcr_int_fields: np.ndarray
    raw_csr_tcr_int_field_names: tuple[str, ...]
    _tcr_loops_accessor: RaggedClusterAccessor = field(init=False, repr=False)
    _n_identical_accessor: RaggedClusterAccessor = field(init=False, repr=False)
    _v_gene_id_accessor: RaggedClusterAccessor | None = field(init=False, repr=False)
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
        counts_dense: Optional[np.ndarray],
        pvals_dense: Optional[np.ndarray],
        raw_csr_tcr_loops: np.ndarray,
        raw_csr_tcr_int_fields: np.ndarray,
        raw_csr_tcr_int_field_names: tuple[str, ...],
    ) -> None:
        self.cluster_start = int(cluster_start)
        self.cluster_end = int(cluster_end)
        self.cluster_id = cluster_id
        self.n_donors = n_donors
        self.raw_csr_tcr_indptr = raw_csr_tcr_indptr
        self.counts_dense = counts_dense
        self.pvals_dense = pvals_dense
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


class PublicTcrHlaCsrReader:
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
    ):
        self.path = Path(path)
        self.include_counts = bool(include_counts)
        self.include_pvals = bool(include_pvals)
        self._h5 = None

    def __enter__(self) -> "PublicTcrHlaCsrReader":
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
        start: int = 0,
        stop: Optional[int] = None,
    ) -> Iterator[PublicTcrHlaClusterChunk]:
        self.open()
        h5 = self._h5
        clusters_grp = h5["clusters"]
        cluster_id = clusters_grp["cluster_id"]
        n_donors = clusters_grp["n_donors"]
        tcr_indptr = clusters_grp["tcr_indptr"]
        counts_grp = clusters_grp.get("counts") if self.include_counts else None
        pvals_grp = clusters_grp.get("pvals") if self.include_pvals else None
        if self.include_counts and counts_grp is None:
            raise KeyError("HDF5 missing clusters/counts (include_counts=True).")
        if self.include_pvals and pvals_grp is None:
            raise KeyError("HDF5 missing clusters/pvals (include_pvals=True).")
        counts_indptr = counts_grp["indptr"] if counts_grp is not None else None
        counts_indices = counts_grp["indices"] if counts_grp is not None else None
        counts_data = counts_grp["data"] if counts_grp is not None else None
        pvals_indptr = pvals_grp["indptr"] if pvals_grp is not None else None
        pvals_indices = pvals_grp["indices"] if pvals_grp is not None else None
        pvals_data = pvals_grp["data"] if pvals_grp is not None else None

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

            yield PublicTcrHlaClusterChunk(
                cluster_start=cluster_start,
                cluster_end=cluster_end,
                cluster_id=cluster_chunk,
                n_donors=donors_chunk,
                raw_csr_tcr_indptr=tcr_indptr_chunk,
                counts_dense=counts_dense,
                pvals_dense=pvals_dense,
                raw_csr_tcr_loops=tcr_loops,
                raw_csr_tcr_int_fields=tcr_int_fields,
                raw_csr_tcr_int_field_names=tuple(int_field_names),
            )


class PublicTcrHlaCsrWriter:
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
        self._clusters_written = 0
        self._tcrs_written = 0
        self._counts_nnz_total = 0

        self._buffer_cluster_ids: list[int] = []
        self._buffer_n_donors: list[int] = []
        self._buffer_tcr_counts: list[int] = []
        self._buffer_tcr_loops: list[tuple[str, str, str, str]] = []
        self._buffer_tcr_n_identical: list[int] = []
        self._buffer_tcr_v_gene_ids: list[int] = []
        self._buffer_counts_indices: list[np.ndarray] = []
        self._buffer_counts_data: list[np.ndarray] = []
        self._buffer_counts_nnz: list[int] = []

    def __enter__(self) -> "PublicTcrHlaCsrWriter":
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
    ) -> None:
        tcr_v_gene_ids = [v_gene_id] if self.include_v_genes else None
        self.add_cluster(
            cluster_id=cluster_id,
            n_donors=n_donors,
            counts=counts,
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
