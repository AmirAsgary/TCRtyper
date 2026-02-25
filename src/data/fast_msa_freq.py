"""
fast_tcr_msa_freq.py
====================
Ultra-fast amino-acid frequency profiling for clustered TCR CDR sequences.
- CDR3  : center-star MSA (Numba-accelerated Needleman-Wunsch) → frequency matrix
- CDR1/CDR2/CDR2.5 : direct position-wise frequency counting (no alignment)
Results are appended into the *same* dataset_pval.h5 under clusters/cdr_freq/.
Uses Numba for inner loops, multiprocessing for cluster parallelism,
and chunked HDF5 I/O so RAM stays bounded.
Usage
-----
    python fast_tcr_msa_freq.py \
        --h5-path  dataset_pval.h5 \
        --workers 40               \
        --chunk-size 5000
"""
# ── imports ──────────────────────────────────────────────────
import numpy as np
import numba as nb
import h5py
import argparse
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Dict
from src.mle.utils import PublicTcrHlaCsrReaderChunk, CdrFreqWriter
# ── constants ────────────────────────────────────────────────
AA_CHARS = "ACDEFGHIKLMNPQRSTVWY"   # canonical 20 amino acids
N_AA     = 20                        # number of amino acid types
GAP      = np.int8(20)              # integer code for gap / unknown in final MSA & freq
ALIGN_GAP = np.int8(-1)             # sentinel for gaps inserted by NW alignment (distinct from GAP)
N_SYM    = 21                        # 20 AAs + gap
# column indices inside raw_csr_tcr_loops (n_tcrs, 4) object array
CDR3_COL  = 0
CDR25_COL = 1
CDR1_COL  = 2
CDR2_COL  = 3
CDR_NAMES = ("cdr3", "cdr1", "cdr2", "cdr25")
# ── fast ASCII → int8 lookup table (vectorised encoding) ────
_LUT = np.full(128, GAP, dtype=np.int8)
for _i, _c in enumerate(AA_CHARS):
    _LUT[ord(_c)] = np.int8(_i)
# ── encoding helpers ─────────────────────────────────────────
def encode_seq(s: str) -> np.ndarray:
    """Encode a single amino-acid string to an int8 array.
    Every character not in the canonical 20 AA set (including '.')
    is mapped to GAP (=20).  Vectorised via NumPy lookup."""
    return _LUT[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]
def encode_batch(strings: List[str]) -> List[np.ndarray]:
    """Encode a list of AA strings to a list of int8 arrays."""
    return [encode_seq(s) for s in strings]
# ── Numba : Needleman-Wunsch pairwise global alignment ──────
@nb.njit(cache=True)
def _nw(s1, s2, sc_match=2, sc_mis=-1, sc_gap=-3):
    """Global pairwise alignment (Needleman-Wunsch).
    Parameters
    ----------
    s1, s2    : int8 1-D arrays — encoded amino-acid sequences.
    sc_match  : int — score for an identical pair.
    sc_mis    : int — penalty for a mismatch.
    sc_gap    : int — linear gap penalty (open = extend).
    Returns
    -------
    a1, a2 : int8 1-D arrays — aligned sequences.
             Alignment gaps are marked with ALIGN_GAP = -1 (NOT GAP = 20),
             so that original unknown-AA characters (encoded as 20) are
             distinguishable from gaps introduced by the alignment.
    """
    n = len(s1)
    m = len(s2)
    # Use -1 as the alignment-gap sentinel so it cannot collide
    # with GAP=20 that encodes unknown amino acids in the input.
    g = np.int8(-1)
    # DP score matrix (n+1) × (m+1)
    dp = np.empty((n + 1, m + 1), dtype=np.int32)
    dp[0, 0] = 0
    for i in range(1, n + 1):
        dp[i, 0] = i * sc_gap
    for j in range(1, m + 1):
        dp[0, j] = j * sc_gap
    # fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sc = sc_match if s1[i - 1] == s2[j - 1] else sc_mis
            dp[i, j] = max(dp[i - 1, j - 1] + sc,
                           dp[i - 1, j]     + sc_gap,
                           dp[i, j - 1]     + sc_gap)
    # traceback — build alignment in reverse, then flip
    buf1 = np.empty(n + m, dtype=np.int8)
    buf2 = np.empty(n + m, dtype=np.int8)
    k = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sc = sc_match if s1[i - 1] == s2[j - 1] else sc_mis
            if dp[i, j] == dp[i - 1, j - 1] + sc:
                buf1[k] = s1[i - 1]; buf2[k] = s2[j - 1]
                i -= 1; j -= 1; k += 1; continue
        if i > 0 and dp[i, j] == dp[i - 1, j] + sc_gap:
            buf1[k] = s1[i - 1]; buf2[k] = g
            i -= 1; k += 1; continue
        # gap in s1
        buf1[k] = g; buf2[k] = s2[j - 1]
        j -= 1; k += 1
    return buf1[:k][::-1].copy(), buf2[:k][::-1].copy()
# ── Numba : per-position frequency from an MSA matrix ───────
@nb.njit(cache=True)
def _freq(msa, ns=21):
    """Compute per-column amino-acid frequency from an integer MSA.
    Parameters
    ----------
    msa : int8 2-D array, shape (n_seqs, align_len), values in [0..ns-1].
    ns  : number of symbols (default 21 = 20 AAs + gap).
    Returns
    -------
    freq : float32 2-D array, shape (align_len, ns).
    """
    nr, nc = msa.shape
    f = np.zeros((nc, ns), dtype=np.float32)
    if nr == 0:
        return f
    inv = np.float32(1.0 / nr)
    for j in range(nc):
        for i in range(nr):
            f[j, msa[i, j]] += inv
    return f
# ── center-star MSA ──────────────────────────────────────────
def center_star_msa(enc_seqs: List[np.ndarray]) -> np.ndarray:
    """Center-star multiple sequence alignment for short, similar sequences.
    Algorithm
    ---------
    1. Pick the longest sequence as the *center*.
    2. Pairwise-align every other sequence to the center (Numba NW).
    3. Build a merged gap profile: for each center position, record the
       maximum number of insertions (gaps-in-center) seen across all
       pairwise alignments.
    4. Lay every sequence into the merged coordinate system.
    NOTE: NW returns ALIGN_GAP = -1 for alignment-introduced gaps,
    keeping them distinct from GAP = 20 (unknown amino acid in input).
    The gap profile checks for -1 only, so original unknown-AA chars
    are correctly treated as center characters.
    Parameters
    ----------
    enc_seqs : list of int8 1-D arrays (encoded amino-acid sequences).
    Returns
    -------
    msa : int8 2-D array, shape (n_seqs, msa_length), values in [0..20].
    """
    n = len(enc_seqs)
    # trivial cases
    if n == 0:
        return np.empty((0, 0), dtype=np.int8)
    if n == 1:
        return enc_seqs[0].reshape(1, -1)
    # 1. choose center = longest sequence
    ci = 0
    for i in range(1, n):
        if len(enc_seqs[i]) > len(enc_seqs[ci]):
            ci = i
    center = enc_seqs[ci]
    clen = len(center)
    # 2. pairwise align every other sequence to center
    others = [i for i in range(n) if i != ci]
    pw = []  # list of (aligned_center, aligned_other)
    for oi in others:
        pw.append(_nw(center, enc_seqs[oi]))
    # 3. gap profile per pairwise alignment
    #    ALIGN_GAP = -1 marks alignment-introduced gaps only;
    #    original unknown-AA characters (GAP=20) are NOT counted as gaps.
    gps = []
    for ac, _ in pw:
        gp = np.zeros(clen + 1, dtype=np.int32)
        cp = 0
        for k in range(len(ac)):
            if ac[k] == ALIGN_GAP:       # only true alignment gaps (-1)
                gp[cp] += 1
            else:
                cp += 1
        gps.append(gp)
    # 4. merged gap profile = element-wise max
    mg = np.zeros(clen + 1, dtype=np.int32)
    for gp in gps:
        np.maximum(mg, gp, out=mg)
    msa_len = clen + int(mg.sum())
    # 5. column map: center position p → MSA column
    cmap = np.empty(clen, dtype=np.int32)
    col = 0
    for p in range(clen):
        col += int(mg[p])
        cmap[p] = col
        col += 1
    # 6. build MSA matrix (fill with GAP=20)
    msa = np.full((n, msa_len), GAP, dtype=np.int8)
    # place center sequence
    for p in range(clen):
        msa[ci, cmap[p]] = center[p]
    # place each other sequence using its pairwise alignment
    for pi, oi in enumerate(others):
        ac, ao = pw[pi]
        gp = gps[pi]
        ap = 0  # position pointer into ao
        for p in range(clen + 1):
            n_have = int(gp[p])   # insertions this alignment has at position p
            n_need = int(mg[p])   # max insertions across all alignments at p
            # MSA column where the first insertion at position p starts
            if p < clen:
                gc_start = cmap[p] - n_need
            else:
                gc_start = msa_len - n_need
            # place the insertion characters this alignment contributes
            for g in range(n_have):
                val = ao[ap]
                # convert ALIGN_GAP(-1) → GAP(20) for the final MSA matrix
                msa[oi, gc_start + g] = GAP if val == ALIGN_GAP else val
                ap += 1
            # place the character at this center position (match / mismatch / gap-in-other)
            if p < clen:
                val = ao[ap]
                msa[oi, cmap[p]] = GAP if val == ALIGN_GAP else val
                ap += 1
    return msa
# ── high-level frequency functions ───────────────────────────
def freq_with_msa(str_seqs: List[str]) -> np.ndarray:
    """CDR3 path: encode → center-star MSA → per-position frequency.
    Returns shape (alignment_length, 21) float32."""
    if len(str_seqs) == 0:
        return np.empty((0, N_SYM), dtype=np.float32)
    enc = encode_batch(str_seqs)
    msa = center_star_msa(enc)
    return _freq(msa)
def freq_no_msa(str_seqs: List[str]) -> np.ndarray:
    """CDR1/2/2.5 path: encode, pad to max length, count frequencies.
    Returns shape (max_length, 21) float32."""
    if len(str_seqs) == 0:
        return np.empty((0, N_SYM), dtype=np.float32)
    enc = encode_batch(str_seqs)
    ml = max(len(e) for e in enc)
    n = len(enc)
    mat = np.full((n, ml), GAP, dtype=np.int8)
    for i, e in enumerate(enc):
        mat[i, :len(e)] = e
    return _freq(mat)
# ── single-cluster processor ─────────────────────────────────
def process_one_cluster(tcr_rows: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute frequency profiles for one TCR cluster.
    Parameters
    ----------
    tcr_rows : (n_tcrs_in_cluster, 4) object array of string sequences.
               Columns: CDR3, CDR2.5, CDR1, CDR2.
    Returns
    -------
    dict with keys 'cdr3', 'cdr1', 'cdr2', 'cdr25', each mapping
    to a float32 array of shape (region_length, 21).
    """
    cdr3  = freq_with_msa(list(tcr_rows[:, CDR3_COL]))
    cdr1  = freq_no_msa(list(tcr_rows[:, CDR1_COL]))
    cdr2  = freq_no_msa(list(tcr_rows[:, CDR2_COL]))
    cdr25 = freq_no_msa(list(tcr_rows[:, CDR25_COL]))
    return {"cdr3": cdr3, "cdr1": cdr1, "cdr2": cdr2, "cdr25": cdr25}
# ── worker callable for multiprocessing.Pool.map ─────────────
def _worker(cluster_data_list: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
    """Process a sub-batch of clusters inside a worker process."""
    return [process_one_cluster(cd) for cd in cluster_data_list]
# ── JIT warm-up (avoids cold-start inside workers) ───────────
def _warmup():
    """Trigger Numba compilation before forking workers."""
    s1 = np.array([0, 1, 2], dtype=np.int8)
    s2 = np.array([0, 2, 3], dtype=np.int8)
    _nw(s1, s2)
    m = np.array([[0, 1], [2, 3]], dtype=np.int8)
    _freq(m)
# ── main pipeline ────────────────────────────────────────────
def run(
    h5_path: str,
    chunk_size: int = 5000,
    n_workers: Optional[int] = None,
):
    """Read clustered TCR data in chunks, compute frequency profiles
    in parallel, and write results back into the same HDF5 file
    under ``clusters/cdr_freq/``.
    Parameters
    ----------
    h5_path    : path to the dataset_pval.h5 (read + append)
    chunk_size : number of clusters per I/O chunk
    n_workers  : number of parallel worker processes (default = cpu_count)
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 40)
    # warm up Numba JIT before forking
    _warmup()
    # get total cluster count for the writer
    with PublicTcrHlaCsrReaderChunk(
        h5_path, include_counts=False, include_donors=False, include_pvals=False,
    ) as reader:
        total_clusters = reader.num_clusters
    print(f"[INFO] total_clusters={total_clusters:,}  "
          f"workers={n_workers}  chunk_size={chunk_size}")
    t0 = time.time()
    n_done = 0
    # open writer (appends cdr_freq group into existing H5)
    writer = CdrFreqWriter(
        h5_path,
        num_clusters=total_clusters,
        chunk_nnz=min(50_000, chunk_size * 30),
        compression={"compression": "gzip", "compression_opts": 1},
    )
    writer.open()
    # create the worker pool once (reuse across chunks)
    pool = Pool(n_workers) if n_workers > 1 else None
    try:
        with PublicTcrHlaCsrReaderChunk(
            h5_path,
            include_counts=False,
            include_donors=False,
            include_pvals=False,
        ) as reader:
            for chunk in reader.iter_cluster_chunks(chunk_rows=chunk_size):
                indptr = chunk.raw_csr_tcr_indptr
                loops  = chunk.raw_csr_tcr_loops  # (total_tcrs, 4) object
                n_cl   = len(chunk.cluster_id)
                cl_start = chunk.cluster_start
                cl_end   = chunk.cluster_end
                # split into per-cluster sub-arrays
                cluster_data = [
                    loops[int(indptr[i]):int(indptr[i + 1])]
                    for i in range(n_cl)
                ]
                # parallel processing
                if pool is not None and n_cl > 1:
                    batch_sz = max(1, n_cl // n_workers)
                    batches = [
                        cluster_data[j : j + batch_sz]
                        for j in range(0, n_cl, batch_sz)
                    ]
                    nested = pool.map(_worker, batches)
                    results: List[Dict[str, np.ndarray]] = []
                    for sub in nested:
                        results.extend(sub)
                else:
                    results = _worker(cluster_data)
                # collect into dict-of-lists for the writer
                freq_data: Dict[str, List[np.ndarray]] = {
                    nm: [r[nm] for r in results] for nm in CDR_NAMES
                }
                # write to H5
                writer.write_chunk(cl_start, cl_end, freq_data)
                n_done += n_cl
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                print(
                    f"  clusters={n_done:>10,}  "
                    f"rate={rate:,.0f} cl/s  "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        writer.close()
    total = time.time() - t0
    print(f"[DONE] {n_done:,} clusters → {h5_path}  ({total:.1f}s)")
# ── CLI entry point ──────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Fast MSA frequency profiling for clustered TCR CDR sequences."
    )
    p.add_argument("--h5-path", required=True,
                   help="Path to dataset_pval.h5 (read + append)")
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel workers (default: cpu_count, max 40)")
    p.add_argument("--chunk-size", type=int, default=5000,
                   help="Clusters per I/O chunk (default: 5000)")
    args = p.parse_args()
    run(args.h5_path, args.chunk_size, args.workers)