#!/usr/bin/env python3
"""
Compute per-allele BH FDR thresholds across all clusters.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from scipy.stats import fisher_exact
from tqdm import tqdm

from tcrtyper.dataset_processing.synthetic_analysis_utils import (
    allele_counts_from_freqs,
    load_allele_frequencies,
)

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute per-allele BH FDR thresholds across all clusters."
    )
    ap.add_argument(
        "--public-h5",
        required=True,
        help="Path to public_tcr_hla_counts.h5.",
    )
    ap.add_argument(
        "--donor-frequencies-json",
        required=True,
        help="JSON mapping allele id to donor-level frequency.",
    )
    ap.add_argument(
        "--donor-matrix",
        default=None,
        help="Path to donor×allele matrix to infer total donor count.",
    )
    ap.add_argument(
        "--donor-matrix-key",
        default=None,
        help="Array key to use when loading .npz donor matrices.",
    )
    ap.add_argument(
        "--donor-count",
        type=int,
        default=None,
        help="Total donor count (overrides donor matrix inference).",
    )
    ap.add_argument(
        "--min-overlap",
        type=int,
        default=2,
        help="Minimum overlap for Fisher test (default: 2).",
    )
    ap.add_argument(
        "--min-cluster-size",
        type=int,
        default=1,
        help="Minimum donors per cluster to include (default: 1).",
    )
    ap.add_argument(
        "--chunk-rows",
        type=int,
        default=1_000_000,
        help="Rows to read per HDF5 chunk (default: 1000000).",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="BH FDR alpha (default: 0.05).",
    )
    ap.add_argument(
        "--out-json",
        default=None,
        help="Output JSON path (default: <public-h5-dir>/fdr_thresholds.json).",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging.",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable tqdm progress bar.",
    )
    return ap.parse_args()


def _configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_matrix_from_npz(data, npz_key: Optional[str]) -> np.ndarray:
    keys = list(data.keys())
    if npz_key:
        if npz_key not in data:
            raise KeyError(f"npz key {npz_key!r} not found. Available keys: {keys}")
        return data[npz_key]
    preferred = ["donor_hla_matrix", "donor_matrix", "matrix", "X", "data"]
    for key in preferred:
        if key in data:
            return data[key]
    if len(keys) == 1:
        return data[keys[0]]
    raise KeyError(f"npz contains multiple arrays; available keys: {keys}")


def _load_matrix(path: Path, npz_key: Optional[str]) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            return _load_matrix_from_npz(data, npz_key=npz_key)
    return np.load(path, mmap_mode="r")


def _infer_donor_count(
    donor_matrix: Path, npz_key: Optional[str]
) -> tuple[int, int]:
    x = _load_matrix(donor_matrix, npz_key)
    if x.ndim != 2:
        raise SystemExit(f"donor matrix must be 2D, got shape {x.shape}")
    n_donors, n_alleles = x.shape
    return int(n_donors), int(n_alleles)


def _bh_threshold(pvals: np.ndarray, alpha: float) -> tuple[float, int]:
    if pvals.size == 0:
        return 0.0, 0
    ordered = np.sort(pvals)
    m_tests = int(ordered.size)
    max_p = 0.0
    max_idx = 0
    for i, pval in enumerate(ordered, start=1):
        if pval * float(m_tests) / float(i) <= alpha:
            max_p = float(pval)
            max_idx = i
    return max_p, max_idx


def main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)

    public_h5 = Path(args.public_h5).resolve()
    if not public_h5.exists():
        raise FileNotFoundError(f"public HDF5 not found: {public_h5}")

    donor_freq_path = Path(args.donor_frequencies_json).resolve()
    if not donor_freq_path.exists():
        raise FileNotFoundError(f"donor frequency JSON not found: {donor_freq_path}")

    n_donors_total = args.donor_count
    n_alleles_from_matrix = None
    if args.donor_matrix:
        donor_matrix = Path(args.donor_matrix).resolve()
        if not donor_matrix.exists():
            raise FileNotFoundError(f"donor matrix not found: {donor_matrix}")
        n_donors, n_alleles = _infer_donor_count(donor_matrix, args.donor_matrix_key)
        n_alleles_from_matrix = n_alleles
        if n_donors_total is None:
            n_donors_total = n_donors
        elif n_donors_total != n_donors:
            logger.warning(
                "donor count %d does not match donor matrix rows %d",
                n_donors_total,
                n_donors,
            )
    if n_donors_total is None or n_donors_total <= 0:
        raise SystemExit("Provide a valid --donor-count or --donor-matrix.")

    with h5py.File(public_h5, "r") as h5:
        num_alleles = h5.attrs.get("num_alleles")
        if num_alleles is None:
            raise KeyError("HDF5 missing 'num_alleles' attribute.")
        num_alleles = int(num_alleles)
        num_rows = int(h5["cluster_id"].shape[0])

    if n_alleles_from_matrix is not None and n_alleles_from_matrix != num_alleles:
        logger.warning(
            "donor matrix alleles %d != HDF5 alleles %d",
            n_alleles_from_matrix,
            num_alleles,
        )

    logger.info("Public HDF5 rows=%d alleles=%d", num_rows, num_alleles)
    logger.info("Total donors: %d", n_donors_total)

    freqs = load_allele_frequencies(donor_freq_path, num_alleles)
    allele_counts = allele_counts_from_freqs(freqs, n_donors_total)

    pvals_by_allele: dict[int, list[float]] = {}
    rows_seen = 0
    clusters_seen = 0
    clusters_used = 0
    prev_cluster = None

    if args.chunk_rows < 1:
        raise SystemExit("--chunk-rows must be >= 1.")

    with h5py.File(public_h5, "r") as h5:
        cluster_id = h5["cluster_id"]
        n_donors = h5["n_donors"]
        counts_grp = h5["y_counts"]
        indptr = counts_grp["indptr"]
        indices = counts_grp["indices"]
        data = counts_grp["data"]

        with tqdm(
            total=num_rows,
            desc="Processing clusters",
            unit="row",
            disable=args.no_progress,
        ) as pbar:
            for row_start in range(0, num_rows, args.chunk_rows):
                row_end = min(row_start + args.chunk_rows, num_rows)
                cluster_chunk = cluster_id[row_start:row_end]
                donors_chunk = n_donors[row_start:row_end]
                indptr_chunk = indptr[row_start : row_end + 1]

                rows_seen += int(cluster_chunk.size)
                pbar.update(int(cluster_chunk.size))

                if cluster_chunk.size == 0:
                    continue

                change = np.ones(cluster_chunk.size, dtype=bool)
                if cluster_chunk.size > 1:
                    change[1:] = cluster_chunk[1:] != cluster_chunk[:-1]
                if prev_cluster is not None and cluster_chunk[0] == prev_cluster:
                    change[0] = False
                prev_cluster = int(cluster_chunk[-1])

                new_rows = np.flatnonzero(change)
                if new_rows.size == 0:
                    continue

                for local_i in new_rows:
                    clusters_seen += 1
                    cluster_size = int(donors_chunk[local_i])
                    if cluster_size < args.min_cluster_size:
                        continue
                    clusters_used += 1

                    lo = int(indptr_chunk[local_i])
                    hi = int(indptr_chunk[local_i + 1])
                    if hi <= lo:
                        continue
                    idxs = indices[lo:hi]
                    vals = data[lo:hi]

                    mask = vals >= args.min_overlap
                    if not np.any(mask):
                        continue
                    idxs = idxs[mask]
                    vals = vals[mask]

                    for idx, d_ia in zip(idxs, vals):
                        a_idx = int(idx)
                        d_val = int(d_ia)
                        N_a = int(allele_counts[a_idx])
                        if N_a < d_val:
                            N_a = d_val
                        if N_a > n_donors_total:
                            N_a = n_donors_total
                        n11 = d_val
                        n10 = int(cluster_size - d_val)
                        n01 = int(N_a - d_val)
                        n00 = int(n_donors_total - n11 - n10 - n01)
                        if min(n11, n10, n01, n00) < 0:
                            continue
                        table = np.array([[n11, n10], [n01, n00]], dtype=np.int64)
                        _, pval = fisher_exact(table, alternative="greater")
                        pvals_by_allele.setdefault(a_idx, []).append(float(pval))

    logger.info("Rows processed: %d", rows_seen)
    logger.info("Clusters seen: %d", clusters_seen)
    logger.info("Clusters used: %d", clusters_used)

    thresholds = {}
    for idx in range(num_alleles):
        vals = pvals_by_allele.get(idx, [])
        arr = np.asarray(vals, dtype=np.float64)
        thr, n_sig = _bh_threshold(arr, args.alpha)
        thresholds[str(idx)] = {
            "pval_threshold": float(thr),
            "n_sig": int(n_sig),
            "n_tested": int(arr.size),
        }

    out_json = (
        Path(args.out_json).resolve()
        if args.out_json
        else public_h5.parent / "fdr_thresholds.json"
    )
    payload = {
        "public_h5": str(public_h5),
        "donor_frequencies_json": str(donor_freq_path),
        "donor_count": int(n_donors_total),
        "min_overlap": int(args.min_overlap),
        "min_cluster_size": int(args.min_cluster_size),
        "alpha": float(args.alpha),
        "rows_total": int(rows_seen),
        "clusters_seen": int(clusters_seen),
        "clusters_used": int(clusters_used),
        "num_alleles": int(num_alleles),
        "thresholds": thresholds,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote %s", out_json)


if __name__ == "__main__":
    main()
