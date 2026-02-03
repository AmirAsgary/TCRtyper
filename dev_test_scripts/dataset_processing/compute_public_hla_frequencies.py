#!/usr/bin/env python3
"""
Compute per-allele frequencies from public_tcr_hla_counts.h5.
Cluster IDs are de-duplicated to avoid double-counting loop groups.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute HLA allele frequencies from public TCR HDF5 counts."
    )
    ap.add_argument(
        "--export-root",
        default=None,
        help="Export train dataset root (used for default paths).",
    )
    ap.add_argument(
        "--public-h5",
        default=None,
        help=(
            "Path to public_tcr_hla_counts.h5. "
            "Default: <export_root>/public_tcr_hla_counts.h5."
        ),
    )
    ap.add_argument(
        "--out-json",
        default=None,
        help="Output JSON path (default: <export_root>/synthetic/binder_set/hla_frequencies.json).",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Row chunk size for reading sparse counts (default: 1,000,000).",
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


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.public_h5:
        public_h5 = Path(args.public_h5).resolve()
    elif args.export_root:
        export_root = Path(args.export_root).resolve()
        public_h5 = export_root / "public_tcr_hla_counts.h5"
    else:
        raise SystemExit("Provide --public-h5 or --export-root.")

    if args.out_json:
        out_json = Path(args.out_json).resolve()
    elif args.export_root:
        export_root = Path(args.export_root).resolve()
        out_json = export_root / "synthetic" / "binder_set" / "hla_frequencies.json"
    else:
        out_json = Path("hla_frequencies.json").resolve()

    return public_h5, out_json


def _compute_frequencies(path: Path, chunk_size: int, no_progress: bool) -> np.ndarray:
    if chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1.")

    with h5py.File(path, "r") as h5:
        num_alleles = h5.attrs.get("num_alleles")
        if num_alleles is None:
            raise KeyError("HDF5 missing 'num_alleles' attribute.")
        num_alleles = int(num_alleles)

        cluster_id = h5["cluster_id"]
        counts_grp = h5["y_counts"]
        indptr = counts_grp["indptr"]
        indices = counts_grp["indices"]
        data = counts_grp["data"]

        n_rows = int(cluster_id.shape[0])
        counts = np.zeros(num_alleles, dtype=np.float64)
        rows_seen = 0
        clusters_used = 0
        prev_cluster = None

        with tqdm(
            total=n_rows,
            desc="Aggregating clusters",
            unit="row",
            disable=no_progress,
        ) as pbar:
            for row_start in range(0, n_rows, chunk_size):
                row_end = min(row_start + chunk_size, n_rows)
                cluster_chunk = cluster_id[row_start:row_end]
                if cluster_chunk.size == 0:
                    continue
                rows_seen += int(cluster_chunk.size)
                pbar.update(int(cluster_chunk.size))

                change = np.ones(cluster_chunk.size, dtype=bool)
                if cluster_chunk.size > 1:
                    change[1:] = cluster_chunk[1:] != cluster_chunk[:-1]
                if prev_cluster is not None and cluster_chunk[0] == prev_cluster:
                    change[0] = False
                prev_cluster = int(cluster_chunk[-1])

                if not np.any(change):
                    continue

                clusters_used += int(np.sum(change))

                indptr_chunk = indptr[row_start : row_end + 1]
                nnz_per_row = np.diff(indptr_chunk)
                data_start = int(indptr_chunk[0])
                data_end = int(indptr_chunk[-1])
                if data_end <= data_start:
                    continue

                data_slice = data[data_start:data_end].astype(np.float64, copy=False)
                indices_slice = indices[data_start:data_end].astype(np.int64, copy=False)
                weights = np.repeat(change, nnz_per_row)
                counts += np.bincount(
                    indices_slice,
                    weights=data_slice * weights,
                    minlength=num_alleles,
                )

    logger.info("Rows processed: %d", rows_seen)
    logger.info("Clusters used: %d", clusters_used)

    total = float(counts.sum())
    if total <= 0:
        raise SystemExit("Total allele count is zero; cannot compute frequencies.")
    return counts / total


def main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)

    public_h5, out_json = _resolve_paths(args)
    if not public_h5.exists():
        raise FileNotFoundError(f"public HDF5 not found: {public_h5}")

    logger.info("Reading counts from %s", public_h5)
    freqs = _compute_frequencies(public_h5, args.chunk_size, args.no_progress)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(i): float(freqs[i]) for i in range(freqs.size)}
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote %s", out_json)


if __name__ == "__main__":
    main()
