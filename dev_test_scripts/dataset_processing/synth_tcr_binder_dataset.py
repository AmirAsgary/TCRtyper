#!/usr/bin/env python3
"""
Generate synthetic TCR labels by sampling hidden binder sets from a donor HLA matrix.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from tcrtyper.dataset_processing.utils import PublicTcrHlaCsrWriter

logger = logging.getLogger(__name__)


def _parse_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Expected an integer, got {value!r}."
            ) from exc


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate synthetic TCR labels from a donor×allele matrix."
    )
    ap.add_argument(
        "--export-root",
        default=None,
        help="Export train dataset root (used for default paths).",
    )
    ap.add_argument(
        "--donor-matrix",
        default=None,
        help=(
            "Path to donor×allele matrix (.npz or .npy). "
            "Default: <export_root>/donor_hla_matrix.npz."
        ),
    )
    ap.add_argument(
        "--npz-key",
        default=None,
        help="Array key to use when loading .npz matrices.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <export_root>/synthetic_binder_dataset).",
    )
    ap.add_argument(
        "--dataset-size",
        type=_parse_int,
        default=100_000,
        help="Number of synthetic TCRs to generate (default: 1e5).",
    )
    ap.add_argument(
        "--b-size",
        type=int,
        required=True,
        help="Binder set size |B|.",
    )
    ap.add_argument(
        "--binder-sampling",
        choices=("weighted", "rank_stratified"),
        default="weighted",
        help=(
            "Binder sampling strategy (default: weighted). "
            "weighted uses allele frequencies; rank_stratified draws one allele per "
            "frequency rank stratum."
        ),
    )
    ap.add_argument(
        "--n-donors",
        type=int,
        required=True,
        help="Number of donors to assign per synthetic TCR.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Batch size for streaming writes (default: 10000).",
    )
    ap.add_argument(
        "--out-h5",
        default=None,
        help="Output HDF5 path (default: <out-dir>/synthetic_tcr_hla_counts.h5).",
    )
    ap.add_argument(
        "--allele-frequencies-json",
        default=None,
        help="JSON mapping allele id to frequency (overrides donor-level frequencies).",
    )
    ap.add_argument(
        "--compression",
        choices=["gzip", "lzf", "none"],
        default="gzip",
        help="Compression for HDF5 datasets (default: gzip).",
    )
    ap.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="Gzip compression level (default: 4).",
    )
    ap.add_argument(
        "--chunk-rows",
        type=int,
        default=4096,
        help="Chunk size for row-wise datasets (default: 4096).",
    )
    ap.add_argument(
        "--chunk-nnz",
        type=int,
        default=1_000_000,
        help="Chunk size for CSR indices/data (default: 1,000,000).",
    )
    ap.add_argument(
        "--flush-rows",
        type=int,
        default=10_000,
        help="Flush HDF5 buffer every N rows (default: 10000).",
    )
    ap.add_argument(
        "--no-v-genes",
        action="store_true",
        default=False,
        help="Omit v_gene_ids dataset from HDF5 output.",
    )
    ap.add_argument(
        "--max-attempts",
        type=int,
        default=500,
        help="Max attempts per TCR before failing (default: 500).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed (default: 13).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing outputs.",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging.",
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
    if args.donor_matrix:
        donor_matrix = Path(args.donor_matrix).resolve()
    elif args.export_root:
        export_root = Path(args.export_root).resolve()
        candidates = [
            export_root / "donor_hla_matrix.npz",
            export_root / "donor_hla_matrix.npy",
        ]
        donor_matrix = next((p for p in candidates if p.exists()), candidates[0])
    else:
        raise SystemExit("Provide --donor-matrix or --export-root.")

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    elif args.export_root:
        out_dir = Path(args.export_root).resolve() / "synthetic_binder_dataset"
    else:
        out_dir = Path("synthetic_binder_dataset").resolve()

    return donor_matrix, out_dir


def _load_matrix_from_npz(data, npz_key: Optional[str] = None) -> np.ndarray:
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


def _load_allele_frequencies(path: Path, n_alleles: int) -> np.ndarray:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, list):
        freqs = np.asarray(payload, dtype=np.float64)
        if freqs.size != n_alleles:
            raise SystemExit(
                f"Frequency list length {freqs.size} != num alleles {n_alleles}."
            )
        return freqs

    if isinstance(payload, dict):
        freqs = np.zeros(n_alleles, dtype=np.float64)
        for key, val in payload.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                raise SystemExit(f"Invalid allele id key: {key!r}")
            if idx < 0 or idx >= n_alleles:
                raise SystemExit(f"Allele id {idx} out of range 0..{n_alleles - 1}.")
            freqs[idx] = float(val)
        return freqs

    raise SystemExit("Frequency JSON must be a dict or list.")


def _compression_kwargs(compression: str, level: int) -> dict:
    if compression == "none":
        return {}
    if compression == "lzf":
        return {"compression": "lzf"}
    return {"compression": "gzip", "compression_opts": int(level)}


def _iter_unique_arrays(indices: Sequence[int], sources: Sequence[np.ndarray]) -> np.ndarray:
    if not indices:
        return np.array([], dtype=int)
    arrays: list[np.ndarray] = []
    for idx in indices:
        arrays.append(sources[idx])
    if len(arrays) == 1:
        return arrays[0]
    return np.unique(np.concatenate(arrays))


def _build_donor_lists(x: np.ndarray, allele_indices: Iterable[int]) -> list[np.ndarray | None]:
    donor_lists: list[np.ndarray | None] = [None] * x.shape[1]
    for idx in allele_indices:
        donor_lists[idx] = np.flatnonzero(x[:, idx])
    return donor_lists


def _sample_binder_weighted(
    rng: np.random.Generator,
    allele_pool: np.ndarray,
    allele_probs: np.ndarray,
    b_size: int,
) -> np.ndarray:
    return rng.choice(allele_pool, size=b_size, replace=False, p=allele_probs)


def _sample_binder_rank_stratified(
    rng: np.random.Generator,
    allele_pool: np.ndarray,
    rank_scores: np.ndarray,
    b_size: int,
) -> np.ndarray:
    order = np.argsort(-rank_scores, kind="mergesort")
    ranked = allele_pool[order]
    strata = np.array_split(ranked, b_size)
    picks = np.empty(b_size, dtype=ranked.dtype)
    for i, stratum in enumerate(strata):
        if stratum.size == 0:
            raise SystemExit("Rank strata are empty; check b_size and allele pool size.")
        picks[i] = rng.choice(stratum, size=1, replace=False)[0]
    return picks


def main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)

    if args.dataset_size < 1:
        raise SystemExit("--dataset-size must be >= 1.")
    if args.b_size < 1:
        raise SystemExit("--b-size must be >= 1.")
    if args.n_donors < 1:
        raise SystemExit("--n-donors must be >= 1.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1.")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be >= 1.")

    donor_matrix_path, out_dir = _resolve_paths(args)
    if not donor_matrix_path.exists():
        raise FileNotFoundError(f"donor matrix not found: {donor_matrix_path}")

    x = _load_matrix(donor_matrix_path, args.npz_key)
    if x.ndim != 2:
        raise SystemExit(f"donor matrix must be 2D, got shape {x.shape}")

    n_donors, n_alleles = x.shape
    if args.n_donors > n_donors:
        raise SystemExit("--n-donors exceeds number of donors in the matrix.")

    allele_counts = np.count_nonzero(x, axis=0)
    if allele_counts.ndim != 1 or allele_counts.size != n_alleles:
        raise SystemExit("Failed to compute allele counts from donor matrix.")

    freqs = None
    if args.allele_frequencies_json:
        freq_path = Path(args.allele_frequencies_json).resolve()
        if not freq_path.exists():
            raise FileNotFoundError(f"allele frequencies not found: {freq_path}")
        logger.info("Using allele frequencies from %s", freq_path)
        freqs = _load_allele_frequencies(freq_path, n_alleles)
        if np.any(freqs < 0):
            raise SystemExit("Allele frequencies contain negative values.")

    nonzero_mask = allele_counts > 0
    if freqs is not None:
        nonzero_mask = nonzero_mask & (freqs > 0)
    allele_pool = np.flatnonzero(nonzero_mask)
    if allele_pool.size < args.b_size:
        raise SystemExit(
            f"Not enough alleles with nonzero counts for b_size={args.b_size}."
        )

    if freqs is not None:
        allele_probs = freqs[allele_pool].astype(np.float64)
        score_source = "freqs"
    else:
        allele_probs = allele_counts[allele_pool].astype(np.float64)
        score_source = "counts"
    if float(allele_probs.sum()) <= 0:
        raise SystemExit("Allele probability mass is zero after filtering.")
    allele_probs /= allele_probs.sum()
    rank_scores = allele_probs.copy()

    logger.info(
        "Matrix: donors=%d alleles=%d nonzero_alleles=%d",
        n_donors,
        n_alleles,
        allele_pool.size,
    )
    logger.info(
        "Generating dataset_size=%d b_size=%d n_donors=%d",
        args.dataset_size,
        args.b_size,
        args.n_donors,
    )
    logger.info("Binder sampling: %s (score_source=%s)", args.binder_sampling, score_source)

    logger.info("Precomputing donor lists per allele (nonzero only).")
    donor_lists = _build_donor_lists(x, allele_pool.tolist())

    out_dir.mkdir(parents=True, exist_ok=True)
    binder_path = out_dir / "synthetic_binder_sets.npy"
    donor_path = out_dir / "synthetic_donor_indices.npy"
    meta_path = out_dir / "synthetic_meta.json"
    if args.out_h5:
        h5_path = Path(args.out_h5).resolve()
    else:
        h5_path = out_dir / "synthetic_tcr_hla_counts.h5"

    if not args.overwrite:
        for path in (binder_path, donor_path, meta_path, h5_path):
            if path.exists():
                raise SystemExit(f"Output already exists: {path}")

    binder_sets = np.lib.format.open_memmap(
        binder_path,
        mode="w+",
        dtype=np.int32,
        shape=(args.dataset_size, args.b_size),
    )
    donor_indices = np.lib.format.open_memmap(
        donor_path,
        mode="w+",
        dtype=np.int32,
        shape=(args.dataset_size, args.n_donors),
    )

    rng = np.random.default_rng(args.seed)
    total = args.dataset_size
    batch_size = args.batch_size
    comp = _compression_kwargs(args.compression, args.compression_level)
    counts_dtype = np.uint16
    max_index = np.iinfo(np.uint16).max
    indices_dtype = np.uint16 if n_alleles <= max_index else np.uint32
    loops = ("X" * 12, "X" * 7, "X" * 7, "X" * 7)

    writer_attrs = {
        "synthetic": True,
        "donor_matrix": str(donor_matrix_path),
        "dataset_size": int(args.dataset_size),
        "b_size": int(args.b_size),
        "n_donors": int(args.n_donors),
        "binder_sampling": args.binder_sampling,
        "binder_score_source": score_source,
    }
    if args.binder_sampling == "rank_stratified":
        writer_attrs["binder_rank_strata"] = int(args.b_size)
    if args.allele_frequencies_json:
        writer_attrs["allele_frequencies_json"] = str(
            Path(args.allele_frequencies_json).resolve()
        )

    with PublicTcrHlaCsrWriter(
        h5_path,
        num_alleles=n_alleles,
        counts_dtype=counts_dtype,
        indices_dtype=indices_dtype,
        chunk_rows=args.chunk_rows,
        chunk_nnz=args.chunk_nnz,
        flush_rows=args.flush_rows,
        compression=comp,
        include_v_genes=not args.no_v_genes,
        attrs=writer_attrs,
    ) as writer:
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            for row in range(start, end):
                ok = False
                for _ in range(args.max_attempts):
                    if args.binder_sampling == "rank_stratified":
                        binder = _sample_binder_rank_stratified(
                            rng,
                            allele_pool,
                            rank_scores,
                            args.b_size,
                        )
                    else:
                        binder = _sample_binder_weighted(
                            rng,
                            allele_pool,
                            allele_probs,
                            args.b_size,
                        )
                    candidates = _iter_unique_arrays(binder.tolist(), donor_lists)
                    if candidates.size < args.n_donors:
                        continue
                    donors = rng.choice(candidates, size=args.n_donors, replace=False)
                    binder_sets[row] = np.sort(binder)
                    donor_indices[row] = donors
                    counts = np.sum(x[donors], axis=0, dtype=np.int32)
                    writer.add_row(
                        loops=loops,
                        n_donors=args.n_donors,
                        cluster_id=row,
                        n_identical=args.n_donors,
                        counts=counts,
                        v_gene_id=None,
                    )
                    ok = True
                    break
                if not ok:
                    raise SystemExit(
                        f"Failed to generate synthetic TCR at row {row} after {args.max_attempts} attempts."
                    )
            if (end % max(batch_size, 1)) == 0 or end == total:
                logger.info("Generated %d/%d synthetic TCRs", end, total)

    meta = {
        "synthetic": True,
        "donor_matrix": str(donor_matrix_path),
        "dataset_size": int(args.dataset_size),
        "b_size": int(args.b_size),
        "n_donors": int(args.n_donors),
        "seed": int(args.seed),
        "binder_sampling": args.binder_sampling,
        "binder_score_source": score_source,
        "binder_sets": binder_path.name,
        "donor_indices": donor_path.name,
        "alleles": int(n_alleles),
        "donors": int(n_donors),
        "nonzero_alleles": int(allele_pool.size),
        "allele_counts": allele_counts.astype(int).tolist(),
        "h5": h5_path.name,
    }
    if args.binder_sampling == "rank_stratified":
        meta["binder_rank_strata"] = int(args.b_size)
    if args.allele_frequencies_json:
        meta["allele_frequencies_json"] = str(
            Path(args.allele_frequencies_json).resolve()
        )
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    logger.info("Wrote %s", binder_path)
    logger.info("Wrote %s", donor_path)
    logger.info("Wrote %s", meta_path)
    logger.info("Wrote %s", h5_path)


if __name__ == "__main__":
    main()
