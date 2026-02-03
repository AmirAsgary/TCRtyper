#!/usr/bin/env python3
"""
Generate synthetic TCR clusters with known solution allele sets.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _parse_int_list(value: str) -> list[int]:
    items = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        items.append(int(chunk))
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.")
    return items


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate synthetic TCR clusters with known solution sets."
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
        "--out-json",
        default=None,
        help="Output JSON path (default: <export_root>/synthetic_tcrs.json).",
    )
    ap.add_argument(
        "--solution-sizes",
        type=_parse_int_list,
        default="1,2,3",
        help="Comma-separated solution sizes to generate (default: 1,2,3).",
    )
    ap.add_argument(
        "--per-size",
        type=int,
        default=10,
        help="Synthetic clusters per solution size (default: 10).",
    )
    ap.add_argument(
        "--min-donors",
        type=int,
        default=8,
        help="Minimum donors per synthetic TCR (default: 8).",
    )
    ap.add_argument(
        "--max-donors",
        type=int,
        default=80,
        help="Maximum donors per synthetic TCR (default: 80).",
    )
    ap.add_argument(
        "--min-allele-coverage",
        type=int,
        default=1,
        help="Minimum donors per solution allele (default: 1).",
    )
    ap.add_argument(
        "--min-unique-coverage",
        type=int,
        default=0,
        help="Minimum donors uniquely covered per solution allele (default: 0).",
    )
    ap.add_argument(
        "--max-attempts",
        type=int,
        default=200,
        help="Max attempts per synthetic TCR before giving up (default: 200).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed (default: 13).",
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

    if args.out_json:
        out_json = Path(args.out_json).resolve()
    elif args.export_root:
        out_json = Path(args.export_root).resolve() / "synthetic_tcrs.json"
    else:
        out_json = Path("synthetic_tcrs.json").resolve()

    return donor_matrix, out_json


def _load_matrix(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            keys = list(data.keys())
            preferred = ["donor_hla_matrix", "donor_matrix", "matrix", "X", "data"]
            for key in preferred:
                if key in data:
                    return data[key]
            if len(keys) == 1:
                return data[keys[0]]
            raise KeyError(f"npz contains multiple arrays; available keys: {keys}")
    return np.load(path, mmap_mode="r")


def _coverage_stats(x_sub: np.ndarray) -> tuple[list[int], list[int]]:
    if x_sub.size == 0:
        return [], []
    if x_sub.ndim != 2:
        raise ValueError("Expected a 2D submatrix.")
    counts = np.sum(x_sub, axis=1)
    cov = []
    unique = []
    for j in range(x_sub.shape[1]):
        col = x_sub[:, j] != 0
        cov.append(int(np.sum(col)))
        unique.append(int(np.sum(col & (counts == 1))))
    return cov, unique


def main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)

    if args.min_donors < 1:
        raise SystemExit("--min-donors must be >= 1.")
    if args.max_donors < args.min_donors:
        raise SystemExit("--max-donors must be >= --min-donors.")
    if args.per_size < 1:
        raise SystemExit("--per-size must be >= 1.")
    if args.min_allele_coverage < 0:
        raise SystemExit("--min-allele-coverage must be >= 0.")
    if args.min_unique_coverage < 0:
        raise SystemExit("--min-unique-coverage must be >= 0.")

    donor_matrix_path, out_json = _resolve_paths(args)
    if not donor_matrix_path.exists():
        raise FileNotFoundError(f"donor matrix not found: {donor_matrix_path}")

    x = _load_matrix(donor_matrix_path)
    if x.ndim != 2:
        raise SystemExit(f"donor matrix must be 2D, got shape {x.shape}")

    n_donors, n_alleles = x.shape
    rng = np.random.default_rng(args.seed)

    clusters = []
    for size in args.solution_sizes:
        if size < 1 or size > n_alleles:
            raise SystemExit(f"solution size {size} is invalid for A={n_alleles}.")
        for tcr_idx in range(args.per_size):
            ok = False
            for attempt in range(args.max_attempts):
                sol = rng.choice(n_alleles, size=size, replace=False)
                sol = np.sort(sol)
                mask = np.any(x[:, sol] != 0, axis=1)
                candidates = np.flatnonzero(mask)
                if candidates.size < args.min_donors:
                    continue
                max_n = min(args.max_donors, int(candidates.size))
                if max_n < args.min_donors:
                    continue
                n_pick = int(rng.integers(args.min_donors, max_n + 1))
                donor_idx = rng.choice(candidates, size=n_pick, replace=False)
                donor_idx = np.sort(donor_idx)
                cov, unique = _coverage_stats(x[donor_idx][:, sol] != 0)
                minimal = all(u > 0 for u in unique) if unique else False
                if cov and any(c < args.min_allele_coverage for c in cov):
                    continue
                if unique and any(u < args.min_unique_coverage for u in unique):
                    continue
                clusters.append(
                    {
                        "cid": f"syn_s{size}_{tcr_idx:04d}",
                        "donor_indices": donor_idx.tolist(),
                        "solution": sol.tolist(),
                        "solution_size": size,
                        "n_donors": int(donor_idx.size),
                        "solution_is_minimal": bool(minimal),
                        "allele_coverage": cov,
                        "unique_coverage": unique,
                    }
                )
                ok = True
                break
            if not ok:
                raise SystemExit(
                    f"Failed to generate synthetic TCR for size={size} after {args.max_attempts} attempts."
                )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "synthetic": True,
        "donor_matrix": str(donor_matrix_path),
        "solution_sizes": args.solution_sizes,
        "per_size": args.per_size,
        "min_donors": args.min_donors,
        "max_donors": args.max_donors,
        "min_allele_coverage": args.min_allele_coverage,
        "min_unique_coverage": args.min_unique_coverage,
        "clusters": clusters,
    }
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote %d synthetic clusters to %s", len(clusters), out_json)


if __name__ == "__main__":
    main()
