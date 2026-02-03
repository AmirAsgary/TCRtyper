#!/usr/bin/env python3
"""
Compute per-allele donor frequencies from a donor HLA matrix.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute HLA allele frequencies from donor HLA matrix."
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
        "--out-json",
        default=None,
        help="Output JSON path (default: <export_root>/synthetic/binder_set/donor_hla_frequencies.json).",
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
        export_root = Path(args.export_root).resolve()
        out_json = export_root / "synthetic" / "binder_set" / "donor_hla_frequencies.json"
    else:
        out_json = Path("donor_hla_frequencies.json").resolve()

    return donor_matrix, out_json


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


def main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)

    donor_matrix, out_json = _resolve_paths(args)
    if not donor_matrix.exists():
        raise FileNotFoundError(f"donor matrix not found: {donor_matrix}")

    x = _load_matrix(donor_matrix, args.npz_key)
    if x.ndim != 2:
        raise SystemExit(f"donor matrix must be 2D, got shape {x.shape}")

    n_donors, n_alleles = x.shape
    if n_donors <= 0:
        raise SystemExit("donor matrix has zero rows.")

    counts = np.count_nonzero(x, axis=0).astype(np.float64)
    freqs = counts / float(n_donors)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(i): float(freqs[i]) for i in range(n_alleles)}
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Wrote %s", out_json)


if __name__ == "__main__":
    main()
