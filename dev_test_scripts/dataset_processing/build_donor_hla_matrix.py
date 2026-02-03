#!/usr/bin/env python3
"""
Build and persist donor×allele matrix from per-donor mask files.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build donor×allele matrix from patients_index.tsv and mask files."
    )
    ap.add_argument(
        "--export-root",
        default="export_train_dataset",
        help="Export train dataset root (default: export_train_dataset).",
    )
    ap.add_argument(
        "--out-matrix",
        default=None,
        help=(
            "Output matrix path (.npz or .npy). "
            "Default: <export_root>/donor_hla_matrix.npz."
        ),
    )
    ap.add_argument(
        "--out-donors",
        default=None,
        help=(
            "Optional donor keys JSON output (default: only for .npy outputs)."
        ),
    )
    ap.add_argument(
        "--dtype",
        default="uint8",
        choices=("uint8", "bool"),
        help="Matrix dtype (default: uint8).",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
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


def _load_id_to_hla(export_root: Path) -> dict[int, str]:
    candidates = [
        export_root / "id_to_hla.json",
        export_root / "hla_id_to_name.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            out: dict[int, str] = {}
            for k, v in raw.items():
                try:
                    out[int(k)] = str(v)
                except Exception:
                    continue
            if out:
                return out
    raise FileNotFoundError("id_to_hla.json not found under export_root.")


def _load_patient_index(export_root: Path) -> tuple[list[str], list[Path]]:
    idx_path = export_root / "patients_index.tsv"
    if not idx_path.exists():
        raise FileNotFoundError(f"patients_index.tsv not found: {idx_path}")

    donor_keys: list[str] = []
    mask_paths: list[Path] = []

    with open(idx_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"sample_id", "relpath_mask", "dataset"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"patients_index.tsv missing columns: {sorted(missing)}"
            )
        for row in reader:
            dataset = (row.get("dataset") or "").strip()
            sample_id = (row.get("sample_id") or "").strip()
            relpath_mask = (row.get("relpath_mask") or "").strip()
            if not dataset or not sample_id or not relpath_mask:
                continue
            donor_keys.append(f"{dataset}/{sample_id}")
            mask_paths.append(export_root / relpath_mask)

    if not donor_keys:
        raise SystemExit("patients_index.tsv has no valid donor rows.")

    return donor_keys, mask_paths


def main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)

    export_root = Path(args.export_root).resolve()
    out_matrix = (
        Path(args.out_matrix).resolve()
        if args.out_matrix
        else (export_root / "donor_hla_matrix.npz")
    )
    out_donors = Path(args.out_donors).resolve() if args.out_donors else None
    out_ext = out_matrix.suffix.lower()
    if out_ext not in {".npy", ".npz"}:
        raise SystemExit("out-matrix must end with .npy or .npz.")
    write_npz = out_ext == ".npz"
    if out_donors is None and not write_npz:
        out_donors = export_root / "donor_hla_matrix_donors.json"

    id_to_hla = _load_id_to_hla(export_root)
    num_alleles = len(id_to_hla)

    donor_keys, mask_paths = _load_patient_index(export_root)
    n_donors = len(donor_keys)
    dtype = np.uint8 if args.dtype == "uint8" else np.bool_

    out_matrix.parent.mkdir(parents=True, exist_ok=True)
    if out_donors is not None:
        out_donors.parent.mkdir(parents=True, exist_ok=True)

    if write_npz:
        import tempfile

        tmp_fh = tempfile.NamedTemporaryFile(
            prefix="donor_hla_matrix_",
            suffix=".npy",
            dir=str(out_matrix.parent),
            delete=False,
        )
        tmp_path = Path(tmp_fh.name)
        tmp_fh.close()
        matrix_path = tmp_path
    else:
        matrix_path = out_matrix

    logger.info("Writing donor×allele matrix: %s", out_matrix)
    mm = np.lib.format.open_memmap(
        matrix_path,
        mode="w+",
        dtype=dtype,
        shape=(n_donors, num_alleles),
    )

    iterator = enumerate(mask_paths)
    if not args.no_progress:
        iterator = tqdm(iterator, total=n_donors, desc="masks", unit="donor")

    for i, path in iterator:
        if not path.exists():
            raise FileNotFoundError(f"Mask file not found: {path}")
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 1 or arr.shape[0] != num_alleles:
            raise SystemExit(
                f"Mask {path} has shape {arr.shape}, expected ({num_alleles},)"
            )
        row = (arr != 0)
        if dtype == np.uint8:
            row = row.astype(np.uint8, copy=False)
        mm[i] = row

    mm.flush()

    if write_npz:
        donor_arr = np.array(donor_keys, dtype=str)
        np.savez(out_matrix, donor_hla_matrix=mm, donor_keys=donor_arr)
        try:
            matrix_path.unlink()
        except OSError:
            pass
    if out_donors is not None:
        with open(out_donors, "w", encoding="utf-8") as fh:
            json.dump(donor_keys, fh, indent=2)

    logger.info("Wrote %d donors, %d alleles", n_donors, num_alleles)
    if out_donors is not None:
        logger.info("Donor keys JSON: %s", out_donors)


if __name__ == "__main__":
    main()
