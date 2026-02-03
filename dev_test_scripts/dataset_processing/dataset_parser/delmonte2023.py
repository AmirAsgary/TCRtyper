# src/dataset_parser/delmonte2023.py
#!/usr/bin/env python3
"""
delmonte2023 donor mapping parser (pre-export):

Data sources (under <root>/metadata):
  - 1-s2.0-S0091674923025447-mmc2.xlsx  (HGRepo ID code ↔ sample_name)
      header: Excel row 3  (pandas header=2)
      data:   rows 4..680  (inclusive) → nrows = 680 - 3 = 677
  - 1-s2.0-S0091674923025447-mmc3.xlsx  (Patient_ID ↔ sample_name)
      header: Excel row 3  (pandas header=2)
      data:   rows 4..104  (inclusive) → nrows = 104 - 3 = 101
      normalize sample_name: '.' → '-' (matches filenames on disk)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Ensure sibling import works when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_donor_samples_from_mapping import merge_donor_samples_from_mapping  # noqa: E402
from tcrtyper.dataset_processing.path_utils import processed_dataset_root  # noqa: E402


def _xlsx_path(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / p)


def load_mmc2(path: Path) -> List[Tuple[str, str]]:
    """Read HGRepo ID code ↔ sample_name (rows 4..680 inclusive)."""
    HEADER_ROW = 2  # 0-based
    NROWS = 680 - 3  # 677 data rows
    need = ["HGRepo ID code", "sample_name"]

    df = pd.read_excel(path, header=HEADER_ROW, nrows=NROWS, engine=None)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")

    out: List[Tuple[str, str]] = []
    for _, r in df.iterrows():
        donor = str(r.get("HGRepo ID code") or "").strip()
        sample = str(r.get("sample_name") or "").strip()
        if donor and sample and not donor.lower().startswith("note"):
            out.append((donor, sample))
    return out


def load_mmc3(path: Path) -> List[Tuple[str, str]]:
    """Read Patient_ID ↔ sample_name (rows 4..104 inclusive); normalize '.'→'-' in sample_name."""
    HEADER_ROW = 2  # 0-based
    NROWS = 104 - 3  # 101 data rows
    need = ["Patient_ID", "sample_name"]

    df = pd.read_excel(path, header=HEADER_ROW, nrows=NROWS, engine=None)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")

    out: List[Tuple[str, str]] = []
    renamed = 0
    for _, r in df.iterrows():
        donor = str(r.get("Patient_ID") or "").strip()
        sample = str(r.get("sample_name") or "").strip()
        if not donor or not sample:
            continue
        sample_norm = sample.replace(".", "-")
        if sample_norm != sample:
            renamed += 1
        out.append((donor, sample_norm))

    if renamed:
        print(f"[info] mmc3: normalized '.' → '-' in {renamed} sample_name(s)")
    return out


def build_map(m2: List[Tuple[str, str]], m3: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    donors: Dict[str, List[str]] = {}
    for donor, sample in m2 + m3:
        donors.setdefault(donor, [])
        if sample not in donors[donor]:
            donors[donor].append(sample)
    # normalize: unique + sort for stability
    for k in donors:
        donors[k] = sorted(donors[k])
    return donors


def main():
    ap = argparse.ArgumentParser(description="delmonte2023: build donor map from Excel, write JSON, and merge donor samples.")
    ap.add_argument("--root", required=True, help="Dataset root (inputs; outputs go to ../processed/<dataset>)")
    ap.add_argument("--mmc2", default="metadata/1-s2.0-S0091674923025447-mmc2.xlsx", help="Path to mmc2.xlsx (relative to --root or absolute)")
    ap.add_argument("--mmc3", default="metadata/1-s2.0-S0091674923025447-mmc3.xlsx", help="Path to mmc3.xlsx (relative to --root or absolute)")
    ap.add_argument("--map-out", default="donor_samples_map.json", help="Output mapping JSON filename (relative to processed/<dataset>)")
    ap.add_argument("--keep-sources", action="store_true", help="Keep original per-sample TSVs after merge")
    ap.add_argument("--encoding", default="utf-8")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Root not found: {root}")
    out_root = processed_dataset_root(root)
    out_root.mkdir(parents=True, exist_ok=True)

    mmc2_path = _xlsx_path(root, args.mmc2)
    mmc3_path = _xlsx_path(root, args.mmc3)
    if not mmc2_path.exists():
        raise SystemExit(f"mmc2 not found: {mmc2_path}")
    if not mmc3_path.exists():
        raise SystemExit(f"mmc3 not found: {mmc3_path}")

    m2 = load_mmc2(mmc2_path)
    m3 = load_mmc3(mmc3_path)
    donors = build_map(m2, m3)

    # Write standardized donor map
    out_map = out_root / args.map_out
    with open(out_map, "w", encoding="utf-8") as f:
        json.dump({"donors": donors}, f, indent=2, ensure_ascii=False)

    # Quick mapping summary
    n_donors = len(donors)
    n_samples = sum(len(v) for v in donors.values())
    singletons = sum(1 for v in donors.values() if len(v) == 1)
    multis = n_donors - singletons
    print(f"Wrote {out_map} | donors={n_donors}, samples={n_samples}, singletons={singletons}, multi={multis}")

    # Merge + HLA composition via the shared merger (unchanged)
    merge_donor_samples_from_mapping(
        out_root,
        donors,
        encoding=args.encoding,
        keep_sources=args.keep_sources,
    )


if __name__ == "__main__":
    main()
