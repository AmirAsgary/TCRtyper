# src/dataset_parser/russell.py
#!/usr/bin/env python3
"""
russell donor mapping + HLA parser (pre-export):

Data sources (under <root>):
  - metadata Excel with columns:
      Cohort, ID,
      A_1, A_2, B_1, B_2, C_1, C_2,
      DPA1_1, DPA1_2, DPB1_1, DPB1_2,
      DQA1_1, DQA1_2, DQB1_1, DQB1_2,
      DRB1_1, DRB1_2, DRB345_1, DRB345_2

    Layout:
        header: Excel row 1  (pandas header=0)
        data:   rows 2..150 inclusive - nrows = 150 - 1 = 149
        donor ID: column "ID" (B)

  - fetch/*.tsv
        sample files, e.g. nica_358_run4_umi5_B.tsv
        donor ID is sample_name.split('_')[1]  ( "358")
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


HLA_I_COLS = ["A_1", "A_2", "B_1", "B_2", "C_1", "C_2"]
HLA_II_COLS = [
    "DPA1_1",
    "DPA1_2",
    "DPB1_1",
    "DPB1_2",
    "DQA1_1",
    "DQA1_2",
    "DQB1_1",
    "DQB1_2",
    "DRB1_1",
    "DRB1_2",
    "DRB345_1",
    "DRB345_2",
]


def _xlsx_path(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / p)


def _normalize_donor_id(v) -> str:
    """Normalize donor ID from Excel into a clean string."""
    if pd.isna(v):
        return ""
    # Excel often stores numeric IDs as floats like 358.0
    try:
        as_int = int(v)
        if float(v) == float(as_int):
            return str(as_int)
    except Exception:
        pass
    return str(v).strip()


def _normalize_hla_allele(v) -> str:
    """
    Normalize HLA allele strings from Excel.

    Examples:
      "A*03:01"     -> "HLA-A*03:01"
      "DPB1*04:01"  -> "HLA-DPB1*04:01"
      "HLA-B*0801"  -> "HLA-B*0801"  (unchanged, normalized prefix)
    """
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if not s:
        return ""
    # treat typical missing markers as empty
    if s.upper() in {"NA", "N/A", "ND", "NONE", "0"}:
        return ""
    if s.upper().startswith("HLA-"):
        # normalize prefix capitalization
        return "HLA-" + s.split("HLA-", 1)[1]
    return f"HLA-{s}"


def load_russell_hla_table(path: Path) -> Dict[str, dict]:
    """
    Load Russell HLA table from Excel.

    Returns:
      { donor_id: {
          "donor_id": <str>,
          "cohort": <str or None>,
          "hla_i": [str],
          "hla_ii": [str],
          "hla_types": [str],
        }, ... }
    """
    HEADER_ROW = 0  # 0-based (Excel row 1)
    NROWS = 150 - 1  # rows 2..150 inclusive

    # Restrict to A..T to match layout if there are extra trailing columns
    df = pd.read_excel(path, header=HEADER_ROW, nrows=NROWS, usecols="A:T", engine=None)

    needed_cols = ["Cohort", "ID"] + HLA_I_COLS + HLA_II_COLS
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")

    by_donor: Dict[str, dict] = {}

    for _, r in df.iterrows():
        donor_id = _normalize_donor_id(r.get("ID"))
        if not donor_id:
            continue

        cohort = str(r.get("Cohort") or "").strip() or None

        def collect(cols: List[str]) -> List[str]:
            alleles: List[str] = []
            for c in cols:
                val = r.get(c)
                allele = _normalize_hla_allele(val)
                if allele:
                    alleles.append(allele)
            return alleles

        hla_i = collect(HLA_I_COLS)
        hla_ii = collect(HLA_II_COLS)
        hla_types = hla_i + hla_ii

        # Overwrite on duplicate IDs, but warn once
        if donor_id in by_donor:
            print(f"[warn] duplicate HLA row for donor ID {donor_id}; overwriting previous entry")

        by_donor[donor_id] = {
            "donor_id": donor_id,
            "cohort": cohort,
            "hla_i": hla_i,
            "hla_ii": hla_ii,
            "hla_types": hla_types,
        }

    return by_donor


def build_map_and_hla(
    root: Path,
    hla_by_donor: Dict[str, dict],
    fetch_dir: Path,
) -> Tuple[Dict[str, List[str]], List[dict]]:
    """
    Build donor_samples_map and per-sample HLA assignments.

    - discovers samples under fetch_dir/*.tsv
    - assumes donor_id == sample_name.split('_')[1]
    """
    if not fetch_dir.is_dir():
        raise SystemExit(f"fetch dir not found: {fetch_dir}")

    donors: Dict[str, List[str]] = {}
    hla_entries: List[dict] = []

    tsv_paths = sorted(fetch_dir.glob("*.tsv"))
    if not tsv_paths:
        print(f"[warn] no TSV files found under {fetch_dir}")

    for tsv in tsv_paths:
        sample_name = tsv.stem
        parts = sample_name.split("_")
        if len(parts) < 2:
            print(f"[warn] could not infer donor ID from sample name '{sample_name}' (expected '<cohort>_<ID>_...'); skipping")
            continue

        donor_id = parts[1]
        donors.setdefault(donor_id, [])
        if sample_name not in donors[donor_id]:
            donors[donor_id].append(sample_name)

        info = hla_by_donor.get(donor_id)
        if info is None:
            print(f"[warn] no HLA row for donor ID {donor_id} (sample {sample_name}); marking as no_hla")
            hla_i: List[str] = []
            hla_ii: List[str] = []
            cohort = None
        else:
            hla_i = list(info.get("hla_i") or [])
            hla_ii = list(info.get("hla_ii") or [])
            cohort = info.get("cohort")

        hla_types = hla_i + hla_ii
        num_hla = len(hla_types)
        status = "ok" if hla_types else "no_hla"

        entry = {
            "sample_name": sample_name,
            "hla_i": hla_i,
            "hla_ii": hla_ii,
            "hla_types": hla_types,
            "num_hla": num_hla,
            # Russell raw sample files live under fetch/, not export/
            "path": f"fetch/{sample_name}.tsv",
            "status": status,
            "donor_id": donor_id,
        }
        if cohort is not None:
            entry["cohort"] = cohort

        hla_entries.append(entry)

    # Normalize: sort and deduplicate sample lists for stability
    for d in donors:
        donors[d] = sorted(set(donors[d]))
    donors = dict(sorted(donors.items(), key=lambda kv: kv[0]))
    hla_entries.sort(key=lambda e: e["sample_name"])

    return donors, hla_entries


def main():
    ap = argparse.ArgumentParser(
        description=(
            "russell: build donor map + per-sample HLA assignments from Excel, "
            "write JSONs, and merge donor samples."
        )
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Dataset root (inputs; outputs go to ../processed/<dataset>)",
    )
    ap.add_argument(
        "--xlsx",
        default="metadata/russell_hla.xlsx",
        help="Path to Russell HLA Excel file (relative to --root or absolute)",
    )
    ap.add_argument(
        "--fetch-dir",
        default="fetch",
        help="Directory (relative to --root or absolute) containing sample TSVs (default: fetch)",
    )
    ap.add_argument(
        "--map-out",
        default="donor_samples_map.json",
        help="Output donor mapping JSON filename (relative to processed/<dataset>)",
    )
    ap.add_argument(
        "--hla-out",
        default="hla_assignments.json",
        help="Output per-sample HLA JSON filename (relative to processed/<dataset>)",
    )
    ap.add_argument("--keep-sources", action="store_true", help="Keep original per-sample TSVs after merge")
    ap.add_argument("--encoding", default="utf-8")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Root not found: {root}")
    out_root = processed_dataset_root(root)
    out_root.mkdir(parents=True, exist_ok=True)

    xlsx_path = _xlsx_path(root, args.xlsx)
    if not xlsx_path.exists():
        raise SystemExit(f"Excel file not found: {xlsx_path}")

    fetch_dir = Path(args.fetch_dir)
    if not fetch_dir.is_absolute():
        fetch_dir = root / fetch_dir

    # Load HLA table from Excel
    hla_by_donor = load_russell_hla_table(xlsx_path)

    # Build donor map + per-sample HLA assignments
    donors, hla_entries = build_map_and_hla(root, hla_by_donor, fetch_dir)

    # Write standardized donor map
    map_path = out_root / args.map_out
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump({"donors": donors}, f, indent=2, ensure_ascii=False)

    # Write per-sample HLA assignments
    hla_path = out_root / args.hla_out
    with open(hla_path, "w", encoding="utf-8") as f:
        json.dump(hla_entries, f, indent=2, ensure_ascii=False)

    # Quick summaries
    n_donors = len(donors)
    n_samples = sum(len(v) for v in donors.values())
    print(f"Wrote {map_path} | donors={n_donors}, samples={n_samples}")

    print(f"Wrote {hla_path} | hla_samples={len(hla_entries)}")

    # Merge + HLA composition via the shared merger (unchanged)
    merge_donor_samples_from_mapping(
        out_root,
        donors,
        encoding=args.encoding,
        keep_sources=args.keep_sources,
    )


if __name__ == "__main__":
    main()
