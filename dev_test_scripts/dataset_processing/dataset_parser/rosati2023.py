#!/usr/bin/env python3
"""
rosati donor mapping + HLA parser (pre-export):

Data sources (under <root>):
  - meta/filereport_read_run_PRJEB50045.tsv
        columns: run_accession, sample_alias, ...
        sample_alias examples:
            1_CD_3_AlphaBeta.v2
            1_CD_62_Beta.v2
            1_Healthy_100_AlphaBeta.v2
        we parse sample_alias → donor_id ("CD_3", "Healthy_100") and chain type.

  - metadata/HLA_grouped_patients_data3.tsv
        Ortega/HLAGuessr HLA summary for Rosati "data3" patients.
        columns: HLA, locus, n_positive, pos_patients, n_negative, neg_patients
        pos_patients contains comma-separated patient IDs like "CD_60, Healthy_26, ...".

  - processed/ERR10360928_TRB.tsv
        tcrdist-standardized TRB repertoires per run
        (output of tcrdist_rosati_pipeline.py; sample_name = "ERR10360928_TRB").

  - processed_data/ERR10360928_TRB.miXcr_like.tsv
        original miXcr-like TRB clone tables from Ortega.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Ensure sibling import works when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_donor_samples_from_mapping import merge_donor_samples_from_mapping  # noqa: E402
from tcrtyper.dataset_processing.path_utils import processed_dataset_root  # noqa: E402


CLASS_I_LOCI = {"A", "B", "C"}
CLASS_II_LOCI = {"DPA1", "DPB1", "DQA1", "DQB1", "DRB1", "DRB3", "DRB4", "DRB5"}


def _resolve(root: Path, p: str) -> Path:
    """Resolve a possibly-relative path against the dataset root."""
    path = Path(p)
    return path if path.is_absolute() else (root / path)


def parse_sample_alias(alias: str) -> dict:
    """
    Parse Rosati sample_alias strings like:

      1_CD_3_AlphaBeta.v2
      1_CD_62_Beta.v2
      1_CD_71_Alpha.v2
      1_Healthy_100_AlphaBeta.v2

    Returns dict with keys:
      - cohort: "CD" or "Healthy"
      - donor_num: numeric string, e.g. "3" or "100"
      - chain: "Alpha", "Beta", or "AlphaBeta"
      - patient_id: "CD_3" or "Healthy_100"

    If parsing fails, all fields are None.
    """
    s = str(alias or "").strip()
    m = re.match(r"^1_(CD|Healthy)_([0-9]+)_(AlphaBeta|Beta|Alpha)\.v2$", s)
    if not m:
        return {"cohort": None, "donor_num": None, "chain": None, "patient_id": None}
    cohort, num, chain = m.groups()
    patient_id = f"{cohort}_{num}"
    return {"cohort": cohort, "donor_num": num, "chain": chain, "patient_id": patient_id}


def load_rosati_hla_table(hla_grouped_path: Path) -> Dict[str, dict]:
    """
    Load HLA_grouped_patients_data3.tsv and build:

        { donor_id: {
            "donor_id": <str>,
            "hla_i": [str],
            "hla_ii": [str],
            "hla_types": [str],
        }, ... }

    Only patients with IDs like "CD_<num>" or "Healthy_<num>" are kept.
    """
    df = pd.read_csv(hla_grouped_path, sep="\t", dtype=str)
    df = df.rename(columns=lambda c: c.strip())

    needed_cols = {"HLA", "locus", "pos_patients"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise SystemExit(f"{hla_grouped_path}: missing columns {sorted(missing)}")

    df["HLA"] = df["HLA"].astype(str).str.strip()
    df["locus"] = df["locus"].astype(str).str.strip()
    df["pos_patients"] = df["pos_patients"].fillna("").astype(str)

    patient_to_hla_i: Dict[str, set] = defaultdict(set)
    patient_to_hla_ii: Dict[str, set] = defaultdict(set)

    for _, row in df.iterrows():
        allele = row["HLA"]
        locus = row["locus"]
        if not allele or not locus:
            continue

        # Grouped file uses e.g. "A*01:01"; normalize to "HLA-A*01:01".
        full_allele = f"HLA-{allele}"

        if locus in CLASS_I_LOCI:
            target = patient_to_hla_i
        elif locus in CLASS_II_LOCI:
            target = patient_to_hla_ii
        else:
            # Ignore loci tcrdist / downstream code doesn't expect.
            continue

        pats_raw = row["pos_patients"]
        if not isinstance(pats_raw, str):
            continue

        for p in pats_raw.split(","):
            p = p.strip()
            if not p:
                continue
            # Only keep Rosati CD/Healthy donors (data3).
            if not re.match(r"^(CD|Healthy)_\d+$", p):
                continue
            target[p].add(full_allele)

    all_patients = sorted(set(patient_to_hla_i.keys()) | set(patient_to_hla_ii.keys()))

    by_donor: Dict[str, dict] = {}
    for pid in all_patients:
        hla_i = sorted(patient_to_hla_i.get(pid, set()))
        hla_ii = sorted(patient_to_hla_ii.get(pid, set()))
        by_donor[pid] = {
            "donor_id": pid,
            "hla_i": hla_i,
            "hla_ii": hla_ii,
            "hla_types": hla_i + hla_ii,
        }

    print(f"{hla_grouped_path}: HLA entries for {len(by_donor)} Rosati donors (CD_/Healthy_)")
    return by_donor


def build_map_and_hla(
    root: Path,
    meta_path: Path,
    hla_by_donor: Dict[str, dict],
) -> Tuple[Dict[str, List[str]], List[dict]]:
    """
    Build donor_samples_map and per-sample HLA assignments for Rosati.

    - Uses filereport_read_run_PRJEB50045.tsv to map run_accession → patient_id.
    - Only keeps Beta / AlphaBeta runs (TRB-containing).
    - Sample name convention: <run_accession>_TRB
      (matches tcrdist_rosati_pipeline -> processed/<sample_name>.tsv)
    """
    df_meta = pd.read_csv(meta_path, sep="\t", dtype=str)
    df_meta = df_meta.rename(columns=lambda c: c.strip())

    needed_cols = {"run_accession", "sample_alias"}
    missing = needed_cols - set(df_meta.columns)
    if missing:
        raise SystemExit(f"{meta_path}: missing columns {sorted(missing)}")

    df_meta["sample_alias"] = df_meta["sample_alias"].astype(str).str.strip()

    parsed = df_meta["sample_alias"].apply(parse_sample_alias)
    parsed_df = pd.DataFrame(list(parsed))
    df_meta = pd.concat([df_meta, parsed_df], axis=1)

    df_meta_patients = df_meta[df_meta["patient_id"].notna()].copy()
    df_meta_trb = df_meta_patients[df_meta_patients["chain"].isin(["Beta", "AlphaBeta"])].copy()

    print(f"{meta_path}: total rows={len(df_meta)}, parsed_patients={len(df_meta_patients)}, TRB rows={len(df_meta_trb)}")

    donors: Dict[str, List[str]] = {}
    hla_entries: List[dict] = []

    if df_meta_trb.empty:
        print(f"[warn] no TRB (Beta/AlphaBeta) rows found in {meta_path}")

    for _, r in df_meta_trb.iterrows():
        run = str(r["run_accession"]).strip()
        patient_id = r.get("patient_id")
        cohort = r.get("cohort")

        if not patient_id:
            continue

        donor_id = patient_id  # e.g. "CD_3" or "Healthy_100"
        sample_name = f"{run}_TRB"

        donors.setdefault(donor_id, [])
        if sample_name not in donors[donor_id]:
            donors[donor_id].append(sample_name)

        info = hla_by_donor.get(donor_id)
        if info is None:
            print(f"[warn] no HLA entry for donor {donor_id} (run {run}); marking as no_hla")
            hla_i: List[str] = []
            hla_ii: List[str] = []
        else:
            hla_i = list(info.get("hla_i") or [])
            hla_ii = list(info.get("hla_ii") or [])

        hla_types = hla_i + hla_ii
        num_hla = len(hla_types)
        status = "ok" if hla_types else "no_hla"

        # For Rosati, the "raw" TRB clones live in processed_data/<run>_TRB.miXcr_like.tsv
        rel_path = f"processed_data/{run}_TRB.miXcr_like.tsv"

        entry = {
            "sample_name": sample_name,
            "hla_i": hla_i,
            "hla_ii": hla_ii,
            "hla_types": hla_types,
            "num_hla": num_hla,
            "path": rel_path,
            "status": status,
            "donor_id": donor_id,
        }
        if cohort:
            entry["cohort"] = str(cohort)

        hla_entries.append(entry)

    # Normalize for stability: sort and deduplicate
    for d in donors:
        donors[d] = sorted(set(donors[d]))
    donors = dict(sorted(donors.items(), key=lambda kv: kv[0]))
    hla_entries.sort(key=lambda e: e["sample_name"])

    n_donors = len(donors)
    n_samples = sum(len(v) for v in donors.values())
    print(f"Built donors map: donors={n_donors}, samples={n_samples}")
    print(f"Per-sample HLA entries: {len(hla_entries)}")

    # Sanity: warn if some HLA donors never appear in meta
    donors_from_hla = set(hla_by_donor.keys())
    donors_from_meta = set(donors.keys())
    missing_in_meta = sorted(donors_from_hla - donors_from_meta)
    if missing_in_meta:
        print(f"[info] HLA donors not present as TRB runs in meta: {len(missing_in_meta)} "
              f"(examples: {', '.join(missing_in_meta[:20])})")

    return donors, hla_entries


def main():
    ap = argparse.ArgumentParser(
        description=(
            "rosati: build donor map + per-sample HLA assignments "
            "from Ortega/HLAGuessr (data3) + Rosati filereport, then merge donor samples."
        )
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Rosati dataset root (inputs; outputs go to ../processed/<dataset>).",
    )
    ap.add_argument(
        "--meta",
        default="meta/filereport_read_run_PRJEB50045.tsv",
        help="Path to Rosati filereport TSV (relative to --root or absolute).",
    )
    ap.add_argument(
        "--hla-grouped",
        default="metadata/HLA_grouped_patients_data3.tsv",
        help="Path to HLA_grouped_patients_data3.tsv (relative to --root or absolute).",
    )
    ap.add_argument(
        "--map-out",
        default="donor_samples_map.json",
        help="Output donor mapping JSON filename (relative to processed/<dataset>).",
    )
    ap.add_argument(
        "--hla-out",
        default="hla_assignments.json",
        help="Output per-sample HLA JSON filename (relative to processed/<dataset>).",
    )
    ap.add_argument(
        "--keep-sources",
        action="store_true",
        help="Keep per-sample processed TSVs after donor merge (see merge_donor_samples_from_mapping).",
    )
    ap.add_argument(
        "--encoding",
        default="utf-8",
        help="Encoding for JSON output files (default: utf-8).",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Root not found: {root}")
    out_root = processed_dataset_root(root)
    out_root.mkdir(parents=True, exist_ok=True)

    meta_path = _resolve(root, args.meta)
    if not meta_path.exists():
        raise SystemExit(f"Meta file not found: {meta_path}")

    hla_grouped_path = _resolve(root, args.hla_grouped)
    if not hla_grouped_path.exists():
        raise SystemExit(f"HLA grouped file not found: {hla_grouped_path}")

    # 1) Load HLA table from HLAGuessr grouped file
    hla_by_donor = load_rosati_hla_table(hla_grouped_path)

    # 2) Build donor → samples mapping + per-sample HLA entries
    donors, hla_entries = build_map_and_hla(root, meta_path, hla_by_donor)

    # 3) Write standardized donor map JSON
    map_path = out_root / args.map_out
    with open(map_path, "w", encoding=args.encoding) as f:
        json.dump({"donors": donors}, f, indent=2, ensure_ascii=False)
    print(f"Wrote {map_path} | donors={len(donors)}, samples={sum(len(v) for v in donors.values())}")

    # 4) Write per-sample HLA assignments JSON
    hla_path = out_root / args.hla_out
    with open(hla_path, "w", encoding=args.encoding) as f:
        json.dump(hla_entries, f, indent=2, ensure_ascii=False)
    print(f"Wrote {hla_path} | hla_samples={len(hla_entries)}")

    # 5) Merge processed/<sample_name>.tsv into donor_*.tsv + donor-level HLA json
    merge_donor_samples_from_mapping(
        out_root,
        donors,
        encoding=args.encoding,
        keep_sources=args.keep_sources,
    )


if __name__ == "__main__":
    main()
