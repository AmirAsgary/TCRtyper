#!/usr/bin/env python3
# src/assemble_train_dataset.py
#
# Assemble donor-level export from per-dataset donor files + HLA donor assignments.
#
# Expected layout:
#   <BASE>/<processed_subdir_name>/<dataset>/
#       <tcrdist_processed_subdir_name>/<donor_file>.tsv
#       <hla_donor_assignments_filename>
#
# Outputs:
#   <BASE>/<train_export_root_name>/
#       <train_hla_to_id_filename>, <train_id_to_hla_filename>
#       <train_v_gene_to_id_filename>, <train_id_to_v_gene_filename>
#       <train_j_gene_to_id_filename>, <train_id_to_j_gene_filename>
#       <train_patients_index_filename>
#       <dataset>/
#           <train_samples_subdir_name>/<donor_file>.tsv
#           <train_masks_subdir_name>/<donor_file>.npy
#           <train_rewriting_report_filename>
#
# Row keep rule: require cdr3aa present and mapped v_b/j_b (>= 0). Only gapped loops kept.

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from tcrtyper.config import config
from tcrtyper.dataset_processing.hla_utils import donor_hla_typing_flags, static_hla_loci

logger = logging.getLogger(__name__)

OUTPUT_COLS = [
    "cdr3aa",
    "cdr2aa_gapped",
    "cdr1aa_gapped",
    "cdr2.5aa_gapped",
    "v_b",
    "j_b",
    "count",
]
MISSING_STRINGS = {"", "nan", "none", "na", "n/a", "x"}
MAX_LOG_ROWS_PER_SAMPLE = 1000


# ---------------- utils ----------------


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def tsv_header(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        return next(reader, [])


def norm_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() in MISSING_STRINGS else s


def is_missing_value(x) -> bool:
    return norm_str(x) == ""


def pick_count_column(df: pd.DataFrame) -> pd.Series:
    if "count (templates/reads)<old>" in df.columns:
        return (
            pd.to_numeric(df["count (templates/reads)<old>"], errors="coerce")
            .fillna(1)
            .astype(int)
        )
    if "templates" in df.columns:
        return pd.to_numeric(df["templates"], errors="coerce").fillna(1).astype(int)
    if "count" in df.columns:
        return pd.to_numeric(df["count"], errors="coerce").fillna(1).astype(int)
    return pd.Series(1, index=df.index, dtype=int)



def classify_hla(a: str) -> str:
    u = a.upper()
    if u.startswith(("HLA-A", "HLA-B", "HLA-C")):
        return "I"
    if u.startswith(("HLA-DP", "HLA-DQ", "HLA-DR")):
        return "II"
    return "OTHER"


def donor_hlas(entry: Mapping) -> set:
    return set(map(str, (entry.get("hla_i") or []) + (entry.get("hla_ii") or [])))


def compute_hla_frequencies_donors(
    all_assign: Mapping[str, Sequence[Mapping]],
) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for entries in all_assign.values():
        for e in entries:
            for a in donor_hlas(e):
                if a:
                    freq[a] = freq.get(a, 0) + 1
    return freq


def build_hla_id_mappings(
    freq: Mapping[str, int],
) -> Tuple[Dict[str, int], Dict[str, str]]:
    mhc_i = sorted(
        [(a, c) for a, c in freq.items() if classify_hla(a) == "I"],
        key=lambda t: (-t[1], t[0]),
    )
    mhc_ii = sorted(
        [(a, c) for a, c in freq.items() if classify_hla(a) == "II"],
        key=lambda t: (-t[1], t[0]),
    )
    ordered = [a for a, _ in mhc_i] + [a for a, _ in mhc_ii]
    hla_to_id = {a: i for i, a in enumerate(ordered)}
    id_to_hla = {str(i): a for a, i in hla_to_id.items()}
    return hla_to_id, id_to_hla


def unique_hla_mask(hla_to_id: Mapping[str, int], alleles: Iterable[str]) -> np.ndarray:
    m = np.zeros(len(hla_to_id), dtype=np.uint8)
    for a in alleles:
        idx = hla_to_id.get(a)
        if idx is not None:
            m[idx] = 1
    return m



def dataset_roots_strict(base: Path) -> List[Path]:
    ds_root = base / config.data.processed_subdir_name
    if not ds_root.is_dir():
        raise FileNotFoundError(
            f"'{config.data.processed_subdir_name}' subfolder not found under base: {base}"
        )
    roots = sorted([p for p in ds_root.iterdir() if p.is_dir()])
    if not roots:
        raise FileNotFoundError(f"No dataset directories under: {ds_root}")
    return roots


_REQUIRED_FIELDS = {
    "donor_file": str,
    "path": str,
    "used_samples": list,
    "status": str,
    "hla_i": list,
    "hla_ii": list,
    "hla_types": list,
    "num_hla": (int, float),  # accept numeric, cast to int
}


def _validate_entry_types(ds_name: str, idx: int, e: dict) -> dict:
    if not isinstance(e, dict):
        raise TypeError(
            f"[{ds_name}] hla_donor_assignments[{idx}] must be an object/dict, "
            f"got {type(e).__name__}"
        )
    for k, tp in _REQUIRED_FIELDS.items():
        if k not in e:
            raise KeyError(
                f"[{ds_name}] hla_donor_assignments[{idx}] missing required field '{k}'"
            )
        if not isinstance(e[k], tp):
            raise TypeError(
                f"[{ds_name}] field '{k}' has wrong type: expected {tp}, "
                f"got {type(e[k]).__name__}"
            )
    if not all(isinstance(s, str) for s in e["used_samples"]):
        raise TypeError(f"[{ds_name}] 'used_samples' must be a list[str]")
    e = dict(e)
    e["num_hla"] = int(e["num_hla"])
    return e


def load_all_donor_assignments(base: Path) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for droot in dataset_roots_strict(base):
        p = droot / config.data.hla_donor_assignments_filename
        if not p.exists():
            raise FileNotFoundError(
                f"[{droot.name}] missing {config.data.hla_donor_assignments_filename}: {p}"
            )
        with open(p, "r", encoding="utf-8") as fh:
            items = json.load(fh)
        if not isinstance(items, list):
            raise TypeError(
                f"[{droot.name}] {config.data.hla_donor_assignments_filename} "
                f"must be a top-level list"
            )
        validated = [_validate_entry_types(droot.name, i, e) for i, e in enumerate(items)]
        out[droot.name] = validated
    logger.info("Loaded donor assignments for %d dataset(s)", len(out))
    return out



def donor_input_files(
    base: Path, donor_assign: Mapping[str, Sequence[dict]]
) -> Dict[str, List[Path]]:
    per_ds: Dict[str, List[Path]] = {}
    roots_by_name = {r.name: r for r in dataset_roots_strict(base)}
    for ds_name, entries in donor_assign.items():
        droot = roots_by_name[ds_name]
        proc_dir = droot / config.data.tcrdist_processed_subdir_name
        if not proc_dir.is_dir():
            raise FileNotFoundError(
                f"[{ds_name}] '{config.data.tcrdist_processed_subdir_name}' "
                f"directory not found: {proc_dir}"
            )
        allowed = {
            str(e.get("donor_file", "")).strip() for e in entries if e.get("donor_file")
        }
        files = [f for f in sorted(proc_dir.glob("*.tsv")) if f.stem in allowed]
        per_ds[ds_name] = files
    return per_ds


def _ensure_hla_flags(entry: dict, *, ignore_dra1_locus: bool) -> dict:
    if "fully_typed_all_locuses_contain_allele" in entry:
        stored_ignore = entry.get("hla_typing_ignore_dra1_locus")
        if stored_ignore is None or stored_ignore == ignore_dra1_locus:
            return entry
    hla_types = entry.get("hla_types") or entry.get("hla_i") or []
    if not hla_types:
        hla_types = (entry.get("hla_i") or []) + (entry.get("hla_ii") or [])
    entry.update(
        donor_hla_typing_flags(hla_types, ignore_dra1_locus=ignore_dra1_locus)
    )
    return entry



def build_gene_maps(
    per_ds_files: Mapping[str, Sequence[Path]],
    disable_progress: bool = False,
) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, int], Dict[str, str]]:
    v_counter, j_counter = Counter(), Counter()
    all_files = [f for lst in per_ds_files.values() for f in lst]
    for f in tqdm(
        all_files,
        desc="Counting V/J genes",
        unit="file",
        disable=disable_progress,
    ):
        header = tsv_header(f)
        usecols = [c for c in ("v_b_gene", "j_b_gene") if c in header]
        if not usecols:
            continue
        df = pd.read_csv(f, sep="\t", usecols=usecols)
        if "v_b_gene" in df.columns:
            v_counter.update(
                [norm_str(x) for x in df["v_b_gene"] if not is_missing_value(x)]
            )
        if "j_b_gene" in df.columns:
            j_counter.update(
                [norm_str(x) for x in df["j_b_gene"] if not is_missing_value(x)]
            )

    v_sorted = sorted(v_counter.items(), key=lambda t: (-t[1], t[0]))
    j_sorted = sorted(j_counter.items(), key=lambda t: (-t[1], t[0]))

    v_gene_to_id = {g: i for i, (g, _) in enumerate(v_sorted)}
    id_to_v_gene = {str(i): g for g, i in v_gene_to_id.items()}

    j_gene_to_id = {g: i for i, (g, _) in enumerate(j_sorted)}
    id_to_j_gene = {str(i): g for g, i in j_gene_to_id.items()}

    logger.info(
        "Built V/J maps: %d V genes, %d J genes",
        len(v_gene_to_id),
        len(j_gene_to_id),
    )
    return v_gene_to_id, id_to_v_gene, j_gene_to_id, id_to_j_gene



def rewrite_sample_file(
    in_path: Path,
    v_map: Mapping[str, int],
    j_map: Mapping[str, int],
) -> Tuple[pd.DataFrame, dict]:
    header = tsv_header(in_path)
    desired = [
        "cdr3aa",
        "cdr2aa_gapped",
        "cdr1aa_gapped",
        "cdr2.5aa_gapped",
        "v_b_gene",
        "j_b_gene",
        "count (templates/reads)<old>",
        "templates",
        "count",
    ]
    usecols = [c for c in desired if c in header]
    if usecols:
        df = pd.read_csv(in_path, sep="\t", usecols=usecols)
    else:
        # Keep row count without loading unused columns.
        df = pd.read_csv(in_path, sep="\t", usecols=[0])
    log = {
        "input_rows": int(len(df)),
        "output_rows": 0,
        "dropped_rows": 0,
        "dropped_by_col": {},
        "drops_sampled": [],
    }

    out = pd.DataFrame(index=df.index)
    out["cdr3aa"] = df["cdr3aa"].apply(norm_str) if "cdr3aa" in df.columns else ""

    for c in ["cdr2aa_gapped", "cdr1aa_gapped", "cdr2.5aa_gapped"]:
        out[c] = df[c].apply(norm_str) if c in df.columns else ""

    if "v_b_gene" in df.columns:
        out["v_b"] = df["v_b_gene"].apply(norm_str).map(lambda s: v_map.get(s, -1))
    else:
        out["v_b"] = -1
    if "j_b_gene" in df.columns:
        out["j_b"] = df["j_b_gene"].apply(norm_str).map(lambda s: j_map.get(s, -1))
    else:
        out["j_b"] = -1

    out["count"] = pick_count_column(df)

    keep = (
        (~out["cdr3aa"].apply(is_missing_value))
        & (out["v_b"] >= 0)
        & (out["j_b"] >= 0)
    )

    first_missing = pd.Series("", index=out.index, dtype=object)
    first_missing.loc[
        out["cdr3aa"].apply(is_missing_value) & (first_missing == "")
    ] = "cdr3aa"
    first_missing.loc[(out["v_b"] < 0) & (first_missing == "")] = "v_b_gene_unmapped"
    first_missing.loc[(out["j_b"] < 0) & (first_missing == "")] = "j_b_gene_unmapped"

    dropped = out.index[~keep]
    log["dropped_rows"] = int((~keep).sum())
    log["output_rows"] = int(keep.sum())
    log["dropped_by_col"] = {
        "cdr3aa": int((first_missing == "cdr3aa").sum()),
        "v_b_gene_unmapped": int((first_missing == "v_b_gene_unmapped").sum()),
        "j_b_gene_unmapped": int((first_missing == "j_b_gene_unmapped").sum()),
    }
    if log["dropped_rows"] > 0:
        n_sample = min(MAX_LOG_ROWS_PER_SAMPLE, log["dropped_rows"])
        for ridx in dropped[:n_sample]:
            log["drops_sampled"].append(
                {
                    "row_index": int(ridx)
                    if isinstance(ridx, (int, np.integer))
                    else str(ridx),
                    "missing_col": str(first_missing.loc[ridx]),
                }
            )

    cleaned = out.loc[keep].reset_index(drop=True)[OUTPUT_COLS]
    return cleaned, log



def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Assemble donor-level export from per-dataset donor files and "
            "hla_donor_assignments.json. Strict format, fail fast."
        )
    )
    ap.add_argument(
        "--base",
        default=config.data.base_dir,
        help=(
            "Base directory with a "
            f"'{config.data.processed_subdir_name}/' subfolder "
            f"(default: {config.data.base_dir})."
        ),
    )
    ap.add_argument(
        "--out",
        dest="out_name",
        default=config.data.train_export_root_name,
        help=(
            "Output folder name under base (alias of --out-name; "
            f"default: {config.data.train_export_root_name})."
        ),
    )
    ap.add_argument("--out-name", dest="out_name", help=argparse.SUPPRESS)
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging.",
    )
    ap.add_argument(
        "--require-complete-hla",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep only donors fully typed across expected HLA loci "
            "(A/B/C + DPA1/DPB1/DQA1/DQB1/DRA1/DRB1; DRB3/4/5 excluded)."
        ),
    )
    ap.add_argument(
        "--ignore-DRA1-locuse",
        "--ignore-DRA1-locus",
        dest="ignore_dra1_locus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Ignore the DRA1 locus when computing completeness flags. "
            "Enabled by default because DRA1 is often monomorphic and untyped."
        ),
    )
    return ap.parse_args()


def _configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)

    base = Path(args.base).resolve()
    out_root = ensure_dir(base / args.out_name)

    logger.info("Assembling train dataset")
    logger.info("Base: %s", base)
    logger.info("Output root: %s", out_root)
    loci_info = static_hla_loci(ignore_dra1_locus=args.ignore_dra1_locus)
    logger.info(
        "HLA completeness loci (static): class I=%s; class II=%s; "
        "excluded loci=%s; ignore DRA1=%s (monomorphic/untyped in many datasets).",
        ",".join(loci_info["class_i"]),
        ",".join(loci_info["class_ii"]),
        ",".join(sorted({"DRB3", "DRB4", "DRB5"})),
        args.ignore_dra1_locus,
    )

    # 1) Load and validate donor HLA assignments (strict)
    all_donor_assign = load_all_donor_assignments(base)
    total_donors = sum(len(entries) for entries in all_donor_assign.values())
    filter_summary = {
        "enabled": bool(args.require_complete_hla),
        "criteria": "fully_typed_all_locuses_contain_allele",
        "ignore_dra1_locus": bool(args.ignore_dra1_locus),
        "total_donors": int(total_donors),
        "kept_donors": int(total_donors),
        "dropped_donors": 0,
        "dropped_by_dataset": {},
    }
    if args.require_complete_hla:
        filtered: Dict[str, List[dict]] = {}
        dropped_total = 0
        for ds_name, entries in all_donor_assign.items():
            kept: List[dict] = []
            for e in entries:
                _ensure_hla_flags(e, ignore_dra1_locus=args.ignore_dra1_locus)
                if e.get("fully_typed_all_locuses_contain_allele"):
                    kept.append(e)
            dropped = len(entries) - len(kept)
            dropped_total += dropped
            filter_summary["dropped_by_dataset"][ds_name] = {
                "total": len(entries),
                "kept": len(kept),
                "dropped": dropped,
            }
            filtered[ds_name] = kept
        all_donor_assign = filtered
        filter_summary["kept_donors"] = int(total_donors - dropped_total)
        filter_summary["dropped_donors"] = int(dropped_total)
        logger.warning(
            "Dropped %d/%d donors without complete HLA typing.",
            dropped_total,
            total_donors,
        )

    # 2) Build HLA id maps from donor universe
    hla_freq = compute_hla_frequencies_donors(all_donor_assign)
    hla_to_id, id_to_hla = build_hla_id_mappings(hla_freq)
    (out_root / config.data.train_hla_to_id_filename).write_text(
        json.dumps(hla_to_id, indent=2), encoding="utf-8"
    )
    (out_root / config.data.train_id_to_hla_filename).write_text(
        json.dumps(id_to_hla, indent=2), encoding="utf-8"
    )

    # 3) Gather donor TSV inputs
    per_ds_files = donor_input_files(base, all_donor_assign)

    # 4) Build global V/J maps from donor TSVs
    v_gene_to_id, id_to_v_gene, j_gene_to_id, id_to_j_gene = build_gene_maps(
        per_ds_files, disable_progress=args.no_progress
    )
    (out_root / config.data.train_v_gene_to_id_filename).write_text(
        json.dumps(v_gene_to_id, indent=2), encoding="utf-8"
    )
    (out_root / config.data.train_id_to_v_gene_filename).write_text(
        json.dumps(id_to_v_gene, indent=2), encoding="utf-8"
    )
    (out_root / config.data.train_j_gene_to_id_filename).write_text(
        json.dumps(j_gene_to_id, indent=2), encoding="utf-8"
    )
    (out_root / config.data.train_id_to_j_gene_filename).write_text(
        json.dumps(id_to_j_gene, indent=2), encoding="utf-8"
    )

    # 5) Rewrite donors and build masks
    index_rows: List[dict] = []
    for ds_name in tqdm(
        sorted(per_ds_files.keys()),
        desc="Datasets",
        unit="dataset",
        disable=args.no_progress,
    ):
        ds_out = ensure_dir(out_root / ds_name)
        masks_dir = ensure_dir(ds_out / config.data.train_masks_subdir_name)
        samples_dir = ensure_dir(ds_out / config.data.train_samples_subdir_name)

        entries = all_donor_assign.get(ds_name, [])
        by_donor = {
            str(e.get("donor_file", "")).strip(): e
            for e in entries
            if e.get("donor_file")
        }
        ds_report = {"dataset": ds_name, "samples": {}}

        files = per_ds_files.get(ds_name, [])
        if not files:
            (ds_out / config.data.train_rewriting_report_filename).write_text(
                json.dumps(ds_report, indent=2), encoding="utf-8"
            )
            continue

        for in_tsv in tqdm(
            files,
            desc=f"{ds_name} donors",
            unit="donor",
            leave=False,
            disable=args.no_progress,
        ):
            donor_stem = in_tsv.stem
            if donor_stem not in by_donor:
                raise KeyError(
                    f"[{ds_name}] donor TSV '{in_tsv.name}' not present in "
                    f"{config.data.hla_donor_assignments_filename}"
                )

            cleaned_df, log = rewrite_sample_file(in_tsv, v_gene_to_id, j_gene_to_id)
            out_tsv = samples_dir / f"{donor_stem}.tsv"
            cleaned_df.to_csv(out_tsv, sep="\t", index=False)

            mask = unique_hla_mask(hla_to_id, donor_hlas(by_donor[donor_stem]))
            mask_path = masks_dir / f"{donor_stem}.npy"
            np.save(mask_path, mask)

            log["status"] = "ok"
            log["output_file"] = str(out_tsv.relative_to(out_root))
            log["mask_file"] = str(mask_path.relative_to(out_root))
            ds_report["samples"][in_tsv.name] = log

            index_rows.append(
                {
                    "sample_id": in_tsv.name,
                    "relpath_tcr": str(out_tsv.relative_to(out_root)),
                    "relpath_mask": str(mask_path.relative_to(out_root)),
                    "dataset": ds_name,
                }
            )

        (ds_out / config.data.train_rewriting_report_filename).write_text(
            json.dumps(ds_report, indent=2), encoding="utf-8"
        )

    # 6) patients_index.tsv
    idx_path = out_root / config.data.train_patients_index_filename
    if index_rows:
        pd.DataFrame(
            index_rows,
            columns=["sample_id", "relpath_tcr", "relpath_mask", "dataset"],
        ).to_csv(idx_path, sep="\t", index=False)
    else:
        pd.DataFrame(
            columns=["sample_id", "relpath_tcr", "relpath_mask", "dataset"]
        ).to_csv(idx_path, sep="\t", index=False)

    # 7) summary
    summary = {
        "base": str(base),
        "out_root": str(out_root),
        "n_datasets": len(per_ds_files),
        "n_unique_hla": len(hla_to_id),
        "n_patients_indexed": len(index_rows),
        "n_v_genes": len(v_gene_to_id),
        "n_j_genes": len(j_gene_to_id),
        "hla_completeness_filter": filter_summary,
        "hla_completeness_loci": {
            "class_i": loci_info["class_i"],
            "class_ii": loci_info["class_ii"],
            "excluded_loci": ["DRB3", "DRB4", "DRB5"],
            "ignore_dra1_locus": bool(args.ignore_dra1_locus),
            "reason": (
                "Static loci list for consistency across datasets; "
                "DRB3/4/5 excluded from completeness flags; "
                "DRA1 ignored by default because it is often monomorphic and untyped."
            ),
        },
    }
    (out_root / config.data.train_summary_filename).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    logger.info(
        "Assembled train dataset: %d dataset(s), %d patients, %d HLA alleles, "
        "%d V genes, %d J genes",
        summary["n_datasets"],
        summary["n_patients_indexed"],
        summary["n_unique_hla"],
        summary["n_v_genes"],
        summary["n_j_genes"],
    )
    logger.info("Summary written to %s", out_root / config.data.train_summary_filename)


if __name__ == "__main__":
    main()
