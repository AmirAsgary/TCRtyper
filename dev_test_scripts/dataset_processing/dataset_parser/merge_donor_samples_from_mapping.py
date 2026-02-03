# src/merge_donor_samples_from_mapping.py
#!/usr/bin/env python3
"""
Merge donor samples from a standardized mapping JSON and compose HLA donor assignments.

Input (under <root>):
  - donor_samples_map.json
      {
        "donors": {
          "<donor_id>": ["sample_a", "sample_b", ...],
          ...
        }
      }
  - processed/<sample>.tsv
  - hla_assignments.json (per-sample; UNCHANGED)

Outputs:
  - processed/donor_<DONOR>.tsv       (for singleton donors)
  - processed/merge_<DONOR>.tsv       (for multi-sample donors)
  - hla_donor_assignments.json        (per donor; explicit status)
  - console summary

Status rules (per donor):
  - ok                    : all non-empty HLA sets equal and none empty
  - partial_hla           : non-empty sets equal but some samples have empty HLA
  - conflict_hla_mismatch : two or more non-empty sets differ
  - no_hla                : all samples have empty (or missing) HLA
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

from tcrtyper.dataset_processing.path_utils import processed_dataset_root
from tcrtyper.dataset_processing.hla_utils import (
    HLAClass,
    classify_hla_tag,
    donor_hla_typing_flags,
    normalize_hla_list_with_metadata,
)

logger = logging.getLogger(__name__)
SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")
DATA_SUBDIR = "processed"


def sanitize(x: str, max_len: int = 120) -> str:
    s = (SAFE_CHARS_RE.sub("_", str(x).strip())).strip("_")
    return s[:max_len] if s else "unknown"

def singleton_basename(donor: str) -> str:
    return f"donor_{sanitize(donor)}"

def multi_basename(donor: str) -> str:
    return f"merge_{sanitize(donor)}"

def src_tsv(root: Path, sample: str) -> Path:
    return root / DATA_SUBDIR / f"{sample}.tsv"

def dst_tsv_singleton(root: Path, donor: str) -> Path:
    return root / DATA_SUBDIR / f"{singleton_basename(donor)}.tsv"

def dst_tsv_multi(root: Path, donor: str) -> Path:
    return root / DATA_SUBDIR / f"{multi_basename(donor)}.tsv"

def concat_tsvs(sources: List[Path], dest: Path, encoding: str) -> int:
    """Write header from first file, then append data lines from all; returns data rows written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    header_written = False
    with open(dest, "w", encoding=encoding, newline="") as out_f:
        for src in sources:
            if not src.exists():
                print(f"[warn] missing TSV: {src}")
                continue
            with open(src, "r", encoding=encoding, newline="") as in_f:
                for i, line in enumerate(in_f):
                    if i == 0:
                        if not header_written:
                            out_f.write(line)
                            header_written = True
                        continue
                    out_f.write(line)
                    rows += 1
    return rows


def _load_hla_by_sample(root: Path) -> Dict[str, dict]:
    p = root / "hla_assignments.json"
    if not p.exists():
        raise SystemExit(f"hla_assignments.json not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise SystemExit("hla_assignments.json: expected a list")
    by_name: Dict[str, dict] = {}
    for e in items:
        if isinstance(e, dict):
            name = str(e.get("sample_name") or "").strip()
            if name and name not in by_name:
                by_name[name] = e
    return by_name

def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def _update_stats(
    stats: dict,
    meta: List[dict],
    *,
    dataset: str,
    sample_name: str,
) -> tuple[bool, bool]:
    donor_changed = False
    donor_dropped = False
    for info in meta:
        stats["alleles_total"] += 1
        raw = str(info.get("raw") or "").strip()
        raw_label = raw or (info.get("coerced") or "").strip()
        raw_key = (info.get("coerced") or raw_label).strip()
        if info.get("status") == "dropped":
            donor_dropped = True
            stats["alleles_dropped"] += 1
            reason = info.get("drop_reason") or "unknown"
            stats["drop_reasons"][reason] = stats["drop_reasons"].get(reason, 0) + 1
            stats["dropped_counts"][raw_key] = stats["dropped_counts"].get(raw_key, 0) + 1
            logger.info(
                "[%s][%s] dropped allele '%s' (reason=%s)",
                dataset,
                sample_name,
                raw_label,
                reason,
            )
            continue

        rules = info.get("change_rules") or []
        if rules:
            donor_changed = True
            stats["alleles_changed"] += 1
            for rule in rules:
                stats["change_rules"][rule] = stats["change_rules"].get(rule, 0) + 1
            normalized = info.get("normalized") or raw_label
            key = f"{raw_key} -> {normalized}"
            stats["changed_pairs"][key] = stats["changed_pairs"].get(key, 0) + 1
            logger.info(
                "[%s][%s] normalized allele '%s' -> '%s' (rules=%s)",
                dataset,
                sample_name,
                raw_label,
                normalized,
                ",".join(rules),
            )

    return donor_changed, donor_dropped


def _compose_hla_for_donors(
    root: Path,
    donor_map: Dict[str, List[str]],
) -> Tuple[List[dict], dict]:
    """Return list of donor HLA dicts and write hla_donor_assignments.json."""
    by_sample = _load_hla_by_sample(root)
    out: List[dict] = []
    status_counter = Counter()
    dataset = root.name
    stats = {
        "dataset": dataset,
        "donors_total": 0,
        "donors_with_changes": 0,
        "donors_with_drops": 0,
        "donors_empty_after_drop": 0,
        "alleles_total": 0,
        "alleles_dropped": 0,
        "alleles_changed": 0,
        "drop_reasons": {},
        "change_rules": {},
        "changed_pairs": {},
        "dropped_counts": {},
    }

    for donor, samples in sorted(donor_map.items()):
        stats["donors_total"] += 1
        donor_changed = False
        donor_dropped = False
        had_raw_alleles = False
        had_norm_alleles = False
        # Decide output basename/path based on multiplicity
        if len(samples) == 1:
            base = singleton_basename(donor)
        else:
            base = multi_basename(donor)
        export_path = f"export/{base}.tsv"
        used_samples = [f"export/{s}.tsv" for s in samples]

        # Gather entries and HLA sets
        entries = [by_sample[s] for s in samples if s in by_sample]
        sets = []
        first_nonempty = None
        first_nonempty_types: List[str] = []
        for e in entries:
            sample_name = str(e.get("sample_name") or "").strip()
            raw_list = (
                e.get("hla_types")
                or e.get("hla_all")
                or (e.get("hla_i") or []) + (e.get("hla_ii") or [])
            )
            if raw_list:
                had_raw_alleles = True
            htypes, meta = normalize_hla_list_with_metadata(raw_list)
            if htypes:
                had_norm_alleles = True
            changed, dropped = _update_stats(
                stats,
                meta,
                dataset=dataset,
                sample_name=sample_name,
            )
            donor_changed = donor_changed or changed
            donor_dropped = donor_dropped or dropped
            sset = frozenset(htypes)
            sets.append(sset)
            if sset and first_nonempty is None:
                first_nonempty = e
                first_nonempty_types = htypes

        if first_nonempty:
            nonempty_sets = [s for s in sets if s]
            equal = all(s == nonempty_sets[0] for s in nonempty_sets)
            empties = any(not s for s in sets)
            if not equal:
                status = "conflict_hla_mismatch"
            elif empties:
                status = "partial_hla"
            else:
                status = "ok"

            hla_i = [a for a in first_nonempty_types if classify_hla_tag(a) == HLAClass.I]
            hla_ii = [a for a in first_nonempty_types if classify_hla_tag(a) == HLAClass.II]
            hla_types = hla_i + hla_ii
            num_hla = len(hla_types)
        else:
            status = "no_hla"
            hla_i, hla_ii, hla_types, num_hla = [], [], [], 0

        status_counter[status] += 1

        if donor_changed:
            stats["donors_with_changes"] += 1
        if donor_dropped:
            stats["donors_with_drops"] += 1
        if had_raw_alleles and not had_norm_alleles:
            stats["donors_empty_after_drop"] += 1

        flags = donor_hla_typing_flags(hla_types, ignore_dra1_locus=True)
        out.append({
            "donor_file": base,          # without .tsv
            "path": export_path,
            "used_samples": used_samples,
            "status": status,
            "hla_i": hla_i,
            "hla_ii": hla_ii,
            "hla_types": hla_types,
            "num_hla": num_hla,
            **flags,
        })

    out_path = root / "hla_donor_assignments.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    summary_path = root / "hla_donor_assignments_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Print concise summary
    total = sum(status_counter.values())
    parts = ", ".join(f"{k}={status_counter[k]}" for k in sorted(status_counter.keys()))
    print(f"Wrote {out_path} with {total} donors | {parts}")
    print(f"Wrote {summary_path}")
    return out, stats


def merge_donor_samples_from_mapping(
    root: Path,
    donor_map: Dict[str, List[str]],
    *,
    encoding: str = "utf-8",
    keep_sources: bool = False,
) -> dict:
    """Merge TSVs per donor (singleton→donor_, multi→merge_) and compose HLA donor assignments."""
    _configure_logging()
    # Merge TSVs
    (root / DATA_SUBDIR).mkdir(exist_ok=True)

    missing_counts = 0
    merged_single = merged_multi = 0

    for donor, samples in sorted(donor_map.items()):
        sources = [src_tsv(root, s) for s in samples if src_tsv(root, s).exists()]
        if not sources:
            print(f"[warn] donor {donor}: no source TSVs found for samples {samples[:5]}{' ...' if len(samples)>5 else ''}")
            missing_counts += 1
            continue

        if len(sources) == 1:
            dest = dst_tsv_singleton(root, donor)
            if sources[0] != dest:
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(sources[0], dest)
            else:
                dest.touch(exist_ok=True)
            merged_single += 1
        else:
            dest = dst_tsv_multi(root, donor)
            concat_tsvs(sources, dest, encoding)
            if not keep_sources:
                for s in sources:
                    try:
                        if s.exists() and s != dest:
                            s.unlink()
                    except Exception as e:
                        print(f"[warn] could not remove {s}: {e}")
            merged_multi += 1

    # Compose HLA donor file
    donor_hla, hla_stats = _compose_hla_for_donors(root, donor_map)

    # Summary
    summary = {
        "donors_total": len(donor_map),
        "merged_singleton": merged_single,
        "merged_multi": merged_multi,
        "donors_missing_sources": missing_counts,
        "hla_donors_written": len(donor_hla),
        "hla_summary_path": str((root / "hla_donor_assignments_summary.json")),
    }
    print(f"Merge summary: {summary}")
    return summary


def _load_map_json(path: Path) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "donors" not in data:
        raise SystemExit(f"{path}: expected an object with a 'donors' field")
    donors = data["donors"]
    if not isinstance(donors, dict):
        raise SystemExit(f"{path}: 'donors' must be an object mapping donor_id -> [samples]")
    # normalize
    out: Dict[str, List[str]] = {}
    for k, v in donors.items():
        if not isinstance(v, list):
            raise SystemExit(f"{path}: donor '{k}' must map to a list of sample names")
        out[k] = sorted({str(x).strip() for x in v if str(x).strip()})
    return out

def main():
    ap = argparse.ArgumentParser(description="Merge donor samples from mapping and compose HLA donor assignments.")
    ap.add_argument("--root", required=True, help="Dataset root (outputs go to ../processed/<dataset>)")
    ap.add_argument("--map", default="donor_samples_map.json", help="Path to donor mapping JSON (default: <root>/donor_samples_map.json)")
    ap.add_argument("--keep-sources", action="store_true", help="Keep original per-sample TSVs after merge")
    ap.add_argument("--encoding", default="utf-8")
    args = ap.parse_args()
    _configure_logging()

    root = Path(args.root).resolve()
    out_root = processed_dataset_root(root)
    out_root.mkdir(parents=True, exist_ok=True)

    if Path(args.map).is_absolute():
        map_path = Path(args.map)
    else:
        map_path = out_root / args.map
        if not map_path.exists():
            map_path = root / args.map
    donor_map = _load_map_json(map_path)

    print(f"Loaded donor map: donors={len(donor_map)} | samples={sum(len(v) for v in donor_map.values())}")
    merge_donor_samples_from_mapping(
        out_root,
        donor_map,
        encoding=args.encoding,
        keep_sources=args.keep_sources,
    )

if __name__ == "__main__":
    main()
