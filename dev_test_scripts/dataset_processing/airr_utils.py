#!/usr/bin/env python3
"""
Helpers for parsing AIRR Data Commons metadata JSON exports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from tcrtyper.dataset_processing.hla_utils import (
    HLAClass,
    classify_hla_tag,
    coerce_hla_allele,
)


def load_airr_metadata(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"AIRR metadata JSON not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    reps = payload.get("Repertoire")
    if reps is None:
        raise KeyError(f"Missing 'Repertoire' in AIRR metadata: {path}")
    if not isinstance(reps, list):
        raise TypeError("'Repertoire' must be a list")
    return [r for r in reps if isinstance(r, dict)]


def find_airr_metadata(dataset_dir: Path) -> Path:
    candidates = sorted(dataset_dir.glob("*metadata.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No metadata JSON found under {dataset_dir} (expected *metadata.json)."
        )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise FileNotFoundError(
            f"Multiple metadata JSON files found under {dataset_dir}: {names}"
        )
    return candidates[0]


def get_repertoire_id(rep: dict) -> Optional[str]:
    rid = rep.get("repertoire_id")
    return str(rid).strip() if rid else None


def get_subject_id(rep: dict) -> Optional[str]:
    subject = rep.get("subject") or {}
    sid = subject.get("subject_id")
    return str(sid).strip() if sid else None


def _dedupe_preserve(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_hla_from_repertoire(rep: dict) -> Tuple[List[str], List[str]]:
    subject = rep.get("subject") or {}
    genotype = subject.get("genotype") or {}
    mhc_set = genotype.get("mhc_genotype_set") or {}
    mhc_list = mhc_set.get("mhc_genotype_list") or []

    hla_i: List[str] = []
    hla_ii: List[str] = []

    for entry in mhc_list if isinstance(mhc_list, list) else []:
        if not isinstance(entry, dict):
            continue
        mhc_class = str(entry.get("mhc_class") or "").upper()
        alleles = entry.get("mhc_alleles") or []
        is_class_ii = "MHC-II" in mhc_class or "MHC II" in mhc_class
        is_class_i = "MHC-I" in mhc_class or "MHC I" in mhc_class
        for allele in alleles if isinstance(alleles, list) else []:
            if not isinstance(allele, dict):
                continue
            raw = allele.get("allele_designation")
            norm = coerce_hla_allele(raw)
            if not norm:
                continue
            if is_class_ii:
                hla_ii.append(norm)
            elif is_class_i:
                hla_i.append(norm)
            else:
                cls = classify_hla_tag(norm)
                if cls == HLAClass.I:
                    hla_i.append(norm)
                elif cls == HLAClass.II:
                    hla_ii.append(norm)

    return _dedupe_preserve(hla_i), _dedupe_preserve(hla_ii)
