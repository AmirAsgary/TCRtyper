#!/usr/bin/env python3
"""
Shared helpers for HLA parsing and assignment status computation.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Mapping, Optional, Sequence, Tuple


class HLAClass(str, Enum):
    I = "I"
    II = "II"
    DROP = "DROP"


class HLAStatus(str, Enum):
    OK = "ok"
    PARTIAL = "partial"
    NO_HLA = "no_hla"


CLASS_I_GROUP_PREFIXES = ("HLA-A", "HLA-B", "HLA-C")
CLASS_II_GROUP_PREFIXES = ("HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1", "HLA-DRB1")
DROP_PREFIXES = ("HLA-E", "HLA-F", "HLA-G")

_ALLELE_CORE_RE = re.compile(r"^(?:HLA-)?(?P<gene>[A-Z0-9]+)\*(?P<allele>.+)$")
_LOCUS_RE = re.compile(r"^HLA-([A-Z0-9]+)\*")

CLASS_I_LOCI_ORDER = ["A", "B", "C"]
CLASS_II_LOCI_ORDER = [
    "DPA1",
    "DPB1",
    "DQA1",
    "DQB1",
    "DRA1",
    "DRB1",
    "DRB3",
    "DRB4",
    "DRB5",
]
DRB_EXCLUDE_LOCI = {"DRB3", "DRB4", "DRB5"}


def static_hla_loci(*, ignore_dra1_locus: bool = True) -> dict:
    class_i = list(CLASS_I_LOCI_ORDER)
    class_ii = list(CLASS_II_LOCI_ORDER)
    if ignore_dra1_locus:
        class_ii = [l for l in class_ii if l != "DRA1"]
    expected_loci = class_i + class_ii
    flag_loci = [l for l in expected_loci if l not in DRB_EXCLUDE_LOCI]
    class_ii_flag_loci = [l for l in class_ii if l not in DRB_EXCLUDE_LOCI]
    return {
        "class_i": class_i,
        "class_ii": class_ii,
        "expected": expected_loci,
        "flag_loci": flag_loci,
        "class_ii_flag_loci": class_ii_flag_loci,
    }


def classify_hla_tag(tag: Optional[str]) -> Optional[HLAClass]:
    if not tag:
        return None
    u = tag.strip().upper()
    if not u.startswith("HLA"):
        return None
    if u.startswith(DROP_PREFIXES):
        return HLAClass.DROP
    if u.startswith(CLASS_I_GROUP_PREFIXES):
        return HLAClass.I
    if u.startswith(CLASS_II_GROUP_PREFIXES):
        return HLAClass.II
    return None


def split_hla_tags(sample_tags_raw: Optional[str]) -> Tuple[List[str], List[str]]:
    """Split a comma-separated sample_tags string into (hla_i, hla_ii)."""
    if sample_tags_raw is None:
        sample_tags_raw = ""
    tags = [t.strip() for t in sample_tags_raw.split(",") if t.strip()]
    hla_i: List[str] = []
    hla_ii: List[str] = []
    for t in tags:
        cls = classify_hla_tag(t)
        if cls is HLAClass.I:
            hla_i.append(t)
        elif cls is HLAClass.II:
            hla_ii.append(t)
    return hla_i, hla_ii


def class_status(hla_tags: Sequence[str], group_prefixes: Sequence[str]) -> str:
    if not hla_tags:
        return HLAStatus.NO_HLA.value

    present_groups = set()
    for tag in hla_tags:
        u = str(tag).upper()
        for gp in group_prefixes:
            if u.startswith(gp):
                present_groups.add(gp)

    if len(present_groups) == len(group_prefixes):
        return HLAStatus.OK.value
    return HLAStatus.PARTIAL.value


def global_status(has_tcr: bool, hla_i_status: str, hla_ii_status: str) -> str:
    if not has_tcr:
        return "no_tcr_seq"

    if (
        hla_i_status == HLAStatus.NO_HLA.value
        and hla_ii_status == HLAStatus.NO_HLA.value
    ):
        return HLAStatus.NO_HLA.value

    if hla_i_status == HLAStatus.OK.value and hla_ii_status == HLAStatus.OK.value:
        return HLAStatus.OK.value

    return HLAStatus.PARTIAL.value


def coerce_hla_allele(allele: Optional[str]) -> Optional[str]:
    """Normalize HLA allele prefix/casing without altering field separators."""
    if not allele:
        return None
    s = re.sub(r"\s+", "", str(allele).strip().upper())
    if not s:
        return None
    m = _ALLELE_CORE_RE.match(s)
    if not m:
        return None
    gene, rest = m.group("gene"), m.group("allele")
    if not gene or not rest:
        return None
    return f"HLA-{gene}*{rest}"


def normalize_allele_imgt(allele: Optional[str]) -> Optional[str]:
    """Normalize allele names to IMGT-like colon-delimited format."""
    norm, _ = normalize_allele_imgt_with_metadata(allele)
    return norm


def normalize_allele_imgt_with_metadata(
    allele: Optional[str],
) -> Tuple[Optional[str], dict]:
    """
    Normalize allele names to IMGT-like colon-delimited format with metadata.
    Drops incomplete alleles with <4 digits (after removing non-digits).
    """
    meta = {
        "raw": "" if allele is None else str(allele),
        "coerced": None,
        "normalized": None,
        "status": "dropped",
        "drop_reason": "",
        "change_rules": [],
    }
    if allele is None:
        meta["drop_reason"] = "missing"
        return None, meta
    raw_str = str(allele).strip()
    if not raw_str:
        meta["drop_reason"] = "missing"
        return None, meta
    cleaned = re.sub(r"\s+", "", raw_str).upper()
    if not cleaned:
        meta["drop_reason"] = "missing"
        return None, meta
    m = _ALLELE_CORE_RE.match(cleaned)
    if not m:
        meta["drop_reason"] = "unparseable"
        return None, meta
    gene = m.group("gene")
    rest = m.group("allele")
    if not gene or not rest:
        meta["drop_reason"] = "unparseable"
        return None, meta
    gene_prefix = f"HLA-{gene}"
    coerced = f"{gene_prefix}*{rest}"
    meta["coerced"] = coerced

    rules = set()
    if not cleaned.startswith("HLA-"):
        rules.add("add_prefix")

    suf_m = re.search(r"([A-Z]+)$", rest)
    suffix = suf_m.group(1) if suf_m else ""
    core = rest[: -len(suffix)] if suffix else rest
    if not core:
        meta["drop_reason"] = "empty_core"
        return None, meta

    def _pad_group(group: str, length: int) -> str:
        return group.rjust(length, "0")

    if ":" in core:
        fields = core.split(":")
        norm_fields: List[str] = []
        for i, field in enumerate(fields):
            digits = re.sub(r"\D", "", field)
            if not digits:
                meta["drop_reason"] = "invalid_field"
                return None, meta
            if i == 0:
                if len(digits) == 1:
                    digits = _pad_group(digits, 2)
                    rules.add("pad_first_field")
                if len(digits) == 3:
                    rules.add("three_digit_first_field")
                elif len(digits) != 2:
                    meta["drop_reason"] = "invalid_first_field"
                    return None, meta
            elif i == 1:
                if len(digits) == 1:
                    digits = _pad_group(digits, 2)
                    rules.add("pad_second_field")
                if len(digits) not in (2, 3):
                    meta["drop_reason"] = "invalid_second_field"
                    return None, meta
                if len(digits) == 3:
                    rules.add("three_digit_second_field")
            else:
                if len(digits) == 1:
                    digits = _pad_group(digits, 2)
                    rules.add("pad_field")
                if len(digits) != 2:
                    meta["drop_reason"] = "invalid_field_length"
                    return None, meta
            norm_fields.append(digits)
        total_digits = sum(len(f) for f in norm_fields)
        if total_digits < 4:
            meta["drop_reason"] = "insufficient_digits"
            return None, meta
        allele_norm = ":".join(norm_fields)
    else:
        digits = re.sub(r"\D", "", core)
        if len(digits) < 4:
            meta["drop_reason"] = "insufficient_digits"
            return None, meta
        rules.add("insert_colons")
        if len(digits) == 5 and digits.endswith("01"):
            first = digits[:3]
            second = digits[3:]
            rules.add("suffix01_first_field")
            rules.add("three_digit_first_field")
            fields = [first, second]
            rest_digits = ""
        else:
            first = digits[:2]
            rest_digits = digits[2:]
            if len(rest_digits) % 2 == 1:
                second = rest_digits[:3]
                rest_digits = rest_digits[3:]
                rules.add("three_digit_second_field")
            else:
                second = rest_digits[:2]
                rest_digits = rest_digits[2:]
            fields = [first, second]
        for i in range(0, len(rest_digits), 2):
            chunk = rest_digits[i : i + 2]
            if len(chunk) != 2:
                meta["drop_reason"] = "invalid_field_length"
                return None, meta
            fields.append(chunk)
        allele_norm = ":".join(fields)

    normalized = f"{gene_prefix}*{allele_norm}{suffix}"
    meta["normalized"] = normalized
    meta["status"] = "ok"
    meta["change_rules"] = sorted(rules)
    return normalized, meta


def normalize_allele(allele: Optional[str]) -> Optional[str]:
    """Legacy entrypoint: preserve colon formatting and normalize prefix only."""
    return coerce_hla_allele(allele)


def normalize_hla_list_with_metadata(
    alleles: Sequence[str],
) -> Tuple[List[str], List[dict]]:
    """Normalize a list of alleles to IMGT, returning metadata per input."""
    out: List[str] = []
    meta: List[dict] = []
    seen = set()
    for a in alleles or []:
        norm, info = normalize_allele_imgt_with_metadata(a)
        meta.append(info)
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out, meta


def normalize_hla_list(alleles: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Normalize a list of alleles to IMGT, dropping invalid/incomplete entries."""
    out, meta = normalize_hla_list_with_metadata(alleles)
    dropped = [m["raw"] for m in meta if m.get("status") == "dropped"]
    return out, dropped


def normalize_hla_entry(
    entry: Mapping[str, Sequence[str]],
) -> Tuple[List[str], List[str], List[str], List[dict]]:
    raw_types = list(entry.get("hla_types") or entry.get("hla_all") or [])
    if not raw_types:
        raw_types = list(entry.get("hla_i") or []) + list(entry.get("hla_ii") or [])
    hla_types, meta = normalize_hla_list_with_metadata(raw_types)
    hla_i = [a for a in hla_types if classify_hla_tag(a) == HLAClass.I]
    hla_ii = [a for a in hla_types if classify_hla_tag(a) == HLAClass.II]
    return hla_i, hla_ii, hla_types, meta


def build_hla_entry(
    *,
    sample_name: str,
    rel_path: Optional[str],
    hla_i: Sequence[str],
    hla_ii: Sequence[str],
    has_tcr: bool,
) -> dict:
    hla_i_list = list(hla_i)
    hla_ii_list = list(hla_ii)
    hla_types = hla_i_list + hla_ii_list

    hla_i_status = class_status(hla_i_list, CLASS_I_GROUP_PREFIXES)
    hla_ii_status = class_status(hla_ii_list, CLASS_II_GROUP_PREFIXES)
    status = global_status(has_tcr, hla_i_status, hla_ii_status)

    return {
        "sample_name": sample_name,
        "path": rel_path,
        "status": status,
        "num_hla": len(hla_types),
        "hla_i": hla_i_list,
        "hla_ii": hla_ii_list,
        "hla_types": hla_types,
        "hla_all": hla_types,
        "hla_i_status": hla_i_status,
        "hla_ii_status": hla_ii_status,
    }


def _parse_hla_locus(allele: Optional[str]) -> Optional[str]:
    if not allele:
        return None
    m = _LOCUS_RE.match(str(allele).strip().upper())
    return m.group(1) if m else None


def donor_hla_typing_flags(
    hla_types: Sequence[str], *, ignore_dra1_locus: bool = True
) -> dict:
    """
    Compute donor HLA completeness flags using the same locus logic as
    plot_donor_hla_typing.py (A/B/C + DPA1/DPB1/DQA1/DQB1/DRA1/DRB1; DRB3/4/5 excluded for flags).
    """
    seen = set()
    alleles: List[str] = []
    for a in hla_types or []:
        s = str(a).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        alleles.append(s)

    locus_counts: dict[str, int] = {}
    for allele in alleles:
        locus = _parse_hla_locus(allele)
        if not locus:
            continue
        locus_counts[locus] = locus_counts.get(locus, 0) + 1

    def _all_loci_at_least(loci: List[str], min_count: int) -> bool:
        return all(locus_counts.get(locus, 0) >= min_count for locus in loci)

    loci = static_hla_loci(ignore_dra1_locus=ignore_dra1_locus)
    expected_loci = loci["expected"]
    flag_loci = loci["flag_loci"]
    class_ii_flag_loci = loci["class_ii_flag_loci"]

    fully_typed_all_locuses_two_alleles = _all_loci_at_least(flag_loci, 2)
    fully_typed_all_locuses_contain_allele = _all_loci_at_least(flag_loci, 1)
    not_fully_typed = not fully_typed_all_locuses_contain_allele
    fully_typed_hla_i = _all_loci_at_least(loci["class_i"], 1)
    fully_typed_hla_ii = _all_loci_at_least(class_ii_flag_loci, 1)
    full_hla_i_only = not_fully_typed and fully_typed_hla_i
    full_hla_ii_only = not_fully_typed and fully_typed_hla_ii

    num_hla_i = sum(locus_counts.get(locus, 0) for locus in loci["class_i"])
    num_hla_ii = sum(locus_counts.get(locus, 0) for locus in loci["class_ii"])
    num_hla_total = len(alleles)

    return {
        "fully_typed_all_locuses_two_alleles": fully_typed_all_locuses_two_alleles,
        "fully_typed_all_locuses_contain_allele": fully_typed_all_locuses_contain_allele,
        "full_hla_i_only": full_hla_i_only,
        "full_hla_ii_only": full_hla_ii_only,
        "not_fully_typed": not_fully_typed,
        "fully_typed_hla_i": fully_typed_hla_i,
        "fully_typed_hla_ii": fully_typed_hla_ii,
        "num_hla_i": int(num_hla_i),
        "num_hla_ii": int(num_hla_ii),
        "num_hla_total": int(num_hla_total),
        "hla_typing_ignore_dra1_locus": bool(ignore_dra1_locus),
    }
