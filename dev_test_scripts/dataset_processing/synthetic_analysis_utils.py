#!/usr/bin/env python3
"""
Shared helpers for synthetic binder dataset analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact


def load_allele_frequencies(path: Path, n_alleles: int) -> np.ndarray:
    """
    Load allele frequency JSON as a dense float array.
    Accepts dict (id -> freq) or list (index -> freq).
    """
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, list):
        freqs = np.asarray(payload, dtype=np.float64)
        if freqs.size != n_alleles:
            raise SystemExit(
                f"Frequency list length {freqs.size} != num alleles {n_alleles}."
            )
        return freqs
    if isinstance(payload, dict):
        freqs = np.zeros(n_alleles, dtype=np.float64)
        for key, val in payload.items():
            idx = int(key)
            if idx < 0 or idx >= n_alleles:
                raise SystemExit(f"Allele id {idx} out of range 0..{n_alleles - 1}.")
            freqs[idx] = float(val)
        return freqs
    raise SystemExit("Frequency JSON must be a dict or list.")


def allele_counts_from_freqs(freqs: np.ndarray, n_donors_total: int) -> np.ndarray:
    counts = np.rint(freqs * float(n_donors_total)).astype(np.int64)
    return np.clip(counts, 0, int(n_donors_total))


def compute_pvals(
    counts: np.ndarray,
    n_donors_i: int,
    allele_counts: np.ndarray,
    n_donors_total: int,
    min_overlap: int,
) -> np.ndarray:
    """
    Fisher exact p-values (greater) for each allele in a counts row.
    Alleles with counts < min_overlap keep p=1.0.
    """
    pvals = np.ones_like(counts, dtype=np.float64)
    idxs = np.flatnonzero(counts >= min_overlap)
    for a in idxs:
        d_ia = int(counts[a])
        N_a = int(allele_counts[a])
        if N_a < d_ia:
            N_a = d_ia
        if N_a > n_donors_total:
            N_a = n_donors_total
        n11 = d_ia
        n10 = int(n_donors_i - d_ia)
        n01 = int(N_a - d_ia)
        n00 = int(n_donors_total - n11 - n10 - n01)
        if min(n11, n10, n01, n00) < 0:
            continue
        table = np.array([[n11, n10], [n01, n00]], dtype=np.int64)
        _, p = fisher_exact(table, alternative="greater")
        pvals[a] = float(p)
    return pvals


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    if pvals.size == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    m = float(len(ranked))
    qvals = np.empty_like(ranked)
    prev = 1.0
    for i in range(len(ranked) - 1, -1, -1):
        rank = float(i + 1)
        q = ranked[i] * m / rank
        if q > prev:
            q = prev
        prev = q
        qvals[i] = min(q, 1.0)
    out = np.empty_like(qvals)
    out[order] = qvals
    return out


def scaled_counts_norm(counts_sel: np.ndarray, freq_sub: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where(freq_sub > 0.0, counts_sel / freq_sub, 0.0)
    max_s = np.nanmax(scaled) if np.any(np.isfinite(scaled)) else 0.0
    if max_s > 0:
        return scaled / max_s
    return np.zeros_like(scaled, dtype=np.float64)


def z_scores(counts_sel: np.ndarray, n_donors_i: int, freq_sub: np.ndarray) -> np.ndarray:
    expected = float(n_donors_i) * freq_sub
    var = float(n_donors_i) * freq_sub * (1.0 - freq_sub)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(var > 0.0, (counts_sel - expected) / np.sqrt(var), 0.0)
