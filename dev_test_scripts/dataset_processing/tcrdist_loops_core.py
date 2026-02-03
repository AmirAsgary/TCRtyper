#!/usr/bin/env python3
"""
tcrdist_loops_core.py

Minimal, dataset-agnostic wrapper around tcrdist3 for a single beta chain.

Input:  a Pandas DataFrame with at least
    - 'cdr3_b_aa'  (beta-chain CDR3 amino acid)
    - 'v_b_gene'   (IMGT V gene, e.g. 'TRBV5-1*01')
    - 'j_b_gene'   (IMGT J gene, e.g. 'TRBJ1-2*01')
    - 'count'      (clone abundance; if missing, set to 1)

This module calls tcrdist3.TCRrep with:
    compute_distances = False
    deduplicate       = False

So each input row yields exactly one output row, and we only use tcrdist
for germline loop inference.

Output columns:
    cdr3aa, cdr2aa_gapped, cdr1aa_gapped, cdr2.5aa_gapped,
    v_b_gene, j_b_gene, count
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from tcrdist.repertoire import TCRrep
import tcrdist.repertoire_db as rdb

OUTPUT_COLUMNS = [
    "cdr3aa",
    "cdr2aa_gapped",
    "cdr1aa_gapped",
    "cdr2.5aa_gapped",
    "v_b_gene",
    "j_b_gene",
    "count",
]


def resolve_db_file() -> Optional[str]:
    """
    Return the db filename tcrdist3 expects (basename only),
    e.g. 'alphabeta_gammadelta_db.tsv' if present.
    """
    base = getattr(rdb, "path_to_db", None)
    if not base or not os.path.isdir(base):
        return None
    for fn in ("alphabeta_gammadelta_db.tsv", "alphabeta_db.tsv"):
        p = os.path.join(base, fn)
        if os.path.exists(p):
            return os.path.basename(p)
    return None


def run_tcrdist_loops(
    cell_df: pd.DataFrame,
    *,
    organism: str = "human",
    chain: str = "beta",
    db_file: Optional[str] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Given a standardized beta-chain DataFrame, infer CDR1/CDR2/CDR2.5 from V gene.

    Parameters
    ----------
    cell_df : pd.DataFrame
        Must contain 'cdr3_b_aa', 'v_b_gene', 'j_b_gene'.
        If 'count' is missing, it is created and set to 1.
    organism : str
        'human' or 'mouse'. Default: 'human'.
    chain : str
        TCR chain. For now only 'beta' is used here.
    db_file : str or None
        tcrdist3 db filename, e.g. 'alphabeta_gammadelta_db.tsv'.
        If None, resolve_db_file() is used.
    debug : bool
        If True, print simple diagnostics.

    Returns
    -------
    pd.DataFrame with columns:
        cdr3aa, cdr2aa_gapped, cdr1aa_gapped, cdr2.5aa_gapped,
        v_b_gene, j_b_gene, count
    """
    required = ["cdr3_b_aa", "v_b_gene", "j_b_gene"]
    missing = [c for c in required if c not in cell_df.columns]
    if missing:
        raise ValueError(f"cell_df missing required columns: {missing}")

    df = cell_df.copy()

    if "count" not in df.columns:
        df["count"] = 1

    # Basic sanity: keep only rows with non-empty V/J
    mask = (
        df["v_b_gene"].notna()
        & (df["v_b_gene"].astype(str) != "")
        & df["j_b_gene"].notna()
        & (df["j_b_gene"].astype(str) != "")
    )
    dropped = int((~mask).sum())
    if dropped and debug:
        print(
            f"[tcrdist_loops_core] dropping {dropped} rows with missing V/J before TCRrep"
        )

    df = df.loc[mask].copy()
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    if db_file is None:
        db_file = resolve_db_file()

    # Use tcrdist only to infer loops; no distance computation.
    tr = TCRrep(
        cell_df=df,
        organism=organism,
        chains=[chain],
        db_file=db_file,
        compute_distances=False,
        deduplicate=False,
    )

    cdf = tr.cell_df.copy()
    for need in ["cdr1_b_aa", "cdr2_b_aa", "pmhc_b_aa"]:
        if need not in cdf.columns:
            cdf[need] = ""

    out = pd.DataFrame(
        {
            "cdr3aa": cdf["cdr3_b_aa"].astype(str),
            "cdr2aa_gapped": cdf["cdr2_b_aa"].astype(str),
            "cdr1aa_gapped": cdf["cdr1_b_aa"].astype(str),
            "cdr2.5aa_gapped": cdf["pmhc_b_aa"].astype(str),
            "v_b_gene": cdf["v_b_gene"].astype(str),
            "j_b_gene": cdf["j_b_gene"].astype(str),
            "count": cdf["count"],
        }
    )

    return out
