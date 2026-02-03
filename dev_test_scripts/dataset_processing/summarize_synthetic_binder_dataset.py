#!/usr/bin/env python3
"""
Summarize Fisher p-values for synthetic binder datasets.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata, wilcoxon

from tcrtyper.dataset_processing.utils import PublicTcrHlaCsrReader
from tcrtyper.dataset_processing.synthetic_analysis_utils import (
    allele_counts_from_freqs,
    bh_fdr,
    compute_pvals,
    load_allele_frequencies,
    z_scores,
)

logger = logging.getLogger(__name__)

CLEAN_KS = (5, 10, 15)
DIRTY_KS = (10, 15)
SIMES_QUANTILES = tuple(range(10, 101, 10))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Summarize Fisher p-values for synthetic binder datasets."
    )
    ap.add_argument(
        "--dataset-dir",
        required=True,
        help="Synthetic dataset directory (b*/n*/N*).",
    )
    ap.add_argument(
        "--h5",
        default=None,
        help="HDF5 counts path (default: <dataset-dir>/synthetic_tcr_hla_counts.h5).",
    )
    ap.add_argument(
        "--binder-sets",
        default=None,
        help="Binder set .npy (default: <dataset-dir>/synthetic_binder_sets.npy).",
    )
    ap.add_argument(
        "--meta",
        default=None,
        help="Metadata JSON (default: <dataset-dir>/synthetic_meta.json).",
    )
    ap.add_argument(
        "--donor-frequencies-json",
        required=True,
        help="JSON mapping allele id to donor-level frequency.",
    )
    ap.add_argument(
        "--out-json",
        default=None,
        help="Output JSON (default: <dataset-dir>/processed/synthetic_metrics.json).",
    )
    ap.add_argument(
        "--min-overlap",
        type=int,
        default=2,
        help="Minimum overlap for Fisher test (default: 2).",
    )
    ap.add_argument(
        "--fdr-alpha",
        type=float,
        default=0.05,
        help="FDR threshold (default: 0.05).",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Write simple p-value distribution plots.",
    )
    ap.add_argument(
        "--max-tcrs",
        type=int,
        default=0,
        help="Optional cap on number of TCRs to process (default: 0 = all).",
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


def _safe_mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(values))


def _safe_nanmean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return None
    return float(np.nanmean(arr))


def _masked_nanmean(values: list[float], mask: np.ndarray) -> Optional[float]:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size != mask.size:
        raise ValueError("Mask size does not match values.")
    sub = arr[mask]
    if sub.size == 0 or np.all(np.isnan(sub)):
        return None
    return float(np.nanmean(sub))


def _masked_frac_ge(values: np.ndarray, mask: np.ndarray, threshold: float) -> Optional[float]:
    if values.size != mask.size:
        raise ValueError("Mask size does not match values.")
    denom = int(np.sum(mask))
    if denom == 0:
        return None
    return float(np.sum((values >= threshold) & mask)) / float(denom)


def _safe_mean_log10(values: list[float]) -> Optional[float]:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(-np.log10(np.maximum(arr, 1e-300))))


def _safe_median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.median(values))


def _safe_std(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.std(values))


def _auc_from_scores(scores_b: np.ndarray, scores_nonb: np.ndarray) -> Optional[float]:
    """
    Compute per-TCR AUC with Mann-Whitney U using -log10(p) scores.
    Returns P(score_B > score_nonB); 0.5 means no separation.
    """
    if scores_b.size == 0 or scores_nonb.size == 0:
        return None
    combined = np.concatenate([scores_b, scores_nonb])
    ranks = rankdata(combined, method="average")
    n_b = float(scores_b.size)
    n_nonb = float(scores_nonb.size)
    rank_sum = float(np.sum(ranks[: scores_b.size]))
    u_stat = rank_sum - n_b * (n_b + 1.0) / 2.0
    return float(u_stat / (n_b * n_nonb))


def _bag_purity(bag: np.ndarray, binder: np.ndarray, *, mode: str) -> Optional[float]:
    # TODO: Emit positive-rate metrics directly once we can regenerate datasets.
    if bag.size == 0:
        return None
    in_b = np.isin(bag, binder)
    if mode == "pos":
        return float(np.mean(in_b))
    if mode == "neg":
        return float(np.mean(~in_b))
    raise ValueError(f"Unknown bag purity mode: {mode}")


def _simes_pvalue(pvals: np.ndarray) -> Optional[float]:
    if pvals.size == 0:
        return None
    ordered = np.sort(pvals)
    m = float(ordered.size)
    ranks = np.arange(1.0, m + 1.0, dtype=np.float64)
    simes = np.min(ordered * m / ranks)
    return float(min(simes, 1.0))


def main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)

    dataset_dir = Path(args.dataset_dir).resolve()
    if args.h5:
        h5_path = Path(args.h5).resolve()
    else:
        h5_path = dataset_dir / "synthetic_tcr_hla_counts.h5"
    if args.binder_sets:
        binder_path = Path(args.binder_sets).resolve()
    else:
        binder_path = dataset_dir / "synthetic_binder_sets.npy"
    if args.meta:
        meta_path = Path(args.meta).resolve()
    else:
        meta_path = dataset_dir / "synthetic_meta.json"
    if args.out_json:
        out_json = Path(args.out_json).resolve()
    else:
        out_json = dataset_dir / "processed" / "synthetic_metrics.json"

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {h5_path}")
    if not binder_path.exists():
        raise FileNotFoundError(f"Binder sets not found: {binder_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta JSON not found: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    n_donors_total = int(meta.get("donors", 0))
    if n_donors_total <= 0:
        raise SystemExit("Meta JSON missing valid 'donors' count.")
    b_size = int(meta.get("b_size", 0))
    dataset_size = int(meta.get("dataset_size", 0))
    n_donors_param = int(meta.get("n_donors", 0))

    with PublicTcrHlaCsrReader(h5_path) as reader:
        num_rows = reader.num_rows
        num_alleles = reader.num_alleles

    freqs = load_allele_frequencies(Path(args.donor_frequencies_json).resolve(), num_alleles)
    allele_counts = allele_counts_from_freqs(freqs, n_donors_total)

    binder_sets = np.load(binder_path, mmap_mode="r")
    if binder_sets.shape[0] != num_rows:
        raise SystemExit(
            f"Binder sets rows {binder_sets.shape[0]} != HDF5 rows {num_rows}."
        )

    pvals_all: list[float] = []
    all_b_pvals: list[float] = []
    all_non_b_pvals: list[float] = []
    b_pval_indices: list[int] = []
    b_present_counts: list[int] = []
    b_present_fractions: list[float] = []
    aucs: list[float] = []
    clean_neg_purity = {k: [] for k in CLEAN_KS}
    clean_pos_purity = {k: [] for k in CLEAN_KS}
    dirty_neg_purity = {k: [] for k in DIRTY_KS}
    dirty_pos_purity = {k: [] for k in DIRTY_KS}
    max_neg_purity: list[float] = []
    max_pos_purity: list[float] = []
    neg_sizes: list[float] = []
    simes_scores: list[float] = []

    max_tcrs = int(args.max_tcrs)
    if max_tcrs < 0:
        raise SystemExit("--max-tcrs must be >= 0.")
    if max_tcrs > 0:
        num_rows = min(num_rows, max_tcrs)

    logger.info("Processing %d TCR rows", num_rows)
    log_every = 25000
    allele_ids = np.arange(num_alleles, dtype=np.int64)

    with PublicTcrHlaCsrReader(h5_path) as reader:
        processed = 0
        for row in reader.iter_rows(stop=num_rows):
            processed += 1
            counts = row.counts.astype(np.int64)
            binder = np.asarray(binder_sets[row.row], dtype=np.int64)
            b_present_mask = counts[binder] > 0
            b_present = binder[b_present_mask]
            b_present_set = set(int(x) for x in b_present.tolist())

            b_present_count = int(b_present.size)
            b_present_fraction = (
                float(b_present_count) / float(len(binder)) if len(binder) > 0 else 0.0
            )
            b_present_counts.append(b_present_count)
            b_present_fractions.append(b_present_fraction)

            pvals_row = compute_pvals(
                counts,
                int(row.n_donors),
                allele_counts,
                n_donors_total,
                args.min_overlap,
            )
            scores_all = -np.log10(np.maximum(pvals_row, 1e-300))

            z = z_scores(counts.astype(np.float64), int(row.n_donors), freqs)
            eligible_mask = counts >= args.min_overlap
            neg_mask = (z < 0.0) & eligible_mask
            neg_indices = np.flatnonzero(neg_mask)
            neg_scores = scores_all[neg_indices]
            neg_order = neg_indices[np.argsort(neg_scores, kind="stable")]
            neg_size = int(neg_order.size)
            neg_sizes.append(float(neg_size))
            simes_score = _simes_pvalue(pvals_row[neg_indices])
            simes_scores.append(float(simes_score) if simes_score is not None else np.nan)

            pos_order = np.argsort(scores_all, kind="stable")[::-1]

            if neg_size > 0:
                max_neg_purity.append(_bag_purity(neg_order, binder, mode="neg"))
                max_pos_purity.append(
                    _bag_purity(pos_order[:neg_size], binder, mode="pos")
                )
            else:
                max_neg_purity.append(np.nan)
                max_pos_purity.append(np.nan)

            for k in CLEAN_KS:
                if neg_size >= k:
                    clean_neg_purity[k].append(
                        _bag_purity(neg_order[:k], binder, mode="neg")
                    )
                    clean_pos_purity[k].append(
                        _bag_purity(pos_order[:k], binder, mode="pos")
                    )
                else:
                    clean_neg_purity[k].append(np.nan)
                    clean_pos_purity[k].append(np.nan)

            eligible_count = int(np.sum(eligible_mask))
            for k in DIRTY_KS:
                if eligible_count < k:
                    dirty_neg_purity[k].append(np.nan)
                    dirty_pos_purity[k].append(np.nan)
                    continue
                if neg_size >= k:
                    dirty_neg = neg_order[:k]
                else:
                    needed = k - neg_size
                    nonneg_indices = np.flatnonzero(eligible_mask & ~neg_mask)
                    nonneg_scores = scores_all[nonneg_indices]
                    nonneg_order = nonneg_indices[
                        np.argsort(nonneg_scores, kind="stable")
                    ]
                    dirty_neg = np.concatenate([neg_order, nonneg_order[:needed]])
                dirty_neg_purity[k].append(
                    _bag_purity(dirty_neg, binder, mode="neg")
                )
                dirty_pos_purity[k].append(
                    _bag_purity(pos_order[:k], binder, mode="pos")
                )
            idxs = np.flatnonzero(counts >= args.min_overlap)
            if idxs.size > 0:
                scores_sel = scores_all[idxs]
                in_b_mask = np.isin(idxs, binder)
                auc = _auc_from_scores(scores_sel[in_b_mask], scores_sel[~in_b_mask])
                if auc is not None:
                    aucs.append(auc)
            for a in idxs:
                pval = float(pvals_row[a])
                pvals_all.append(pval)
                current_idx = len(pvals_all) - 1
                if int(a) in b_present_set:
                    all_b_pvals.append(pval)
                    b_pval_indices.append(current_idx)
                else:
                    all_non_b_pvals.append(pval)
            if processed % log_every == 0 or processed == num_rows:
                logger.info("Processed %d/%d TCR rows", processed, num_rows)

    pvals_arr = np.asarray(pvals_all, dtype=np.float64)
    qvals_arr = bh_fdr(pvals_arr)

    all_b_qvals = [float(qvals_arr[i]) for i in b_pval_indices]

    wilcoxon_p = None
    if aucs:
        diffs = np.asarray(aucs, dtype=np.float64) - 0.5
        if np.allclose(diffs, 0.0):
            wilcoxon_p = 1.0
        else:
            try:
                _, pval = wilcoxon(diffs, zero_method="zsplit", alternative="greater")
                wilcoxon_p = float(pval)
            except ValueError:
                wilcoxon_p = None

    neg_size_arr = np.asarray(neg_sizes, dtype=np.float64)
    neg_size_fracs = {
        k: float(np.sum(neg_size_arr >= k)) / float(processed) if processed else None
        for k in set(CLEAN_KS + DIRTY_KS)
    }
    neg_size_stats = {
        "mean": _safe_mean(neg_sizes),
        "min": float(np.min(neg_size_arr)) if neg_size_arr.size else None,
        "max": float(np.max(neg_size_arr)) if neg_size_arr.size else None,
        "std": _safe_std(neg_sizes),
        "frac_gt_0": float(np.sum(neg_size_arr > 0.0)) / float(neg_size_arr.size)
        if neg_size_arr.size
        else None,
    }
    bag_purity = {
        "clean": {
            "k_values": list(CLEAN_KS),
            "by_k": {
                str(k): {
                    "neg_purity_mean": _safe_nanmean(clean_neg_purity[k]),
                    "pos_purity_mean": _safe_nanmean(clean_pos_purity[k]),
                    "frac_neg_size_ge_k": neg_size_fracs.get(k),
                }
                for k in CLEAN_KS
            },
            "max": {
                "neg_purity_mean": _safe_nanmean(max_neg_purity),
                "pos_purity_mean": _safe_nanmean(max_pos_purity),
                "neg_size": neg_size_stats,
            },
        },
        "dirty": {
            "k_values": list(DIRTY_KS),
            "by_k": {
                str(k): {
                    "neg_purity_mean": _safe_nanmean(dirty_neg_purity[k]),
                    "pos_purity_mean": _safe_nanmean(dirty_pos_purity[k]),
                    "frac_neg_size_ge_k": neg_size_fracs.get(k),
                }
                for k in DIRTY_KS
            },
        },
    }

    simes_arr = np.asarray(simes_scores, dtype=np.float64)
    simes_valid = np.isfinite(simes_arr)
    simes_quantiles: dict[str, dict] = {}
    for q in SIMES_QUANTILES:
        if q == 100:
            mask = np.ones(simes_arr.size, dtype=bool)
            threshold = None
        elif np.any(simes_valid):
            threshold = float(np.nanpercentile(simes_arr[simes_valid], 100 - q))
            mask = simes_valid & (simes_arr >= threshold)
        else:
            threshold = None
            mask = np.zeros(simes_arr.size, dtype=bool)

        entry = {
            "quantile": int(q),
            "simes_threshold": threshold,
            "n_tcrs": int(np.sum(mask)),
            "clean": {
                "k_values": list(CLEAN_KS),
                "by_k": {
                    str(k): {
                        "neg_purity_mean": _masked_nanmean(
                            clean_neg_purity[k], mask
                        ),
                        "pos_purity_mean": _masked_nanmean(
                            clean_pos_purity[k], mask
                        ),
                        "frac_neg_size_ge_k": _masked_frac_ge(neg_size_arr, mask, k),
                    }
                    for k in CLEAN_KS
                },
                "max": {
                    "neg_purity_mean": _masked_nanmean(max_neg_purity, mask),
                    "pos_purity_mean": _masked_nanmean(max_pos_purity, mask),
                    "neg_size": {
                        "mean": _masked_nanmean(neg_sizes, mask),
                        "min": float(np.min(neg_size_arr[mask]))
                        if np.any(mask)
                        else None,
                        "max": float(np.max(neg_size_arr[mask]))
                        if np.any(mask)
                        else None,
                        "std": _safe_std(neg_size_arr[mask].tolist())
                        if np.any(mask)
                        else None,
                        "frac_gt_0": float(np.sum((neg_size_arr > 0.0) & mask))
                        / float(np.sum(mask))
                        if np.any(mask)
                        else None,
                    },
                },
            },
            "dirty": {
                "k_values": list(DIRTY_KS),
                "by_k": {
                    str(k): {
                        "neg_purity_mean": _masked_nanmean(
                            dirty_neg_purity[k], mask
                        ),
                        "pos_purity_mean": _masked_nanmean(
                            dirty_pos_purity[k], mask
                        ),
                        "frac_neg_size_ge_k": _masked_frac_ge(neg_size_arr, mask, k),
                    }
                    for k in DIRTY_KS
                },
            },
        }
        simes_quantiles[f"simes_{q:02d}"] = entry

    summary = {
        "dataset_dir": str(dataset_dir),
        "b_size": b_size,
        "n_donors": n_donors_param,
        "dataset_size": dataset_size,
        "num_tcrs": int(processed),
        "min_overlap": int(args.min_overlap),
        "fdr_alpha": float(args.fdr_alpha),
        "median_auc": _safe_median(aucs),
        "mean_auc": _safe_mean(aucs),
        "frac_auc_>_0_5": float(sum(1 for a in aucs if a > 0.5)) / float(len(aucs))
        if aucs
        else None,
        "wilcoxon_p": wilcoxon_p,
        "b_present": {
            "mean_fraction": _safe_mean(b_present_fractions),
            "mean_count": _safe_mean(b_present_counts),
        },
        "b_pvals": {
            "count": int(len(all_b_pvals)),
            "mean": _safe_mean(all_b_pvals),
            "mean_log10": _safe_mean_log10(all_b_pvals),
            "frac_gt_0_05": float(sum(1 for p in all_b_pvals if p > 0.05))
            / float(len(all_b_pvals))
            if all_b_pvals
            else None,
        },
        "non_b_pvals": {
            "count": int(len(all_non_b_pvals)),
            "mean": _safe_mean(all_non_b_pvals),
            "mean_log10": _safe_mean_log10(all_non_b_pvals),
        },
        "b_qvals": {
            "count": int(len(all_b_qvals)),
            "mean": _safe_mean(all_b_qvals),
            "mean_log10": _safe_mean_log10(all_b_qvals),
            "frac_le_alpha": float(sum(1 for q in all_b_qvals if q <= args.fdr_alpha))
            / float(len(all_b_qvals))
            if all_b_qvals
            else None,
        },
        "bag_purity": bag_purity,
        "simes_quantiles": simes_quantiles,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote %s", out_json)

    if args.plot:
        plot_dir = out_json.parent / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        if all_b_pvals and all_non_b_pvals:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(
                all_b_pvals,
                bins=50,
                alpha=0.7,
                label="B alleles",
            )
            ax.hist(
                all_non_b_pvals,
                bins=50,
                alpha=0.7,
                label="Non-B alleles",
            )
            ax.set_xlabel("Fisher p-value")
            ax.set_ylabel("Count")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / "pvalue_histogram.png", dpi=120)
            plt.close(fig)


if __name__ == "__main__":
    main()
