#!/usr/bin/env python3
"""
Aggregate synthetic binder metrics JSONs and create summary heatmaps.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

CLEAN_KS = (5, 10, 15)
DIRTY_KS = (10, 15)
SIMES_QUANTILES = tuple(range(10, 101, 10))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot summary heatmaps from synthetic binder metrics JSONs."
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Root directory containing synthetic datasets (binder_set).",
    )
    ap.add_argument(
        "--metrics-name",
        default="synthetic_metrics.json",
        help="Metrics JSON filename (default: synthetic_metrics.json).",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <root>/processed/summary_plots).",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Plot DPI (default: 120).",
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


def _find_metrics(root: Path, name: str) -> List[Path]:
    return sorted(root.rglob(f"processed/{name}"))


def _heatmap(
    data: np.ndarray,
    x_labels: List[int],
    y_labels: List[int],
    title: str,
    out_path: Path,
    dpi: int,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("b_size")
    ax.set_ylabel("n_donors")
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isfinite(val):
                continue
            rgba = im.cmap(im.norm(val))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if luminance < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _bag_value(record: dict, *, bag: str, k: int, key: str) -> Optional[float]:
    return (
        record.get("bag_purity", {})
        .get(bag, {})
        .get("by_k", {})
        .get(str(k), {})
        .get(key)
    )


def _bag_max_value(record: dict, key: str) -> Optional[float]:
    return (
        record.get("bag_purity", {})
        .get("clean", {})
        .get("max", {})
        .get(key)
    )


def _bag_pos_rate(record: dict, *, bag: str, k: int, neg: bool) -> Optional[float]:
    # TODO: Emit positive-rate metrics at the source once we can regenerate datasets.
    key = "neg_purity_mean" if neg else "pos_purity_mean"
    val = _bag_value(record, bag=bag, k=k, key=key)
    if val is None:
        return None
    if neg:
        return 1.0 - float(val)
    return float(val)


def _bag_max_pos_rate(record: dict, *, neg: bool) -> Optional[float]:
    key = "neg_purity_mean" if neg else "pos_purity_mean"
    val = _bag_max_value(record, key)
    if val is None:
        return None
    if neg:
        return 1.0 - float(val)
    return float(val)


def _simes_entry(record: dict, quantile: int) -> Optional[dict]:
    key = f"simes_{quantile:02d}"
    return record.get("simes_quantiles", {}).get(key)


def _simes_pos_rate(entry: Optional[dict], *, bag: str, k: int, neg: bool) -> Optional[float]:
    if entry is None:
        return None
    key = "neg_purity_mean" if neg else "pos_purity_mean"
    val = (
        entry.get(bag, {})
        .get("by_k", {})
        .get(str(k), {})
        .get(key)
    )
    if val is None:
        return None
    if neg:
        return 1.0 - float(val)
    return float(val)


def _simes_max_pos_rate(entry: Optional[dict], *, neg: bool) -> Optional[float]:
    if entry is None:
        return None
    key = "neg_purity_mean" if neg else "pos_purity_mean"
    val = entry.get("clean", {}).get("max", {}).get(key)
    if val is None:
        return None
    if neg:
        return 1.0 - float(val)
    return float(val)


def main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else root / "processed" / "summary_plots"

    files = _find_metrics(root, args.metrics_name)
    if not files:
        raise SystemExit(f"No metrics files found under {root}")

    records = []
    for path in files:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        records.append(data)

    b_sizes = sorted({int(r.get("b_size", 0)) for r in records})
    n_donors = sorted({int(r.get("n_donors", 0)) for r in records})
    if not b_sizes or not n_donors:
        raise SystemExit("Missing b_size or n_donors in metrics JSONs.")

    b_index = {b: i for i, b in enumerate(b_sizes)}
    n_index = {n: i for i, n in enumerate(n_donors)}

    def _init_matrix() -> np.ndarray:
        return np.full((len(n_donors), len(b_sizes)), np.nan, dtype=np.float64)

    mat_b_mean = _init_matrix()
    mat_nonb_mean = _init_matrix()
    mat_b_present = _init_matrix()
    mat_b_qsig = _init_matrix()

    for r in records:
        b = int(r.get("b_size", 0))
        n = int(r.get("n_donors", 0))
        i = n_index.get(n)
        j = b_index.get(b)
        if i is None or j is None:
            continue
        b_mean = r.get("b_pvals", {}).get("mean_log10")
        if b_mean is None:
            raw = r.get("b_pvals", {}).get("mean")
            if raw is not None:
                b_mean = -np.log10(max(float(raw), 1e-300))
        nonb_mean = r.get("non_b_pvals", {}).get("mean_log10")
        if nonb_mean is None:
            raw = r.get("non_b_pvals", {}).get("mean")
            if raw is not None:
                nonb_mean = -np.log10(max(float(raw), 1e-300))
        if b_mean is not None:
            mat_b_mean[i, j] = b_mean
        if nonb_mean is not None:
            mat_nonb_mean[i, j] = nonb_mean
        mat_b_present[i, j] = r.get("b_present", {}).get("mean_fraction")
        mat_b_qsig[i, j] = r.get("b_qvals", {}).get("frac_le_alpha")

    _heatmap(
        mat_b_mean,
        b_sizes,
        n_donors,
        "Mean -log10(p) (B alleles)",
        out_dir / "mean_pvals_b.png",
        args.dpi,
    )
    _heatmap(
        mat_nonb_mean,
        b_sizes,
        n_donors,
        "Mean -log10(p) (non-B alleles)",
        out_dir / "mean_pvals_nonb.png",
        args.dpi,
    )
    _heatmap(
        mat_b_present,
        b_sizes,
        n_donors,
        "Mean fraction of B alleles observed",
        out_dir / "b_present_fraction.png",
        args.dpi,
    )
    _heatmap(
        mat_b_qsig,
        b_sizes,
        n_donors,
        "Fraction of B alleles with q <= alpha",
        out_dir / "b_qsig_fraction.png",
        args.dpi,
    )

    for k in CLEAN_KS:
        mat_clean_neg = _init_matrix()
        mat_clean_pos = _init_matrix()
        for r in records:
            b = int(r.get("b_size", 0))
            n = int(r.get("n_donors", 0))
            i = n_index.get(n)
            j = b_index.get(b)
            if i is None or j is None:
                continue
            neg_val = _bag_pos_rate(r, bag="clean", k=k, neg=True)
            pos_val = _bag_pos_rate(r, bag="clean", k=k, neg=False)
            if neg_val is not None:
                mat_clean_neg[i, j] = neg_val
            if pos_val is not None:
                mat_clean_pos[i, j] = pos_val
        _heatmap(
            mat_clean_neg,
            b_sizes,
            n_donors,
            f"Clean neg bag positive rate (K={k})",
            out_dir / f"clean_neg_purity_k{k}.png",
            args.dpi,
            vmin=0.0,
            vmax=1.0,
        )
        _heatmap(
            mat_clean_pos,
            b_sizes,
            n_donors,
            f"Clean pos bag positive rate (K={k})",
            out_dir / f"clean_pos_purity_k{k}.png",
            args.dpi,
            vmin=0.0,
            vmax=1.0,
        )

    for k in DIRTY_KS:
        mat_dirty_neg = _init_matrix()
        mat_dirty_pos = _init_matrix()
        for r in records:
            b = int(r.get("b_size", 0))
            n = int(r.get("n_donors", 0))
            i = n_index.get(n)
            j = b_index.get(b)
            if i is None or j is None:
                continue
            neg_val = _bag_pos_rate(r, bag="dirty", k=k, neg=True)
            pos_val = _bag_pos_rate(r, bag="dirty", k=k, neg=False)
            if neg_val is not None:
                mat_dirty_neg[i, j] = neg_val
            if pos_val is not None:
                mat_dirty_pos[i, j] = pos_val
        _heatmap(
            mat_dirty_neg,
            b_sizes,
            n_donors,
            f"Dirty neg bag positive rate (K={k})",
            out_dir / f"dirty_neg_purity_k{k}.png",
            args.dpi,
            vmin=0.0,
            vmax=1.0,
        )
        _heatmap(
            mat_dirty_pos,
            b_sizes,
            n_donors,
            f"Dirty pos bag positive rate (K={k})",
            out_dir / f"dirty_pos_purity_k{k}.png",
            args.dpi,
            vmin=0.0,
            vmax=1.0,
        )

    mat_max_neg = _init_matrix()
    mat_max_pos = _init_matrix()
    for r in records:
        b = int(r.get("b_size", 0))
        n = int(r.get("n_donors", 0))
        i = n_index.get(n)
        j = b_index.get(b)
        if i is None or j is None:
            continue
        neg_val = _bag_max_pos_rate(r, neg=True)
        pos_val = _bag_max_pos_rate(r, neg=False)
        if neg_val is not None:
            mat_max_neg[i, j] = neg_val
        if pos_val is not None:
            mat_max_pos[i, j] = pos_val
    _heatmap(
        mat_max_neg,
        b_sizes,
        n_donors,
        "Max neg bag positive rate",
        out_dir / "max_neg_purity.png",
        args.dpi,
        vmin=0.0,
        vmax=1.0,
    )
    _heatmap(
        mat_max_pos,
        b_sizes,
        n_donors,
        "Max pos bag positive rate",
        out_dir / "max_pos_purity.png",
        args.dpi,
        vmin=0.0,
        vmax=1.0,
    )

    simes_root = out_dir / "simes"
    for q in SIMES_QUANTILES:
        simes_dir = simes_root / f"simes_{q:02d}"

        for k in CLEAN_KS:
            mat_clean_neg = _init_matrix()
            mat_clean_pos = _init_matrix()
            for r in records:
                b = int(r.get("b_size", 0))
                n = int(r.get("n_donors", 0))
                i = n_index.get(n)
                j = b_index.get(b)
                if i is None or j is None:
                    continue
                entry = _simes_entry(r, q)
                neg_val = _simes_pos_rate(entry, bag="clean", k=k, neg=True)
                pos_val = _simes_pos_rate(entry, bag="clean", k=k, neg=False)
                if neg_val is not None:
                    mat_clean_neg[i, j] = neg_val
                if pos_val is not None:
                    mat_clean_pos[i, j] = pos_val
            _heatmap(
                mat_clean_neg,
                b_sizes,
                n_donors,
                f"Clean neg bag positive rate (K={k}, simes top {q}%)",
                simes_dir / f"clean_neg_purity_k{k}.png",
                args.dpi,
                vmin=0.0,
                vmax=1.0,
            )
            _heatmap(
                mat_clean_pos,
                b_sizes,
                n_donors,
                f"Clean pos bag positive rate (K={k}, simes top {q}%)",
                simes_dir / f"clean_pos_purity_k{k}.png",
                args.dpi,
                vmin=0.0,
                vmax=1.0,
            )

        mat_clean_neg_all = _init_matrix()
        mat_clean_pos_all = _init_matrix()
        for r in records:
            b = int(r.get("b_size", 0))
            n = int(r.get("n_donors", 0))
            i = n_index.get(n)
            j = b_index.get(b)
            if i is None or j is None:
                continue
            entry = _simes_entry(r, q)
            neg_val = _simes_max_pos_rate(entry, neg=True)
            pos_val = _simes_max_pos_rate(entry, neg=False)
            if neg_val is not None:
                mat_clean_neg_all[i, j] = neg_val
            if pos_val is not None:
                mat_clean_pos_all[i, j] = pos_val
        _heatmap(
            mat_clean_neg_all,
            b_sizes,
            n_donors,
            f"Clean neg bag positive rate (K=all, simes top {q}%)",
            simes_dir / "clean_neg_purity_kall.png",
            args.dpi,
            vmin=0.0,
            vmax=1.0,
        )
        _heatmap(
            mat_clean_pos_all,
            b_sizes,
            n_donors,
            f"Clean pos bag positive rate (K=all, simes top {q}%)",
            simes_dir / "clean_pos_purity_kall.png",
            args.dpi,
            vmin=0.0,
            vmax=1.0,
        )

        for k in DIRTY_KS:
            mat_dirty_neg = _init_matrix()
            mat_dirty_pos = _init_matrix()
            for r in records:
                b = int(r.get("b_size", 0))
                n = int(r.get("n_donors", 0))
                i = n_index.get(n)
                j = b_index.get(b)
                if i is None or j is None:
                    continue
                entry = _simes_entry(r, q)
                neg_val = _simes_pos_rate(entry, bag="dirty", k=k, neg=True)
                pos_val = _simes_pos_rate(entry, bag="dirty", k=k, neg=False)
                if neg_val is not None:
                    mat_dirty_neg[i, j] = neg_val
                if pos_val is not None:
                    mat_dirty_pos[i, j] = pos_val
            _heatmap(
                mat_dirty_neg,
                b_sizes,
                n_donors,
                f"Dirty neg bag positive rate (K={k}, simes top {q}%)",
                simes_dir / f"dirty_neg_purity_k{k}.png",
                args.dpi,
                vmin=0.0,
                vmax=1.0,
            )
            _heatmap(
                mat_dirty_pos,
                b_sizes,
                n_donors,
                f"Dirty pos bag positive rate (K={k}, simes top {q}%)",
                simes_dir / f"dirty_pos_purity_k{k}.png",
                args.dpi,
                vmin=0.0,
                vmax=1.0,
            )

    logger.info("Wrote summary plots to %s", out_dir)


if __name__ == "__main__":
    main()
