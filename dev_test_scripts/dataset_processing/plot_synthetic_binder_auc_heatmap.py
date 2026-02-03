#!/usr/bin/env python3
"""
Plot a global heatmap of binder identifiability (AUC) across b_size and n_donors.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot a global AUC heatmap for synthetic binder datasets."
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
        "--metric",
        default="median_auc",
        choices=("median_auc", "mean_auc", "frac_auc_>_0_5"),
        help="Metric to plot (default: median_auc).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output path (default: <root>/../binder_auc_heatmap.png).",
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
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
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


def main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_path = root.parent / "binder_auc_heatmap.png"

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

    mat = np.full((len(n_donors), len(b_sizes)), np.nan, dtype=np.float64)
    for r in records:
        b = int(r.get("b_size", 0))
        n = int(r.get("n_donors", 0))
        i = n_index.get(n)
        j = b_index.get(b)
        if i is None or j is None:
            continue
        mat[i, j] = r.get(args.metric)

    title = f"{args.metric} (-log10(p) AUC, B vs non-B)"
    _heatmap(mat, b_sizes, n_donors, title, out_path, args.dpi)
    logger.info("Wrote AUC heatmap to %s", out_path)


if __name__ == "__main__":
    main()
