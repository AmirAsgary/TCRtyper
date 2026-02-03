#!/usr/bin/env python3
"""
Plot per-cluster distributions and class I/II heatmaps for public TCR HLA counts.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tcrtyper.dataset_processing.synthetic_analysis_utils import (
    allele_counts_from_freqs,
    bh_fdr,
    compute_pvals,
    load_allele_frequencies,
    z_scores,
)
from tcrtyper.dataset_processing.utils import PublicTcrHlaCsrReader

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot per-cluster distributions and heatmaps for public TCR HLA counts."
    )
    ap.add_argument(
        "--public-h5",
        required=True,
        help="Path to public_tcr_hla_counts.h5.",
    )
    ap.add_argument(
        "--donor-frequencies-json",
        required=True,
        help="JSON mapping allele id to donor-level frequency.",
    )
    ap.add_argument(
        "--donor-matrix",
        default=None,
        help="Path to donor×allele matrix to infer total donor count.",
    )
    ap.add_argument(
        "--donor-matrix-key",
        default=None,
        help="Array key to use when loading .npz donor matrices.",
    )
    ap.add_argument(
        "--donor-count",
        type=int,
        default=None,
        help="Total donor count (overrides donor matrix inference).",
    )
    ap.add_argument(
        "--id-to-hla",
        default=None,
        help="Allele id->name JSON (default: <export_root>/id_to_hla.json).",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <public-h5-dir>/plots).",
    )
    ap.add_argument(
        "--max-tcrs",
        type=int,
        default=50,
        help="Max clusters to plot and include in heatmaps (default: 50).",
    )
    ap.add_argument(
        "--min-cluster-size",
        type=int,
        default=200,
        help="Minimum donors per cluster to plot (default: 200).",
    )
    ap.add_argument(
        "--min-overlap",
        type=int,
        default=2,
        help="Minimum overlap for Fisher test (default: 2).",
    )
    ap.add_argument(
        "--max-rank",
        type=int,
        default=60,
        help="Max allele rank to show in heatmaps (default: 60).",
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


def _load_id_map(path: Path) -> dict[int, str]:
    if not path.exists():
        logger.warning("id_to_hla.json not found at %s", path)
        return {}
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    out: dict[int, str] = {}
    for key, val in raw.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        out[idx] = str(val)
    return out


def _resolve_export_root(public_h5: Path) -> Optional[Path]:
    parts = list(public_h5.parts)
    for marker in ("clusters", "mmseqs", "synthetic"):
        if marker in parts:
            idx = parts.index(marker)
            if idx > 0:
                return Path(*parts[:idx])
    return public_h5.parent


def _get_hla_class(name: str) -> str:
    u = name.upper()
    if u.startswith("HLA-A") or u.startswith("HLA-B") or u.startswith("HLA-C"):
        return "I"
    return "II"


def _load_matrix_from_npz(data, npz_key: Optional[str]) -> np.ndarray:
    keys = list(data.keys())
    if npz_key:
        if npz_key not in data:
            raise KeyError(f"npz key {npz_key!r} not found. Available keys: {keys}")
        return data[npz_key]
    preferred = ["donor_hla_matrix", "donor_matrix", "matrix", "X", "data"]
    for key in preferred:
        if key in data:
            return data[key]
    if len(keys) == 1:
        return data[keys[0]]
    raise KeyError(f"npz contains multiple arrays; available keys: {keys}")


def _load_matrix(path: Path, npz_key: Optional[str]) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            return _load_matrix_from_npz(data, npz_key=npz_key)
    return np.load(path, mmap_mode="r")


def _infer_donor_count(
    donor_matrix: Path, npz_key: Optional[str]
) -> tuple[int, int]:
    x = _load_matrix(donor_matrix, npz_key)
    if x.ndim != 2:
        raise SystemExit(f"donor matrix must be 2D, got shape {x.shape}")
    n_donors, n_alleles = x.shape
    return int(n_donors), int(n_alleles)


def _compute_tcr_metrics(
    counts_row: np.ndarray,
    n_donors_i: int,
    allele_counts: np.ndarray,
    n_donors_total: int,
    freq_vec: np.ndarray,
    min_overlap: int,
) -> Optional[dict]:
    counts = counts_row.astype(np.int64)
    mask = counts >= min_overlap
    allele_indices = np.flatnonzero(mask)
    if allele_indices.size == 0:
        return None

    counts_sel = counts[allele_indices]
    order = np.argsort(counts_sel)[::-1]
    allele_indices = allele_indices[order]
    counts_sel = counts_sel[order]

    freq_sub = freq_vec[allele_indices]
    z_vals = z_scores(counts_sel, n_donors_i, freq_sub)

    pvals_all = compute_pvals(
        counts,
        n_donors_i,
        allele_counts,
        n_donors_total,
        min_overlap,
    )
    pvals_sel = pvals_all[allele_indices]
    neglog10_p = -np.log10(np.maximum(pvals_sel, 1e-300))
    qvals_all = bh_fdr(pvals_all)
    qvals_sel = qvals_all[allele_indices]
    neglog10_q = -np.log10(np.maximum(qvals_sel, 1e-300))

    return {
        "allele_indices": allele_indices,
        "counts_sel": counts_sel,
        "neglog10_p": neglog10_p,
        "neglog10_q": neglog10_q,
        "z_scores": z_vals,
    }


def _plot_tcr_panel(
    row_idx: int,
    cluster_id: int,
    n_donors_i: int,
    n_identical: int,
    loops: Sequence[str],
    allele_indices: np.ndarray,
    counts_sel: np.ndarray,
    neglog10_p: np.ndarray,
    neglog10_q: np.ndarray,
    z_scores_vals: np.ndarray,
    allele_name_map: dict[int, str],
    out_dir: Path,
    dpi: int,
) -> None:
    logger.info("Plotting TCR row %d (cluster_id=%d).", row_idx, cluster_id)
    allele_labels = [
        allele_name_map.get(int(a), str(int(a))) for a in allele_indices
    ]
    classes = np.array([_get_hla_class(name) for name in allele_labels], dtype=object)
    mask_I = classes == "I"
    mask_II = classes == "II"

    counts_I = counts_sel[mask_I]
    counts_II = counts_sel[mask_II]
    neglog_I = neglog10_p[mask_I]
    neglog_II = neglog10_p[mask_II]
    neglogq_I = neglog10_q[mask_I]
    neglogq_II = neglog10_q[mask_II]
    z_I = z_scores_vals[mask_I]
    z_II = z_scores_vals[mask_II]
    labels_I = np.asarray(allele_labels, dtype=object)[mask_I]
    labels_II = np.asarray(allele_labels, dtype=object)[mask_II]

    max_len = max(len(labels_I), len(labels_II), 1)
    base_width = max(6.0, 0.25 * max_len)
    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(2.0 * base_width, 9.5),
        sharey="row",
    )

    rows = [
        ("Counts", counts_I, counts_II),
        ("-log10(p)", neglog_I, neglog_II),
        ("BH FDR -log10(p)", neglogq_I, neglogq_II),
        ("z = (observed - expected) / std", z_I, z_II),
    ]

    bar_color = "#4c78a8"
    for r, (title, vals_I, vals_II) in enumerate(rows):
        ax_I, ax_II = axes[r]
        if len(vals_I) > 0:
            xs = np.arange(len(vals_I))
            ax_I.bar(xs, vals_I, edgecolor="black", color=bar_color)
            ax_I.set_xticks(xs)
            ax_I.set_xticklabels(labels_I, rotation=90, fontsize=6)
        else:
            ax_I.text(0.5, 0.5, "No class I alleles", ha="center", va="center")
            ax_I.set_xticks([])
        if len(vals_II) > 0:
            xs = np.arange(len(vals_II))
            ax_II.bar(xs, vals_II, edgecolor="black", color=bar_color)
            ax_II.set_xticks(xs)
            ax_II.set_xticklabels(labels_II, rotation=90, fontsize=6)
        else:
            ax_II.text(0.5, 0.5, "No class II alleles", ha="center", va="center")
            ax_II.set_xticks([])
        ax_I.set_title(f"{title} (Class I)")
        ax_II.set_title(f"{title} (Class II)")

    loops_str = " ".join(loops)
    fig.suptitle(
        "Cluster {cluster_id} (row {row}, n_donors={n_donors}, n_identical={n_identical})\n"
        "loops: {loops}".format(
            cluster_id=cluster_id,
            row=row_idx,
            n_donors=n_donors_i,
            n_identical=n_identical,
            loops=loops_str,
        ),
        y=0.99,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cluster_{row_idx:06d}_summary.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Wrote per-TCR plot: %s", out_path)


def _plot_heatmap(
    data: np.ndarray,
    title: str,
    out_path: Path,
    dpi: int,
    vmax: Optional[float] = None,
) -> None:
    logger.info("Rendering heatmap: %s", out_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    if vmax is None:
        vmax = np.nanquantile(data, 0.98) if np.any(np.isfinite(data)) else 1.0
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Allele rank")
    ax.set_ylabel("Cluster index (sampled)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Wrote heatmap: %s", out_path)


def main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)

    public_h5 = Path(args.public_h5).resolve()
    if not public_h5.exists():
        raise FileNotFoundError(f"public HDF5 not found: {public_h5}")

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = public_h5.parent / "plots"

    if args.id_to_hla:
        id_to_hla_path = Path(args.id_to_hla).resolve()
    else:
        export_root = _resolve_export_root(public_h5)
        id_to_hla_path = export_root / "id_to_hla.json" if export_root else Path("id_to_hla.json")

    donor_freq_path = Path(args.donor_frequencies_json).resolve()
    if not donor_freq_path.exists():
        raise FileNotFoundError(f"donor frequency JSON not found: {donor_freq_path}")

    n_donors_total = args.donor_count
    n_alleles_from_matrix = None
    if args.donor_matrix:
        donor_matrix = Path(args.donor_matrix).resolve()
        if not donor_matrix.exists():
            raise FileNotFoundError(f"donor matrix not found: {donor_matrix}")
        n_donors, n_alleles = _infer_donor_count(donor_matrix, args.donor_matrix_key)
        n_alleles_from_matrix = n_alleles
        if n_donors_total is None:
            n_donors_total = n_donors
        elif n_donors_total != n_donors:
            logger.warning(
                "donor count %d does not match donor matrix rows %d",
                n_donors_total,
                n_donors,
            )
    if n_donors_total is None or n_donors_total <= 0:
        raise SystemExit("Provide a valid --donor-count or --donor-matrix.")

    with PublicTcrHlaCsrReader(public_h5) as reader:
        num_rows = reader.num_rows
        num_alleles = reader.num_alleles

    if n_alleles_from_matrix is not None and n_alleles_from_matrix != num_alleles:
        logger.warning(
            "donor matrix alleles %d != HDF5 alleles %d",
            n_alleles_from_matrix,
            num_alleles,
        )

    logger.info("Public HDF5 rows=%d alleles=%d", num_rows, num_alleles)
    logger.info("Total donors: %d", n_donors_total)

    freqs = load_allele_frequencies(donor_freq_path, num_alleles)
    allele_counts = allele_counts_from_freqs(freqs, n_donors_total)

    id_to_hla = _load_id_map(id_to_hla_path)
    logger.info("Loaded id->HLA map entries: %d", len(id_to_hla))

    max_tcrs = int(args.max_tcrs)
    if max_tcrs < 0:
        raise SystemExit("--max-tcrs must be >= 0.")
    if max_tcrs == 0:
        logger.info("Skipping plots (max_tcrs=0).")
        return

    heat_p_I = []
    heat_p_II = []
    plotted = 0
    seen_clusters = set()

    with PublicTcrHlaCsrReader(public_h5) as reader:
        for row in reader:
            if row.cluster_id in seen_clusters:
                continue
            if row.n_donors < args.min_cluster_size:
                seen_clusters.add(row.cluster_id)
                continue
            seen_clusters.add(row.cluster_id)

            metrics = _compute_tcr_metrics(
                row.counts,
                row.n_donors,
                allele_counts,
                n_donors_total,
                freqs,
                args.min_overlap,
            )
            if metrics is None:
                continue

            allele_indices = metrics["allele_indices"]
            allele_labels = [
                id_to_hla.get(int(a), str(int(a))) for a in allele_indices
            ]
            classes = np.array(
                [_get_hla_class(name) for name in allele_labels], dtype=object
            )
            mask_I = classes == "I"
            mask_II = classes == "II"

            neglogp = metrics["neglog10_p"]
            heat_p_I.append(neglogp[mask_I])
            heat_p_II.append(neglogp[mask_II])

            _plot_tcr_panel(
                row.row,
                row.cluster_id,
                row.n_donors,
                row.n_identical_sequences,
                (row.cdr3aa, row.cdr2aa_gapped, row.cdr1aa_gapped, row.cdr2_5aa_gapped),
                allele_indices,
                metrics["counts_sel"],
                metrics["neglog10_p"],
                metrics["neglog10_q"],
                metrics["z_scores"],
                id_to_hla,
                out_dir / "per_tcr",
                args.dpi,
            )
            plotted += 1
            if plotted >= max_tcrs:
                break

    if not heat_p_I and not heat_p_II:
        logger.info("No TCRs selected for plotting.")
        return

    max_rank = int(args.max_rank)
    if max_rank < 1:
        raise SystemExit("--max-rank must be >= 1.")

    def _build_heat_matrix(rows: list[np.ndarray]) -> np.ndarray:
        max_len = max((len(r) for r in rows), default=0)
        rank = min(max_len, max_rank)
        mat = np.full((len(rows), rank), np.nan, dtype=np.float64)
        for i, arr in enumerate(rows):
            if arr.size == 0:
                continue
            cut = min(rank, arr.size)
            mat[i, :cut] = arr[:cut]
        return mat

    p_I_mat = _build_heat_matrix(heat_p_I)
    p_II_mat = _build_heat_matrix(heat_p_II)

    _plot_heatmap(
        p_I_mat,
        "-log10(p) (Class I)",
        out_dir / "heatmaps" / "pvals_class_I.png",
        args.dpi,
    )
    _plot_heatmap(
        p_II_mat,
        "-log10(p) (Class II)",
        out_dir / "heatmaps" / "pvals_class_II.png",
        args.dpi,
    )

    logger.info("Wrote plots to %s", out_dir)


if __name__ == "__main__":
    main()
