#!/usr/bin/env python3
"""
Plot per-TCR distributions and heatmaps for synthetic binder datasets.
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

from tcrtyper.dataset_processing.utils import PublicTcrHlaCsrReader
from tcrtyper.dataset_processing.synthetic_analysis_utils import (
    allele_counts_from_freqs,
    bh_fdr,
    compute_pvals,
    load_allele_frequencies,
    scaled_counts_norm,
    z_scores,
)

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot per-TCR distributions and heatmaps for synthetic binder datasets."
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
        "--meta",
        default=None,
        help="Metadata JSON (default: <dataset-dir>/synthetic_meta.json).",
    )
    ap.add_argument(
        "--binder-sets",
        default=None,
        help="Binder set .npy (default: <dataset-dir>/synthetic_binder_sets.npy).",
    )
    ap.add_argument(
        "--donor-frequencies-json",
        required=True,
        help="JSON mapping allele id to donor-level frequency.",
    )
    ap.add_argument(
        "--id-to-hla",
        default=None,
        help="Allele id->name JSON (default: <export_root>/id_to_hla.json).",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <dataset-dir>/processed/plots).",
    )
    ap.add_argument(
        "--max-tcrs",
        type=int,
        default=50,
        help="Max TCRs to plot and include in heatmaps (default: 50).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for sampling TCRs (default: 13).",
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


def _resolve_export_root(dataset_dir: Path) -> Optional[Path]:
    parts = list(dataset_dir.parts)
    if "synthetic" in parts:
        idx = parts.index("synthetic")
        if idx > 0:
            return Path(*parts[:idx])
    return None


def _get_hla_class(name: str) -> str:
    u = name.upper()
    if u.startswith("HLA-A") or u.startswith("HLA-B") or u.startswith("HLA-C"):
        return "I"
    return "II"


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
    scaled_norm = scaled_counts_norm(counts_sel, freq_sub)
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
        "scaled_norm": scaled_norm,
        "neglog10_p": neglog10_p,
        "neglog10_q": neglog10_q,
        "z_scores": z_vals,
    }


def _plot_tcr_panel(
    idx: int,
    n_donors_i: int,
    loops: Sequence[str],
    allele_indices: np.ndarray,
    counts_sel: np.ndarray,
    scaled_norm: np.ndarray,
    neglog10_p: np.ndarray,
    neglog10_q: np.ndarray,
    z_scores: np.ndarray,
    binder_set: set[int],
    b_present_count: int,
    b_size: int,
    allele_name_map: dict[int, str],
    out_dir: Path,
    dpi: int,
) -> None:
    logger.info("Plotting TCR %d (n_donors=%d).", idx, n_donors_i)
    allele_labels = [
        allele_name_map.get(int(a), str(int(a))) for a in allele_indices
    ]
    classes = np.array([_get_hla_class(name) for name in allele_labels], dtype=object)
    mask_I = classes == "I"
    mask_II = classes == "II"

    counts_I = counts_sel[mask_I]
    counts_II = counts_sel[mask_II]
    scaled_I = scaled_norm[mask_I]
    scaled_II = scaled_norm[mask_II]
    neglog_I = neglog10_p[mask_I]
    neglog_II = neglog10_p[mask_II]
    neglogq_I = neglog10_q[mask_I]
    neglogq_II = neglog10_q[mask_II]
    z_I = z_scores[mask_I]
    z_II = z_scores[mask_II]
    labels_I = np.asarray(allele_labels, dtype=object)[mask_I]
    labels_II = np.asarray(allele_labels, dtype=object)[mask_II]
    binder_flags = np.array(
        [int(a) in binder_set for a in allele_indices], dtype=bool
    )
    colors = np.where(binder_flags, "#f28e2b", "#4c78a8")
    colors_I = np.asarray(colors, dtype=object)[mask_I]
    colors_II = np.asarray(colors, dtype=object)[mask_II]

    max_len = max(len(labels_I), len(labels_II), 1)
    base_width = max(6.0, 0.25 * max_len)
    fig, axes = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(2.0 * base_width, 11.0),
        sharey="row",
    )

    rows = [
        ("Counts", counts_I, counts_II),
        ("Scaled", scaled_I, scaled_II),
        ("-log10(p)", neglog_I, neglog_II),
        ("FDR 0.05 -log10(p)", neglogq_I, neglogq_II),
        ("z = (observed − expected) / std", z_I, z_II),
    ]

    for r, (title, vals_I, vals_II) in enumerate(rows):
        ax_I, ax_II = axes[r]
        if len(vals_I) > 0:
            xs = np.arange(len(vals_I))
            ax_I.bar(xs, vals_I, edgecolor="black", color=colors_I)
            ax_I.set_xticks(xs)
            ax_I.set_xticklabels(labels_I, rotation=90, fontsize=6)
        else:
            ax_I.text(0.5, 0.5, "No class I alleles", ha="center", va="center")
            ax_I.set_xticks([])
        if len(vals_II) > 0:
            xs = np.arange(len(vals_II))
            ax_II.bar(xs, vals_II, edgecolor="black", color=colors_II)
            ax_II.set_xticks(xs)
            ax_II.set_xticklabels(labels_II, rotation=90, fontsize=6)
        else:
            ax_II.text(0.5, 0.5, "No class II alleles", ha="center", va="center")
            ax_II.set_xticks([])
        ax_I.set_title(f"{title} (Class I)")
        ax_II.set_title(f"{title} (Class II)")

    b_pct = 0.0
    if b_size > 0:
        b_pct = 100.0 * float(b_present_count) / float(b_size)
    loops_str = " ".join(loops)
    fig.suptitle(
        "TCR {idx} (n_donors={n_donors}, B present {b_present}/{b_size} ({b_pct:.1f}%))\n"
        "loops: {loops}".format(
            idx=idx,
            n_donors=n_donors_i,
            b_present=b_present_count,
            b_size=b_size,
            b_pct=b_pct,
            loops=loops_str,
        ),
        y=0.99,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tcr_{idx:06d}_summary.png"
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
    ax.set_ylabel("TCR index (sampled)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Wrote heatmap: %s", out_path)


def main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)
    logger.info("Starting synthetic binder plotting.")

    dataset_dir = Path(args.dataset_dir).resolve()
    if args.h5:
        h5_path = Path(args.h5).resolve()
    else:
        h5_path = dataset_dir / "synthetic_tcr_hla_counts.h5"
    if args.meta:
        meta_path = Path(args.meta).resolve()
    else:
        meta_path = dataset_dir / "synthetic_meta.json"
    if args.binder_sets:
        binder_path = Path(args.binder_sets).resolve()
    else:
        binder_path = dataset_dir / "synthetic_binder_sets.npy"

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = dataset_dir / "processed" / "plots"
    logger.info("Dataset dir: %s", dataset_dir)
    logger.info("HDF5: %s", h5_path)
    logger.info("Meta: %s", meta_path)
    logger.info("Binder sets: %s", binder_path)
    logger.info("Output dir: %s", out_dir)

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {h5_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta JSON not found: {meta_path}")
    if not binder_path.exists():
        raise FileNotFoundError(f"Binder sets not found: {binder_path}")

    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    n_donors_total = int(meta.get("donors", 0))
    if n_donors_total <= 0:
        raise SystemExit("Meta JSON missing valid 'donors' count.")
    logger.info("Total donors: %d", n_donors_total)

    with PublicTcrHlaCsrReader(h5_path) as reader:
        num_rows = reader.num_rows
        num_alleles = reader.num_alleles
    logger.info("HDF5 rows=%d alleles=%d", num_rows, num_alleles)
    binder_sets = np.load(binder_path, mmap_mode="r")
    if binder_sets.shape[0] != num_rows:
        raise SystemExit(
            f"Binder sets rows {binder_sets.shape[0]} != HDF5 rows {num_rows}."
        )
    logger.info("Binder sets shape: %s", binder_sets.shape)

    freq_path = Path(args.donor_frequencies_json).resolve()
    logger.info("Loading donor frequencies: %s", freq_path)
    freqs = load_allele_frequencies(freq_path, num_alleles)
    allele_counts = allele_counts_from_freqs(freqs, n_donors_total)
    logger.info("Loaded donor frequencies and allele counts.")

    export_root = _resolve_export_root(dataset_dir)
    if args.id_to_hla:
        id_to_hla_path = Path(args.id_to_hla).resolve()
    elif export_root is not None:
        id_to_hla_path = export_root / "id_to_hla.json"
    else:
        id_to_hla_path = Path("id_to_hla.json")
    id_to_hla = _load_id_map(id_to_hla_path)
    logger.info("Loaded id->HLA map entries: %d", len(id_to_hla))

    max_tcrs = int(args.max_tcrs)
    if max_tcrs < 0:
        raise SystemExit("--max-tcrs must be >= 0.")
    if max_tcrs == 0:
        logger.info("Skipping per-TCR plots and heatmaps (max_tcrs=0).")
        return
    if max_tcrs > num_rows:
        max_tcrs = num_rows
    logger.info("Plotting up to %d TCRs.", max_tcrs)

    heat_scaled_I = []
    heat_scaled_II = []
    heat_p_I = []
    heat_p_II = []
    sample_rows = []
    plotted = 0

    with PublicTcrHlaCsrReader(h5_path) as reader:
        for row in reader:
            logger.info("Processing row %d.", row.row)
            metrics = _compute_tcr_metrics(
                row.counts,
                row.n_donors,
                allele_counts,
                n_donors_total,
                freqs,
                args.min_overlap,
            )
            if metrics is None:
                logger.info("Skipping row %d (no nonzero alleles).", row.row)
                continue

            allele_indices = metrics["allele_indices"]
            logger.info("Row %d has %d alleles.", row.row, allele_indices.size)
            binder = np.asarray(binder_sets[row.row], dtype=np.int64)
            b_size = int(binder.size)
            b_present_count = int(np.sum(row.counts[binder] > 0))
            binder_set = set(int(x) for x in binder.tolist())
            allele_labels = [
                id_to_hla.get(int(a), str(int(a))) for a in allele_indices
            ]
            classes = np.array(
                [_get_hla_class(name) for name in allele_labels], dtype=object
            )
            mask_I = classes == "I"
            mask_II = classes == "II"

            scaled = metrics["scaled_norm"]
            neglogp = metrics["neglog10_p"]

            heat_scaled_I.append(scaled[mask_I])
            heat_scaled_II.append(scaled[mask_II])
            heat_p_I.append(neglogp[mask_I])
            heat_p_II.append(neglogp[mask_II])
            sample_rows.append(row.row)

            _plot_tcr_panel(
                row.row,
                row.n_donors,
                (row.cdr3aa, row.cdr2aa_gapped, row.cdr1aa_gapped, row.cdr2_5aa_gapped),
                allele_indices,
                metrics["counts_sel"],
                metrics["scaled_norm"],
                metrics["neglog10_p"],
                metrics["neglog10_q"],
                metrics["z_scores"],
                binder_set,
                b_present_count,
                b_size,
                id_to_hla,
                out_dir / "per_tcr",
                args.dpi,
            )
            plotted += 1
            logger.info("Plotted %d/%d TCRs.", plotted, max_tcrs)
            if plotted >= max_tcrs:
                break

    if not sample_rows:
        logger.info("No TCRs selected for plotting.")
        return

    max_rank = int(args.max_rank)
    if max_rank < 1:
        raise SystemExit("--max-rank must be >= 1.")
    logger.info("Building heatmaps with max rank %d.", max_rank)

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

    scaled_I_mat = _build_heat_matrix(heat_scaled_I)
    scaled_II_mat = _build_heat_matrix(heat_scaled_II)
    p_I_mat = _build_heat_matrix(heat_p_I)
    p_II_mat = _build_heat_matrix(heat_p_II)

    _plot_heatmap(
        scaled_I_mat,
        "Scaled counts (Class I)",
        out_dir / "heatmaps" / "scaled_class_I.png",
        args.dpi,
        vmax=1.0,
    )
    _plot_heatmap(
        scaled_II_mat,
        "Scaled counts (Class II)",
        out_dir / "heatmaps" / "scaled_class_II.png",
        args.dpi,
        vmax=1.0,
    )
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
