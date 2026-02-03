#!/usr/bin/env python3
"""
Aggregate per-cluster metrics into donor-size bins and plot summaries.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tcrtyper.dataset_processing.synthetic_analysis_utils import (
    allele_counts_from_freqs,
    compute_pvals,
    load_allele_frequencies,
)
from tcrtyper.dataset_processing.utils import PublicTcrHlaCsrReader

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Bin clusters by donor size, summarize metrics, and plot."
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
        "--fdr-thresholds-json",
        required=True,
        help="JSON produced by compute_public_hla_fdr_thresholds.py.",
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
        "--min-overlap",
        type=int,
        default=None,
        help="Minimum overlap for Fisher test (default: from thresholds JSON).",
    )
    ap.add_argument(
        "--min-cluster-size",
        type=int,
        default=20,
        help="Minimum donors per cluster to include (default: 20).",
    )
    ap.add_argument(
        "--small-bin-max",
        type=int,
        default=520,
        help="Upper bound for small bins (default: 520).",
    )
    ap.add_argument(
        "--small-bins",
        type=int,
        default=100,
        help="Number of bins for sizes 20..520 (default: 100).",
    )
    ap.add_argument(
        "--large-bin-min",
        type=int,
        default=500,
        help="Lower bound for large bins (default: 500).",
    )
    ap.add_argument(
        "--large-bins",
        type=int,
        default=50,
        help="Number of bins for sizes >=500 (default: 50).",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <public-h5-dir>/cluster_bins).",
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


def _build_bins(min_size: int, max_size: int, n_bins: int, label_prefix: str) -> list[dict]:
    if n_bins < 1:
        raise SystemExit("Number of bins must be >= 1.")
    if max_size <= min_size:
        raise SystemExit("Bin max must be > min.")
    edges = np.linspace(min_size, max_size, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        bins.append(
            {
                "label": f"{label_prefix}_{i}",
                "min": lo,
                "max": hi,
                "center": 0.5 * (lo + hi),
                "cluster_sizes": [],
                "v_gene_purity": [],
                "n_sig_alleles": [],
            }
        )
    return bins


def _assign_bin(size: int, edges: np.ndarray) -> Optional[int]:
    idx = int(np.searchsorted(edges, size, side="right") - 1)
    if idx < 0 or idx >= len(edges) - 1:
        return None
    return idx


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "median": None, "mean": None, "p25": None, "p75": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def _plot_metric(
    ax,
    centers: np.ndarray,
    medians: np.ndarray,
    p25: np.ndarray,
    p75: np.ndarray,
    title: str,
    ylabel: str,
):
    ax.plot(centers, medians, color="#1f77b4", linewidth=2)
    ax.fill_between(centers, p25, p75, color="#1f77b4", alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("Cluster size (binned)")
    ax.set_ylabel(ylabel)


def main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)

    public_h5 = Path(args.public_h5).resolve()
    if not public_h5.exists():
        raise FileNotFoundError(f"public HDF5 not found: {public_h5}")

    donor_freq_path = Path(args.donor_frequencies_json).resolve()
    if not donor_freq_path.exists():
        raise FileNotFoundError(f"donor frequency JSON not found: {donor_freq_path}")

    thresholds_path = Path(args.fdr_thresholds_json).resolve()
    if not thresholds_path.exists():
        raise FileNotFoundError(f"thresholds JSON not found: {thresholds_path}")

    with thresholds_path.open("r", encoding="utf-8") as fh:
        thresholds_payload = json.load(fh)

    min_overlap = (
        int(args.min_overlap)
        if args.min_overlap is not None
        else int(thresholds_payload.get("min_overlap", 2))
    )
    if "min_overlap" in thresholds_payload and int(thresholds_payload["min_overlap"]) != min_overlap:
        logger.warning(
            "min_overlap %d does not match thresholds JSON (%s)",
            min_overlap,
            thresholds_payload.get("min_overlap"),
        )

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

    freqs = load_allele_frequencies(donor_freq_path, num_alleles)
    allele_counts = allele_counts_from_freqs(freqs, n_donors_total)

    thresholds = np.zeros(num_alleles, dtype=np.float64)
    for key, entry in thresholds_payload.get("thresholds", {}).items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= num_alleles:
            continue
        thresholds[idx] = float(entry.get("pval_threshold", 0.0))

    n_alleles_fdr = int(np.sum(thresholds > 0.0))
    logger.info("Alleles with FDR threshold: %d", n_alleles_fdr)

    min_size = int(args.min_cluster_size)
    small_max = min(int(args.small_bin_max), int(n_donors_total))
    large_min = int(args.large_bin_min)
    if min_size >= small_max:
        raise SystemExit("--min-cluster-size must be < --small-bin-max.")
    if large_min < min_size:
        raise SystemExit("--large-bin-min must be >= --min-cluster-size.")

    small_bins = _build_bins(min_size, small_max, int(args.small_bins), "small")
    small_edges = np.linspace(min_size, small_max, int(args.small_bins) + 1)

    large_bins = []
    large_edges = None
    if n_donors_total > large_min:
        large_bins = _build_bins(large_min, int(n_donors_total), int(args.large_bins), "large")
        large_edges = np.linspace(
            large_min, int(n_donors_total), int(args.large_bins) + 1
        )

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else public_h5.parent / "cluster_bins"
    )
    out_json = out_dir / "cluster_bin_metrics.json"
    out_plot = out_dir / "cluster_bin_metrics.png"

    current_cluster = None
    counts_row = None
    n_donors = None
    v_gene_counts = {}
    total_identical = 0
    rows_total = 0
    clusters_seen = 0
    clusters_used = 0

    def flush_cluster() -> None:
        nonlocal counts_row, n_donors, v_gene_counts, total_identical, clusters_used
        if counts_row is None or n_donors is None:
            return
        size = int(n_donors)
        if size < min_size:
            return

        if size >= large_min and large_edges is not None:
            idx = _assign_bin(size, large_edges)
            if idx is None:
                return
            bin_entry = large_bins[idx]
        else:
            idx = _assign_bin(size, small_edges)
            if idx is None:
                return
            bin_entry = small_bins[idx]

        if total_identical > 0 and v_gene_counts:
            v_gene_purity = max(v_gene_counts.values()) / float(total_identical)
        else:
            v_gene_purity = 0.0

        pvals = compute_pvals(
            counts_row,
            int(n_donors),
            allele_counts,
            n_donors_total,
            min_overlap,
        )
        sig_mask = (pvals <= thresholds) & (thresholds > 0.0)
        n_sig = int(np.sum(sig_mask))

        bin_entry["cluster_sizes"].append(size)
        bin_entry["v_gene_purity"].append(float(v_gene_purity))
        bin_entry["n_sig_alleles"].append(n_sig)
        clusters_used += 1

    with PublicTcrHlaCsrReader(public_h5) as reader:
        for row in reader:
            rows_total += 1
            if current_cluster is None:
                clusters_seen += 1
                current_cluster = row.cluster_id
                counts_row = row.counts
                n_donors = row.n_donors
                v_gene_counts = {}
                total_identical = 0
            if row.cluster_id != current_cluster:
                flush_cluster()
                clusters_seen += 1
                current_cluster = row.cluster_id
                counts_row = row.counts
                n_donors = row.n_donors
                v_gene_counts = {}
                total_identical = 0

            weight = int(row.n_identical_sequences) if row.n_identical_sequences else 1
            total_identical += weight
            if row.v_gene_id is not None:
                key = int(row.v_gene_id)
                v_gene_counts[key] = v_gene_counts.get(key, 0) + weight

        flush_cluster()

    logger.info("Rows processed: %d", rows_total)
    logger.info("Clusters seen: %d", clusters_seen)
    logger.info("Clusters used: %d", clusters_used)

    bin_payload = []
    for entry in small_bins + large_bins:
        bin_payload.append(
            {
                "label": entry["label"],
                "min": entry["min"],
                "max": entry["max"],
                "center": entry["center"],
                "cluster_sizes": entry["cluster_sizes"],
                "v_gene_purity": entry["v_gene_purity"],
                "n_sig_alleles": entry["n_sig_alleles"],
                "summary": {
                    "v_gene_purity": _summarize(entry["v_gene_purity"]),
                    "n_sig_alleles": _summarize(entry["n_sig_alleles"]),
                },
            }
        )

    payload = {
        "public_h5": str(public_h5),
        "donor_frequencies_json": str(donor_freq_path),
        "thresholds_json": str(thresholds_path),
        "donor_count": int(n_donors_total),
        "min_overlap": int(min_overlap),
        "min_cluster_size": int(min_size),
        "rows_total": int(rows_total),
        "clusters_seen": int(clusters_seen),
        "clusters_used": int(clusters_used),
        "num_alleles": int(num_alleles),
        "n_alleles_fdr": int(n_alleles_fdr),
        "bins": bin_payload,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote %s", out_json)

    centers = []
    purity_med = []
    purity_p25 = []
    purity_p75 = []
    hits_med = []
    hits_p25 = []
    hits_p75 = []
    for entry in bin_payload:
        summ_p = entry["summary"]["v_gene_purity"]
        summ_h = entry["summary"]["n_sig_alleles"]
        if summ_p["count"] == 0:
            continue
        centers.append(entry["center"])
        purity_med.append(summ_p["median"])
        purity_p25.append(summ_p["p25"])
        purity_p75.append(summ_p["p75"])
        hits_med.append(summ_h["median"])
        hits_p25.append(summ_h["p25"])
        hits_p75.append(summ_h["p75"])

    if centers:
        centers_arr = np.asarray(centers, dtype=np.float64)
        order = np.argsort(centers_arr)
        centers_arr = centers_arr[order]
        purity_med = np.asarray(purity_med, dtype=np.float64)[order]
        purity_p25 = np.asarray(purity_p25, dtype=np.float64)[order]
        purity_p75 = np.asarray(purity_p75, dtype=np.float64)[order]
        hits_med = np.asarray(hits_med, dtype=np.float64)[order]
        hits_p25 = np.asarray(hits_p25, dtype=np.float64)[order]
        hits_p75 = np.asarray(hits_p75, dtype=np.float64)[order]

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
        _plot_metric(
            axes[0],
            centers_arr,
            purity_med,
            purity_p25,
            purity_p75,
            "V-gene purity by cluster size",
            "V-gene purity",
        )
        _plot_metric(
            axes[1],
            centers_arr,
            hits_med,
            hits_p25,
            hits_p75,
            "FDR<=0.05 allele hits by cluster size",
            "Alleles with FDR<=0.05",
        )
        fig.suptitle(
            "Alleles with FDR<=0.05: {n_alleles} | clusters used: {clusters} | "
            "cluster-deduped".format(
                n_alleles=n_alleles_fdr,
                clusters=clusters_used,
            )
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_plot, dpi=args.dpi)
        plt.close(fig)
        logger.info("Wrote %s", out_plot)
    else:
        logger.info("No clusters in bins; skipping plot.")


if __name__ == "__main__":
    main()
