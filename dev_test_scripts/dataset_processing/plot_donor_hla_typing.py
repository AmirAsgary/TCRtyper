#!/usr/bin/env python3
"""
Summarize donor HLA typing completeness and generate dataset-wide plots.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tcrtyper.config import config

logger = logging.getLogger(__name__)

DEFAULT_PLOT_SUBDIR = "donor_hla_typing"
DEFAULT_TOP_N = 25
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
LOCUS_RE = re.compile(r"^HLA-([A-Z0-9]+)\*")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot donor HLA typing completeness from a donor×allele matrix."
    )
    ap.add_argument(
        "--matrix",
        default=None,
        help="Path to donor×allele matrix (.npy, .npz, or .h5/.hdf5).",
    )
    ap.add_argument(
        "--npz-key",
        default=None,
        help="Array key to use when loading .npz matrices.",
    )
    ap.add_argument(
        "--h5-dataset",
        default="donor_hla_matrix",
        help="HDF5 dataset path for the matrix (default: donor_hla_matrix).",
    )
    ap.add_argument(
        "--h5-donors-dataset",
        default=None,
        help="HDF5 dataset path for donor keys (optional).",
    )
    ap.add_argument(
        "--id-to-hla",
        default=None,
        help="Path to id_to_hla.json (default: export_train_dataset/id_to_hla.json).",
    )
    ap.add_argument(
        "--donors",
        default=None,
        help=(
            "Optional donor keys JSON (default: donor_hla_matrix_donors.json next to matrix). "
            "If matrix is .npz/.h5, donor_keys inside the file are used when available."
        ),
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory (default: <matrix_dir>/plots/donor_hla_typing or "
            "<export_root>/plots/donor_hla_typing if matrix is under export root)."
        ),
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Top-N alleles to show (default: {DEFAULT_TOP_N}).",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=config.data.plots_default_dpi,
        help=f"Figure DPI (default: {config.data.plots_default_dpi}).",
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


def _default_export_root() -> Path:
    return Path(config.data.base_dir) / config.data.train_export_root_name


def _resolve_default_matrix(export_root: Path) -> Optional[Path]:
    candidates = [
        export_root / "donor_hla_matrix.npz",
        export_root / "donor_hla_matrix.npy",
        export_root / "donor_hla_matrix.h5",
        export_root / "donor_hla_matrix.hdf5",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_id_to_hla(path: Path) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    out: Dict[int, str] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            continue
    if not out:
        raise ValueError(f"id_to_hla.json is empty or invalid: {path}")
    return out


def _load_matrix(
    path: Path,
    npz_key: Optional[str],
    h5_dataset: Optional[str],
) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            keys = list(data.keys())
            if npz_key:
                if npz_key not in data:
                    raise KeyError(
                        f"npz key {npz_key!r} not found. Available keys: {keys}"
                    )
                return data[npz_key]
            if len(keys) == 1:
                return data[keys[0]]
            preferred = ["donor_hla_matrix", "donor_matrix", "matrix", "X", "data"]
            for key in preferred:
                if key in data:
                    return data[key]
            raise KeyError(
                "npz contains multiple arrays; pass --npz-key. "
                f"Available keys: {keys}"
            )
    if path.suffix.lower() in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as h5:
            dataset_path = (h5_dataset or "").strip()
            if dataset_path and dataset_path in h5:
                return np.asarray(h5[dataset_path])

            preferred = ["donor_hla_matrix", "donor_matrix", "matrix", "X", "data"]
            for key in preferred:
                if key in h5:
                    return np.asarray(h5[key])

            if dataset_path:
                fallback = f"datasets/{dataset_path}"
                if fallback in h5:
                    return np.asarray(h5[fallback])

            keys = list(h5.keys())
            raise KeyError(
                "HDF5 matrix dataset not found; "
                f"tried {dataset_path!r} and common names. Top-level keys: {keys}"
            )
    return np.load(path, mmap_mode="r")


def _load_donor_keys_from_npz(data) -> List[str] | None:
    for key in ["donor_keys", "donors", "donor_hla_matrix_donors"]:
        if key in data:
            vals = data[key]
            return [str(v) for v in vals.tolist()]
    return None


def _load_donor_keys_from_h5(h5, dataset_path: Optional[str]) -> List[str] | None:
    if dataset_path and dataset_path in h5:
        vals = h5[dataset_path]
        if hasattr(vals, "asstr"):
            vals = vals.asstr()
        return [str(v) for v in np.asarray(vals).tolist()]

    for key in ["donor_keys", "donors", "donor_hla_matrix_donors"]:
        if key in h5:
            vals = h5[key]
            if hasattr(vals, "asstr"):
                vals = vals.asstr()
            return [str(v) for v in np.asarray(vals).tolist()]
    return None


def _parse_locus(allele: Optional[str]) -> Optional[str]:
    if not allele:
        return None
    m = LOCUS_RE.match(allele.strip().upper())
    if not m:
        return None
    return m.group(1)


def _build_locus_indices(
    id_to_hla: Dict[int, str], num_alleles: int
) -> Tuple[Dict[str, List[int]], List[int]]:
    locus_to_indices: Dict[str, List[int]] = {}
    unknown: List[int] = []
    for idx in range(num_alleles):
        allele = id_to_hla.get(idx)
        locus = _parse_locus(allele)
        if locus is None:
            unknown.append(idx)
            continue
        locus_to_indices.setdefault(locus, []).append(idx)
    return locus_to_indices, unknown


def _order_loci(loci: Iterable[str]) -> Tuple[List[str], List[str], List[str]]:
    loci_set = set(loci)
    class_i = [l for l in CLASS_I_LOCI_ORDER if l in loci_set]

    class_ii_candidates = sorted([l for l in loci_set if l.startswith("D")])
    class_ii: List[str] = [l for l in CLASS_II_LOCI_ORDER if l in class_ii_candidates]
    for locus in class_ii_candidates:
        if locus not in class_ii:
            class_ii.append(locus)

    other = sorted([l for l in loci_set if l not in class_i and not l.startswith("D")])
    return class_i, class_ii, other


def _stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def _plot_flag_counts(
    out_path: Path,
    counts: Dict[str, int],
    n_donors: int,
    dpi: int,
) -> None:
    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    fig = plt.figure(figsize=(9, 4))
    ax = plt.gca()
    x = np.arange(len(labels))
    ax.bar(x, values, color="#4C78A8")
    ax.set_ylabel("Donors")
    ax.set_title("Donor HLA typing completeness")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")

    ymax = max(values) if values else 0
    for i, val in enumerate(values):
        pct = (val / n_donors) if n_donors else 0.0
        ax.text(i, val + ymax * 0.02, f"{val} ({pct:.1%})", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_count_hist(
    out_path: Path,
    counts_i: np.ndarray,
    counts_ii: np.ndarray,
    counts_total: np.ndarray,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)

    def _hist(ax, values: np.ndarray, title: str, color: str) -> None:
        maxv = int(values.max()) if values.size else 0
        bins = np.arange(-0.5, maxv + 1.5, 1.0)
        ax.hist(values, bins=bins, color=color, edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel("Alleles per donor")
        ax.set_ylabel("Donors")

    _hist(axes[0], counts_i, "Class I (A/B/C)", "#F58518")
    _hist(axes[1], counts_ii, "Class II (D*)", "#54A24B")
    _hist(axes[2], counts_total, "Total", "#4C78A8")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_locus_stacked(
    out_path: Path,
    locus_bins: Dict[str, Dict[str, int]],
    locus_order: List[str],
    dpi: int,
) -> None:
    labels = locus_order
    bins = ["0", "1", "2", "3+"]
    values = {b: [locus_bins[l][b] for l in labels] for b in bins}

    fig = plt.figure(figsize=(max(9, len(labels) * 0.7), 4))
    ax = plt.gca()
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    colors = {"0": "#E0E0E0", "1": "#F58518", "2": "#54A24B", "3+": "#4C78A8"}
    for b in bins:
        ax.bar(x, values[b], bottom=bottom, label=b, color=colors[b])
        bottom += np.array(values[b])

    ax.set_ylabel("Donors")
    ax.set_title("Alleles per locus (0/1/2/3+)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(title="Alleles", loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_top_alleles(
    out_path: Path,
    allele_counts: np.ndarray,
    id_to_hla: Dict[int, str],
    n_donors: int,
    top_n: int,
    dpi: int,
) -> List[Dict[str, float]]:
    if allele_counts.size == 0:
        return []
    top_n = min(top_n, allele_counts.size)
    order = np.argsort(-allele_counts)[:top_n]
    labels = [id_to_hla.get(int(i), f"allele_{i}") for i in order]
    values = allele_counts[order] / n_donors if n_donors else allele_counts[order]

    fig_height = max(4.0, 0.3 * top_n)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.barh(labels[::-1], values[::-1], color="#72B7B2")
    ax.set_xlabel("Fraction of donors" if n_donors else "Donors")
    ax.set_title(f"Top {top_n} alleles by donor frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)

    out = []
    for idx in order:
        count = int(allele_counts[idx])
        pct = (count / n_donors) if n_donors else 0.0
        out.append(
            {
                "allele": id_to_hla.get(int(idx), f"allele_{idx}"),
                "count": count,
                "pct": pct,
            }
        )
    return out


def _extract_dataset(key: str) -> str:
    if not key:
        return ""
    if "/" in key:
        return key.split("/", 1)[0]
    return key


def _plot_dataset_typing(
    out_path: Path,
    dataset_stats: List[Dict[str, float]],
    dpi: int,
) -> None:
    if not dataset_stats:
        return
    labels = [d["dataset"] for d in dataset_stats]
    values_i = [d["pct_hla_i"] for d in dataset_stats]
    values_ii = [d["pct_hla_ii"] for d in dataset_stats]
    counts = [d["n_donors"] for d in dataset_stats]

    fig = plt.figure(figsize=(max(8, len(labels) * 0.9), 4.2))
    ax = plt.gca()
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, values_i, width=width, color="#F58518", label="HLA I")
    ax.bar(x + width / 2, values_ii, width=width, color="#54A24B", label="HLA II")
    ax.set_ylabel("Fully typed fraction")
    ax.set_title("Fully typed donors by dataset (HLA I/II)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")

    for i, n in enumerate(counts):
        ymax = max(values_i[i], values_ii[i])
        ax.text(i, ymax + 0.02, f"n={n}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)

    export_root = _default_export_root()
    matrix_path = (
        Path(args.matrix).resolve()
        if args.matrix
        else (_resolve_default_matrix(export_root) or None)
    )
    if matrix_path is None:
        raise FileNotFoundError("Matrix path not provided and no default found.")
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix not found: {matrix_path}")

    if matrix_path.suffix.lower() in {".h5", ".hdf5"}:
        npz_candidates = [
            matrix_path.with_suffix(".npz"),
            export_root / "donor_hla_matrix.npz",
        ]
        for candidate in npz_candidates:
            if candidate.exists():
                logger.info(
                    "Using NPZ donor matrix %s instead of HDF5 %s",
                    candidate,
                    matrix_path,
                )
                matrix_path = candidate
                break

    id_to_hla_path = (
        Path(args.id_to_hla).resolve()
        if args.id_to_hla
        else (export_root / config.data.train_id_to_hla_filename)
    )
    if not id_to_hla_path.exists():
        raise FileNotFoundError(f"id_to_hla.json not found: {id_to_hla_path}")

    donors_path = (
        Path(args.donors).resolve()
        if args.donors
        else matrix_path.with_name("donor_hla_matrix_donors.json")
    )
    donor_keys: List[str] = []
    if matrix_path.suffix.lower() == ".npz":
        with np.load(matrix_path) as data:
            donor_keys = _load_donor_keys_from_npz(data) or []
    if matrix_path.suffix.lower() in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(matrix_path, "r") as h5:
            donor_keys = _load_donor_keys_from_h5(h5, args.h5_donors_dataset) or []
    if not donor_keys and donors_path.exists():
        with open(donors_path, "r", encoding="utf-8") as fh:
            donor_keys = [str(v) for v in json.load(fh)]

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else (
            matrix_path.parent
            / config.data.plots_root_subdir_name
            / DEFAULT_PLOT_SUBDIR
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading matrix: %s", matrix_path)
    matrix = _load_matrix(matrix_path, args.npz_key, args.h5_dataset)
    if matrix.ndim != 2:
        raise ValueError(f"Matrix must be 2D; got shape {matrix.shape}")

    n_donors, num_alleles = matrix.shape
    id_to_hla = _load_id_to_hla(id_to_hla_path)
    if max(id_to_hla.keys()) >= num_alleles:
        logger.warning(
            "id_to_hla contains indices beyond matrix columns (%d >= %d)",
            max(id_to_hla.keys()),
            num_alleles,
        )

    locus_to_indices, unknown_indices = _build_locus_indices(id_to_hla, num_alleles)
    class_i_loci, class_ii_loci, other_loci = _order_loci(locus_to_indices.keys())
    expected_loci = class_i_loci + class_ii_loci
    flag_loci = [l for l in expected_loci if l not in DRB_EXCLUDE_LOCI]
    class_ii_flag_loci = [l for l in class_ii_loci if l not in DRB_EXCLUDE_LOCI]

    if not expected_loci:
        raise ValueError("No recognizable HLA loci found in id_to_hla.json.")

    if other_loci:
        logger.info("Other loci detected (ignored for flags): %s", ", ".join(other_loci))
    if unknown_indices:
        logger.info("Alleles without locus match: %d", len(unknown_indices))

    present = (matrix != 0)

    class_i_indices = [idx for l in class_i_loci for idx in locus_to_indices[l]]
    class_ii_indices = [idx for l in class_ii_loci for idx in locus_to_indices[l]]

    counts_i = (
        present[:, class_i_indices].sum(axis=1)
        if class_i_indices
        else np.zeros(n_donors, dtype=int)
    )
    counts_ii = (
        present[:, class_ii_indices].sum(axis=1)
        if class_ii_indices
        else np.zeros(n_donors, dtype=int)
    )
    counts_total = present.sum(axis=1)

    locus_counts: Dict[str, np.ndarray] = {}
    for locus in expected_loci:
        idxs = locus_to_indices[locus]
        locus_counts[locus] = present[:, idxs].sum(axis=1)

    def _all_loci_at_least(loci: List[str], min_count: int) -> np.ndarray:
        mask = np.ones(n_donors, dtype=bool)
        for locus in loci:
            mask &= locus_counts[locus] >= min_count
        return mask

    fully_typed_all_locuses_two_alleles = _all_loci_at_least(flag_loci, 2)
    fully_typed_all_locuses_contain_allele = _all_loci_at_least(flag_loci, 1)
    not_fully_typed = ~fully_typed_all_locuses_contain_allele
    fully_typed_hla_i = (
        _all_loci_at_least(class_i_loci, 1)
        if class_i_loci
        else np.zeros(n_donors, dtype=bool)
    )
    fully_typed_hla_ii = (
        _all_loci_at_least(class_ii_flag_loci, 1)
        if class_ii_flag_loci
        else np.zeros(n_donors, dtype=bool)
    )
    full_hla_i_only = not_fully_typed & fully_typed_hla_i
    full_hla_ii_only = not_fully_typed & fully_typed_hla_ii

    flag_counts = {
        "fully_typed_all_locuses_two_alleles": int(
            fully_typed_all_locuses_two_alleles.sum()
        ),
        "fully_typed_all_locuses_contain_allele": int(
            fully_typed_all_locuses_contain_allele.sum()
        ),
        "full_hla_i_only": int(full_hla_i_only.sum()),
        "full_hla_ii_only": int(full_hla_ii_only.sum()),
        "not_fully_typed": int(not_fully_typed.sum()),
    }

    locus_bins: Dict[str, Dict[str, int]] = {}
    for locus in expected_loci:
        vals = locus_counts[locus]
        locus_bins[locus] = {
            "0": int((vals == 0).sum()),
            "1": int((vals == 1).sum()),
            "2": int((vals == 2).sum()),
            "3+": int((vals >= 3).sum()),
        }

    summary = {
        "matrix_path": str(matrix_path),
        "id_to_hla_path": str(id_to_hla_path),
        "donors_path": str(donors_path) if donors_path.exists() else None,
        "n_donors": int(n_donors),
        "n_alleles": int(num_alleles),
        "class_i_loci": class_i_loci,
        "class_ii_loci": class_ii_loci,
        "other_loci": other_loci,
        "unknown_locus_alleles": len(unknown_indices),
        "flags": flag_counts,
        "flags_pct": {
            k: (v / n_donors) if n_donors else 0.0 for k, v in flag_counts.items()
        },
        "allele_counts": {
            "class_i": _stats(counts_i),
            "class_ii": _stats(counts_ii),
            "total": _stats(counts_total),
        },
        "locus_bins": locus_bins,
    }

    flags_path = out_dir / "donor_hla_typing_flags.tsv"
    if not donor_keys:
        donor_keys = [f"donor_{i}" for i in range(n_donors)]
    if len(donor_keys) != n_donors:
        logger.warning(
            "Donor keys length (%d) does not match matrix rows (%d); using indices.",
            len(donor_keys),
            n_donors,
        )
        donor_keys = [f"donor_{i}" for i in range(n_donors)]

    with open(flags_path, "w", encoding="utf-8") as fh:
        fh.write(
            "\t".join(
                [
                    "donor_key",
                    "fully_typed_all_locuses_two_alleles",
                    "fully_typed_all_locuses_contain_allele",
                    "full_hla_i_only",
                    "full_hla_ii_only",
                    "not_fully_typed",
                    "num_hla_i",
                    "num_hla_ii",
                    "num_hla_total",
                ]
            )
            + "\n"
        )
        for i, key in enumerate(donor_keys):
            fh.write(
                "\t".join(
                    [
                        str(key),
                        "1" if fully_typed_all_locuses_two_alleles[i] else "0",
                        "1" if fully_typed_all_locuses_contain_allele[i] else "0",
                        "1" if full_hla_i_only[i] else "0",
                        "1" if full_hla_ii_only[i] else "0",
                        "1" if not_fully_typed[i] else "0",
                        str(int(counts_i[i])),
                        str(int(counts_ii[i])),
                        str(int(counts_total[i])),
                    ]
                )
                + "\n"
            )

    flag_plot = out_dir / "donor_hla_typing_flags.png"
    counts_plot = out_dir / "donor_hla_allele_counts.png"
    locus_plot = out_dir / "donor_hla_locus_counts.png"
    top_plot = out_dir / "donor_hla_top_alleles.png"
    dataset_plot = out_dir / "donor_hla_typing_by_dataset.png"

    _plot_flag_counts(flag_plot, flag_counts, n_donors, args.dpi)
    _plot_count_hist(counts_plot, counts_i, counts_ii, counts_total, args.dpi)
    _plot_locus_stacked(locus_plot, locus_bins, expected_loci, args.dpi)
    top_alleles = _plot_top_alleles(
        top_plot, present.sum(axis=0), id_to_hla, n_donors, args.top_n, args.dpi
    )

    dataset_stats: List[Dict[str, float]] = []
    if donor_keys:
        dataset_to_indices: Dict[str, List[int]] = {}
        for idx, key in enumerate(donor_keys):
            dataset = _extract_dataset(str(key))
            if dataset:
                dataset_to_indices.setdefault(dataset, []).append(idx)

        for dataset in sorted(dataset_to_indices.keys()):
            indices = dataset_to_indices[dataset]
            if not indices:
                continue
            typed = fully_typed_all_locuses_contain_allele[indices].sum()
            typed_i = fully_typed_hla_i[indices].sum()
            typed_ii = fully_typed_hla_ii[indices].sum()
            n = len(indices)
            dataset_stats.append(
                {
                    "dataset": dataset,
                    "n_donors": int(n),
                    "n_fully_typed": int(typed),
                    "pct_fully_typed": float(typed / n) if n else 0.0,
                    "n_hla_i_typed": int(typed_i),
                    "pct_hla_i": float(typed_i / n) if n else 0.0,
                    "n_hla_ii_typed": int(typed_ii),
                    "pct_hla_ii": float(typed_ii / n) if n else 0.0,
                }
            )
        _plot_dataset_typing(dataset_plot, dataset_stats, args.dpi)
    else:
        logger.info("No donor keys available; skipping dataset typing plot.")

    summary["top_alleles"] = top_alleles
    summary["dataset_stats"] = dataset_stats
    summary_path = out_dir / "donor_hla_typing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Wrote summary: %s", summary_path)
    logger.info("Wrote flags TSV: %s", flags_path)
    logger.info("Saved plots to: %s", out_dir)


if __name__ == "__main__":
    main()
