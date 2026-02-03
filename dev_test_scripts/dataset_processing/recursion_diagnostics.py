#!/usr/bin/env python3
"""
Compute recursion diagnostics and baseline comparisons for public TCR clusters.

Outputs plots and JSON summaries under <export_root>/figures.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import random
import time
from datetime import datetime
from functools import partial
from collections import Counter
from pathlib import Path
from typing import List, Optional

import ijson
import matplotlib
import numpy as np
from tqdm import tqdm

from tcrtyper.dataset_processing.recursion_utils import (
    DEFAULT_MIN_CHI_SQUARED,
    find_solutions_from_indices,
    rank_alleles,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)


DEFAULT_CHI_GRID = [0.0, 1.0, 2.0, 3.841, 5.0, 8.0, 12.0]

_WORK_X = None
_WORK_X_BOOL = None
_WORK_COUNTS_A = None
_WORK_COUNTS_TOTAL = None
_WORK_ALLELE_CLASS = None
_WORK_ALLELE_GENE = None
_WORK_SETTINGS = {}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Diagnostics for recursion vs naive top-p on public TCR clusters."
    )
    ap.add_argument(
        "--export-root",
        default="export_train_dataset",
        help="Export train dataset root (default: export_train_dataset).",
    )
    ap.add_argument(
        "--public-json",
        default=None,
        help="Path to public_tcrs.json (default: <export_root>/public_tcrs.json).",
    )
    ap.add_argument(
        "--synthetic-json",
        default=None,
        help="Path to synthetic TCR JSON (overrides --public-json).",
    )
    ap.add_argument(
        "--synthetic-dir",
        default=None,
        help="Directory of synthetic JSON files to merge (overrides --public-json).",
    )
    ap.add_argument(
        "--donor-matrix",
        default=None,
        help=(
            "Path to donor×allele matrix (.npz or .npy). "
            "Default: <export_root>/donor_hla_matrix.npz."
        ),
    )
    ap.add_argument(
        "--donor-keys",
        default=None,
        help=(
            "Path to donor keys JSON (optional if .npz contains donor_keys). "
            "Default: <export_root>/donor_hla_matrix_donors.json."
        ),
    )
    ap.add_argument(
        "--min-donors",
        type=int,
        default=2,
        help="Minimum donors per cluster to include (default: 2).",
    )
    ap.add_argument(
        "--min-chi-squared",
        type=float,
        default=DEFAULT_MIN_CHI_SQUARED,
        help=f"Min chi-squared for recursion (default: {DEFAULT_MIN_CHI_SQUARED}).",
    )
    ap.add_argument(
        "--chi-grid",
        default=None,
        help="Comma-separated min_chi_squared grid (default: built-in list).",
    )
    ap.add_argument(
        "--skip-chi-sweep",
        action="store_true",
        help="Skip the chi-squared sweep (no solved-fraction plot).",
    )
    ap.add_argument(
        "--max-clusters",
        type=int,
        default=5000,
        help=(
            "Max clusters to process (reservoir sample). "
            "Use 0 to disable sampling and process all."
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for sampling clusters (default: 13).",
    )
    ap.add_argument(
        "--posterior",
        choices=("uniform", "mi_inverse", "mi_inverse_size"),
        default="mi_inverse",
        help="Posterior weight proxy over solutions (default: mi_inverse).",
    )
    ap.add_argument(
        "--search-mode",
        choices=("dfs", "beam", "mc"),
        default="beam",
        help="Search mode for solution enumeration (default: beam).",
    )
    ap.add_argument(
        "--beam-width",
        type=int,
        default=256,
        help="Beam width for beam search (default: 256).",
    )
    ap.add_argument(
        "--beam-score",
        choices=("remaining", "chi_sum", "hybrid"),
        default="chi_sum",
        help="Beam pruning score (default: chi_sum).",
    )
    ap.add_argument(
        "--mc-num-rollouts",
        type=int,
        default=1024,
        help="Monte-Carlo rollouts for mc mode (default: 1024).",
    )
    ap.add_argument(
        "--mc-top-k",
        type=int,
        default=20,
        help="Top-k candidates to sample from per rollout (default: 20).",
    )
    ap.add_argument(
        "--mc-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for mc sampling (default: 1.0).",
    )
    ap.add_argument(
        "--max-solutions",
        type=int,
        default=1000,
        help="Cap solutions per cluster (0 disables, default: 1000).",
    )
    ap.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Max alleles per solution (0 disables, default: 6).",
    )
    ap.add_argument(
        "--max-branch",
        type=int,
        default=30,
        help="Cap candidate alleles per recursion level (0 disables, default: 30).",
    )
    ap.add_argument(
        "--max-runtime-sec",
        type=float,
        default=2.0,
        help="Max seconds per cluster recursion (0 disables, default: 2.0).",
    )
    ap.add_argument(
        "--bootstrap-samples",
        type=int,
        default=20,
        help="Bootstrap replicates per cluster (default: 20; 0 disables).",
    )
    ap.add_argument(
        "--bootstrap-max-clusters",
        type=int,
        default=200,
        help="Max clusters to use for bootstrap stability (default: 200).",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Worker processes for per-cluster loops (default: 4).",
    )
    ap.add_argument(
        "--bootstrap-repeat",
        action="store_true",
        help="Compute repeat stability on identical donors (meaningful for mc mode).",
    )
    ap.add_argument(
        "--figures-dir",
        default=None,
        help="Output figures root dir (default: <export_root>/figures).",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
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


def _parse_chi_grid(arg: Optional[str]) -> List[float]:
    if not arg:
        return list(DEFAULT_CHI_GRID)
    out = []
    for raw in arg.split(","):
        raw = raw.strip()
        if not raw:
            continue
        out.append(float(raw))
    return out or list(DEFAULT_CHI_GRID)


def _can_fork() -> bool:
    try:
        return "fork" in mp.get_all_start_methods()
    except Exception:
        return False


def _load_id_to_hla(export_root: Path) -> dict[int, str]:
    candidates = [
        export_root / "id_to_hla.json",
        export_root / "hla_id_to_name.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            out: dict[int, str] = {}
            for k, v in raw.items():
                try:
                    out[int(k)] = str(v)
                except Exception:
                    continue
            if out:
                return out
    raise FileNotFoundError("id_to_hla.json not found under export_root.")


def _hla_class(name: str) -> str:
    u = str(name).upper()
    if u.startswith(("HLA-A", "HLA-B", "HLA-C")):
        return "I"
    if u.startswith(("HLA-DP", "HLA-DQ", "HLA-DR")):
        return "II"
    return "OTHER"


def _hla_gene(name: str) -> str:
    u = str(name).upper()
    if u.startswith("HLA-"):
        u = u[4:]
    gene = u.split("*", 1)[0]
    return gene


def _load_donor_keys_json(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"donor keys JSON not found: {path} (run build_donor_hla_matrix.py)"
        )
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list) or not data:
        raise SystemExit(f"donor keys JSON must be a non-empty list: {path}")
    return [str(x) for x in data]


def _load_donor_keys_from_npz(data) -> List[str] | None:
    for key in ["donor_keys", "donors", "donor_hla_matrix_donors"]:
        if key in data:
            vals = data[key]
            return [str(v) for v in vals.tolist()]
    return None


def _load_matrix_from_npz(data, npz_key: str | None = None) -> np.ndarray:
    keys = list(data.keys())
    if npz_key:
        if npz_key not in data:
            raise KeyError(f"npz key {npz_key!r} not found. Available keys: {keys}")
        return data[npz_key]
    if len(keys) == 1:
        return data[keys[0]]
    preferred = ["donor_hla_matrix", "donor_matrix", "matrix", "X", "data"]
    for key in preferred:
        if key in data:
            return data[key]
    raise KeyError(f"npz contains multiple arrays; available keys: {keys}")


def _load_donor_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"donor matrix not found: {path} (run build_donor_hla_matrix.py)"
        )
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            arr = _load_matrix_from_npz(data)
    else:
        arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise SystemExit(f"donor matrix must be 2D, got shape {arr.shape}")
    return arr


def _stream_clusters(
    public_json: Path,
    donor_to_idx: dict[str, int],
    min_donors: int,
    max_clusters: Optional[int],
    seed: int,
    show_progress: bool,
):
    clusters = []
    missing_donors = 0
    total_clusters = 0
    eligible_clusters = 0
    rng = random.Random(seed)

    with open(public_json, "r", encoding="utf-8") as fh:
        iterator = ijson.kvitems(fh, "")
        if show_progress:
            iterator = tqdm(iterator, desc="public_tcrs", unit="cluster")
        for cid, donors in iterator:
            total_clusters += 1
            if not isinstance(donors, list):
                continue
            uniq = []
            seen = set()
            for d in donors:
                if d in seen:
                    continue
                seen.add(d)
                idx = donor_to_idx.get(d)
                if idx is None:
                    missing_donors += 1
                    continue
                uniq.append(idx)
            if len(uniq) < min_donors:
                continue
            eligible_clusters += 1
            entry = {"cid": cid, "indices": np.array(uniq, dtype=int)}
            if max_clusters is None or max_clusters <= 0:
                clusters.append(entry)
                continue
            if len(clusters) < max_clusters:
                clusters.append(entry)
                continue
            j = rng.randrange(eligible_clusters)
            if j < max_clusters:
                clusters[j] = entry

    return clusters, total_clusters, eligible_clusters, missing_donors


def _load_synthetic_file(path: Path) -> tuple[list[dict], dict]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    meta = {}
    if isinstance(payload, dict):
        clusters = payload.get("clusters", [])
        for key in (
            "min_donors",
            "max_donors",
            "min_allele_coverage",
            "min_unique_coverage",
            "solution_sizes",
        ):
            if key in payload:
                meta[key] = payload[key]
    elif isinstance(payload, list):
        clusters = payload
    else:
        raise SystemExit(f"Synthetic JSON must be a dict or list: {path}")
    if not isinstance(clusters, list):
        raise SystemExit(f"Synthetic JSON clusters must be a list: {path}")
    return clusters, meta


def _load_synthetic_clusters(
    synthetic_json: Optional[Path],
    synthetic_dir: Optional[Path],
    min_donors: int,
    max_clusters: Optional[int],
    seed: int,
) -> tuple[list[dict], int, int, dict]:
    files: list[Path] = []
    if synthetic_json is not None:
        files.append(synthetic_json)
    if synthetic_dir is not None:
        files.extend(sorted(Path(synthetic_dir).glob("*.json")))
    if not files:
        raise SystemExit("Synthetic input requested but no JSON files were found.")

    all_entries = []
    meta_list: list[dict] = []
    total_clusters = 0
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"Synthetic JSON not found: {path}")
        clusters, meta = _load_synthetic_file(path)
        if meta:
            meta_list.append(meta)
        for entry in clusters:
            total_clusters += 1
            if not isinstance(entry, dict):
                continue
            donors = entry.get("donor_indices") or entry.get("indices")
            if donors is None:
                continue
            indices = np.asarray(donors, dtype=int)
            if indices.size < min_donors:
                continue
            cid = entry.get("cid") or f"syn_{total_clusters:06d}"
            cluster = {"cid": cid, "indices": indices}
            sol = entry.get("solution")
            if sol is not None:
                cluster["synthetic_solution"] = sorted(int(x) for x in sol)
                cluster["synthetic_solution_size"] = len(cluster["synthetic_solution"])
                cluster["synthetic_solution_minimal"] = bool(entry.get("solution_is_minimal", False))
            all_entries.append(cluster)

    eligible_clusters = len(all_entries)
    if max_clusters is not None and max_clusters > 0 and eligible_clusters > max_clusters:
        rng = random.Random(seed)
        rng.shuffle(all_entries)
        all_entries = all_entries[:max_clusters]
    meta_summary: dict = {}
    for key in ("min_donors", "max_donors", "min_allele_coverage", "min_unique_coverage"):
        vals = [m.get(key) for m in meta_list if isinstance(m.get(key), int)]
        if not vals:
            continue
        uniq = sorted(set(vals))
        meta_summary[key] = uniq[0] if len(uniq) == 1 else "mix"
    sizes = []
    for meta in meta_list:
        if isinstance(meta.get("solution_sizes"), list):
            for s in meta["solution_sizes"]:
                if isinstance(s, int):
                    sizes.append(s)
    if sizes:
        meta_summary["solution_size_range"] = (min(sizes), max(sizes))
    return all_entries, total_clusters, eligible_clusters, meta_summary


def _synthetic_label(meta: dict) -> str:
    if not meta:
        return "syn"
    parts = ["syn"]
    for key, prefix in (
        ("min_donors", "mindon"),
        ("max_donors", "maxdon"),
        ("min_allele_coverage", "cov"),
        ("min_unique_coverage", "ucov"),
    ):
        val = meta.get(key)
        if val is None:
            continue
        parts.append(f"{prefix}{val}")
    if "solution_size_range" in meta:
        lo, hi = meta["solution_size_range"]
        parts.append(f"s{lo}..{hi}")
    return "_".join(parts)

def _solution_mi(x_bool: np.ndarray, allele_idx: np.ndarray) -> int:
    if allele_idx.size == 0:
        return 0
    covered = np.any(x_bool[:, allele_idx], axis=1)
    return int(np.count_nonzero(covered))


def _solution_mi_all(x_bool: np.ndarray, allele_idx: np.ndarray) -> Optional[int]:
    if allele_idx.size == 0:
        return None
    covered = np.all(x_bool[:, allele_idx], axis=1)
    return int(np.count_nonzero(covered))


def _metrics_worker(cluster: dict) -> dict:
    if _WORK_X is None:
        raise RuntimeError("Worker globals are not initialized.")
    x = _WORK_X
    x_bool = _WORK_X_BOOL
    counts_a = _WORK_COUNTS_A
    counts_total = _WORK_COUNTS_TOTAL
    allele_class = _WORK_ALLELE_CLASS
    allele_gene = _WORK_ALLELE_GENE
    settings = _WORK_SETTINGS

    idx = cluster["indices"]
    n_i = int(idx.size)
    x_prime = x[idx]
    y_vec = np.ones(x_prime.shape[0], dtype=np.uint8)
    cand = np.ones(x_prime.shape[1], dtype=np.uint8)
    ranked = rank_alleles(
        y_vec,
        x_prime,
        cand,
        counts_a,
        counts_total,
        min_chi_squared=None,
        require_coverage=True,
        enrichment_only=True,
    )
    top_allele = int(ranked[0]) if ranked.size else -1
    baseline_sig = rank_alleles(
        y_vec,
        x_prime,
        cand,
        counts_a,
        counts_total,
        min_chi_squared=settings["min_chi_squared"],
        require_coverage=True,
        enrichment_only=True,
    )
    baseline_mi = int(counts_a[top_allele]) if top_allele >= 0 else None

    solutions, stats = find_solutions_from_indices(
        idx,
        x,
        min_chi_squared=settings["min_chi_squared"],
        counts_a=counts_a,
        counts_total=counts_total,
        search_mode=settings["search_mode"],
        beam_width=settings["beam_width"],
        beam_score=settings["beam_score"],
        mc_num_rollouts=settings["mc_num_rollouts"],
        mc_top_k=settings["mc_top_k"],
        mc_temperature=settings["mc_temperature"],
        max_solutions=settings["max_solutions"],
        max_depth=settings["max_depth"],
        max_branch=settings["max_branch"],
        max_runtime_sec=settings["max_runtime_sec"],
        return_stats=True,
    )
    num_solutions = len(solutions)
    no_candidate_final = (
        stats.get("no_candidate_hit", False)
        and num_solutions == 0
        and not stats.get("truncated", False)
    )
    synthetic_solution = cluster.get("synthetic_solution")
    synthetic_found = None
    if synthetic_solution is not None:
        sol_set = np.array(sorted(int(x) for x in synthetic_solution), dtype=int)
        synthetic_found = False
        for z in solutions:
            idx = np.flatnonzero(z)
            if idx.size != sol_set.size:
                continue
            if np.array_equal(idx, sol_set):
                synthetic_found = True
                break

    min_size = None
    min_mi = None
    best_idx, best_mi, best_size = _best_solution(solutions, x_bool)
    if best_idx is not None:
        min_size = best_size
        min_mi = best_mi
    min_mi_all = _solution_mi_all(x_bool, best_idx) if best_idx is not None else None

    baseline_any = _solution_mi(x_bool, baseline_sig) if baseline_sig.size else None
    baseline_all = _solution_mi_all(x_bool, baseline_sig) if baseline_sig.size else None

    mi_list = []
    size_list = []
    for z in solutions:
        z_idx = np.flatnonzero(z)
        size_list.append(int(z_idx.size))
        mi_list.append(_solution_mi(x_bool, z_idx))

    purity_rec = _class_purity(best_idx, allele_class) if best_idx is not None else None
    purity_base = _class_purity(baseline_sig, allele_class) if baseline_sig.size else None

    posterior = _posterior_weights(mi_list, size_list, settings["posterior"])
    top1_mass = float(np.max(posterior)) if posterior.size else None
    ent = _entropy(posterior) if posterior.size else None
    if ent is not None and math.isnan(ent):
        ent = None

    return {
        "cid": cluster["cid"],
        "n_donors": n_i,
        "num_solutions": num_solutions,
        "truncated": stats.get("truncated", False),
        "max_solutions_hit": stats.get("max_solutions_hit", False),
        "max_depth_hit": stats.get("max_depth_hit", False),
        "max_branch_capped": stats.get("max_branch_capped", False),
        "timeout_hit": stats.get("timeout_hit", False),
        "no_candidate_hit": stats.get("no_candidate_hit", False),
        "no_candidate_final": no_candidate_final,
        "min_solution_size": min_size,
        "min_mi": min_mi,
        "min_mi_all": min_mi_all,
        "baseline_top_allele": top_allele,
        "baseline_mi": baseline_mi,
        "baseline_mi_all": baseline_all,
        "class_purity_recursion": purity_rec,
        "class_purity_baseline": purity_base,
        "posterior_top1_mass": top1_mass,
        "posterior_entropy": ent,
        "top_class_recursion": _top_class(best_idx, allele_class) if best_idx is not None else "",
        "top_gene_recursion": _top_gene(best_idx, allele_gene) if best_idx is not None else "",
        "baseline_any": baseline_any,
        "synthetic_solution": synthetic_solution,
        "synthetic_solution_found": synthetic_found,
        "synthetic_solution_size": cluster.get("synthetic_solution_size"),
        "synthetic_solution_minimal": cluster.get("synthetic_solution_minimal"),
    }


def _chi_sweep_worker(cluster: dict, *, chi: float) -> tuple[int, int]:
    if _WORK_X is None:
        raise RuntimeError("Worker globals are not initialized.")
    settings = _WORK_SETTINGS
    x = _WORK_X
    counts_a = _WORK_COUNTS_A
    counts_total = _WORK_COUNTS_TOTAL
    idx = cluster["indices"]
    solutions, stats = find_solutions_from_indices(
        idx,
        x,
        min_chi_squared=chi,
        counts_a=counts_a,
        counts_total=counts_total,
        search_mode=settings["search_mode"],
        beam_width=settings["beam_width"],
        beam_score=settings["beam_score"],
        mc_num_rollouts=settings["mc_num_rollouts"],
        mc_top_k=settings["mc_top_k"],
        mc_temperature=settings["mc_temperature"],
        max_solutions=settings["max_solutions"],
        max_depth=settings["max_depth"],
        max_branch=settings["max_branch"],
        max_runtime_sec=settings["max_runtime_sec"],
        return_stats=True,
    )
    return (1 if solutions else 0, 1 if stats.get("truncated", False) else 0)


def _best_solution(
    solutions: List[np.ndarray],
    x_bool: np.ndarray,
):
    if not solutions:
        return None, None, None
    best = None
    best_mi = None
    best_size = None
    for z in solutions:
        idx = np.flatnonzero(z)
        mi = _solution_mi(x_bool, idx)
        size = int(idx.size)
        if best is None or mi < best_mi or (mi == best_mi and size < best_size):
            best = idx
            best_mi = mi
            best_size = size
    return best, best_mi, best_size


def _class_purity(allele_idx: np.ndarray, allele_class: List[str]) -> Optional[float]:
    if allele_idx.size == 0:
        return None
    n_i = sum(1 for i in allele_idx if allele_class[i] == "I")
    n_ii = sum(1 for i in allele_idx if allele_class[i] == "II")
    denom = allele_idx.size
    if denom <= 0:
        return None
    return max(n_i, n_ii) / float(denom)


def _top_class(allele_idx: np.ndarray, allele_class: List[str]) -> str:
    n_i = sum(1 for i in allele_idx if allele_class[i] == "I")
    n_ii = sum(1 for i in allele_idx if allele_class[i] == "II")
    if n_i == 0 and n_ii == 0:
        return ""
    return "I" if n_i >= n_ii else "II"


def _top_gene(allele_idx: np.ndarray, allele_gene: List[str]) -> str:
    counts = Counter()
    for i in allele_idx:
        g = allele_gene[i]
        if g:
            counts[g] += 1
    if not counts:
        return ""
    gene, _ = max(counts.items(), key=lambda t: (t[1], t[0]))
    return gene


def _posterior_weights(mi_list: List[int], size_list: List[int], method: str) -> np.ndarray:
    if not mi_list:
        return np.array([], dtype=np.float64)
    mi = np.array(mi_list, dtype=np.float64)
    sizes = np.array(size_list, dtype=np.float64)
    if method == "uniform":
        w = np.ones_like(mi)
    elif method == "mi_inverse_size":
        w = 1.0 / np.maximum(mi, 1.0)
        w /= np.maximum(sizes, 1.0)
    else:
        w = 1.0 / np.maximum(mi, 1.0)
    total = float(np.sum(w))
    if total <= 0:
        return np.ones_like(mi) / float(len(mi))
    return w / total


def _entropy(prob: np.ndarray) -> float:
    if prob.size == 0:
        return float("nan")
    mask = prob > 0
    vals = prob[mask]
    return float(-np.sum(vals * np.log(vals)))


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _plot_solved_fraction(fig_path: Path, chi_grid, solved_frac, runtime_sec) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(chi_grid, solved_frac, marker="o", color="tab:blue", label="Solved fraction")
    ax1.set_xlabel("min_chi_squared")
    ax1.set_ylabel("Solved fraction", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(chi_grid, runtime_sec, marker="s", color="tab:red", label="Runtime (s)")
    ax2.set_ylabel("Runtime (s)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def _plot_histograms(
    fig_path: Path,
    num_solutions,
    min_sizes,
    *,
    n_with: int,
    n_without: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    subtitle = f"Clusters with ≥1 solution: {n_with} | without solution: {n_without}"

    if num_solutions:
        max_sol = max(num_solutions)
        bins = np.linspace(1, max_sol + 1, min(40, max_sol + 1))
        axes[0].hist(num_solutions, bins=bins, color="tab:blue", alpha=0.8)
    axes[0].set_xlabel("|Z_i| (#solutions)")
    axes[0].set_ylabel("TCR clusters")
    axes[0].set_title(subtitle, fontsize=9)

    if min_sizes:
        max_size = max(min_sizes)
        bins = np.arange(1, max_size + 2)
        axes[1].hist(min_sizes, bins=bins, color="tab:green", alpha=0.8)
    axes[1].set_xlabel("min |z|")
    axes[1].set_ylabel("TCR clusters")
    axes[1].set_title(subtitle, fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def _plot_long_template(
    fig_path: Path,
    *,
    x_left,
    y_left,
    x_right,
    y_right,
    left_label: str,
    right_label: str,
    right_log: bool = False,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> None:
    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax1.scatter(x_left, y_left, s=36, color="tab:blue", alpha=0.85)
    ax1.set_xlabel("Cluster index")
    ax1.set_ylabel(left_label, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.scatter(x_right, y_right, s=36, color="tab:orange", alpha=0.85)
    ax2.set_ylabel(right_label, color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    if right_log:
        ax2.set_yscale("log")

    if title:
        fig.suptitle(title)
    if subtitle:
        ax1.set_title(subtitle, fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def _plot_specificity_scatter_pair(
    fig_path: Path,
    *,
    rec_any_n,
    rec_any_y,
    base_any_n,
    base_any_y,
    rec_all_n,
    rec_all_y,
    base_all_n,
    base_all_y,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: any allele coverage
    ax = axes[0]
    if rec_any_n and rec_any_y:
        ax.scatter(rec_any_n, rec_any_y, s=12, alpha=0.5, label="Recursion")
    if base_any_n and base_any_y:
        ax.scatter(base_any_n, base_any_y, s=12, alpha=0.5, label="Naive p<0.05")
    max_val = 0
    if rec_any_n:
        max_val = max(max_val, max(rec_any_n))
    if rec_any_y:
        max_val = max(max_val, max(rec_any_y))
    if base_any_n:
        max_val = max(max_val, max(base_any_n))
    if base_any_y:
        max_val = max(max_val, max(base_any_y))
    if max_val > 0:
        ax.plot([0, max_val], [0, max_val], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("N_i (# donors with TCR)")
    ax.set_ylabel("min M_i(z) (# donors with any allele in solution)")
    ax.legend()

    # Right: all alleles coverage
    ax = axes[1]
    if rec_all_n and rec_all_y:
        ax.scatter(rec_all_n, rec_all_y, s=12, alpha=0.5, label="Recursion")
    if base_all_n and base_all_y:
        ax.scatter(base_all_n, base_all_y, s=12, alpha=0.5, label="Naive p<0.05")
    max_val = 0
    if rec_all_n:
        max_val = max(max_val, max(rec_all_n))
    if rec_all_y:
        max_val = max(max_val, max(rec_all_y))
    if base_all_n:
        max_val = max(max_val, max(base_all_n))
    if base_all_y:
        max_val = max(max_val, max(base_all_y))
    if max_val > 0:
        ax.plot([0, max_val], [0, max_val], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("N_i (# donors with TCR)")
    ax.set_ylabel("M_i^all(z) (# donors with all alleles in solution)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def _plot_class_purity(fig_path: Path, purity_rec, purity_base) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 11)
    if purity_rec:
        ax.hist(purity_rec, bins=bins, alpha=0.6, label="Recursion")
    if purity_base:
        ax.hist(purity_base, bins=bins, alpha=0.6, label="Naive p<0.05")
    ax.set_xlabel("Class purity (I vs II)")
    ax.set_ylabel("TCR clusters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def _plot_n_donors_hist(fig_path: Path, n_donors_list) -> None:
    if not n_donors_list:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    max_val = max(n_donors_list)
    bins = np.arange(0, max_val + 2)
    ax.hist(n_donors_list, bins=bins, color="tab:blue", alpha=0.8)
    ax.set_xlabel("N_i (# donors with TCR)")
    ax.set_ylabel("TCR clusters")
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def _plot_num_solutions_hist(fig_path: Path, num_solutions_list) -> None:
    if not num_solutions_list:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    max_val = max(num_solutions_list)
    if max_val <= 50:
        bins = np.arange(0, max_val + 2)
    else:
        bins = np.linspace(0, max_val, 40)
    ax.hist(num_solutions_list, bins=bins, color="tab:purple", alpha=0.8)
    ax.set_xlabel("|Z_i| (#solutions, includes zeros)")
    ax.set_ylabel("TCR clusters")
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def _plot_truncation_summary(fig_path: Path, cluster_metrics: List[dict]) -> None:
    if not cluster_metrics:
        return
    total = len(cluster_metrics)
    counts = {
        "truncated": sum(1 for c in cluster_metrics if c.get("truncated")),
        "timeout": sum(1 for c in cluster_metrics if c.get("timeout_hit")),
        "max_branch": sum(1 for c in cluster_metrics if c.get("max_branch_capped")),
        "max_depth": sum(1 for c in cluster_metrics if c.get("max_depth_hit")),
        "max_solutions": sum(1 for c in cluster_metrics if c.get("max_solutions_hit")),
        "no_candidate_final": sum(1 for c in cluster_metrics if c.get("no_candidate_final")),
        "no_solution": sum(1 for c in cluster_metrics if (c.get("num_solutions") or 0) == 0),
    }
    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="tab:gray", alpha=0.8)
    ax.set_ylabel("TCR clusters")
    ax.set_title(f"Counts over {total} clusters")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def _plot_status_wide(fig_path: Path, cluster_metrics: List[dict]) -> None:
    if not cluster_metrics:
        return
    n = len(cluster_metrics)
    xs = np.arange(1, n + 1)

    class_labels = ["I", "II", "none"]
    class_colors = ["#1f77b4", "#ff7f0e", "#b0b0b0"]
    class_map = {"I": 0, "II": 1, "": 2, None: 2}
    class_codes = [class_map.get(c.get("top_class_recursion"), 2) for c in cluster_metrics]

    flag_specs = [
        ("ok", "#2ca02c", lambda c: (c.get("num_solutions") or 0) > 0 and not c.get("truncated")),
        ("no_solution", "#c7c7c7", lambda c: (c.get("num_solutions") or 0) == 0),
        ("no_candidate_final", "#aec7e8", lambda c: c.get("no_candidate_final")),
        ("timeout", "#d62728", lambda c: c.get("timeout_hit")),
        ("max_solutions", "#9467bd", lambda c: c.get("max_solutions_hit")),
        ("max_depth", "#ff7f0e", lambda c: c.get("max_depth_hit")),
        ("max_branch", "#8c564b", lambda c: c.get("max_branch_capped")),
        ("truncated", "#7f7f7f", lambda c: c.get("truncated")),
    ]
    off_rgb = np.array(to_rgb("#f0f0f0"))
    flag_img = np.zeros((len(flag_specs), n, 3), dtype=np.float32)
    for i, (label, color, pred) in enumerate(flag_specs):
        row = np.tile(off_rgb, (n, 1))
        on_rgb = np.array(to_rgb(color))
        for j, c in enumerate(cluster_metrics):
            if pred(c):
                row[j] = on_rgb
        flag_img[i] = row

    fig, axes = plt.subplots(2, 1, figsize=(14, 4), sharex=True)

    axes[0].imshow(
        [class_codes],
        aspect="auto",
        cmap=ListedColormap(class_colors),
        vmin=0,
        vmax=len(class_colors) - 1,
    )
    axes[0].set_yticks([])
    axes[0].set_title("Top class (recursion)")

    axes[1].imshow(flag_img, aspect="auto")
    axes[1].set_yticks(np.arange(len(flag_specs)))
    axes[1].set_yticklabels([f[0] for f in flag_specs], fontsize=8)
    axes[1].set_xlabel("Cluster index")
    axes[1].set_title("Truncation / outcome flags")

    if n <= 60:
        axes[1].set_xticks(xs - 1)
        axes[1].set_xticklabels([str(x) for x in xs], fontsize=8)
    else:
        step = max(1, n // 20)
        ticks = np.arange(0, n, step)
        axes[1].set_xticks(ticks)
        axes[1].set_xticklabels([str(i + 1) for i in ticks], fontsize=8)

    class_legend = [Patch(facecolor=class_colors[i], label=class_labels[i]) for i in range(len(class_labels))]
    flag_legend = [Patch(facecolor=f[1], label=f[0]) for f in flag_specs]
    axes[0].legend(handles=class_legend, loc="upper right", ncol=len(class_labels), fontsize=8)
    axes[1].legend(handles=flag_legend, loc="upper right", ncol=3, fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def _plot_synthetic_per_tcr(fig_path: Path, cluster_metrics: List[dict]) -> None:
    syn = [m for m in cluster_metrics if m.get("synthetic_solution") is not None]
    if not syn:
        return
    n = len(syn)
    xs = np.arange(1, n + 1)
    num_solutions = [int(m.get("num_solutions") or 0) for m in syn]
    indicator = [1 if m.get("synthetic_solution") is not None else 0 for m in syn]

    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax1.bar(xs, num_solutions, color="tab:blue", alpha=0.7, label="#solutions")
    ax1.set_ylabel("#solutions")
    ax1.set_xlabel("Synthetic cluster index")

    ax2 = ax1.twinx()
    ax2.bar(xs, indicator, color="tab:green", alpha=0.6, width=0.4, label="predefined solution")
    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel("Predefined solution (0/1)")

    if n <= 60:
        ax1.set_xticks(xs)
        ax1.set_xticklabels([str(x) for x in xs], fontsize=8)
    else:
        step = max(1, n // 20)
        ticks = np.arange(1, n + 1, step)
        ax1.set_xticks(ticks)
        ax1.set_xticklabels([str(x) for x in ticks], fontsize=8)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def _plot_synthetic_by_size(fig_path: Path, cluster_metrics: List[dict]) -> None:
    syn = [m for m in cluster_metrics if m.get("synthetic_solution_size") is not None]
    if not syn:
        return
    size_bins: dict[int, dict[str, int]] = {}
    for m in syn:
        size = int(m.get("synthetic_solution_size"))
        if size not in size_bins:
            size_bins[size] = {"found": 0, "any": 0}
        if m.get("synthetic_solution_found"):
            size_bins[size]["found"] += 1
        if (m.get("num_solutions") or 0) > 0:
            size_bins[size]["any"] += 1

    sizes = sorted(size_bins)
    found = [size_bins[s]["found"] for s in sizes]
    any_sol = [size_bins[s]["any"] for s in sizes]
    xs = np.arange(len(sizes))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(xs - width / 2, found, width=width, color="tab:green", alpha=0.8, label="predefined solution found")
    ax.bar(xs + width / 2, any_sol, width=width, color="tab:blue", alpha=0.6, label=">0 solutions")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("True solution size")
    ax.set_ylabel("TCR clusters")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def _plot_posterior(fig_path: Path, top1_mass, entropy_vals) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if top1_mass:
        axes[0].hist(top1_mass, bins=20, color="tab:purple", alpha=0.8)
    axes[0].set_xlabel("Posterior top-1 mass")

    if entropy_vals:
        axes[1].hist(entropy_vals, bins=20, color="tab:orange", alpha=0.8)
    axes[1].set_xlabel("Posterior entropy")
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def _plot_bootstrap(fig_path: Path, n_donors, stability_class, stability_gene) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if stability_class:
        ax.scatter(n_donors, stability_class, s=12, alpha=0.6, label="Top class")
    if stability_gene:
        ax.scatter(n_donors, stability_gene, s=12, alpha=0.6, label="Top gene")
    ax.set_xlabel("N_i")
    ax.set_ylabel("Bootstrap stability")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)
    if args.max_clusters is not None and args.max_clusters < 0:
        logger.warning("max-clusters < 0 is invalid; using 0 (no sampling).")
        args.max_clusters = 0
    if args.max_solutions is not None and args.max_solutions < 0:
        logger.warning("max-solutions < 0 is invalid; using 0 (no cap).")
        args.max_solutions = 0
    if args.max_depth is not None and args.max_depth < 0:
        logger.warning("max-depth < 0 is invalid; using 0 (no cap).")
        args.max_depth = 0
    if args.max_branch is not None and args.max_branch < 0:
        logger.warning("max-branch < 0 is invalid; using 0 (no cap).")
        args.max_branch = 0
    if args.max_runtime_sec is not None and args.max_runtime_sec < 0:
        logger.warning("max-runtime-sec < 0 is invalid; using 0 (no cap).")
        args.max_runtime_sec = 0
    if args.threads < 1:
        logger.warning("threads < 1 is invalid; using 1.")
        args.threads = 1

    export_root = Path(args.export_root).resolve()
    use_synthetic = bool(args.synthetic_json or args.synthetic_dir)
    public_json = None
    if not use_synthetic:
        public_json = (
            Path(args.public_json).resolve()
            if args.public_json
            else (export_root / "public_tcrs.json")
        )
        if not public_json.exists():
            raise FileNotFoundError(f"public_tcrs.json not found: {public_json}")

    if args.donor_matrix:
        donor_matrix_path = Path(args.donor_matrix).resolve()
    else:
        candidates = [
            export_root / "donor_hla_matrix.npz",
            export_root / "donor_hla_matrix.npy",
        ]
        donor_matrix_path = next((p for p in candidates if p.exists()), candidates[0])
    donor_keys_path = Path(args.donor_keys).resolve() if args.donor_keys else None

    figures_root = Path(args.figures_dir).resolve() if args.figures_dir else (export_root / "figures")
    max_solutions = args.max_solutions if args.max_solutions and args.max_solutions > 0 else None
    max_depth = args.max_depth if args.max_depth and args.max_depth > 0 else None
    max_branch = args.max_branch if args.max_branch and args.max_branch > 0 else None
    max_runtime_sec = (
        args.max_runtime_sec if args.max_runtime_sec and args.max_runtime_sec > 0 else None
    )
    max_sol_label = str(max_solutions) if max_solutions is not None else "none"
    max_branch_label = str(max_branch) if max_branch is not None else "none"
    if args.search_mode == "mc":
        params_label = f"mc{args.mc_num_rollouts}_temp{args.mc_temperature:g}"
    elif args.search_mode == "beam":
        params_label = f"beam{args.beam_width}_{args.beam_score}"
    else:
        params_label = "dfs"
    id_to_hla = _load_id_to_hla(export_root)
    num_alleles = len(id_to_hla)
    allele_class = ["" for _ in range(num_alleles)]
    allele_gene = ["" for _ in range(num_alleles)]
    for idx in range(num_alleles):
        name = id_to_hla.get(idx, "")
        allele_class[idx] = _hla_class(name)
        allele_gene[idx] = _hla_gene(name)

    donor_keys = None
    if donor_matrix_path.suffix.lower() == ".npz":
        with np.load(donor_matrix_path) as data:
            donor_keys = _load_donor_keys_from_npz(data)
    if donor_keys is None and donor_keys_path is not None:
        donor_keys = _load_donor_keys_json(donor_keys_path)
    if donor_keys is None:
        raise SystemExit("donor keys not found in .npz and no JSON provided.")
    donor_to_idx = {k: i for i, k in enumerate(donor_keys)}
    x = _load_donor_matrix(donor_matrix_path)
    if x.shape[0] != len(donor_keys):
        raise SystemExit(
            f"donor matrix rows ({x.shape[0]}) != donor keys ({len(donor_keys)})"
        )
    if x.shape[1] != num_alleles:
        raise SystemExit(
            f"donor matrix cols ({x.shape[1]}) != num_alleles ({num_alleles})"
        )
    x_bool = x.astype(bool, copy=False)
    counts_a = np.sum(x, axis=0).astype(np.int64, copy=False)
    counts_total = x.shape[0]

    logger.info("Loaded %d donors, %d alleles", counts_total, num_alleles)

    synthetic_meta = {}
    if use_synthetic:
        clusters, total_clusters, eligible_clusters, synthetic_meta = _load_synthetic_clusters(
            Path(args.synthetic_json).resolve() if args.synthetic_json else None,
            Path(args.synthetic_dir).resolve() if args.synthetic_dir else None,
            args.min_donors,
            args.max_clusters,
            args.seed,
        )
        missing_refs = 0
    else:
        clusters, total_clusters, eligible_clusters, missing_refs = _stream_clusters(
            public_json,
            donor_to_idx,
            args.min_donors,
            args.max_clusters,
            args.seed,
            show_progress=not args.no_progress,
        )

    ntcr_label = str(args.max_clusters) if args.max_clusters and args.max_clusters > 0 else "all"
    run_label = (
        f"{args.search_mode}_branch{max_branch_label}_don{args.min_donors}"
        f"_ntcr{ntcr_label}_maxsol{max_sol_label}_params{params_label}"
    )
    if use_synthetic:
        run_label = f"{_synthetic_label(synthetic_meta)}_{run_label}"
    run_label = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{run_label}"
    figures_dir = figures_root / run_label
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Clusters: %d total, %d eligible (min_donors=%d), sampled=%d, missing_refs=%d",
        total_clusters,
        eligible_clusters,
        args.min_donors,
        len(clusters),
        missing_refs,
    )
    logger.info(
        "Recursion limits: max_solutions=%s max_depth=%s max_branch=%s max_runtime_sec=%s",
        args.max_solutions,
        args.max_depth,
        args.max_branch,
        args.max_runtime_sec,
    )
    if args.search_mode == "beam":
        logger.info(
            "Search mode: %s (beam_width=%d, beam_score=%s)",
            args.search_mode,
            args.beam_width,
            args.beam_score,
        )
    elif args.search_mode == "mc":
        logger.info(
            "Search mode: %s (mc_rollouts=%d, mc_top_k=%d, mc_temp=%.3f)",
            args.search_mode,
            args.mc_num_rollouts,
            args.mc_top_k,
            args.mc_temperature,
        )
    else:
        logger.info("Search mode: %s", args.search_mode)
    use_pool = args.threads > 1
    if use_pool and not _can_fork():
        logger.warning("Multiprocessing requires fork; falling back to threads=1.")
        use_pool = False
        args.threads = 1
    if use_pool:
        logger.info("Using %d worker processes for per-cluster loops", args.threads)
    if args.max_clusters and args.max_clusters > 0 and eligible_clusters > args.max_clusters:
        logger.info(
            "Sampling enabled: using %d of %d eligible clusters.",
            args.max_clusters,
            eligible_clusters,
        )

    global _WORK_X, _WORK_X_BOOL, _WORK_COUNTS_A, _WORK_COUNTS_TOTAL
    global _WORK_ALLELE_CLASS, _WORK_ALLELE_GENE, _WORK_SETTINGS
    _WORK_X = x
    _WORK_X_BOOL = x_bool
    _WORK_COUNTS_A = counts_a
    _WORK_COUNTS_TOTAL = counts_total
    _WORK_ALLELE_CLASS = allele_class
    _WORK_ALLELE_GENE = allele_gene
    _WORK_SETTINGS.clear()
    _WORK_SETTINGS.update(
        {
            "min_chi_squared": args.min_chi_squared,
            "search_mode": args.search_mode,
            "beam_width": args.beam_width,
            "beam_score": args.beam_score,
            "mc_num_rollouts": args.mc_num_rollouts,
            "mc_top_k": args.mc_top_k,
            "mc_temperature": args.mc_temperature,
            "max_solutions": max_solutions,
            "max_depth": max_depth,
            "max_branch": max_branch,
            "max_runtime_sec": max_runtime_sec,
            "posterior": args.posterior,
        }
    )

    chunksize = max(1, len(clusters) // (args.threads * 4)) if clusters else 1
    cluster_metrics = []
    chi_grid = []
    solved_frac = []
    runtime_sec = []
    truncated_counts = []

    if use_pool:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=args.threads) as pool:
            metrics_iter = pool.imap(_metrics_worker, clusters, chunksize=chunksize)
            if not args.no_progress:
                metrics_iter = tqdm(metrics_iter, total=len(clusters), desc="metrics", unit="cluster")
            cluster_metrics = list(metrics_iter)
            pool_metrics = pool
            if not args.skip_chi_sweep:
                # chi sweep uses same pool
                chi_grid = _parse_chi_grid(args.chi_grid)
                sweep_iter = chi_grid
                if not args.no_progress:
                    sweep_iter = tqdm(chi_grid, desc="chi_sweep", unit="chi", position=0)
                for chi in sweep_iter:
                    start = time.perf_counter()
                    solved = 0
                    truncated = 0
                    func = partial(_chi_sweep_worker, chi=chi)
                    chi_iter = pool_metrics.imap(func, clusters, chunksize=chunksize)
                    if not args.no_progress:
                        chi_iter = tqdm(
                            chi_iter,
                            total=len(clusters),
                            desc=f"chi={chi:g}",
                            unit="cluster",
                            leave=False,
                            position=1,
                        )
                    for has_sol, trunc in chi_iter:
                        solved += has_sol
                        truncated += trunc
                    elapsed = time.perf_counter() - start
                    solved_frac.append(solved / float(len(clusters)) if clusters else 0.0)
                    runtime_sec.append(elapsed)
                    truncated_counts.append(truncated)
    else:
        metrics_iter = clusters
        if not args.no_progress:
            metrics_iter = tqdm(metrics_iter, desc="metrics", unit="cluster")
        cluster_metrics = [_metrics_worker(c) for c in metrics_iter]
        if not args.skip_chi_sweep:
            chi_grid = _parse_chi_grid(args.chi_grid)
            sweep_iter = chi_grid
            if not args.no_progress:
                sweep_iter = tqdm(chi_grid, desc="chi_sweep", unit="chi", position=0)
            for chi in sweep_iter:
                start = time.perf_counter()
                solved = 0
                truncated = 0
                cluster_iter = clusters
                if not args.no_progress:
                    cluster_iter = tqdm(
                        clusters,
                        desc=f"chi={chi:g}",
                        unit="cluster",
                        leave=False,
                        position=1,
                    )
                for cluster in cluster_iter:
                    has_sol, trunc = _chi_sweep_worker(cluster, chi=chi)
                    solved += has_sol
                    truncated += trunc
                elapsed = time.perf_counter() - start
                solved_frac.append(solved / float(len(clusters)) if clusters else 0.0)
                runtime_sec.append(elapsed)
                truncated_counts.append(truncated)

    if use_synthetic:
        found_list = [
            m.get("synthetic_solution_found")
            for m in cluster_metrics
            if m.get("synthetic_solution") is not None
        ]
        if found_list:
            found_count = sum(1 for v in found_list if v)
            logger.info("Synthetic solutions found: %d/%d", found_count, len(found_list))
        size_stats: dict[int, list[int]] = {}
        for m in cluster_metrics:
            size = m.get("synthetic_solution_size")
            found = m.get("synthetic_solution_found")
            if size is None or found is None:
                continue
            size = int(size)
            if size not in size_stats:
                size_stats[size] = [0, 0]
            size_stats[size][1] += 1
            if found:
                size_stats[size][0] += 1
        if size_stats:
            parts = []
            for size in sorted(size_stats):
                found, total = size_stats[size]
                parts.append(f"{size}: {found}/{total}")
            logger.info("Synthetic solutions found by size: %s", ", ".join(parts))

    num_solutions_list = [m.get("num_solutions") for m in cluster_metrics]
    min_size_list = [m.get("min_solution_size") for m in cluster_metrics if m.get("min_solution_size") is not None]
    scatter_rec_n = [m.get("n_donors") for m in cluster_metrics if m.get("min_mi") is not None]
    scatter_rec_mi = [m.get("min_mi") for m in cluster_metrics if m.get("min_mi") is not None]
    scatter_rec_all_n = [m.get("n_donors") for m in cluster_metrics if m.get("min_mi_all") is not None]
    scatter_rec_mi_all = [m.get("min_mi_all") for m in cluster_metrics if m.get("min_mi_all") is not None]
    scatter_base_n = [m.get("n_donors") for m in cluster_metrics if m.get("baseline_any") is not None]
    scatter_base_mi = [m.get("baseline_any") for m in cluster_metrics if m.get("baseline_any") is not None]
    scatter_base_all_n = [m.get("n_donors") for m in cluster_metrics if m.get("baseline_mi_all") is not None]
    scatter_base_mi_all = [m.get("baseline_mi_all") for m in cluster_metrics if m.get("baseline_mi_all") is not None]
    purity_rec_list = [m.get("class_purity_recursion") for m in cluster_metrics if m.get("class_purity_recursion") is not None]
    purity_base_list = [m.get("class_purity_baseline") for m in cluster_metrics if m.get("class_purity_baseline") is not None]
    posterior_top1_list = [m.get("posterior_top1_mass") for m in cluster_metrics if m.get("posterior_top1_mass") is not None]
    posterior_entropy_list = [m.get("posterior_entropy") for m in cluster_metrics if m.get("posterior_entropy") is not None]
    n_donors_all = [m.get("n_donors") for m in cluster_metrics if m.get("n_donors") is not None]

    bootstrap_payload = {
        "samples": args.bootstrap_samples,
        "max_clusters": args.bootstrap_max_clusters,
        "clusters": [],
    }
    repeat_payload = {
        "samples": args.bootstrap_samples,
        "max_clusters": args.bootstrap_max_clusters,
        "clusters": [],
    }

    bootstrap_indices = None
    if (args.bootstrap_samples > 0 and clusters) and (args.bootstrap_samples > 0):
        rng = np.random.default_rng(args.seed)
        bcount = min(len(clusters), args.bootstrap_max_clusters)
        bootstrap_indices = rng.choice(len(clusters), size=bcount, replace=False)

    if args.bootstrap_samples > 0 and clusters:
        b_indices = bootstrap_indices
        if b_indices is None:
            rng = np.random.default_rng(args.seed)
            bcount = min(len(clusters), args.bootstrap_max_clusters)
            b_indices = rng.choice(len(clusters), size=bcount, replace=False)
        for i in tqdm(b_indices, desc="bootstrap", unit="cluster", disable=args.no_progress):
            cluster = clusters[int(i)]
            idx = cluster["indices"]
            if idx.size == 0:
                continue
            full_solutions = find_solutions_from_indices(
                idx,
                x,
                min_chi_squared=args.min_chi_squared,
                counts_a=counts_a,
                counts_total=counts_total,
                search_mode=args.search_mode,
                beam_width=args.beam_width,
                mc_num_rollouts=args.mc_num_rollouts,
                mc_top_k=args.mc_top_k,
                mc_temperature=args.mc_temperature,
                max_solutions=max_solutions,
                max_depth=max_depth,
                max_branch=max_branch,
                max_runtime_sec=max_runtime_sec,
            )
            best_idx, _, _ = _best_solution(full_solutions, x_bool)
            if best_idx is None:
                continue
            ref_class = _top_class(best_idx, allele_class)
            ref_gene = _top_gene(best_idx, allele_gene)
            match_class = 0
            match_gene = 0
            for _ in range(args.bootstrap_samples):
                boot_idx = rng.choice(idx, size=idx.size, replace=True)
                boot_solutions = find_solutions_from_indices(
                    boot_idx,
                    x,
                    min_chi_squared=args.min_chi_squared,
                    counts_a=counts_a,
                    counts_total=counts_total,
                    search_mode=args.search_mode,
                    beam_width=args.beam_width,
                    mc_num_rollouts=args.mc_num_rollouts,
                    mc_top_k=args.mc_top_k,
                    mc_temperature=args.mc_temperature,
                    max_solutions=max_solutions,
                    max_depth=max_depth,
                    max_branch=max_branch,
                    max_runtime_sec=max_runtime_sec,
                )
                boot_best, _, _ = _best_solution(boot_solutions, x_bool)
                if boot_best is None:
                    continue
                if _top_class(boot_best, allele_class) == ref_class:
                    match_class += 1
                if _top_gene(boot_best, allele_gene) == ref_gene:
                    match_gene += 1
            denom = float(args.bootstrap_samples) if args.bootstrap_samples else 1.0
            bootstrap_payload["clusters"].append(
                {
                    "cid": cluster["cid"],
                    "n_donors": int(idx.size),
                    "top_class_stability": match_class / denom,
                    "top_gene_stability": match_gene / denom,
                }
            )

    if args.bootstrap_repeat and args.bootstrap_samples > 0 and clusters:
        b_indices = bootstrap_indices
        if b_indices is None:
            rng = np.random.default_rng(args.seed)
            bcount = min(len(clusters), args.bootstrap_max_clusters)
            b_indices = rng.choice(len(clusters), size=bcount, replace=False)
        deterministic = args.search_mode in ("dfs", "beam")
        for i in tqdm(b_indices, desc="repeat", unit="cluster", disable=args.no_progress):
            cluster = clusters[int(i)]
            idx = cluster["indices"]
            if idx.size == 0:
                continue
            ref_solutions = find_solutions_from_indices(
                idx,
                x,
                min_chi_squared=args.min_chi_squared,
                counts_a=counts_a,
                counts_total=counts_total,
                search_mode=args.search_mode,
                beam_width=args.beam_width,
                mc_num_rollouts=args.mc_num_rollouts,
                mc_top_k=args.mc_top_k,
                mc_temperature=args.mc_temperature,
                max_solutions=max_solutions,
                max_depth=max_depth,
                max_branch=max_branch,
                max_runtime_sec=max_runtime_sec,
            )
            ref_best, _, _ = _best_solution(ref_solutions, x_bool)
            if ref_best is None:
                continue
            ref_class = _top_class(ref_best, allele_class)
            ref_gene = _top_gene(ref_best, allele_gene)
            match_class = 0
            match_gene = 0
            if deterministic:
                match_class = args.bootstrap_samples
                match_gene = args.bootstrap_samples
            else:
                for _ in range(args.bootstrap_samples):
                    repeat_solutions = find_solutions_from_indices(
                        idx,
                        x,
                        min_chi_squared=args.min_chi_squared,
                        counts_a=counts_a,
                        counts_total=counts_total,
                        search_mode=args.search_mode,
                        beam_width=args.beam_width,
                        mc_num_rollouts=args.mc_num_rollouts,
                        mc_top_k=args.mc_top_k,
                        mc_temperature=args.mc_temperature,
                        max_solutions=max_solutions,
                        max_depth=max_depth,
                        max_branch=max_branch,
                        max_runtime_sec=max_runtime_sec,
                    )
                    repeat_best, _, _ = _best_solution(repeat_solutions, x_bool)
                    if repeat_best is None:
                        continue
                    if _top_class(repeat_best, allele_class) == ref_class:
                        match_class += 1
                    if _top_gene(repeat_best, allele_gene) == ref_gene:
                        match_gene += 1
            denom = float(args.bootstrap_samples) if args.bootstrap_samples else 1.0
            repeat_payload["clusters"].append(
                {
                    "cid": cluster["cid"],
                    "n_donors": int(idx.size),
                    "top_class_stability": match_class / denom,
                    "top_gene_stability": match_gene / denom,
                }
            )

    out_payload = {
        "export_root": str(export_root),
        "public_json": str(public_json) if public_json is not None else "",
        "synthetic_json": str(Path(args.synthetic_json).resolve()) if args.synthetic_json else "",
        "synthetic_dir": str(Path(args.synthetic_dir).resolve()) if args.synthetic_dir else "",
        "synthetic_meta": synthetic_meta,
        "figures_dir": str(figures_dir),
        "donor_matrix": str(donor_matrix_path),
        "donor_keys": str(donor_keys_path),
        "min_donors": args.min_donors,
        "min_chi_squared": args.min_chi_squared,
        "posterior": args.posterior,
        "max_clusters": args.max_clusters,
        "seed": args.seed,
        "threads": args.threads,
        "skip_chi_sweep": args.skip_chi_sweep,
        "max_solutions": max_solutions,
        "max_depth": max_depth,
        "max_branch": max_branch,
        "max_runtime_sec": max_runtime_sec,
        "n_donors_total": counts_total,
        "n_alleles": num_alleles,
        "clusters_total": total_clusters,
        "clusters_eligible": eligible_clusters,
        "clusters_used": len(clusters),
        "missing_donor_refs": missing_refs,
        "cluster_metrics": cluster_metrics,
        "solved_fraction_vs_min_chi": (
            [
                {
                    "min_chi_squared": chi_grid[i],
                    "solved_fraction": solved_frac[i],
                    "runtime_sec": runtime_sec[i],
                    "truncated_clusters": truncated_counts[i],
                }
                for i in range(len(chi_grid))
            ]
            if chi_grid
            else []
        ),
        "bootstrap": bootstrap_payload,
        "bootstrap_repeat": repeat_payload,
    }
    search_params = {"search_mode": args.search_mode}
    if args.search_mode == "beam":
        search_params["beam_width"] = args.beam_width
        search_params["beam_score"] = args.beam_score
    elif args.search_mode == "mc":
        search_params["mc_num_rollouts"] = args.mc_num_rollouts
        search_params["mc_top_k"] = args.mc_top_k
        search_params["mc_temperature"] = args.mc_temperature
    out_payload.update(search_params)

    _save_json(figures_dir / "recursion_diagnostics.json", out_payload)

    if chi_grid:
        _plot_solved_fraction(
            figures_dir / "solved_fraction_vs_min_chi.png",
            chi_grid,
            solved_frac,
            runtime_sec,
        )
    _plot_n_donors_hist(
        figures_dir / "n_donors_hist.png",
        n_donors_all,
    )
    _plot_num_solutions_hist(
        figures_dir / "num_solutions_hist.png",
        [v for v in num_solutions_list if v is not None],
    )
    _plot_truncation_summary(
        figures_dir / "truncation_summary.png",
        cluster_metrics,
    )
    _plot_status_wide(
        figures_dir / "status_long.png",
        cluster_metrics,
    )
    if use_synthetic:
        _plot_synthetic_per_tcr(
            figures_dir / "synthetic_per_tcr.png",
            cluster_metrics,
        )
        _plot_synthetic_by_size(
            figures_dir / "synthetic_by_size.png",
            cluster_metrics,
        )
    n_with = sum(1 for v in num_solutions_list if v and v > 0)
    n_without = len(num_solutions_list) - n_with
    _plot_histograms(
        figures_dir / "solutions_histograms.png",
        [v for v in num_solutions_list if v and v > 0],
        min_size_list,
        n_with=n_with,
        n_without=n_without,
    )
    if num_solutions_list:
        xs = list(range(1, len(num_solutions_list) + 1))
        min_sizes_all = [c.get("min_solution_size") for c in cluster_metrics]
        x_min = [xs[i] for i, v in enumerate(min_sizes_all) if v is not None]
        y_min = [v for v in min_sizes_all if v is not None]
        x_sol = [xs[i] for i, v in enumerate(num_solutions_list) if v and v > 0]
        y_sol = [v for v in num_solutions_list if v and v > 0]
        _plot_long_template(
            figures_dir / "solutions_long.png",
            x_left=x_min,
            y_left=y_min,
            x_right=x_sol,
            y_right=y_sol,
            left_label="min |z|",
            right_label="#solutions",
            right_log=True,
            subtitle=f"Clusters with ≥1 solution: {n_with} | without solution: {n_without}",
        )
    _plot_specificity_scatter_pair(
        figures_dir / "specificity_scatter.png",
        rec_any_n=scatter_rec_n,
        rec_any_y=scatter_rec_mi,
        base_any_n=scatter_base_n,
        base_any_y=scatter_base_mi,
        rec_all_n=scatter_rec_all_n,
        rec_all_y=scatter_rec_mi_all,
        base_all_n=scatter_base_all_n,
        base_all_y=scatter_base_mi_all,
    )
    if num_solutions_list and n_donors_all:
        xs = list(range(1, len(cluster_metrics) + 1))
        _plot_long_template(
            figures_dir / "n_donors_solutions_long.png",
            x_left=xs,
            y_left=n_donors_all,
            x_right=xs,
            y_right=[v if v is not None else 0 for v in num_solutions_list],
            left_label="N_i (# donors with TCR)",
            right_label="#solutions",
            right_log=True,
        )
    _plot_class_purity(
        figures_dir / "class_purity_hist.png",
        purity_rec_list,
        purity_base_list,
    )
    if purity_rec_list or purity_base_list:
        xs = list(range(1, len(cluster_metrics) + 1))
        purity_rec_all = [c.get("class_purity_recursion") for c in cluster_metrics]
        purity_base_all = [c.get("class_purity_baseline") for c in cluster_metrics]
        x_rec = [xs[i] for i, v in enumerate(purity_rec_all) if v is not None]
        y_rec = [v for v in purity_rec_all if v is not None]
        x_base = [xs[i] for i, v in enumerate(purity_base_all) if v is not None]
        y_base = [v for v in purity_base_all if v is not None]
        _plot_long_template(
            figures_dir / "class_purity_long.png",
            x_left=x_rec,
            y_left=y_rec,
            x_right=x_base,
            y_right=y_base,
            left_label="Class purity (recursion)",
            right_label="Class purity (baseline)",
            right_log=False,
        )

    if bootstrap_payload["clusters"]:
        xs = list(range(1, len(bootstrap_payload["clusters"]) + 1))
        boot_class = [c["top_class_stability"] for c in bootstrap_payload["clusters"]]
        boot_gene = [c["top_gene_stability"] for c in bootstrap_payload["clusters"]]
        _plot_long_template(
            figures_dir / "bootstrap_stability_long.png",
            x_left=xs,
            y_left=boot_class,
            x_right=xs,
            y_right=boot_gene,
            left_label="Top class stability",
            right_label="Top gene stability",
            right_log=False,
        )
    if repeat_payload["clusters"]:
        xs = list(range(1, len(repeat_payload["clusters"]) + 1))
        rep_class = [c["top_class_stability"] for c in repeat_payload["clusters"]]
        rep_gene = [c["top_gene_stability"] for c in repeat_payload["clusters"]]
        _plot_long_template(
            figures_dir / "bootstrap_repeat_long.png",
            x_left=xs,
            y_left=rep_class,
            x_right=xs,
            y_right=rep_gene,
            left_label="Repeat top class stability",
            right_label="Repeat top gene stability",
            right_log=False,
        )

    logger.info("Wrote diagnostics JSON and plots under %s", figures_dir)


if __name__ == "__main__":
    main()
