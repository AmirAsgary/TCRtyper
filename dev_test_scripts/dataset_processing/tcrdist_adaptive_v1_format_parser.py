#!/usr/bin/env python3
"""
tcrdist_adaptive_v1_format_parser.py

Adaptive ImmunoSEQ v1 (beta chain) -> tcrdist loops.

    Each TSV is expected to have:
        - rearrangement
        - amino_acid
        - frame_type
        - rearrangement_type
        - templates
        - productive_frequency
        - bio_identity
        - v_resolved / j_resolved (or v_gene/j_gene; see tcrdist importer)
"""

# FIXME preload conda's libstdc++.so.6
import os
import glob
import ctypes


def _preload_conda_libstdcxx() -> None:
    try:
        conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
        conda_lib = os.path.join(conda_prefix, "lib") if conda_prefix else ""
        if conda_lib and os.path.isdir(conda_lib):
            candidates = (
                glob.glob(os.path.join(conda_lib, "libstdc++.so.6*"))
                + glob.glob(os.path.join(conda_lib, "libstdc++.so*"))
            )
            for p in candidates:
                try:
                    ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                    os.environ["LD_LIBRARY_PATH"] = (
                        f"{conda_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
                    )
                    break
                except OSError:
                    continue
    except Exception:
        pass

_preload_conda_libstdcxx()

import argparse
import json
import logging
import sys
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, List, Any

import pandas as pd
from tqdm.auto import tqdm

from tcrdist.adpt_funcs import import_adaptive_file
from tcrtyper.dataset_processing.tcrdist_loops_core import run_tcrdist_loops
from tcrtyper.config import config
from tcrtyper.dataset_processing.path_utils import processed_dataset_root

logger = logging.getLogger(__name__)


def discover_export_files(dataset_dir: Path) -> List[Path]:
    """
    Inside a dataset directory, Adaptive v1 exports are under 'export/'.
    """
    exp = dataset_dir / config.data.export_subdir_name
    if not exp.is_dir():
        return []
    return sorted(exp.glob("*.tsv"))


def run_one_sample(
    export_tsv: Path,
    processed_dir: Path,
    tcrdist_debug: bool = False,
) -> Dict[str, Any]:
    """
    Process a single Adaptive v1 TSV file.

    Steps:
        1. Read raw TSV (for n_raw).
        2. Call import_adaptive_file(...) directly on this TSV.
        3. Build standardized cell_df with cdr3_b_aa, v_b_gene, j_b_gene, count.
        4. Call run_tcrdist_loops(...) to infer loops.
        5. Write loops TSV to <dataset>/processed/<sample>.tsv.
    """
    sample_name = export_tsv.stem
    dataset_name = export_tsv.parent.parent.name  # parent of 'export'

    logger.info("[Adaptive v1][%s] processing %s", dataset_name, export_tsv.name)

    df_raw = pd.read_csv(export_tsv, sep="\t", dtype=str)
    n_raw = int(len(df_raw))

    df_adapt = import_adaptive_file(
        adaptive_filename=str(export_tsv),
        organism="human",
        chain="beta",
        version_year=2020,
        sep="\t",
        log=tcrdist_debug,
        count="productive_frequency",
        return_valid_cdr3_only=False,
    )

    n_after_adaptive_import = int(len(df_adapt))

    templates = pd.to_numeric(
        df_adapt.get("templates", 1), errors="coerce"
    ).fillna(0).astype(int)

    cell_df = pd.DataFrame(
        {
            "cdr3_b_aa": df_adapt["cdr3_b_aa"].astype(str),
            "v_b_gene": df_adapt["v_b_gene"].astype(str),
            "j_b_gene": df_adapt["j_b_gene"].astype(str),
            "count": templates,
        }
    )

    loops_df = run_tcrdist_loops(
        cell_df,
        organism="human",
        chain="beta",
        debug=tcrdist_debug,
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_out = processed_dir / f"{sample_name}.tsv"
    loops_df.to_csv(processed_out, sep="\t", index=False)

    return {
        "input": str(export_tsv),
        "processed_out": str(processed_out),
        "n_raw": n_raw,
        "n_after_adaptive_import": n_after_adaptive_import,
        "n_with_loops": int(len(loops_df)),
    }


def _run_one_sample_star(args: Any) -> Dict[str, Any]:
    """Helper for multiprocessing.Pool: unpack tuple into run_one_sample."""
    return run_one_sample(*args)


def _resolve_log_every(log_every: int, n_cores: int, show_progress: bool) -> int:
    if log_every is None:
        log_every = 0
    if log_every > 0:
        return log_every
    if n_cores and n_cores > 1:
        return 200
    if not show_progress:
        return 200
    if not sys.stderr.isatty():
        return 200
    return 0


def process_dataset(
    dataset_dir: str,
    tcrdist_debug: bool,
    show_progress: bool,
    n_cores: int,
    one_per_dataset: bool,
    log_every: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process a single Adaptive v1 dataset:
        dataset_dir/
          export/
          processed/ # created
    """
    dataset_path = Path(dataset_dir)
    dataset_name = dataset_path.name
    out_root = processed_dataset_root(dataset_path)
    processed_dir = out_root / config.data.tcrdist_processed_subdir_name
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = discover_export_files(dataset_path)
    reports: List[Dict[str, Any]] = []

    if not files:
        logger.warning(
            "[Adaptive v1] no .tsv files found in %s",
            dataset_path / config.data.export_subdir_name,
        )
        return {dataset_name: reports}

    if one_per_dataset:
        rep = run_one_sample(files[0], processed_dir, tcrdist_debug=tcrdist_debug)
        reports.append(rep)
        return {dataset_name: reports}

    tasks = [(f, processed_dir, tcrdist_debug) for f in files]
    total = len(tasks)
    log_every = _resolve_log_every(log_every, n_cores, show_progress)
    logger.info("[%s] starting %d sample(s) (n_cores=%s)", dataset_name, total, n_cores)
    if log_every:
        logger.info("[%s] progress 0/%d", dataset_name, total)

    if n_cores is None or n_cores <= 1 or len(tasks) == 1:
        inner = tqdm(
            tasks,
            desc=dataset_name,
            unit="file",
            disable=not show_progress,
            file=sys.stderr,
            mininterval=10,
            leave=True,
        )
        for idx, (f, pdir, dbg) in enumerate(inner, 1):
            rep = run_one_sample(f, pdir, tcrdist_debug=dbg)
            reports.append(rep)
            if log_every and (idx % log_every == 0 or idx == total):
                logger.info("[%s] progress %d/%d", dataset_name, idx, total)
    else:
        with Pool(processes=n_cores) as pool:
            inner = tqdm(
                pool.imap_unordered(_run_one_sample_star, tasks),
                total=len(tasks),
                desc=dataset_name,
                unit="file",
                disable=not show_progress,
                file=sys.stderr,
                mininterval=10,
                leave=True,
            )
            for idx, rep in enumerate(inner, 1):
                reports.append(rep)
                if log_every and (idx % log_every == 0 or idx == total):
                    logger.info("[%s] progress %d/%d", dataset_name, idx, total)

    return {dataset_name: reports}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Call TCRdist loops from Adaptive v1 export datasets.\n"
            f"Reads Adaptive v1 exports under <dataset-dir>/{config.data.export_subdir_name}, "
            "uses import_adaptive_file to map to IMGT, then uses TCRrep (via "
            "tcrdist_loops_core) to infer CDR1/CDR2/CDR2.5.\n"
            "Output per sample: cdr3aa, cdr2aa_gapped, cdr1aa_gapped, "
            "cdr2.5aa_gapped, v_b_gene, j_b_gene, count."
        )
    )
    p.add_argument(
        "--dataset-dir",
        required=True,
        help=(
            "Path to a single Adaptive v1 dataset directory. "
            f"TSVs are expected under <dataset-dir>/{config.data.export_subdir_name}."
        ),
    )
    p.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging and verbose tcrdist internals.",
    )
    p.add_argument(
        "--one-per-dataset",
        action="store_true",
        default=False,
        help="Process only first sample in the dataset (debug / quick run).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable tqdm progress bars.",
    )
    p.add_argument(
        "--summary",
        default=None,
        help=(
            "Path to write run summary JSON "
            f"(default: <dataset-dir>/{config.data.tcrdist_summary_filename})."
        ),
    )
    p.add_argument(
        "--n-cores",
        type=int,
        default=8,
        help="Number of parallel worker processes (default: 8).",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=0,
        help=(
            "Log progress every N samples (0 disables). "
            "If 0, defaults to 200 for non-TTY, --no-progress, or n-cores>1."
        ),
    )
    return p.parse_args()


def _configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    _configure_logging(args.debug)

    reports_by_dataset = process_dataset(
        dataset_dir=args.dataset_dir,
        tcrdist_debug=args.debug,
        show_progress=not args.no_progress,
        n_cores=args.n_cores,
        one_per_dataset=args.one_per_dataset,
        log_every=args.log_every,
    )

    dataset_path = Path(args.dataset_dir)
    out_root = processed_dataset_root(dataset_path)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_default = out_root / config.data.tcrdist_summary_filename
    summary_path = Path(args.summary) if args.summary is not None else summary_default

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(reports_by_dataset, fh, indent=2, ensure_ascii=False)

    for ds, reps in reports_by_dataset.items():
        logger.info(
            "[Adaptive v1][%s] processed %d sample%s; one_per_dataset=%s",
            ds,
            len(reps),
            "" if len(reps) == 1 else "s",
            args.one_per_dataset,
        )
        for r in reps:
            logger.info(
                "  %s: raw=%d after_adaptive=%d with_loops=%d -> %s",
                Path(r["input"]).name,
                r["n_raw"],
                r["n_after_adaptive_import"],
                r["n_with_loops"],
                r["processed_out"],
            )

    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
