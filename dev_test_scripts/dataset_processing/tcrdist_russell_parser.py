#!/usr/bin/env python3
"""
tcrdist_russell_pipeline.py

Russell dataset (beta chain) -> tcrdist loops.
    - Reads each Russell TSV in dataset_dir / tsv_subfolder.
    - Builds a minimal TCRdist input DataFrame with:
           cdr3_b_aa, v_b_gene, j_b_gene, count
       (count is set to 1 per row).
    - Calls tcrdist_loops_core.run_tcrdist_loops to infer:
           cdr1_b_aa, cdr2_b_aa, pmhc_b_aa (CDR2.5)
    - Writes per-file TSVs into dataset_dir / processed with columns:
           cdr3aa, cdr2aa_gapped, cdr1aa_gapped, cdr2.5aa_gapped,
           v_b_gene, j_b_gene, count
"""

# FIXME preload conda's libstdc++.so.6
import os
import glob
import ctypes


def _preload_conda_libstdcxx():
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
from typing import List, Dict, Any

import pandas as pd
from tqdm.auto import tqdm

from tcrtyper.dataset_processing.tcrdist_loops_core import run_tcrdist_loops
from tcrtyper.config import config
from tcrtyper.dataset_processing.path_utils import processed_dataset_root

logger = logging.getLogger(__name__)


def discover_russell_files(tsv_dir: Path) -> List[Path]:
    if not tsv_dir.is_dir():
        return []
    return sorted(p for p in tsv_dir.glob("*.tsv") if p.is_file())


def run_one_sample(
    tsv_path: Path,
    processed_dir: Path,
    tcrdist_debug: bool = False,
) -> Dict[str, Any]:
    """
    Process a single Russell TSV 
        1. Read TSV.
        2. Build standardized TCRdist input DataFrame:
               cdr3_b_aa, v_b_gene, j_b_gene, count
           (count = 1 per row).
        3. Call run_tcrdist_loops(...) to infer loops.
        4. Write processed TSV and return summary dict.
    """
    sample_name = tsv_path.stem
    dataset_name = tsv_path.parent.parent.name
    logger.info("[Russell][%s] processing %s", dataset_name, tsv_path.name)

    df_raw = pd.read_csv(tsv_path, sep="\t", dtype=str)

    required = ["cdr3", "v_gene", "j_gene"]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"{tsv_path}: missing required columns: {missing}")

    cdr3 = df_raw["cdr3"].astype(str)
    v_gene = df_raw["v_gene"].astype(str)
    j_gene = df_raw["j_gene"].astype(str)

    mask = (
        cdr3.notna()
        & (cdr3.astype(str) != "")
        & v_gene.notna()
        & (v_gene.astype(str) != "")
        & j_gene.notna()
        & (j_gene.astype(str) != "")
    )

    n_raw = int(len(df_raw))
    n_after_basic_filter = int(mask.sum())

    cell_df = pd.DataFrame(
        {
            "cdr3_b_aa": cdr3[mask].astype(str),
            "v_b_gene": v_gene[mask].astype(str),
            "j_b_gene": j_gene[mask].astype(str),
            # Russell dataset doesn't carry explicit clone counts here, use 1 per row.
            "count": 1,
        }
    )

    # Infer CDR1/CDR2/CDR2.5 from V gene via TCRrep
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
        "input": str(tsv_path),
        "processed_out": str(processed_out),
        "n_raw": n_raw,
        "n_after_basic_filter": n_after_basic_filter,
        "n_with_loops": int(len(loops_df)),
    }


def _run_one_sample_star(args):
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
    tsv_subfolder: str,
    tcrdist_debug: bool,
    show_progress: bool,
    n_cores: int,
    log_every: int,
) -> List[Dict[str, Any]]:
    """
    Process all Russell TSV files in a single dataset folder in parallel.

    dataset_dir/
      <tsv_subfolder>/    
      processed/ # creates
    """
    dataset_path = Path(dataset_dir)
    tsv_dir = dataset_path / tsv_subfolder
    out_root = processed_dataset_root(dataset_path)
    processed_dir = out_root / config.data.tcrdist_processed_subdir_name
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = discover_russell_files(tsv_dir)
    reports: List[Dict[str, Any]] = []

    if not files:
        logger.warning("[Russell] no .tsv files found in %s", tsv_dir)
        return reports

    tasks = [(f, processed_dir, tcrdist_debug) for f in files]
    total = len(tasks)
    log_every = _resolve_log_every(log_every, n_cores, show_progress)
    logger.info("[%s] starting %d sample(s) (n_cores=%s)", tsv_dir.name, total, n_cores)
    if log_every:
        logger.info("[%s] progress 0/%d", tsv_dir.name, total)

    if n_cores is None or n_cores <= 1 or len(tasks) == 1:
        inner = tqdm(
            tasks,
            desc=tsv_dir.name,
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
                logger.info("[%s] progress %d/%d", tsv_dir.name, idx, total)
    else:
        with Pool(processes=n_cores) as pool:
            inner = tqdm(
                pool.imap_unordered(_run_one_sample_star, tasks),
                total=len(tasks),
                desc=tsv_dir.name,
                unit="file",
                disable=not show_progress,
                file=sys.stderr,
                mininterval=10,
                leave=True,
            )
            for idx, rep in enumerate(inner, 1):
                reports.append(rep)
                if log_every and (idx % log_every == 0 or idx == total):
                    logger.info("[%s] progress %d/%d", tsv_dir.name, idx, total)

    return reports


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Call CDR3 loops via tcrdist for the Russell TCR beta dataset.\n\n"
            "requires IMGT format.\n"
            "Input:  .tsv files in dataset-dir/tsv-subfolder.\n"
            "Output: dataset-dir/processed/<file>.tsv with columns:\n"
            "        cdr3aa, cdr2aa_gapped, cdr1aa_gapped, "
            "cdr2.5aa_gapped, v_b_gene, j_b_gene, count."
        )
    )
    p.add_argument(
        "--dataset-dir",
        required=True,
        help="Main directory of the Russell dataset.",
    )
    p.add_argument(
        "--tsv-subfolder",
        default=config.data.russell_tsv_subfolder,
        help=(
            "Relative subfolder under dataset-dir containing TSV files "
            f"(default: '{config.data.russell_tsv_subfolder}')."
        ),
    )
    p.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging and verbose tcrdist internals.",
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

    reports = process_dataset(
        dataset_dir=args.dataset_dir,
        tsv_subfolder=args.tsv_subfolder,
        tcrdist_debug=args.debug,
        show_progress=not args.no_progress,
        n_cores=args.n_cores,
        log_every=args.log_every,
    )

    dataset_path = Path(args.dataset_dir)
    out_root = processed_dataset_root(dataset_path)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(args.summary)
        if args.summary is not None
        else out_root / config.data.tcrdist_summary_filename
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2, ensure_ascii=False)

    logger.info(
        "[Russell] processed %d file%s in %s",
        len(reports),
        "" if len(reports) == 1 else "s",
        dataset_path,
    )
    for r in reports:
        logger.info(
            "  %s: raw=%d after_basic_filter=%d with_loops=%d -> %s",
            Path(r["input"]).name,
            r["n_raw"],
            r["n_after_basic_filter"],
            r["n_with_loops"],
            r["processed_out"],
        )

    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
