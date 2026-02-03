#!/usr/bin/env python3
"""
tcrdist_rosati_pipeline.py

Rosati dataset (TRB miXcr-like clones) -> tcrdist loops.

Input (per file): miXcr-like TSV with at least
    - aaSeqCDR3          (beta CDR3 aa; we fall back to aaSeqImputedCDR3 if needed)
    - bestVHit           (IMGT V gene, e.g. TRBV24-1*00)
    - bestJHit           (IMGT J gene, e.g. TRBJ2-7*00)
    - readCount          (clone count)
"""

# Preload conda's libstdc++.so.6 so SciPy/pwseqdist find GLIBCXX symbols
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
from typing import Any, Dict, List, Union
import re

import pandas as pd
from tqdm.auto import tqdm

from tcrtyper.dataset_processing.tcrdist_loops_core import run_tcrdist_loops
from tcrtyper.config import config
from tcrtyper.dataset_processing.path_utils import processed_dataset_root

logger = logging.getLogger(__name__)


def discover_rosati_files(tsv_dir: Union[Path, str]) -> List[Path]:
    """
    Return all *_TRB.miXcr_like.tsv files in the given TSV directory.
    """
    tsv_dir = Path(tsv_dir)
    if not tsv_dir.is_dir():
        return []
    return sorted(p for p in tsv_dir.glob("*_TRB.miXcr_like.tsv") if p.is_file())


def _clean_gene(g: Any) -> str:
    """
    Normalize miXcr bestVHit / bestJHit to tcrdist input

    - strip whitespace
    - keep only first gene if comma-separated
    - coerce '*00' allele to '*01'
    """
    if g is None:
        return ""
    s = str(g).strip()
    if not s:
        return ""
    if "," in s:
        s = s.split(",", 1)[0].strip()
    s = re.sub(r"\*00$", "*01", s)
    return s


def run_one_sample(
    tsv_path: Path,
    processed_dir: Path,
    tcrdist_debug: bool = False,
) -> Dict[str, Any]:
    """
    Process a single Rosati miXcr-like tsv
        cloneId, readCount, readFraction, targetSequences, ...,

        1. Read TSV.
        2. Build standardized TCRdist input DataFrame:
               cdr3_b_aa, v_b_gene, j_b_gene, count
           (count = readCount per clone).
        3. Call run_tcrdist_loops(...) to infer loops.
        4. Write processed TSV and return summary dict.
    """
    tsv_path = Path(tsv_path)
    name = tsv_path.name
    base = name[:-4] if name.endswith(".tsv") else name
    sample_name = base.replace("_TRB.miXcr_like", "_TRB")

    dataset_name = tsv_path.parent.parent.name if tsv_path.parent.parent.name else ""
    logger.info(
        "[Rosati][%s] processing %s -> sample %s",
        dataset_name,
        tsv_path.name,
        sample_name,
    )

    df_raw = pd.read_csv(tsv_path, sep="\t", dtype=str)

    required_cols = ["bestVHit", "bestJHit"]
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"{tsv_path}: missing required columns: {missing}")

    if "aaSeqCDR3" in df_raw.columns:
        cdr3 = df_raw["aaSeqCDR3"].copy()
        if "aaSeqImputedCDR3" in df_raw.columns:
            cdr3 = cdr3.where(
                cdr3.notna() & (cdr3.astype(str) != ""),
                df_raw["aaSeqImputedCDR3"],
            )
    elif "aaSeqImputedCDR3" in df_raw.columns:
        cdr3 = df_raw["aaSeqImputedCDR3"].copy()
    else:
        raise ValueError(
            f"{tsv_path}: missing 'aaSeqCDR3' / 'aaSeqImputedCDR3' columns"
        )

    v_gene = df_raw["bestVHit"].astype(str).apply(_clean_gene)
    j_gene = df_raw["bestJHit"].astype(str).apply(_clean_gene)

    # clone count
    if "readCount" in df_raw.columns:
        read_count = pd.to_numeric(df_raw["readCount"], errors="coerce").fillna(0)
        read_count = read_count.astype(int)
    else:
        read_count = pd.Series(1, index=df_raw.index, dtype=int)

    cdr3 = cdr3.astype(str)

    mask = (
        cdr3.notna()
        & (cdr3 != "")
        & v_gene.notna()
        & (v_gene != "")
        & j_gene.notna()
        & (j_gene != "")
        & (read_count > 0)
    )

    n_raw = int(len(df_raw))
    n_after_basic_filter = int(mask.sum())

    cell_df = pd.DataFrame(
        {
            "cdr3_b_aa": cdr3[mask].astype(str),
            "v_b_gene": v_gene[mask].astype(str),
            "j_b_gene": j_gene[mask].astype(str),
            "count": read_count[mask].astype(int),
        }
    )

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_out = processed_dir / f"{sample_name}.tsv"

    if cell_df.empty:
        logger.warning(
            "[Rosati] %s: no valid rows after basic filter", tsv_path.name
        )
        cell_df.to_csv(processed_out, sep="\t", index=False)
        return {
            "input": str(tsv_path),
            "processed_out": str(processed_out),
            "n_raw": n_raw,
            "n_after_basic_filter": n_after_basic_filter,
            "n_with_loops": 0,
        }

    loops_df = run_tcrdist_loops(
        cell_df,
        organism="human",
        chain="beta",
        debug=tcrdist_debug,
    )

    # Identify rows where loops couldn't be inferred (e.g. unknown V genes)
    loop_cols = [c for c in ["cdr1_b_aa", "cdr2_b_aa", "pmhc_b_aa"] if c in loops_df.columns]
    if loop_cols:
        bad_mask = loops_df[loop_cols].isna().any(axis=1)
        n_bad = int(bad_mask.sum())
        n_total = int(len(loops_df))
        if n_bad > 0:
            bad_v_genes = sorted(loops_df.loc[bad_mask, "v_b_gene"].unique())
            logger.warning(
                "[Rosati] %s: %d/%d rows missing loops; example V genes: %s",
                tsv_path.name,
                n_bad,
                n_total,
                ", ".join(bad_v_genes[:10]),
            )
        loops_df = loops_df.loc[~bad_mask].copy()

    loops_df.to_csv(processed_out, sep="\t", index=False)

    return {
        "input": str(tsv_path),
        "processed_out": str(processed_out),
        "n_raw": n_raw,
        "n_after_basic_filter": n_after_basic_filter,
        "n_with_loops": int(len(loops_df)),
    }


def _run_one_sample_star(args: Any) -> Dict[str, Any]:
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
    dataset_dir: Union[str, Path],
    tsv_subfolder: str,
    tcrdist_debug: bool,
    show_progress: bool,
    n_cores: int,
    log_every: int,
) -> List[Dict[str, Any]]:
    """
    Process all Rosati TRB miXcr-like TSV files in a single dataset folder in parallel.

    dataset_dir/
      <tsv_subfolder>/
      processed/ # created
    """
    dataset_path = Path(dataset_dir)
    tsv_dir = dataset_path / tsv_subfolder
    out_root = processed_dataset_root(dataset_path)
    processed_dir = out_root / config.data.tcrdist_processed_subdir_name
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = discover_rosati_files(tsv_dir)
    reports: List[Dict[str, Any]] = []

    if not files:
        logger.warning("[Rosati] no *_TRB.miXcr_like.tsv files found in %s", tsv_dir)
        return reports

    tasks = [(f, processed_dir, tcrdist_debug) for f in files]
    total = len(tasks)
    log_every = _resolve_log_every(log_every, n_cores, show_progress)
    logger.info("[%s] starting %d sample(s) (n_cores=%s)", tsv_dir.name, total, n_cores)
    if log_every:
        logger.info("[%s] progress 0/%d", tsv_dir.name, total)

    if n_cores <= 1 or len(tasks) == 1:
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


# ------------------- CLI ------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Call TCRdist loops from Rosati dataset\n"
            "Assumes miXcr-like columns: aaSeqCDR3/aaSeqImputedCDR3, bestVHit, "
            "bestJHit, readCount.\n"
            "Input:  *_TRB.miXcr_like.tsv files in dataset-dir/tsv-subfolder.\n"
            "Output: dataset-dir/processed/<run>_TRB.tsv with columns:\n"
            "        cdr3aa, cdr2aa_gapped, cdr1aa_gapped, "
            "cdr2.5aa_gapped, v_b_gene, j_b_gene, count."
        )
    )
    p.add_argument(
        "--dataset-dir",
        required=True,
        help="Main directory of the Rosati dataset (e.g. .../rosati2023).",
    )
    p.add_argument(
        "--tsv-subfolder",
        default=config.data.rosati_tsv_subfolder,
        help=(
            "Relative subfolder under dataset-dir containing miXcr-like TSV files "
            f"(default: '{config.data.rosati_tsv_subfolder}')."
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
    summary_default = out_root / config.data.tcrdist_summary_filename
    summary_path = Path(args.summary) if args.summary is not None else summary_default

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2, ensure_ascii=False)

    logger.info(
        "[Rosati] processed %d file%s in %s",
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
