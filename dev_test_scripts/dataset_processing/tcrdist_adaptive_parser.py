#!/usr/bin/env python3
"""
tcrdist_adaptive_pipeline.py

Adaptive ImmunoSEQ (beta chain) -> tcrdist loops.
    <dataset-dir>/
      export/        *.tsv  (Adaptive ImmunoSEQ exports)
      processed/     *.tsv  (tcrdist loops; created)
      temp/          *.adaptive2020.tsv (intermediate; created)

  - read Adaptive export TSV.
  - construct a minimal TSV with columns:
         rearrangement, amino_acid, bio_identity,
         templates, productive_frequency,
         frame_type, rearrangement_type, subject
  - use tcrdist.adpt_funcs.import_adaptive_file to:
         - convert Adaptive gene tokens to IMGT v_b_gene / j_b_gene
         - standardize cdr3_b_aa
  - pass standardized DataFrame (cdr3_b_aa, v_b_gene, j_b_gene, count)
     into tcrdist_loops_core.run_tcrdist_loops to infer:
         - cdr1_b_aa, cdr2_b_aa, pmhc_b_aa (CDR2.5)
  - write per-sample TSVs with:
         cdr3aa, cdr2aa_gapped, cdr1aa_gapped, cdr2.5aa_gapped,
         v_b_gene, j_b_gene, count
"""

# FIXME preload conda's libstdc++.so.6, SciPy/pwseqdist issue with GLIBCXX symbols
import os
import glob
import ctypes
import sys


def _preload_conda_libstdcxx():
    """
    Work around GLIBCXX mismatches by explicitly loading libstdc++
    from the active conda environment before importing SciPy/pwseqdist.
    """
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
import re
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from tcrdist.adpt_funcs import import_adaptive_file
from tcrtyper.dataset_processing.tcrdist_loops_core import run_tcrdist_loops
from tcrtyper.config import config
from tcrtyper.dataset_processing.path_utils import processed_dataset_root

logger = logging.getLogger(__name__)

def strip_allele(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return re.sub(r"\*[A-Za-z0-9]+$", "", name)


def _pad2(n: str) -> str:
    try:
        return f"{int(n):02d}"
    except Exception:
        return n


def to_adaptive_bio_token(s: str) -> str:
    """Replace gene names to Adaptive format"""
    s = (s or "").strip()
    s = (
        s.replace("TRBBV", "TCRBV")
        .replace("TRBV", "TCRBV")
        .replace("TRAV", "TCRAV")
        .replace("TRBJ", "TCRBJ")
    )
    s = strip_allele(s)

    m = re.match(r"^(TCRB[VDJ])(\d+)-(\d+)$", s)
    if m:
        return f"{m.group(1)}{_pad2(m.group(2))}-{_pad2(m.group(3))}"

    m = re.match(r"^(TCRB[VDJ])(\d+)$", s)
    if m:
        return f"{m.group(1)}{int(m.group(2))}"

    return s




def build_adaptive_like_df(df_raw: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    """
    Convert ImmunoSEQ export to a minimal tsv for import_adaptive_file()
        - cdr3_b_aa
        - v_b_gene (IMGT)
        - j_b_gene (IMGT)
        - templates / productive_frequency
    """
    required = ["nucleotide", "aminoAcid", "vMaxResolved", "jMaxResolved"]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Input TSV missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "rearrangement": df_raw["nucleotide"].astype(str),
            "amino_acid": df_raw["aminoAcid"].astype(str),
        }
    )

    # Templates / read counts
    if "count (templates/reads)" in df_raw.columns:
        templates = pd.to_numeric(
            df_raw["count (templates/reads)"], errors="coerce"
        ).fillna(0)
    elif "templates" in df_raw.columns:
        templates = pd.to_numeric(df_raw["templates"], errors="coerce").fillna(0)
    else:
        templates = 1
    out["templates"] = templates.astype(int)

    # Productive frequency
    if "frequencyCount (%)" in df_raw.columns:
        fr = pd.to_numeric(df_raw["frequencyCount (%)"], errors="coerce") / 100.0
        out["productive_frequency"] = fr.fillna(0.0)
    elif "productive_frequency" in df_raw.columns:
        out["productive_frequency"] = pd.to_numeric(
            df_raw["productive_frequency"], errors="coerce"
        ).fillna(0.0)
    else:
        total = templates.sum()
        out["productive_frequency"] = (templates / total) if total > 0 else 0.0

    # Frame type
    if "sequenceStatus" in df_raw.columns:
        ft = df_raw["sequenceStatus"].astype(str).str.strip()
        ft = ft.where(ft.isin(["In", "Out"]), "In")
        out["frame_type"] = ft
    else:
        out["frame_type"] = "In"

    out["rearrangement_type"] = "VDJ"
    out["subject"] = sample_name

    # Adaptive-style bio_identity: "<cdr3>+<TCRBVxx-yy>+<TCRBJxx-yy>"
    v_tok = df_raw["vMaxResolved"].astype(str).map(to_adaptive_bio_token)
    j_tok = df_raw["jMaxResolved"].astype(str).map(to_adaptive_bio_token)
    out["bio_identity"] = out["amino_acid"].astype(str) + "+" + v_tok + "+" + j_tok

    cols = [
        "rearrangement",
        "bio_identity",
        "amino_acid",
        "templates",
        "frame_type",
        "rearrangement_type",
        "productive_frequency",
        "subject",
    ]
    return out[cols]


def discover_export_files(dataset_dir: Path) -> list[Path]:
    """
    Inside a dataset directory, Adaptive exports are under <export_subdir_name>/.
    """
    exp = dataset_dir / config.data.export_subdir_name
    if not exp.is_dir():
        return []
    return sorted(exp.glob("*.tsv"))


def run_one_sample(
    export_tsv: Path,
    processed_dir: Path,
    temp_dir: Path,
    tcrdist_debug: bool,
) -> dict:
    sample_name = export_tsv.stem
    dataset_name = export_tsv.parent.parent.name  # parent of export_subdir

    logger.info("[%s] processing %s", dataset_name, export_tsv.name)

    df_raw = pd.read_csv(export_tsv, sep="\t", dtype=str)

    # build minimal Adaptive2020-like TSV
    adpt_df = build_adaptive_like_df(df_raw, sample_name)
    temp_dir.mkdir(parents=True, exist_ok=True)
    adaptive_tsv = temp_dir / f"{sample_name}.adaptive2020.tsv"
    adpt_df.to_csv(adaptive_tsv, sep="\t", index=False)

    # convert Adaptive naming to IMGT+cdr3 via tcrdist helper
    df_adapt = import_adaptive_file(
        adaptive_filename=str(adaptive_tsv),
        organism="human",
        chain="beta",
        version_year=2020,
        sep="\t",
        log=tcrdist_debug,
        count="productive_frequency",
        return_valid_cdr3_only=False,
    )

    # standardized input
    cell_df = pd.DataFrame(
        {
            "cdr3_b_aa": df_adapt["cdr3_b_aa"].astype(str),
            "v_b_gene": df_adapt["v_b_gene"].astype(str),
            "j_b_gene": df_adapt["j_b_gene"].astype(str),
            # use template counts as abundance
            "count": pd.to_numeric(df_adapt.get("templates", 1), errors="coerce")
            .fillna(0)
            .astype(int),
        }
    )

    # main logic, call CDR1/CDR2/CDR2.5 from V gene via TCRrep
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
        "adaptive_like": str(adaptive_tsv),
        "processed_out": str(processed_out),
        "n_raw": int(len(df_raw)),
        "n_after_adaptive_import": int(len(df_adapt)),
        "n_with_loops": int(len(loops_df)),
    }


def _run_one_sample_star(args):
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


def process_single_dataset(
    dataset_dir: str,
    tcrdist_debug: bool,
    show_progress: bool,
    n_cores: int,
    one_per_dataset: bool,
    log_every: int,
) -> dict[str, list[dict]]:
    """
    Process a single Adaptive dataset:
    """
    dataset_path = Path(dataset_dir)
    dataset_name = dataset_path.name
    out_root = processed_dataset_root(dataset_path)
    processed_dir = out_root / config.data.tcrdist_processed_subdir_name
    temp_dir = out_root / config.data.tcrdist_temp_subdir_name
    processed_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    files = discover_export_files(dataset_path)
    reports: list[dict] = []

    if not files:
        logger.warning(
            "No .tsv files found in %s",
            dataset_path / config.data.export_subdir_name,
        )
        return {dataset_name: reports}

    # debug flag: first sample only
    if one_per_dataset:
        rep = run_one_sample(files[0], processed_dir, temp_dir, tcrdist_debug=tcrdist_debug)
        reports.append(rep)
        return {dataset_name: reports}

    tasks = [(f, processed_dir, temp_dir, tcrdist_debug) for f in files]
    total = len(tasks)
    log_every = _resolve_log_every(log_every, n_cores, show_progress)
    logger.info("[%s] starting %d sample(s) (n_cores=%s)", dataset_name, total, n_cores)
    if log_every:
        logger.info("[%s] progress 0/%d", dataset_name, total)

    if n_cores is None or n_cores <= 1 or len(tasks) == 1:
        iterable = tqdm(
            tasks,
            desc=dataset_name,
            unit="file",
            disable=not show_progress,
            file=sys.stderr,
            mininterval=10,
            leave=True,
        )
        for idx, (f, pdir, tdir, dbg) in enumerate(iterable, 1):
            rep = run_one_sample(f, pdir, tdir, tcrdist_debug=dbg)
            reports.append(rep)
            if log_every and (idx % log_every == 0 or idx == total):
                logger.info("[%s] progress %d/%d", dataset_name, idx, total)
    else:
        with Pool(processes=n_cores) as pool:
            iterable = tqdm(
                pool.imap_unordered(_run_one_sample_star, tasks),
                total=len(tasks),
                desc=dataset_name,
                unit="file",
                disable=not show_progress,
                file=sys.stderr,
                mininterval=10,
                leave=True,
            )
            for idx, rep in enumerate(iterable, 1):
                reports.append(rep)
                if log_every and (idx % log_every == 0 or idx == total):
                    logger.info("[%s] progress %d/%d", dataset_name, idx, total)

    return {dataset_name: reports}



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Call CDR loops from V gene using TCRdist on adaptive dataset.\n"
            "Output per sample: cdr3aa, cdr2aa_gapped, cdr1aa_gapped, "
            "cdr2.5aa_gapped, v_b_gene, j_b_gene, count."
        )
    )
    p.add_argument(
        "--dataset-dir",
        required=True,
        help=(
            "Path to a single Adaptive dataset directory. "
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
        help="Process only the first sample in the dataset (debug / quick run).",
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
            f"(default: <dataset-dir>/{config.data.tcrdist_summary_filename})"
        ),
    )
    p.add_argument(
        "--n-cores",
        type=int,
        default=8,
        help="Number of parallel worker processes per dataset (default: 8).",
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

    reports = process_single_dataset(
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
        json.dump(reports, fh, indent=2, ensure_ascii=False)

    for ds, reps in reports.items():
        logger.info(
            "[%s] processed %d sample(s); one_per_dataset=%s",
            ds,
            len(reps),
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
