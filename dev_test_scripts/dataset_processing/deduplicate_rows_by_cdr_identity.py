#!/usr/bin/env python3
# dedup_by_six_identity_fields_parallel.py
#
# Deduplicate rows within each sample file by exact match on six identity
# columns (everything except `count`):
#   cdr3aa, cdr2aa_gapped, cdr1aa_gapped, cdr2.5aa_gapped, v_b, j_b
# For identical keys, sum `count`, keep first-seen non-key fields.

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
import tempfile
import shutil
import multiprocessing as mp
import logging

from tqdm import tqdm

from tcrtyper.config import config

logger = logging.getLogger(__name__)

KEY_FIELDS = [
    "cdr3aa",
    "cdr2aa_gapped",
    "cdr1aa_gapped",
    "cdr2.5aa_gapped",
    "v_b",
    "j_b",
]
COUNT_FIELD = "count"


def in_ipython() -> bool:
    return ("IPYKERNEL" in os.environ) or ("JPY_PARENT_PID" in os.environ)


def iter_sample_files(base_dir: Path, pattern: str):
    """Yield (dataset_name, sample_path) for <base>/<dataset>/samples/<pattern>."""
    for ds_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        smp_dir = ds_dir / "samples"
        if not smp_dir.is_dir():
            continue
        for f in sorted(smp_dir.glob(pattern)):
            if f.is_file():
                yield ds_dir.name, f


def parse_count(val):
    if val is None:
        return 0.0
    s = str(val).strip()
    if s == "":
        return 0.0
    try:
        return float(int(s))
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return 0.0


def format_count(x: float) -> str:
    if math.isfinite(x) and abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x}"


def build_key(row):
    vals = []
    for col in KEY_FIELDS:
        if col not in row:
            return None
        vals.append(row[col])
    return tuple(vals)


def rewrite_file_atomic(dst_path: Path, header, rows):
    parent = dst_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=parent, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        writer = csv.DictWriter(
            tmp, delimiter="\t", fieldnames=header, extrasaction="ignore"
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    mode = None
    try:
        mode = os.stat(dst_path).st_mode
    except FileNotFoundError:
        pass
    shutil.move(str(tmp_path), str(dst_path))
    if mode is not None:
        os.chmod(dst_path, mode)


@dataclass
class WorkerArgs:
    dataset: str
    path_str: str
    dry_run: bool


def _process_file_worker(args: WorkerArgs):
    ds_name = args.dataset
    fpath = Path(args.path_str)
    try:
        with fpath.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            header = reader.fieldnames or []
            missing_keys = [c for c in KEY_FIELDS if c not in header]
            if missing_keys:
                return {
                    "dataset": ds_name,
                    "file": fpath.name,
                    "skipped": True,
                    "reason": f"missing key columns: {','.join(missing_keys)}",
                }
            if COUNT_FIELD not in header:
                return {
                    "dataset": ds_name,
                    "file": fpath.name,
                    "skipped": True,
                    "reason": f"missing '{COUNT_FIELD}' column",
                }

            agg = OrderedDict()
            order = {}
            rows_before = 0
            counts_before = 0.0
            counts_merged_away = 0.0
            duplicates_merged = 0

            for row in reader:
                rows_before += 1
                key = build_key(row)
                if key is None:
                    key = ("__UNKEYED__", str(rows_before))
                c = parse_count(row.get(COUNT_FIELD))
                counts_before += c
                if key not in agg:
                    order[key] = len(order)
                    rep = dict(row)
                    rep[COUNT_FIELD] = c
                    agg[key] = {"row": rep, "sum": c}
                else:
                    agg[key]["sum"] += c
                    counts_merged_away += c
                    duplicates_merged += 1

            out_rows = []
            for k in sorted(order, key=lambda kk: order[kk]):
                rep = dict(agg[k]["row"])
                rep[COUNT_FIELD] = format_count(agg[k]["sum"])
                out_rows.append(rep)

            rows_after = len(out_rows)
            counts_after = sum(parse_count(r[COUNT_FIELD]) for r in out_rows)

        if not args.dry_run:
            rewrite_file_atomic(fpath, header, out_rows)

        return {
            "dataset": ds_name,
            "file": fpath.name,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "duplicates_merged": duplicates_merged,
            "counts_before": counts_before,
            "counts_after": counts_after,
            "counts_merged_away": counts_merged_away,
            "skipped": False,
        }

    except Exception as e:
        return {
            "dataset": ds_name,
            "file": fpath.name,
            "skipped": True,
            "reason": f"exception: {type(e).__name__}: {e}",
        }


def resolve_out_path(base: Path, out_arg: str, default_name: str) -> Path:
    if not out_arg or out_arg == default_name:
        return base / default_name
    p = Path(out_arg)
    if p.is_absolute() or p.parent != Path("."):
        return p.resolve()
    return base / p.name


def _parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Deduplicate TCR sample TSVs by exact match on six identity columns, "
            "summing counts. Parallel per file."
        )
    )
    ap.add_argument(
        "--base",
        default=config.data.base_dir,
        help=(
            "Root containing <dataset>/samples/*.tsv "
            f"(default: {config.data.base_dir})."
        ),
    )
    ap.add_argument(
        "--out",
        default=config.data.dedup_summary_filename,
        help=(
            "Output JSON report path. "
            f"Default: <base>/{config.data.dedup_summary_filename}"
        ),
    )
    ap.add_argument(
        "--pattern",
        default="*.tsv",
        help="Glob pattern for sample files (default: *.tsv).",
    )
    ap.add_argument(
        "--cores",
        type=int,
        default=8,
        help="Max number of worker processes (default: 8).",
    )
    ap.add_argument(
        "--progress",
        choices=["auto", "on", "off"],
        default="auto",
        help="Progress bar mode: auto (default), on, off.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute report without rewriting files.",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging.",
    )
    return ap.parse_args()


def _configure_logging(debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    args = _parse_args()
    _configure_logging(args.debug)

    base = Path(args.base).resolve()
    if not base.is_dir():
        raise SystemExit(f"Base directory not found: {base}")

    out_path = resolve_out_path(
        base, args.out, default_name=config.data.dedup_summary_filename
    )

    files = list(iter_sample_files(base, args.pattern))
    if not files:
        raise SystemExit(
            f"No sample files found under {base}/<dataset>/samples/{args.pattern}"
        )

    worker_args = [WorkerArgs(ds, str(fp), args.dry_run) for ds, fp in files]
    n_procs = max(1, min(args.cores, len(worker_args)))

    if args.progress == "on":
        disable_pbar = False
    elif args.progress == "off":
        disable_pbar = True
    else:
        disable_pbar = in_ipython()

    logger.info("Deduplicating samples")
    logger.info("Base: %s", base)
    logger.info("Files to process: %d", len(worker_args))
    logger.info("Cores requested: %d, cores used: %d", args.cores, n_procs)
    if args.dry_run:
        logger.info("Dry run mode: no files will be rewritten")

    results = []
    with mp.Pool(processes=n_procs) as pool:
        if disable_pbar:
            for res in pool.imap_unordered(_process_file_worker, worker_args, chunksize=1):
                results.append(res)
        else:
            with tqdm(
                total=len(worker_args),
                desc="Deduplicating samples",
                unit="file",
                mininterval=1.0,
                dynamic_ncols=True,
                leave=True,
                file=sys.stdout,
            ) as pbar:
                for res in pool.imap_unordered(
                    _process_file_worker, worker_args, chunksize=1
                ):
                    results.append(res)
                    pbar.update(1)

    datasets = {}
    totals = {
        "files_processed": 0,
        "rows_before": 0,
        "rows_after": 0,
        "duplicates_merged": 0,
        "counts_merged_away": 0.0,
    }

    for stats in results:
        ds_name = stats["dataset"]
        ds = datasets.setdefault(
            ds_name,
            {
                "summary": {
                    "files": 0,
                    "rows_before": 0,
                    "rows_after": 0,
                    "duplicates_merged": 0,
                    "counts_merged_away": 0.0,
                },
                "samples": [],
            },
        )

        ds["samples"].append({k: v for k, v in stats.items() if k != "dataset"})

        if stats.get("skipped"):
            continue

        ds["summary"]["files"] += 1
        ds["summary"]["rows_before"] += stats["rows_before"]
        ds["summary"]["rows_after"] += stats["rows_after"]
        ds["summary"]["duplicates_merged"] += stats["duplicates_merged"]
        ds["summary"]["counts_merged_away"] += stats["counts_merged_away"]

        totals["files_processed"] += 1
        totals["rows_before"] += stats["rows_before"]
        totals["rows_after"] += stats["rows_after"]
        totals["duplicates_merged"] += stats["duplicates_merged"]
        totals["counts_merged_away"] += stats["counts_merged_away"]

    for ds in datasets.values():
        ds["samples"].sort(key=lambda s: (s.get("skipped", False), s.get("file", "")))

    payload = {
        "base": str(base),
        "key_fields": KEY_FIELDS,
        "totals": totals,
        "datasets": datasets,
        "cores_used": n_procs,
        "dry_run": bool(args.dry_run),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Wrote JSON report: %s", out_path)
    logger.info("  cores used:        %d", n_procs)
    logger.info("  files processed:   %d", totals["files_processed"])
    logger.info("  rows before:       %d", totals["rows_before"])
    logger.info("  rows after:        %d", totals["rows_after"])
    logger.info("  duplicates merged: %d", totals["duplicates_merged"])
    logger.info(
        "  counts merged away: %s  (rows_before - rows_after is structural merge count)",
        totals["counts_merged_away"],
    )


if __name__ == "__main__":
    # TODO remove workaround
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    main()
