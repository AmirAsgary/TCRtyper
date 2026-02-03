#!/usr/bin/env python3
"""
Split a large AIRR rearrangement TSV into per-repertoire TSVs under processed/export.
Only keep selected columns: cdr3_aa, v_call, j_call, count.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

from tcrtyper.config import config
from tcrtyper.dataset_processing.path_utils import processed_dataset_root

logger = logging.getLogger(__name__)


def _pick_count_field(header: Iterable[str], preferred: Optional[str]) -> str:
    header_set = set(header)
    if preferred:
        if preferred not in header_set:
            raise ValueError(f"Requested count field missing: {preferred}")
        return preferred
    for cand in ("consensus_count", "duplicate_count", "umi_count", "count"):
        if cand in header_set:
            return cand
    raise ValueError("No count field found (expected consensus_count/duplicate_count/umi_count/count).")


def _pick_gene(val: str) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if not s:
        return ""
    if "," in s:
        s = s.split(",", 1)[0].strip()
    if ";" in s:
        s = s.split(";", 1)[0].strip()
    return s


def _pick_cdr3_aa(row: dict, primary: str) -> tuple[str, bool]:
    val = (row.get(primary) or "").strip()
    if val:
        return val, False
    if primary != "junction_aa":
        fallback = (row.get("junction_aa") or "").strip()
        if fallback:
            return fallback, True
    return "", False


def _pick_count_value(row: dict, primary: str) -> tuple[int, bool]:
    for key in (primary, "consensus_count", "duplicate_count", "umi_count", "count"):
        raw = row.get(key)
        if raw not in (None, ""):
            try:
                return int(float(raw)), key != primary
            except Exception:
                return 1, key != primary
    return 1, True


def find_single_tsv(dataset_dir: Path) -> Path:
    candidates = sorted(p for p in dataset_dir.glob("*.tsv") if p.is_file())
    if not candidates:
        raise FileNotFoundError(f"No TSV found under {dataset_dir}.")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise FileNotFoundError(f"Multiple TSV files found under {dataset_dir}: {names}")
    return candidates[0]


def _write_block(out_dir: Path, repertoire_id: str, rows: list[list[str]]) -> None:
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{repertoire_id}.tsv"
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        if write_header:
            writer.writerow(["cdr3_aa", "v_call", "j_call", "count"])
        writer.writerows(rows)


def _fmt_gb(n_bytes: int) -> str:
    return f"{n_bytes / (1024 ** 3):.2f}GB"


def split_rearrangement_tsv(
    src: Path,
    export_dir: Path,
    *,
    repertoire_id_field: str,
    locus_field: str,
    cdr3_field: str,
    v_field: str,
    j_field: str,
    count_field: Optional[str],
    max_open_files: int,
    flush_max_rows: int = 0,
    log_new_repertoire: bool = True,
    log_every_gb: float = 0.5,
    debug_log_every_rows: int = 0,
) -> dict:
    counts = {
        "rows_total": 0,
        "rows_written": 0,
        "rows_non_trb": 0,
        "rows_missing_fields": 0,
        "rows_missing_repertoire_id": 0,
        "rows_missing_count": 0,
        "rows_cdr3_fallback": 0,
        "rows_count_fallback": 0,
    }
    per_rep_counts: Dict[str, int] = {}
    seen_reps = set()
    current_rep: Optional[str] = None
    current_rows: list[list[str]] = []

    total_bytes = src.stat().st_size
    last_progress_bytes = 0
    with src.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{src} is missing header row.")
        count_field = _pick_count_field(reader.fieldnames, count_field)
        logger.info(
            "Starting split: %s (%s, columns=%d, rows=unknown until end).",
            src,
            _fmt_gb(total_bytes),
            len(reader.fieldnames),
        )
        logger.info("Using count field: %s", count_field)

        for row in reader:
            counts["rows_total"] += 1
            if debug_log_every_rows and counts["rows_total"] % debug_log_every_rows == 0:
                processed = fh.buffer.tell()
                remaining = max(total_bytes - processed, 0)
                pct = (processed / total_bytes * 100.0) if total_bytes else 0.0
                logger.debug(
                    "Rows=%d processed=%s/%s (%.1f%%) remaining=%s",
                    counts["rows_total"],
                    _fmt_gb(processed),
                    _fmt_gb(total_bytes),
                    pct,
                    _fmt_gb(remaining),
                )

            locus = (row.get(locus_field) or "").strip()
            if locus and locus.upper() != "TRB":
                counts["rows_non_trb"] += 1
                continue
            if not locus:
                counts["rows_missing_fields"] += 1
                continue

            repertoire_id = (row.get(repertoire_id_field) or "").strip()
            if not repertoire_id:
                counts["rows_missing_repertoire_id"] += 1
                continue

            cdr3, used_cdr3_fallback = _pick_cdr3_aa(row, cdr3_field)
            v_call = _pick_gene(row.get(v_field))
            j_call = _pick_gene(row.get(j_field))
            if not cdr3 or not v_call or not j_call:
                counts["rows_missing_fields"] += 1
                continue

            count, used_count_fallback = _pick_count_value(row, count_field)
            if used_cdr3_fallback:
                counts["rows_cdr3_fallback"] += 1
            if used_count_fallback:
                counts["rows_count_fallback"] += 1

            if current_rep is None:
                current_rep = repertoire_id
            elif repertoire_id != current_rep:
                _write_block(export_dir, current_rep, current_rows)
                current_rows = []
                current_rep = repertoire_id

            is_new_rep = repertoire_id not in seen_reps
            current_rows.append([cdr3, v_call, j_call, str(count)])
            counts["rows_written"] += 1
            per_rep_counts[repertoire_id] = per_rep_counts.get(repertoire_id, 0) + 1
            if is_new_rep:
                seen_reps.add(repertoire_id)
            if flush_max_rows and len(current_rows) >= flush_max_rows:
                _write_block(export_dir, current_rep, current_rows)
                current_rows = []
            if log_new_repertoire and is_new_rep:
                processed = fh.buffer.tell()
                remaining = max(total_bytes - processed, 0)
                pct = (processed / total_bytes * 100.0) if total_bytes else 0.0
                logger.info(
                    "Split: new repertoire %s (%d); %s/%s (%.1f%%), remaining %s",
                    repertoire_id,
                    len(per_rep_counts),
                    _fmt_gb(processed),
                    _fmt_gb(total_bytes),
                    pct,
                    _fmt_gb(remaining),
                )
            if log_every_gb and total_bytes:
                processed = fh.buffer.tell()
                if processed - last_progress_bytes >= log_every_gb * 1024 ** 3:
                    remaining = max(total_bytes - processed, 0)
                    pct = (processed / total_bytes * 100.0) if total_bytes else 0.0
                    logger.info(
                        "Split progress: %s/%s (%.1f%%), remaining %s",
                        _fmt_gb(processed),
                        _fmt_gb(total_bytes),
                        pct,
                        _fmt_gb(remaining),
                    )
                    last_progress_bytes = processed

    if current_rep is not None:
        _write_block(export_dir, current_rep, current_rows)

    counts["repertoires"] = len(per_rep_counts)
    counts["count_field"] = count_field
    return {"counts": counts, "per_repertoire_counts": per_rep_counts}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Split AIRR rearrangement TSV into per-repertoire TSVs.",
    )
    ap.add_argument(
        "--dataset-dir",
        required=True,
        help="AIRR dataset directory containing the rearrangement TSV.",
    )
    ap.add_argument(
        "--rearrangement-tsv",
        default=None,
        help="Rearrangement TSV filename under dataset dir (default: auto-detect single *.tsv).",
    )
    ap.add_argument(
        "--export-dir",
        default=None,
        help=(
            "Override export directory "
            "(default: processed/<dataset>/export)."
        ),
    )
    ap.add_argument(
        "--count-field",
        default=None,
        help="Override count field (default: auto-pick from consensus_count/duplicate_count/umi_count).",
    )
    ap.add_argument(
        "--max-open-files",
        type=int,
        default=128,
        help="Maximum number of per-repertoire files kept open while splitting.",
    )
    ap.add_argument(
        "--flush-max-rows",
        type=int,
        default=0,
        help="Flush buffered rows after N rows for a repertoire (default: 0, disabled).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Delete existing export/*.tsv files before splitting.",
    )
    ap.add_argument(
        "--summary",
        default=None,
        help="Optional path to write a JSON split summary.",
    )
    ap.add_argument(
        "--log-every-gb",
        type=float,
        default=0.5,
        help="Emit progress every N GB processed (default: 0.5).",
    )
    ap.add_argument(
        "--debug-log-every-rows",
        type=int,
        default=0,
        help="Emit debug progress every N rows processed (default: 0, disabled).",
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


def main() -> None:
    args = parse_args()
    _configure_logging(args.debug)

    dataset_dir = Path(args.dataset_dir).resolve()
    export_dir = (
        Path(args.export_dir).resolve()
        if args.export_dir
        else processed_dataset_root(dataset_dir) / config.data.export_subdir_name
    )
    src = (
        dataset_dir / args.rearrangement_tsv
        if args.rearrangement_tsv
        else find_single_tsv(dataset_dir)
    )
    if not src.exists():
        raise FileNotFoundError(f"Rearrangement TSV not found: {src}")

    if args.force and export_dir.exists():
        for stale in export_dir.glob("*.tsv"):
            stale.unlink()

    report = split_rearrangement_tsv(
        src,
        export_dir,
        repertoire_id_field="repertoire_id",
        locus_field="locus",
        cdr3_field="cdr3_aa",
        v_field="v_call",
        j_field="j_call",
        count_field=args.count_field,
        max_open_files=args.max_open_files,
        flush_max_rows=args.flush_max_rows,
        log_every_gb=args.log_every_gb,
        debug_log_every_rows=args.debug_log_every_rows,
    )

    if report["counts"]["rows_non_trb"]:
        logger.warning("Dropped %d rows with non-TRB locus.", report["counts"]["rows_non_trb"])
    if report["counts"]["rows_missing_fields"]:
        logger.warning(
            "Dropped %d rows with missing required fields.",
            report["counts"]["rows_missing_fields"],
        )

    logger.info(
        "Wrote %d repertoires to %s (rows=%d).",
        report["counts"]["repertoires"],
        export_dir,
        report["counts"]["rows_written"],
    )

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote split summary: %s", summary_path)


if __name__ == "__main__":
    main()
