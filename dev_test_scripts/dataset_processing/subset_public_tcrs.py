#!/usr/bin/env python3
"""
Create a small public_tcrs.json subset for quick diagnostics.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import ijson

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Subset public_tcrs.json to N clusters with >= min_donors."
    )
    ap.add_argument(
        "--export-root",
        default=None,
        help="Export train dataset root (used for default paths).",
    )
    ap.add_argument(
        "--public-json",
        default=None,
        help="Path to public_tcrs.json (default: <export_root>/public_tcrs.json).",
    )
    ap.add_argument(
        "--meta-json",
        default=None,
        help="Path to public_tcrs_meta.json (default: <export_root>/public_tcrs_meta.json).",
    )
    ap.add_argument(
        "--out-json",
        default=None,
        help="Output subset JSON path (default: <export_root>/public_tcrs_subset.json).",
    )
    ap.add_argument(
        "--out-meta",
        default=None,
        help="Output subset metadata JSON path (default: <export_root>/public_tcrs_meta_subset.json).",
    )
    ap.add_argument(
        "--skip-meta",
        action="store_true",
        default=False,
        help="Skip reading/writing the public_tcrs_meta.json subset.",
    )
    ap.add_argument(
        "--min-donors",
        type=int,
        default=10,
        help="Minimum donors per cluster (default: 10).",
    )
    ap.add_argument(
        "--max-clusters",
        type=int,
        default=50,
        help="Number of clusters to keep (default: 50).",
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


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path | None, Path, Path | None]:
    if args.public_json:
        public_json = Path(args.public_json).resolve()
    elif args.export_root:
        public_json = Path(args.export_root).resolve() / "public_tcrs.json"
    else:
        raise SystemExit("Provide --public-json or --export-root.")

    if args.skip_meta:
        meta_json = None
    elif args.meta_json:
        meta_json = Path(args.meta_json).resolve()
    elif args.export_root:
        meta_json = Path(args.export_root).resolve() / "public_tcrs_meta.json"
    else:
        meta_json = None

    if args.out_json:
        out_json = Path(args.out_json).resolve()
    elif args.export_root:
        out_json = Path(args.export_root).resolve() / "public_tcrs_subset.json"
    else:
        out_json = public_json.with_name("public_tcrs_subset.json")

    if args.skip_meta:
        out_meta = None
    elif args.out_meta:
        out_meta = Path(args.out_meta).resolve()
    elif args.export_root:
        out_meta = Path(args.export_root).resolve() / "public_tcrs_meta_subset.json"
    else:
        out_meta = None

    return public_json, meta_json, out_json, out_meta


def _subset_public_json(
    public_json: Path,
    min_donors: int,
    max_clusters: int,
) -> tuple[dict, list[str], int]:
    subset: dict = {}
    keys: list[str] = []
    eligible = 0
    with open(public_json, "r", encoding="utf-8") as fh:
        for cid, donors in ijson.kvitems(fh, ""):
            if not isinstance(donors, list):
                continue
            if len(donors) < min_donors:
                continue
            eligible += 1
            subset[cid] = donors
            keys.append(cid)
            if max_clusters > 0 and len(keys) >= max_clusters:
                break
    return subset, keys, eligible


def _subset_meta_json(meta_json: Path, keys_set: set[str]) -> dict:
    subset: dict = {}
    with open(meta_json, "r", encoding="utf-8") as fh:
        for cid, meta in ijson.kvitems(fh, ""):
            if cid in keys_set:
                subset[cid] = meta
    return subset


def main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)

    if args.max_clusters < 0:
        raise SystemExit("--max-clusters must be >= 0.")
    if args.min_donors < 1:
        raise SystemExit("--min-donors must be >= 1.")

    public_json, meta_json, out_json, out_meta = _resolve_paths(args)
    if not public_json.exists():
        raise FileNotFoundError(f"public_tcrs.json not found: {public_json}")
    if meta_json is not None and not meta_json.exists():
        logger.warning("public_tcrs_meta.json not found: %s", meta_json)
        meta_json = None

    logger.info("Reading %s", public_json)
    subset, keys, eligible = _subset_public_json(
        public_json,
        args.min_donors,
        args.max_clusters,
    )
    if not subset:
        raise SystemExit("No clusters matched the criteria.")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(subset, fh, indent=2)
    logger.info(
        "Wrote %d clusters to %s (eligible=%d, min_donors=%d)",
        len(keys),
        out_json,
        eligible,
        args.min_donors,
    )

    if meta_json is not None and out_meta is not None:
        keys_set = set(keys)
        logger.info("Reading %s", meta_json)
        meta_subset = _subset_meta_json(meta_json, keys_set)
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        with open(out_meta, "w", encoding="utf-8") as fh:
            json.dump(meta_subset, fh, indent=2)
        logger.info("Wrote metadata for %d clusters to %s", len(meta_subset), out_meta)


if __name__ == "__main__":
    main()
