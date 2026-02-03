#!/usr/bin/env python3
#
# Recover sample HLA typing from Adaptive Biotechnologies sample_overview.tsv per dataset, extract HLA tags, write hla_assignments.json.
# - Drop non-MHC I or MHC II HLA
# - Split remaining tags into class I (HLA-A/B/C) and class II (HLA-DP/DQ/DR)
# - Keep all HLA under "hla_types" (I+II), compute num_hla = len(I)+len(II)
# - JSON status rules:
#     - no_tcr_seq  : export/<sample>.tsv missing
#     - no_hla      : both hla_i and hla_ii have no alleles
#     - ok          : HLA-I and HLA-II both fully typed at the DP/DQ/DR group level
#     - partial     : some HLA typing present but not fully typed across I+II

import os
import json
import csv
import argparse
import logging
from pathlib import Path

from tcrtyper.config import config
from tcrtyper.dataset_processing.hla_utils import (
    build_hla_entry,
    split_hla_tags,
)
from tcrtyper.dataset_processing.path_utils import (
    processed_dataset_root,
    sample_overview_path,
)

logger = logging.getLogger(__name__)


def process_single_dataset(dataset_path):
    """Parse HLA tags for one dataset and write hla_assignments.json."""
    dataset_root = Path(dataset_path)
    overview_path = str(sample_overview_path(dataset_root))
    export_path = os.path.join(dataset_path, config.data.export_subdir_name)
    out_root = processed_dataset_root(dataset_root)
    out_root.mkdir(parents=True, exist_ok=True)
    hla_assignments = []

    if not os.path.exists(overview_path):
        raise FileNotFoundError(f"sample_overview.tsv not found: {overview_path}")

    with open(overview_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"sample_name", "sample_tags"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(
                f"Missing required column(s) in {overview_path}: {missing_cols}"
            )

        for row in reader:
            if not any(row.values()):
                continue

            sample_name = (row.get("sample_name") or "").strip()
            hla_i, hla_ii = split_hla_tags(row.get("sample_tags", ""))

            sample_file_path = os.path.join(export_path, f"{sample_name}.tsv")
            has_tcr = os.path.exists(sample_file_path)
            rel_path = (
                os.path.relpath(sample_file_path, dataset_path) if has_tcr else None
            )

            hla_assignments.append(
                build_hla_entry(
                    sample_name=sample_name,
                    rel_path=rel_path,
                    hla_i=hla_i,
                    hla_ii=hla_ii,
                    has_tcr=has_tcr,
                )
            )

    output_json_path = out_root / config.data.hla_assignments_filename
    with open(output_json_path, "w", encoding="utf-8") as out_f:
        json.dump(hla_assignments, out_f, indent=2, ensure_ascii=False)

    return hla_assignments


def process_all_datasets(base_path):
    """Process all dataset directories under base_path/datasets that contain sample_overview.tsv."""
    results = {}
    datasets_root = os.path.join(base_path, config.data.datasets_subdir_name)
    for dataset_name in sorted(os.listdir(datasets_root)):
        dataset_path = os.path.join(datasets_root, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        if not os.path.exists(str(sample_overview_path(Path(dataset_path)))):
            continue
        logger.debug("Processing dataset %s", dataset_name)
        results[dataset_name] = process_single_dataset(dataset_path)
    return results


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract HLA class I/II assignments from Adaptive Biotechnologies "
            "datasets export v2 based on sample_overview.tsv"
        ),
    )
    parser.add_argument(
        "--base",
        default=config.data.base_dir,
        help=(
            "Base directory with dataset subfolders "
            f"(default: {config.data.base_dir})"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Process only a single dataset under --base.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Log verbosity.",
    )
    return parser.parse_args()


def _configure_logging(verbosity):
    level = logging.DEBUG if verbosity and verbosity > 0 else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    args = _parse_args()
    _configure_logging(args.verbose)
    base = Path(args.base).resolve()
    if base.name == config.data.datasets_subdir_name:
        base = base.parent

    if args.dataset:
        ds_path = os.path.join(
            str(base), config.data.datasets_subdir_name, args.dataset
        )
        entries = process_single_dataset(ds_path)
        logger.info(
            "[%s] wrote %s with %d entries",
            args.dataset,
            config.data.hla_assignments_filename,
            len(entries),
        )
    else:
        results = process_all_datasets(str(base))
        for ds, entries in results.items():
            logger.info(
                "[%s] wrote %s with %d entries",
                ds,
                config.data.hla_assignments_filename,
                len(entries),
            )


if __name__ == "__main__":
    main()
