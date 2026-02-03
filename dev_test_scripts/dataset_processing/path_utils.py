#!/usr/bin/env python3
"""
Path helpers for dataset layout.
"""

from __future__ import annotations

from pathlib import Path

from tcrtyper.config import config


def sample_overview_path(dataset_root: Path) -> Path:
    """
    Resolve sample_overview.tsv under the dataset root.

    Preferred location: <dataset_root>/<metadata_subdir>/<sample_overview_filename>.
    Fallback: <dataset_root>/<sample_overview_filename>.
    """
    preferred = (
        dataset_root
        / config.data.metadata_subdir_name
        / config.data.sample_overview_filename
    )
    if preferred.exists():
        return preferred
    fallback = dataset_root / config.data.sample_overview_filename
    if fallback.exists():
        return fallback
    return preferred


def processed_datasets_root(base: Path) -> Path:
    return base / config.data.processed_subdir_name


def input_datasets_root(base: Path) -> Path:
    return base / config.data.datasets_subdir_name


def dataset_input_root(base: Path, dataset: str) -> Path:
    return input_datasets_root(base) / dataset


def dataset_output_root(base: Path, dataset: str) -> Path:
    return processed_datasets_root(base) / dataset


def processed_dataset_root(dataset_root: Path) -> Path:
    """
    Given a dataset root under datasets/, return the matching processed/<dataset>.
    If the input is already under processed/, return it unchanged.
    """
    if dataset_root.parent.name == config.data.processed_subdir_name:
        return dataset_root
    if dataset_root.parent.name == config.data.datasets_subdir_name:
        return (
            dataset_root.parent.parent
            / config.data.processed_subdir_name
            / dataset_root.name
        )

    # Handle symlink-resolved roots by falling back to the configured base_dir.
    for parent in dataset_root.parents:
        if parent.name == config.data.datasets_subdir_name:
            return parent.parent / config.data.processed_subdir_name / dataset_root.name

    return Path(config.data.base_dir) / config.data.processed_subdir_name / dataset_root.name
