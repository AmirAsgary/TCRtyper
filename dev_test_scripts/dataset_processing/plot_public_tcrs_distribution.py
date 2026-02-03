#!/usr/bin/env python3
# plot_public_tcrs_distribution.py
#
# Stream public_tcrs.json or public_tcr_hla_counts.h5 and build a
# donors-per-TCR histogram.

import argparse
import logging
from pathlib import Path
from collections import Counter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tcrtyper.config import config

logger = logging.getLogger(__name__)

DEFAULT_TITLE = "Public TCRs: donors-per-TCR distribution"
PLOT_SUBDIR_NAME = "public_tcrs"
DEFAULT_FIGURE_FILENAME = "public_tcrs_distribution.png"


def stream_donor_counts_json(
    json_path: Path,
    *,
    show_progress: bool,
    progress_interval: float,
) -> Counter:
    import ijson

    counts = Counter()
    with open(json_path, "r", encoding="utf-8") as f:
        iterator = ijson.kvitems(f, "")
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(
                iterator,
                desc="Public TCRs",
                unit="tcr",
                mininterval=progress_interval,
            )
        for tcr_key, donors in iterator:
            if isinstance(donors, list):
                n = len(donors)
                if n > 0:
                    counts[n] += 1
            else:
                logger.debug(
                    "Skipping key %r with non-list value of type %s",
                    tcr_key,
                    type(donors).__name__,
                )
    return counts


def stream_donor_counts_h5(
    h5_path: Path,
    dataset_path: str,
    chunk_rows: int,
    *,
    show_progress: bool,
    progress_interval: float,
) -> Counter:
    import h5py
    import numpy as np

    counts = Counter()
    with h5py.File(h5_path, "r") as h5:
        if dataset_path not in h5:
            fallback = f"datasets/{dataset_path}"
            if fallback in h5:
                dataset_path = fallback
            else:
                raise KeyError(
                    f"HDF5 dataset not found: {dataset_path!r} (tried {fallback!r})"
                )
        ds = h5[dataset_path]
        if ds.ndim != 1:
            raise ValueError(
                f"HDF5 dataset must be 1D (n_donors), got shape {ds.shape}"
            )
        n_rows = int(ds.shape[0])
        if n_rows == 0:
            return counts
        step = max(1, int(chunk_rows))
        if show_progress:
            from tqdm import tqdm

            pbar = tqdm(
                total=n_rows,
                desc="Public TCRs",
                unit="rows",
                mininterval=progress_interval,
            )
        else:
            pbar = None

        for start in range(0, n_rows, step):
            stop = min(n_rows, start + step)
            chunk = np.asarray(ds[start:stop])
            chunk = chunk[chunk > 0]
            if chunk.size == 0:
                if pbar:
                    pbar.update(stop - start)
                continue
            values, freqs = np.unique(chunk, return_counts=True)
            counts.update(dict(zip(values.tolist(), freqs.tolist())))
            if pbar:
                pbar.update(stop - start)
        if pbar:
            pbar.close()
    return counts


def _parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Plot donors-per-TCR distribution from public_tcrs.json or "
            "public_tcr_hla_counts.h5 (streamed to avoid full memory loads)."
        )
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--json",
        help="Path to public_tcrs.json",
    )
    src.add_argument(
        "--h5",
        help="Path to public_tcr_hla_counts.h5",
    )
    ap.add_argument(
        "--h5-dataset",
        default="n_donors",
        help="HDF5 dataset path for donor counts (default: n_donors).",
    )
    ap.add_argument(
        "--chunk-rows",
        type=int,
        default=1_000_000,
        help="Rows to read per HDF5 chunk (default: 1000000).",
    )
    ap.add_argument(
        "--progress",
        action="store_true",
        help="Show a progress bar while streaming counts.",
    )
    ap.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="Minimum seconds between progress updates (default: 30).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help=(
            "Output PNG path "
            f"(default: <input_dir>/{config.data.plots_root_subdir_name}/"
            f"{PLOT_SUBDIR_NAME}/{DEFAULT_FIGURE_FILENAME})"
        ),
    )
    ap.add_argument(
        "--x-max",
        type=int,
        default=None,
        help=(
            "Max donors to display; values above are folded into this bin "
            "(default: derive from --clip-percentile)."
        ),
    )
    ap.add_argument(
        "--clip-percentile",
        type=float,
        default=99.5,
        help=(
            "Percentile for auto x-axis clipping (default: 99.5). "
            "Use 100 to disable."
        ),
    )
    ap.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help=f"Plot title (default: '{DEFAULT_TITLE}')",
    )
    ap.add_argument(
        "--log-y",
        action="store_true",
        help="Use log scale on Y axis",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=config.data.plots_default_dpi,
        help=f"Figure DPI (default: {config.data.plots_default_dpi})",
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


def _percentile_from_counts(counts: Counter, percentile: float) -> int:
    if not counts:
        raise ValueError("Cannot compute percentile for empty counts.")
    if percentile <= 0:
        return min(counts)
    if percentile >= 100:
        return max(counts)
    total = sum(counts.values())
    threshold = total * (percentile / 100.0)
    cumulative = 0
    for x in sorted(counts):
        cumulative += counts[x]
        if cumulative >= threshold:
            return x
    return max(counts)


def _clip_counts(counts: Counter, x_max: int) -> Counter:
    clipped = Counter()
    for x, y in counts.items():
        clipped[min(x, x_max)] += y
    return clipped


def main():
    args = _parse_args()
    _configure_logging(args.debug)

    input_path = None
    if args.json:
        json_path = Path(args.json).resolve()
        if not json_path.exists():
            raise FileNotFoundError(f"JSON not found: {json_path}")
        input_path = json_path
        logger.info("Streaming donors-per-TCR counts from %s", json_path)
        counts = stream_donor_counts_json(
            json_path,
            show_progress=args.progress,
            progress_interval=args.progress_interval,
        )
    else:
        h5_path = Path(args.h5).resolve()
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 not found: {h5_path}")
        input_path = h5_path
        logger.info(
            "Streaming donors-per-TCR counts from %s (dataset=%s)",
            h5_path,
            args.h5_dataset,
        )
        counts = stream_donor_counts_h5(
            h5_path,
            args.h5_dataset,
            args.chunk_rows,
            show_progress=args.progress,
            progress_interval=args.progress_interval,
        )

    if 0 in counts:
        del counts[0]
    if not counts:
        raise ValueError("No TCRs with donors>0 found.")

    x_max = args.x_max
    if x_max is None and args.clip_percentile < 100:
        x_max = _percentile_from_counts(counts, args.clip_percentile)
    if x_max is not None:
        max_seen = max(counts)
        if x_max < max_seen:
            counts = _clip_counts(counts, x_max)
            logger.info(
                "Clipped donors-per-TCR at %d (max seen=%d).",
                x_max,
                max_seen,
            )

    xs = sorted(counts)
    ys = [counts[k] for k in xs]

    # Output path
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        fig_dir = (
            input_path.parent
            / config.data.plots_root_subdir_name
            / PLOT_SUBDIR_NAME
        )
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_path = fig_dir / DEFAULT_FIGURE_FILENAME

    logger.info(
        "Building plot for %d TCR buckets (min donors=%d, max donors=%d)",
        len(xs),
        xs[0],
        xs[-1],
    )

    fig = plt.figure(figsize=(9, 5))
    plt.bar(xs, ys)
    plt.xlabel("Number of donors per TCR")
    plt.ylabel("Number of TCRs")
    plt.title(args.title)
    if args.log_y:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    total_tcrs = sum(ys)
    head = ", ".join(f"{k}:{counts[k]}" for k in xs[:10])

    logger.info("Saved donors-per-TCR distribution to %s", out_path)
    logger.info("TCRs counted: %d, buckets: %d", total_tcrs, len(xs))
    logger.info("First buckets -> %s", head)


if __name__ == "__main__":
    main()
