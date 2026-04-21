#!/usr/bin/env python3
"""
diagnose_filter.py — Preview what a (S_i + n_donors) filter would look like.
=========================================================================
Without writing a new H5, this script applies a candidate filter to your
merged H5 and produces two plots so you can decide if there is enough
signal to train on. Useful for choosing thresholds before running
filter_tcrs_by_S.py.
PLOTS PRODUCED:
  1. per_allele_tcr_count.png      — TCRs per HLA in log scale (bar)
  2. per_allele_gamma_boxplot.png  — gamma distribution per HLA over the
                                     observed (TCR, HLA) pairs that pass
                                     the filter (boxplot + dots)
=========================================================================
USAGE:
    python diagnose_filter.py \\
        --h5_path /path/to/dataset_pval.h5 \\
        --donor_matrix_path /path/to/donor_hla_matrix.npz \\
        --output_dir /path/to/diagnostic_S_minus1_nd10 \\
        --S_threshold -1.0 \\
        --min_n_donors 10 \\
        --use_without_rare

=========================================================================
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import h5py
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Filter TCRs by S_i AND n_donors, write new H5.")
    p.add_argument("--h5_path", required=True, help="Source H5 with S_i field.")
    p.add_argument("--output_dir", required=True, help="Output directory.")
    p.add_argument("--threshold", type=float, default=-1.0,
                   help="Keep TCRs with S_i >= threshold (default: -1.0).")
    p.add_argument("--min_n_donors", type=int, default=1,
                   help="Keep TCRs with n_donors >= this (default: 1).")
    p.add_argument("--use_without_rare", action="store_true",
                   help="Use S_i_without_rare_alleles instead of S_i.")
    p.add_argument("--chunk_size", type=int, default=200000,
                   help="Chunk size (default: 200000).")
    return p.parse_args()


def detect_clusters_layout(clusters_grp, n_clusters):
    """Walk clusters/ group and classify each entry by role."""
    scalars = []
    csr_groups = []
    passthrough = []
    def walk(grp, prefix=""):
        if "indptr" in grp and isinstance(grp["indptr"], h5py.Dataset):
            ip = grp["indptr"]
            if ip.shape[0] == n_clusters + 1:
                csr_groups.append((prefix.rstrip("/"), grp))
                return
        for name, obj in grp.items():
            sub_path = f"{prefix}{name}"
            if isinstance(obj, h5py.Group):
                walk(obj, prefix=sub_path + "/")
            elif isinstance(obj, h5py.Dataset):
                if obj.shape and obj.shape[0] == n_clusters:
                    scalars.append((sub_path, obj))
                else:
                    passthrough.append((sub_path, obj))
    walk(clusters_grp)
    return scalars, csr_groups, passthrough


def copy_scalar_filtered(out_clusters, path, src_ds, mask, chunk_size):
    """Copy a per-TCR scalar/2D dataset, applying boolean mask."""
    n_keep = int(mask.sum())
    if src_ds.ndim == 1:
        out_shape = (n_keep,)
        out_chunks = (min(chunk_size, n_keep),) if n_keep > 0 else None
    else:
        out_shape = (n_keep,) + src_ds.shape[1:]
        out_chunks = ((min(chunk_size, n_keep),) + src_ds.shape[1:]
                      if n_keep > 0 else None)
    out_ds = out_clusters.create_dataset(
        path, shape=out_shape, dtype=src_ds.dtype,
        chunks=out_chunks,
        compression="gzip", compression_opts=4 if n_keep > 0 else None,
    )
    write_pos = 0
    n_clusters = src_ds.shape[0]
    for cs in range(0, n_clusters, chunk_size):
        ce = min(cs + chunk_size, n_clusters)
        chunk_mask = mask[cs:ce]
        n_chunk_keep = int(chunk_mask.sum())
        if n_chunk_keep == 0:
            continue
        block = src_ds[cs:ce]
        out_ds[write_pos:write_pos + n_chunk_keep] = block[chunk_mask]
        write_pos += n_chunk_keep
    for k, v in src_ds.attrs.items():
        out_ds.attrs[k] = v


def copy_csr_filtered(out_clusters, path, src_grp, mask, chunk_size):
    """Rebuild a CSR group (indptr/indices/[data]) for filtered TCRs.
    Vectorized: builds whole-chunk concatenated arrays in memory and
    writes once per chunk, avoiding millions of tiny per-row h5py writes.
    """
    n_clusters = src_grp["indptr"].shape[0] - 1
    n_keep = int(mask.sum())
    src_indptr = src_grp["indptr"]
    src_indices = src_grp["indices"]
    has_data = "data" in src_grp
    src_data = src_grp["data"] if has_data else None
    # Read full source indptr once (small, shape n_clusters+1)
    full_indptr = np.asarray(src_indptr[:])
    full_row_lengths = (full_indptr[1:] - full_indptr[:-1])
    kept_row_lengths = full_row_lengths[mask]
    new_indptr = np.zeros(n_keep + 1, dtype=np.int64)
    new_indptr[1:] = np.cumsum(kept_row_lengths.astype(np.int64))
    new_nnz = int(new_indptr[-1])
    out_grp = out_clusters.create_group(path)
    out_grp.create_dataset(
        "indptr", data=new_indptr, dtype=np.int64,
        chunks=(min(chunk_size + 1, n_keep + 1),),
        compression="gzip", compression_opts=4,
    )
    out_indices = out_grp.create_dataset(
        "indices", shape=(new_nnz,), dtype=src_indices.dtype,
        chunks=(min(1_000_000, new_nnz),) if new_nnz > 0 else None,
        compression="gzip", compression_opts=4 if new_nnz > 0 else None,
    )
    out_data = None
    if has_data:
        out_data = out_grp.create_dataset(
            "data", shape=(new_nnz,), dtype=src_data.dtype,
            chunks=(min(1_000_000, new_nnz),) if new_nnz > 0 else None,
            compression="gzip", compression_opts=4 if new_nnz > 0 else None,
        )
    # Stream chunked: for each chunk, build a concatenated kept array
    # in memory and do ONE bulk write (huge speedup over per-row writes).
    write_pos = 0
    t0 = time.time()
    last_log = t0
    for cs in range(0, n_clusters, chunk_size):
        ce = min(cs + chunk_size, n_clusters)
        chunk_mask = mask[cs:ce]
        if not chunk_mask.any():
            continue
        ip_chunk = full_indptr[cs:ce + 1]
        s, e = int(ip_chunk[0]), int(ip_chunk[-1])
        if e == s:
            continue
        # Bulk read indices/data for the whole chunk range
        idx_block = np.asarray(src_indices[s:e])
        data_block = np.asarray(src_data[s:e]) if has_data else None
        # Vectorized: build a row->kept selection mask over the flat block.
        # Each element belongs to row k in the chunk; the row is kept iff
        # chunk_mask[k] is True. We construct a per-element mask in one shot.
        ip_local = ip_chunk - s
        row_lengths_local = ip_local[1:] - ip_local[:-1]
        # Element-level boolean mask (length = e - s)
        elem_mask = np.repeat(chunk_mask, row_lengths_local)
        # Bulk select kept entries
        kept_idx = idx_block[elem_mask]
        n_kept_in_chunk = kept_idx.shape[0]
        if n_kept_in_chunk > 0:
            out_indices[write_pos:write_pos + n_kept_in_chunk] = kept_idx
            if has_data:
                kept_data = data_block[elem_mask]
                out_data[write_pos:write_pos + n_kept_in_chunk] = kept_data
            write_pos += n_kept_in_chunk
        # Periodic progress log (every 5s)
        now = time.time()
        if now - last_log > 5.0:
            elapsed = now - t0
            pct = 100 * write_pos / max(new_nnz, 1)
            rate = write_pos / elapsed if elapsed > 0 else 0
            print(f"      ... {path}: {write_pos:,}/{new_nnz:,} nnz "
                  f"({pct:.1f}%) | {rate/1e6:.1f}M/s")
            last_log = now
    for k, v in src_grp.attrs.items():
        out_grp.attrs[k] = v


def compute_per_allele_counts(out_h5_path, num_alleles, chunk_size):
    """Count TCRs per HLA from filtered counts CSR."""
    counts_per_hla = np.zeros(num_alleles, dtype=np.int64)
    with h5py.File(out_h5_path, "r") as f:
        clusters = f["clusters"]
        if "counts" not in clusters:
            return counts_per_hla
        cn_indptr = clusters["counts"]["indptr"]
        cn_indices = clusters["counts"]["indices"]
        n = cn_indptr.shape[0] - 1
        full_ip = np.asarray(cn_indptr[:])
        for cs in range(0, n, chunk_size):
            ce = min(cs + chunk_size, n)
            s, e = int(full_ip[cs]), int(full_ip[ce])
            if e == s:
                continue
            block = np.asarray(cn_indices[s:e])
            counts_per_hla += np.bincount(block, minlength=num_alleles)
    return counts_per_hla


def plot_per_allele_counts(counts_per_hla, output_dir, threshold,
                           min_n_donors, field):
    """Bar plot of TCR count per HLA in log scale."""
    A = len(counts_per_hla)
    fig_w = max(20.0, A * 0.10)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    x = np.arange(A)
    display = np.where(counts_per_hla > 0, counts_per_hla, 0.5)
    colors = np.where(counts_per_hla > 0, "steelblue", "red")
    ax.bar(x, display, color=colors, width=0.8, edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlim(-0.5, A - 0.5)
    ax.set_xlabel("HLA allele index", fontsize=12)
    ax.set_ylabel("Number of TCRs (log scale)", fontsize=12)
    n_zero = int((counts_per_hla == 0).sum())
    ax.set_title(
        f"TCR count per HLA after filter "
        f"({field} >= {threshold} AND n_donors >= {min_n_donors}) | "
        f"{A} alleles | {n_zero} alleles with 0 TCRs",
        fontsize=12,
    )
    tick_step = max(1, A // 50)
    ax.set_xticks(np.arange(0, A, tick_step))
    ax.set_xticklabels(np.arange(0, A, tick_step), fontsize=6, rotation=90)
    ax.grid(axis="y", which="both", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    out_path = Path(output_dir) / "per_allele_tcr_counts.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    """Run the filter pipeline."""
    args = parse_args()
    field = "S_i_without_rare_alleles" if args.use_without_rare else "S_i"
    print("=" * 60)
    print("Filter TCRs by S_i + n_donors pipeline")
    print("=" * 60)
    print(f"  Source H5:    {args.h5_path}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Filter:       {field} >= {args.threshold} "
          f"AND n_donors >= {args.min_n_donors}")
    os.makedirs(args.output_dir, exist_ok=True)
    out_h5_path = Path(args.output_dir) / Path(args.h5_path).stem
    out_h5_path = out_h5_path.with_name(out_h5_path.name + "_filtered.h5")
    # ── compute mask ────────────────────────────────────────────────
    t0 = time.time()
    with h5py.File(args.h5_path, "r") as f:
        clusters = f["clusters"]
        if field not in clusters:
            print(f"ERROR: clusters/{field} not found. Run compute_S_i.py first.")
            sys.exit(1)
        S_vals = np.asarray(clusters[field][:])
        n_donors_vals = np.asarray(clusters["n_donors"][:])
        num_clusters = S_vals.shape[0]
        num_alleles = int(f.attrs.get("num_alleles", 0))
        if num_alleles == 0 and "counts" in clusters:
            num_alleles = int(np.max(clusters["counts"]["indices"][:1000])) + 1
    mask_S = S_vals >= args.threshold
    mask_nd = n_donors_vals >= args.min_n_donors
    mask = mask_S & mask_nd
    n_keep = int(mask.sum())
    n_drop = num_clusters - n_keep
    print(f"  Total TCRs:                  {num_clusters:,}")
    print(f"  Pass S_i >= {args.threshold}:    "
          f"{int(mask_S.sum()):,}")
    print(f"  Pass n_donors >= {args.min_n_donors}: "
          f"{int(mask_nd.sum()):,}")
    print(f"  Pass BOTH (kept):            {n_keep:,} "
          f"({100*n_keep/num_clusters:.2f}%)")
    print(f"  Dropped:                     {n_drop:,} "
          f"({100*n_drop/num_clusters:.2f}%)")
    if n_keep == 0:
        print("ERROR: No TCRs pass filter, aborting.")
        sys.exit(1)
    # ── open source + create output ─────────────────────────────────
    src = h5py.File(args.h5_path, "r")
    out = h5py.File(out_h5_path, "w")
    for k, v in src.attrs.items():
        out.attrs[k] = v
    out.attrs["filtered_by"] = field
    out.attrs["filter_S_threshold"] = float(args.threshold)
    out.attrs["filter_min_n_donors"] = int(args.min_n_donors)
    out.attrs["original_num_clusters"] = num_clusters
    out.attrs["filtered_num_clusters"] = n_keep
    out_clusters = out.create_group("clusters")
    src_clusters = src["clusters"]
    scalars, csr_groups, passthrough = detect_clusters_layout(
        src_clusters, num_clusters)
    print(f"\n  Layout: {len(scalars)} scalars, "
          f"{len(csr_groups)} CSR groups, {len(passthrough)} passthrough")
    for path, ds in scalars:
        print(f"  [scalar] {path}")
        copy_scalar_filtered(out_clusters, path, ds, mask, args.chunk_size)
    for path, grp in csr_groups:
        print(f"  [CSR]    {path}")
        copy_csr_filtered(out_clusters, path, grp, mask, args.chunk_size)
    for path, ds in passthrough:
        print(f"  [pass]   {path}")
        out_clusters.create_dataset(path, data=ds[()])
        for k, v in ds.attrs.items():
            out_clusters[path].attrs[k] = v
    out.close()
    src.close()
    write_time = time.time() - t0
    print(f"\nWrite phase done in {write_time:.1f}s ({write_time/60:.1f}min)")
    print(f"  Output H5: {out_h5_path}")
    # ── per-allele report ───────────────────────────────────────────
    print(f"\nComputing per-allele TCR counts...")
    if num_alleles == 0:
        with h5py.File(out_h5_path, "r") as f:
            num_alleles = int(f.attrs.get("num_alleles", 0))
            if num_alleles == 0 and "counts" in f["clusters"]:
                num_alleles = int(np.max(
                    f["clusters"]["counts"]["indices"][:])) + 1
    counts_per_hla = compute_per_allele_counts(
        out_h5_path, num_alleles, args.chunk_size)
    n_zero_alleles = int((counts_per_hla == 0).sum())
    print(f"  Alleles with 0 TCRs:  {n_zero_alleles}/{num_alleles}")
    print(f"  Mean TCRs per HLA:    {counts_per_hla.mean():.0f}")
    print(f"  Median TCRs per HLA:  {int(np.median(counts_per_hla))}")
    print(f"  Min/Max TCRs per HLA: {counts_per_hla.min()}/{counts_per_hla.max()}")
    plot_path = plot_per_allele_counts(
        counts_per_hla, args.output_dir, args.threshold,
        args.min_n_donors, field)
    print(f"  Plot: {plot_path}")
    report = {
        "source_h5": str(args.h5_path),
        "output_h5": str(out_h5_path),
        "filter_field": field,
        "filter_S_threshold": float(args.threshold),
        "filter_min_n_donors": int(args.min_n_donors),
        "original_num_clusters": int(num_clusters),
        "filtered_num_clusters": int(n_keep),
        "dropped_num_clusters": int(n_drop),
        "kept_fraction": float(n_keep / num_clusters),
        "S_i_stats": {
            "min": float(S_vals.min()),
            "max": float(S_vals.max()),
            "mean": float(S_vals.mean()),
            "median": float(np.median(S_vals)),
        },
        "per_allele_counts": {
            "n_alleles": int(num_alleles),
            "n_alleles_with_zero_tcrs": n_zero_alleles,
            "mean_tcrs_per_hla": float(counts_per_hla.mean()),
            "median_tcrs_per_hla": int(np.median(counts_per_hla)),
            "min_tcrs_per_hla": int(counts_per_hla.min()),
            "max_tcrs_per_hla": int(counts_per_hla.max()),
        },
    }
    report_path = Path(args.output_dir) / "filter_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {report_path}")
    total = time.time() - t0
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f}min)")


if __name__ == "__main__":
    main()