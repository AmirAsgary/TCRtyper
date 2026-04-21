#!/usr/bin/env python3
"""
h5_preparation_for_nn_training.py — Unified pipeline for NN training H5 prep.
=========================================================================
Runs all preparation steps in order, with flags to skip steps if their
outputs already exist in the H5 / output directory.
PIPELINE STEPS:
  1. compute_S_i               — adds clusters/S_i and S_i_without_rare_alleles
  2. diagnose_filter           — preview plots before filtering (no H5 write)
  3. filter_tcrs_by_S          — write new filtered H5
  4. compute_percentile_rank   — adds clusters/percentile_rank to filtered H5
  5. analyze_zprobs (optional) — additional visualizations on filtered H5
=========================================================================
USAGE:
    # Full pipeline (all steps):
    python h5_preparation_for_nn_training.py \\
        --h5_path /path/to/dataset_pval.h5 \\
        --donor_matrix_path /path/to/donor_hla_matrix.npz \\
        --output_dir /path/to/nn_prep_out \\
        --S_threshold -1.0 \\
        --min_n_donors 10 \\
        --use_without_rare \\
        --gpu
    # Skip specific steps (e.g. if you already computed S_i):
    python h5_preparation_for_nn_training.py ... \\
        --skip_compute_S_i
    # Available skip flags:
    #   --skip_compute_S_i
    #   --skip_diagnose
    #   --skip_filter
    #   --skip_percentile_rank
    #   --skip_analyze
    # Add visualization step (analyze_zprobs):
    python h5_preparation_for_nn_training.py ... \\
        --run_analyze_zprobs
=========================================================================
DEPENDENCIES (must be in same directory or src/):
    compute_S_i.py
    diagnose_filter.py
    filter_tcrs_by_S.py
    compute_percentile_rank.py
    analyze_zprobs.py  (optional, only if --run_analyze_zprobs)
=========================================================================
"""
import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
import h5py


def parse_args():
    """Parse CLI arguments for the unified pipeline."""
    p = argparse.ArgumentParser(
        description="Unified H5 preparation pipeline for NN training.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Required I/O
    p.add_argument("--h5_path", required=True,
                   help="Source H5 with z_probs (output of MLE merge step).")
    p.add_argument("--donor_matrix_path", required=True,
                   help="Donor HLA matrix NPZ.")
    p.add_argument("--output_dir", required=True,
                   help="Output directory for filtered H5 and plots.")
    # Step parameters
    p.add_argument("--rare_threshold", type=int, default=5,
                   help="HLAs with < this many donors are 'rare' (default: 5).")
    p.add_argument("--S_threshold", type=float, default=-1.0,
                   help="S_i filter threshold (default: -1.0).")
    p.add_argument("--min_n_donors", type=int, default=10,
                   help="n_donors filter (default: 10).")
    p.add_argument("--use_without_rare", action="store_true",
                   help="Use S_i_without_rare_alleles for filtering.")
    # Performance
    p.add_argument("--gpu", action="store_true", help="Use GPU for compute_S_i.")
    p.add_argument("--chunk_size", type=int, default=200000,
                   help="Chunk size for streaming reads (default: 200000).")
    p.add_argument("--S_chunk_size", type=int, default=5000,
                   help="Chunk size for compute_S_i (default: 5000).")
    p.add_argument("--reservoir_size", type=int, default=5000,
                   help="Reservoir size for diagnose_filter (default: 5000).")
    # Skip flags
    p.add_argument("--skip_compute_S_i", action="store_true",
                   help="Skip step 1: compute_S_i (use if already done).")
    p.add_argument("--skip_diagnose", action="store_true",
                   help="Skip step 2: diagnose_filter preview.")
    p.add_argument("--skip_filter", action="store_true",
                   help="Skip step 3: filter_tcrs_by_S.")
    p.add_argument("--skip_percentile_rank", action="store_true",
                   help="Skip step 4: compute_percentile_rank.")
    p.add_argument("--run_analyze_zprobs", action="store_true",
                   help="Run optional step 5: analyze_zprobs visualizations.")
    p.add_argument("--run_compare_rank", action="store_true",
                   help="Run optional step 6: 2-panel gamma vs rank comparison plot.")
    # Force overwrites
    p.add_argument("--force", action="store_true",
                   help="Force overwrite existing artifacts without prompting.")
    # Script location
    p.add_argument("--scripts_dir", default=None,
                   help="Directory containing the helper scripts "
                        "(default: same dir as this script).")
    return p.parse_args()


def find_script(scripts_dir, name):
    """Locate a helper script in scripts_dir or alongside this script."""
    if scripts_dir is None:
        scripts_dir = Path(__file__).parent
    p = Path(scripts_dir) / name
    if not p.exists():
        # Try src/ subdirectory
        alt = Path(scripts_dir).parent / "src" / name
        if alt.exists():
            return alt
        raise FileNotFoundError(f"Helper script not found: {p}")
    return p


def run_subprocess(cmd, label):
    """Run a subprocess and stream its output. Exit on failure."""
    print("\n" + "=" * 70)
    print(f"  STEP: {label}")
    print("=" * 70)
    print(f"  CMD: {' '.join(str(c) for c in cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{label}' failed with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n[OK] Step '{label}' done in {elapsed:.1f}s ({elapsed/60:.1f}min)")


def has_dataset(h5_path, dataset_path):
    """Check if a dataset exists in H5 file."""
    if not Path(h5_path).exists():
        return False
    try:
        with h5py.File(h5_path, "r") as f:
            return dataset_path in f
    except Exception:
        return False


def main():
    """Run the unified pipeline."""
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("H5 PREPARATION PIPELINE FOR NN TRAINING")
    print("=" * 70)
    print(f"  Source H5:        {args.h5_path}")
    print(f"  Output dir:       {out_dir}")
    print(f"  Donor matrix:     {args.donor_matrix_path}")
    print(f"  S_i threshold:    {args.S_threshold}")
    print(f"  min_n_donors:     {args.min_n_donors}")
    print(f"  Use w/o rare:     {args.use_without_rare}")
    print(f"  GPU:              {args.gpu}")
    pipeline_t0 = time.time()
    # Resolve helper scripts
    s_compute = find_script(args.scripts_dir, "compute_S_i.py")
    s_diag = find_script(args.scripts_dir, "diagnose_filter.py")
    s_filter = find_script(args.scripts_dir, "filter_tcrs_by_S.py")
    s_rank = find_script(args.scripts_dir, "compute_percentile_rank.py")
    # ─────────────────────────────────────────────────────────────────
    # STEP 1: compute_S_i
    # ─────────────────────────────────────────────────────────────────
    s1_done = has_dataset(args.h5_path, "clusters/S_i") and \
              has_dataset(args.h5_path, "clusters/S_i_without_rare_alleles")
    if args.skip_compute_S_i:
        print(f"\n[SKIP] Step 1: compute_S_i (--skip_compute_S_i set)")
    elif s1_done and not args.force:
        print(f"\n[SKIP] Step 1: clusters/S_i already exists in H5 "
              f"(use --force to recompute)")
    else:
        cmd = [
            sys.executable, str(s_compute),
            "--h5_path", args.h5_path,
            "--donor_matrix_path", args.donor_matrix_path,
            "--rare_threshold", str(args.rare_threshold),
            "--chunk_size", str(args.S_chunk_size),
        ]
        if args.gpu:
            cmd.append("--gpu")
        if args.force:
            cmd.append("--force")
        run_subprocess(cmd, "compute_S_i")
    # ─────────────────────────────────────────────────────────────────
    # STEP 2: diagnose_filter (preview before filtering)
    # ─────────────────────────────────────────────────────────────────
    diag_dir = out_dir / "diagnostic_preview"
    if args.skip_diagnose:
        print(f"\n[SKIP] Step 2: diagnose_filter (--skip_diagnose set)")
    elif (diag_dir / "diagnostic_report.json").exists() and not args.force:
        print(f"\n[SKIP] Step 2: diagnostic preview already exists "
              f"(use --force to regenerate)")
    else:
        cmd = [
            sys.executable, str(s_diag),
            "--h5_path", args.h5_path,
            "--donor_matrix_path", args.donor_matrix_path,
            "--output_dir", str(diag_dir),
            "--S_threshold", str(args.S_threshold),
            "--min_n_donors", str(args.min_n_donors),
            "--chunk_size", str(args.chunk_size),
            "--reservoir_size", str(args.reservoir_size),
        ]
        if args.use_without_rare:
            cmd.append("--use_without_rare")
        run_subprocess(cmd, "diagnose_filter")
    # ─────────────────────────────────────────────────────────────────
    # STEP 3: filter_tcrs_by_S
    # ─────────────────────────────────────────────────────────────────
    filter_dir = out_dir / "filtered"
    filtered_h5 = filter_dir / (Path(args.h5_path).stem + "_filtered.h5")
    if args.skip_filter:
        print(f"\n[SKIP] Step 3: filter_tcrs_by_S (--skip_filter set)")
    elif filtered_h5.exists() and not args.force:
        print(f"\n[SKIP] Step 3: filtered H5 exists at {filtered_h5} "
              f"(use --force to regenerate)")
    else:
        cmd = [
            sys.executable, str(s_filter),
            "--h5_path", args.h5_path,
            "--output_dir", str(filter_dir),
            "--threshold", str(args.S_threshold),
            "--min_n_donors", str(args.min_n_donors),
            "--chunk_size", str(args.chunk_size),
        ]
        if args.use_without_rare:
            cmd.append("--use_without_rare")
        run_subprocess(cmd, "filter_tcrs_by_S")
    if not filtered_h5.exists():
        print(f"\n[ERROR] Filtered H5 not found at {filtered_h5}. Aborting.")
        sys.exit(1)
    # ─────────────────────────────────────────────────────────────────
    # STEP 4: compute_percentile_rank (on filtered H5)
    # ─────────────────────────────────────────────────────────────────
    pr_done = has_dataset(filtered_h5, "clusters/percentile_rank")
    if args.skip_percentile_rank:
        print(f"\n[SKIP] Step 4: compute_percentile_rank "
              f"(--skip_percentile_rank set)")
    elif pr_done and not args.force:
        print(f"\n[SKIP] Step 4: clusters/percentile_rank already exists "
              f"(use --force to recompute)")
    else:
        cmd = [
            sys.executable, str(s_rank),
            "--h5_path", str(filtered_h5),
            "--chunk_size", str(args.chunk_size),
        ]
        if args.force:
            cmd.append("--force")
        run_subprocess(cmd, "compute_percentile_rank")
    # ─────────────────────────────────────────────────────────────────
    # STEP 5 (optional): analyze_zprobs
    # ─────────────────────────────────────────────────────────────────
    if args.run_analyze_zprobs:
        try:
            s_analyze = find_script(args.scripts_dir, "analyze_zprobs.py")
        except FileNotFoundError:
            try:
                s_analyze = find_script(
                    Path(args.scripts_dir or Path(__file__).parent) / "mle",
                    "analyze_zprobs.py")
            except FileNotFoundError:
                print(f"\n[WARN] analyze_zprobs.py not found, skipping step 5")
                s_analyze = None
        if s_analyze is not None:
            analyze_dir = out_dir / "analysis"
            cmd = [
                sys.executable, str(s_analyze),
                "--h5_path", str(filtered_h5),
                "--donor_matrix_path", args.donor_matrix_path,
                "--output_dir", str(analyze_dir),
                "--chunk_size", str(args.chunk_size),
                "--all",
            ]
            if args.gpu:
                cmd.append("--gpu")
            run_subprocess(cmd, "analyze_zprobs")
    # ─────────────────────────────────────────────────────────────────
    # STEP 6 (optional): compare_gamma_vs_rank
    # ─────────────────────────────────────────────────────────────────
    if args.run_compare_rank:
        try:
            s_cmp = find_script(args.scripts_dir, "compare_gamma_vs_rank.py")
        except FileNotFoundError:
            print(f"\n[WARN] compare_gamma_vs_rank.py not found, skipping step 6")
            s_cmp = None
        if s_cmp is not None:
            cmp_dir = out_dir / "compare_gamma_vs_rank"
            cmd = [
                sys.executable, str(s_cmp),
                "--h5_path", str(filtered_h5),
                "--donor_matrix_path", args.donor_matrix_path,
                "--output_dir", str(cmp_dir),
                "--chunk_size", str(args.chunk_size),
                "--reservoir_size", str(args.reservoir_size),
            ]
            run_subprocess(cmd, "compare_gamma_vs_rank")
    # ─────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────
    total = time.time() - pipeline_t0
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total time:       {total:.1f}s ({total/60:.1f}min)")
    print(f"  Filtered H5:      {filtered_h5}")
    print(f"  Diagnostic plots: {out_dir / 'diagnostic_preview'}")
    print(f"  Filter report:    {filter_dir / 'filter_report.json'}")
    if args.run_analyze_zprobs:
        print(f"  Analysis dir:     {out_dir / 'analysis'}")
    print()
    print("  Ready for NN training. Use the filtered H5 with the new")
    print("  PublicTcrHlaCsrReaderChunk flags:")
    print("      include_z_probs=True")
    print("      include_S_i=True")
    print("      include_S_i_without_rare_alleles=True")
    print("      include_percentile_rank=True   # NEW")
    print()


if __name__ == "__main__":
    main()