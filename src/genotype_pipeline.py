#!/usr/bin/env python3
"""
genotype_pipeline.py — Unified pipeline for donor HLA genotype prediction.
=========================================================================
Orchestrates stages with caching:
  1. check_donor_overlap       — sanity check train vs valid donors
  2. compute_q_i                — add clusters/q_i to train H5
  3a/3b. precompute_embeddings  — pooled CDR embeddings for train + valid
  4. match_tcrs                 — exact + similarity search
  5. build_per_donor_data       — invert CSR, per-donor bundles
  6. optimize_genotypes         — batched MAP on GPU
  7. evaluate_genotype_predictions (optional)
Each stage auto-skips if output exists (unless --force).
=========================================================================
USAGE:
    python genotype_pipeline.py \\
        --train_h5_filtered /path/to/filtered.h5 \\
        --valid_h5 /path/to/valid_ds.h5 \\
        --donor_matrix_path /path/to/donor_hla_matrix.npz \\
        --hla_to_id /path/to/hla_to_id.json \\
        --output_dir /path/to/out \\
        --gpu \\
        --run_evaluate
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
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Unified donor genotype pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_h5_filtered", required=True)
    p.add_argument("--valid_h5", required=True)
    p.add_argument("--donor_matrix_path", required=True)
    p.add_argument("--hla_to_id", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--train_h5_full", default=None,
                   help="Full unfiltered train H5 for overlap check (optional).")
    # Skip flags
    p.add_argument("--skip_donor_overlap", action="store_true")
    p.add_argument("--skip_compute_q_i", action="store_true")
    p.add_argument("--skip_embeddings", action="store_true")
    p.add_argument("--skip_match", action="store_true")
    p.add_argument("--skip_build_per_donor", action="store_true")
    p.add_argument("--skip_optimize", action="store_true")
    p.add_argument("--run_evaluate", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--force_qi", action="store_true")
    # Shared
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--rare_threshold", type=int, default=5)
    p.add_argument("--beta", type=float, default=2.0)
    # Embedding
    p.add_argument("--n_bins", type=int, default=5)
    p.add_argument("--emb_chunk_size", type=int, default=500000)
    # Match
    p.add_argument("--sim_threshold", type=float, default=0.9)
    p.add_argument("--alpha_conf", type=float, default=20.0)
    p.add_argument("--match_chunk_size", type=int, default=0,
                   help="Block size for similarity search (0 = auto-max).")
    # Per-donor build
    p.add_argument("--min_confidence", type=float, default=0.8)
    p.add_argument("--min_tcrs_per_donor", type=int, default=50)
    # Optimize
    p.add_argument("--lambda_reg", type=float, default=10.0)
    p.add_argument("--n_iters", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--donor_batch_size", type=int, default=8)
    p.add_argument("--max_tcrs_per_donor", type=int, default=15000)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--add_absent", action="store_true")
    # q_i
    p.add_argument("--q_chunk_size", type=int, default=1000)
    p.add_argument("--scripts_dir", default=None)
    return p.parse_args()


def find_script(scripts_dir, name):
    """Locate helper script."""
    if scripts_dir is None:
        scripts_dir = Path(__file__).parent
    p = Path(scripts_dir) / name
    if p.exists():
        return p
    alt = Path(scripts_dir).parent / "src" / name
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Helper script not found: {p}")


def run_subprocess(cmd, label):
    """Run subprocess."""
    print("\n" + "=" * 70)
    print(f"  STEP: {label}")
    print("=" * 70)
    print(f"  CMD: {' '.join(str(c) for c in cmd)}\n")
    t0 = time.time()
    r = subprocess.run(cmd, check=False)
    el = time.time() - t0
    if r.returncode != 0:
        print(f"\n[ERROR] '{label}' failed ({r.returncode})")
        sys.exit(r.returncode)
    print(f"\n[OK] '{label}' {el:.1f}s ({el/60:.1f}min)")


def has_dataset(h5_path, dataset_path):
    """Check if dataset exists in H5."""
    if not Path(h5_path).exists():
        return False
    try:
        with h5py.File(h5_path, "r") as f:
            return dataset_path in f
    except Exception:
        return False


def main():
    """Run the pipeline."""
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("DONOR GENOTYPE PREDICTION PIPELINE")
    print("=" * 70)
    print(f"  Train filtered: {args.train_h5_filtered}")
    print(f"  Valid H5:       {args.valid_h5}")
    print(f"  Output dir:     {out_dir}")
    pipeline_t0 = time.time()
    s_overlap = find_script(args.scripts_dir, "check_donor_overlap.py")
    s_qi = find_script(args.scripts_dir, "compute_q_i.py")
    s_emb = find_script(args.scripts_dir, "precompute_embeddings.py")
    s_match = find_script(args.scripts_dir, "match_tcrs.py")
    s_build = find_script(args.scripts_dir, "build_per_donor_data.py")
    s_opt = find_script(args.scripts_dir, "optimize_genotypes.py")
    s_eval = find_script(args.scripts_dir, "evaluate_genotype_predictions.py")
    # ── STEP 1: donor overlap ────────────────────────────────────────
    overlap_dir = out_dir / "donor_overlap"
    overlap_report = overlap_dir / "donor_overlap_report.json"
    if args.skip_donor_overlap:
        print("\n[SKIP] Step 1: donor_overlap")
    elif overlap_report.exists() and not args.force:
        print("\n[SKIP] Step 1: overlap report exists")
    else:
        train_for_check = args.train_h5_full or args.train_h5_filtered
        cmd = [
            sys.executable, str(s_overlap),
            "--train_h5", train_for_check,
            "--valid_h5", args.valid_h5,
            "--donor_matrix_path", args.donor_matrix_path,
            "--hla_to_id", args.hla_to_id,
            "--output_dir", str(overlap_dir),
            "--rare_threshold", str(args.rare_threshold),
        ]
        run_subprocess(cmd, "check_donor_overlap")
    # ── STEP 2: compute q_i ──────────────────────────────────────────
    qi_exists = has_dataset(args.train_h5_filtered, "clusters/q_i")
    if args.skip_compute_q_i:
        print("\n[SKIP] Step 2: compute_q_i")
    elif qi_exists and not args.force_qi:
        print("\n[SKIP] Step 2: q_i exists")
    else:
        cmd = [
            sys.executable, str(s_qi),
            "--h5_path", args.train_h5_filtered,
            "--donor_matrix_path", args.donor_matrix_path,
            "--beta", str(args.beta),
            "--chunk_size", str(args.q_chunk_size),
        ]
        if args.gpu:
            cmd.append("--gpu")
        if args.force_qi:
            cmd.append("--force")
        run_subprocess(cmd, "compute_q_i")
    # ── STEP 3: precompute embeddings ────────────────────────────────
    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(exist_ok=True)
    train_emb_path = emb_dir / "train_emb.npz"
    valid_emb_path = emb_dir / "valid_emb.npz"
    if args.skip_embeddings:
        print("\n[SKIP] Step 3: embeddings")
    else:
        if not train_emb_path.exists() or args.force:
            cmd = [
                sys.executable, str(s_emb),
                "--h5_path", args.train_h5_filtered,
                "--output", str(train_emb_path),
                "--n_bins", str(args.n_bins),
                "--chunk_size", str(args.emb_chunk_size),
            ]
            if args.force:
                cmd.append("--force")
            run_subprocess(cmd, "precompute_embeddings [train]")
        else:
            print(f"\n[SKIP] Step 3a: {train_emb_path.name} exists")
        if not valid_emb_path.exists() or args.force:
            cmd = [
                sys.executable, str(s_emb),
                "--h5_path", args.valid_h5,
                "--output", str(valid_emb_path),
                "--n_bins", str(args.n_bins),
                "--chunk_size", str(args.emb_chunk_size),
            ]
            if args.force:
                cmd.append("--force")
            run_subprocess(cmd, "precompute_embeddings [valid]")
        else:
            print(f"\n[SKIP] Step 3b: {valid_emb_path.name} exists")
    # ── STEP 4: match TCRs ───────────────────────────────────────────
    matches_path = out_dir / "matches" / "matches.npz"
    if args.skip_match:
        print("\n[SKIP] Step 4: match_tcrs")
    elif matches_path.exists() and not args.force:
        print(f"\n[SKIP] Step 4: matches.npz exists")
    else:
        cmd = [
            sys.executable, str(s_match),
            "--train_emb", str(train_emb_path),
            "--valid_emb", str(valid_emb_path),
            "--train_h5", args.train_h5_filtered,
            "--valid_h5", args.valid_h5,
            "--output", str(matches_path),
            "--sim_threshold", str(args.sim_threshold),
            "--alpha_conf", str(args.alpha_conf),
            "--chunk_size", str(args.match_chunk_size),
        ]
        if args.gpu:
            cmd.append("--gpu")
        if args.force:
            cmd.append("--force")
        run_subprocess(cmd, "match_tcrs")
    # ── STEP 5: build per-donor data ─────────────────────────────────
    per_donor_root = out_dir / "per_donor_data"
    meta_path = per_donor_root / "meta.npz"
    per_donor_bundles = per_donor_root / "per_donor"
    if args.skip_build_per_donor:
        print("\n[SKIP] Step 5: build_per_donor_data")
    elif meta_path.exists() and not args.force:
        print(f"\n[SKIP] Step 5: meta.npz exists")
    else:
        cmd = [
            sys.executable, str(s_build),
            "--train_h5", args.train_h5_filtered,
            "--valid_h5", args.valid_h5,
            "--matches", str(matches_path),
            "--output_dir", str(per_donor_root),
            "--min_confidence", str(args.min_confidence),
            "--min_tcrs_per_donor", str(args.min_tcrs_per_donor),
        ]
        if args.force:
            cmd.append("--force")
        run_subprocess(cmd, "build_per_donor_data")
    # ── STEP 6: optimize ─────────────────────────────────────────────
    predictions_path = out_dir / "predictions" / "predictions.json"
    if args.skip_optimize:
        print("\n[SKIP] Step 6: optimize")
    elif predictions_path.exists() and not args.force:
        print(f"\n[SKIP] Step 6: predictions.json exists")
    else:
        cmd = [
            sys.executable, str(s_opt),
            "--meta", str(meta_path),
            "--per_donor_dir", str(per_donor_bundles),
            "--donor_matrix_path", args.donor_matrix_path,
            "--hla_to_id", args.hla_to_id,
            "--output", str(predictions_path),
            "--rare_threshold", str(args.rare_threshold),
            "--lambda_reg", str(args.lambda_reg),
            "--n_iters", str(args.n_iters),
            "--lr", str(args.lr),
            "--donor_batch_size", str(args.donor_batch_size),
            "--max_tcrs_per_donor", str(args.max_tcrs_per_donor),
            "--topk", str(args.topk),
        ]
        if args.gpu:
            cmd.append("--gpu")
        if args.add_absent:
            cmd.append("--add_absent")
        if args.force:
            cmd.append("--force")
        run_subprocess(cmd, "optimize_genotypes")
    # ── STEP 7: evaluate ─────────────────────────────────────────────
    if args.run_evaluate:
        eval_dir = out_dir / "evaluation"
        cmd = [
            sys.executable, str(s_eval),
            "--predictions_json", str(predictions_path),
            "--donor_matrix_path", args.donor_matrix_path,
            "--hla_to_id", args.hla_to_id,
            "--output_dir", str(eval_dir),
        ]
        run_subprocess(cmd, "evaluate_genotype_predictions")
    total = time.time() - pipeline_t0
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total time:   {total:.1f}s ({total/60:.1f}min)")
    print(f"  Predictions:  {predictions_path}")


if __name__ == "__main__":
    main()