#!/usr/bin/env python3
"""
MLE Pipeline for Real TCR-HLA Data — with checkpointing and parallel support.

This script performs Maximum Likelihood Estimation (MLE) on real TCR-HLA data
stored in a large HDF5 file (~200M clusters). Each chunk of clusters is trained
independently and saved to its own checkpoint file (.npz), enabling:
  - Resume after crash (--resume skips completed chunks)
  - Parallel training across multiple jobs (--chunk_id or --chunk_range)
  - Final merge of all chunks into a single annotated H5

=== MODES ===

  train   Train z_probs for each chunk. Each chunk saves to:
              <output_dir>/chunks/chunk_<NNNNNN>.npz
          Supports --resume, --chunk_id, --chunk_range for flexible scheduling.

  merge   Combine all chunk .npz files into a single output H5:
              <output_dir>/<original_name>.h5
          with the new group clusters/z_probs/{indptr, indices, data}.

  status  Print a summary of which chunks are completed vs missing.

=== PARALLEL USAGE ===

  # Compute total chunks first (use status mode):
  python mle_real_data.py --h5_data_path data.h5 --donor_matrix_path donor.npz \
      --output_dir /out --mode status --chunk_size 100000

  # Then submit array jobs (e.g. SLURM):
  #   SBATCH --array=0-19
  python mle_real_data.py --h5_data_path data.h5 --donor_matrix_path donor.npz \
      --output_dir /out --mode train --chunk_id $SLURM_ARRAY_TASK_ID

  # Or run a range per job:
  python mle_real_data.py ... --mode train --chunk_range 0 10

  # After all jobs finish, merge:
  python mle_real_data.py ... --mode merge

=== SEQUENTIAL USAGE (with resume on crash) ===

  python mle_real_data.py --h5_data_path data.h5 --donor_matrix_path donor.npz \
      --output_dir /out --mode train --resume --epochs 10

  # If it crashes at chunk 47, just re-run the same command.
  # Chunks 0-46 will be skipped automatically.

  python mle_real_data.py ... --mode merge

=== OUTPUT ===

  The merged H5 has the same structure as the input, plus:
    clusters/z_probs/{indptr, indices, data}   (sparse CSR, float32)
  Readable via PublicTcrHlaCsrReaderChunk(path, include_z_probs=True).
"""

import os
import sys
import json
import glob
import shutil
import time
import math
import argparse
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MLE training on real TCR-HLA data (chunk-wise, parallel-safe).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sequential with resume:
  python mle_real_data.py --h5_data_path data.h5 --donor_matrix_path donor.npz --output_dir /out --mode train --resume

  # Single chunk (for cluster array jobs):
  python mle_real_data.py ... --mode train --chunk_id 5

  # Range of chunks:
  python mle_real_data.py ... --mode train --chunk_range 0 10

  # Merge all chunks into final H5:
  python mle_real_data.py ... --mode merge

  # Check progress:
  python mle_real_data.py ... --mode status
        """,
    )

    # --- Mode ---
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "merge", "status"],
        help="Pipeline mode: 'train' (fit chunks), 'merge' (combine into H5), "
             "'status' (show progress). Default: train.",
    )

    # --- I/O ---
    parser.add_argument(
        "--h5_data_path", type=str, required=True,
        help="Path to the input HDF5 dataset (read-only source).",
    )
    parser.add_argument(
        "--donor_matrix_path", type=str, required=True,
        help="Path to donor HLA matrix (.npz with key 'donor_hla_matrix').",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for chunk checkpoints and final merged H5.",
    )

    # --- Training hyperparameters ---
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs per chunk (default: 10).")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for TF training (default: 512).")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Initial learning rate (default: 0.01).")
    parser.add_argument("--beta", type=float, default=4.0,
                        help="Beta hyperparameter for penalty term (default: 4.0).")
    parser.add_argument("--l2_reg", type=float, default=1e-5,
                        help="L2 regularization on z_logits (default: 1e-5).")
    parser.add_argument("--pad_token", type=float, default=-1.0,
                        help="Padding token value (default: -1.0).")

    # --- Chunking ---
    parser.add_argument("--chunk_size", type=int, default=100_000,
                        help="Number of clusters per training chunk (default: 100000).")

    # --- Parallel / Resume ---
    parser.add_argument("--chunk_id", type=int, default=None,
                        help="Train ONLY this chunk index (0-based). For SLURM array jobs.")
    parser.add_argument("--chunk_range", type=int, nargs=2, default=None,
                        metavar=("START", "END"),
                        help="Train chunks in range [START, END). For multi-chunk jobs.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip chunks that already have checkpoint files.")

    # --- Device ---
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "gpu"],
                        help="Device: 'auto' (GPU if available), 'cpu', or 'gpu'. "
                             "Default: auto.")
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="Specific GPU index to use (e.g. 0, 1). "
                             "Only relevant with --device gpu/auto.")
    parser.add_argument("--gpu_memory_limit", type=int, default=None,
                        help="GPU memory limit in MB. If set, enables memory growth "
                             "with a cap.")

    # --- Misc ---
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Keras verbosity: 0=silent, 1=progress bar, "
                             "2=one line/epoch. Default: 1.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Device configuration (must happen BEFORE any TF operations)
# ---------------------------------------------------------------------------

def configure_device(args):
    """
    Configure TensorFlow to run on the requested device.
    Call this BEFORE importing any TF-dependent code or creating tensors.

    Handles:
      - Forcing CPU-only execution (hides all GPUs).
      - Selecting a specific GPU by index.
      - Memory growth / memory cap to prevent OOM.
    """
    import tensorflow as tf

    if args.device == "cpu":
        # Hide all GPUs so TF falls back to CPU
        tf.config.set_visible_devices([], "GPU")
        print("Device: CPU (all GPUs hidden)")
        return

    # GPU or auto mode
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        if args.device == "gpu":
            print("WARNING: --device gpu but no GPUs found. Falling back to CPU.")
        else:
            print("Device: CPU (no GPUs detected)")
        return

    # Select specific GPU if requested
    if args.gpu_id is not None:
        if args.gpu_id >= len(gpus):
            raise ValueError(
                f"--gpu_id {args.gpu_id} but only {len(gpus)} GPU(s) available."
            )
        selected = gpus[args.gpu_id]
        tf.config.set_visible_devices([selected], "GPU")
        gpus = [selected]
        print(f"Device: GPU {args.gpu_id} ({selected.name})")
    else:
        print(f"Device: GPU (found {len(gpus)})")

    # Memory growth / memory cap to prevent OOM
    for gpu in gpus:
        try:
            if args.gpu_memory_limit:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=args.gpu_memory_limit
                    )],
                )
                print(f"  GPU memory limit: {args.gpu_memory_limit} MB")
            else:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  GPU memory growth: enabled")
        except RuntimeError as e:
            print(f"  WARNING: Could not configure GPU {gpu}: {e}")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def chunks_dir(output_dir: str) -> Path:
    """Directory where chunk checkpoint .npz files are stored."""
    return Path(output_dir) / "chunks"


def chunk_path(output_dir: str, chunk_id: int) -> Path:
    """Path to a specific chunk checkpoint file."""
    return chunks_dir(output_dir) / f"chunk_{chunk_id:06d}.npz"


def get_completed_chunk_ids(output_dir: str) -> set:
    """Return set of chunk IDs that have completed checkpoint files."""
    cdir = chunks_dir(output_dir)
    if not cdir.exists():
        return set()
    completed = set()
    for f in cdir.glob("chunk_*.npz"):
        try:
            cid = int(f.stem.split("_")[1])
            completed.add(cid)
        except (IndexError, ValueError):
            pass
    return completed


def compute_chunk_schedule(total_clusters: int, chunk_size: int):
    """
    Compute the mapping from chunk_id to (cluster_start, cluster_end).
    Returns a list of (start, end) tuples indexed by chunk_id.
    """
    schedule = []
    for start in range(0, total_clusters, chunk_size):
        end = min(start + chunk_size, total_clusters)
        schedule.append((start, end))
    return schedule


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def extract_binder_sets_from_counts(counts_dense: np.ndarray, pad_token: float):
    """
    Convert a dense counts matrix to padded binder sets (nonzero allele indices).

    Parameters
    ----------
    counts_dense : np.ndarray, shape (n_clusters, num_alleles)
        Dense count matrix for this chunk.
    pad_token : float
        Padding value.

    Returns
    -------
    binder_sets : np.ndarray, shape (n_clusters, max_nonzero_per_row)
        Padded allele column indices where count > 0.
    max_all : int
        Maximum number of nonzero alleles across all rows in this chunk.
    """
    from utils import pad_list_to_array_without_max

    n = counts_dense.shape[0]
    nonzero_rows, nonzero_cols = np.nonzero(counts_dense)
    counts_set = np.split(
        nonzero_cols, np.searchsorted(nonzero_rows, np.arange(1, n))
    )
    binder_sets, max_all = pad_list_to_array_without_max(counts_set, pad_token)
    return binder_sets, max_all


def split_ragged_to_list(flat_indices: np.ndarray, indptr: np.ndarray):
    """Split flat CSR indices into a list of per-row arrays using indptr."""
    return np.split(flat_indices, indptr[1:-1])


def extract_z_probs_from_model(model, binder_mask: np.ndarray) -> np.ndarray:
    """
    Extract sigmoid(z_logits) * mask from a trained SparseTCRModel.

    Returns z_probs with shape (num_tcrs, max_hlas_per_tcr),
    zeroed at padded positions.
    """
    import tensorflow as tf

    z_logits = model.z_embedding.get_weights()[0]  # (num_tcrs, max_hlas_per_tcr)
    z_probs = tf.sigmoid(z_logits).numpy().astype(np.float32)
    z_probs[~binder_mask] = 0.0
    return z_probs


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_chunk_checkpoint(
    output_dir: str,
    chunk_id: int,
    cluster_start: int,
    cluster_end: int,
    binder_sets: np.ndarray,
    z_probs: np.ndarray,
    metadata: dict,
) -> Path:
    """
    Save a single chunk's training results to a compressed .npz file.

    Contents:
      cluster_start, cluster_end : int scalars
      binder_sets : (n_clusters, max_hlas) padded allele indices
      z_probs     : (n_clusters, max_hlas) sigmoid probabilities (float32)
      metadata_json : JSON string with loss history and timing
    """
    cdir = chunks_dir(output_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    fpath = chunk_path(output_dir, chunk_id)

    # Write to a temp file first, then atomically rename.
    # This prevents partial files if the job is killed mid-write.
    tmp_path = fpath.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path,
        cluster_start=np.array(cluster_start),
        cluster_end=np.array(cluster_end),
        binder_sets=binder_sets,
        z_probs=z_probs.astype(np.float32),
        metadata_json=np.array(json.dumps(metadata)),
    )
    tmp_path.rename(fpath)  # atomic on POSIX
    return fpath


def load_chunk_checkpoint(fpath: Path) -> dict:
    """Load a chunk checkpoint .npz and return its contents."""
    data = np.load(fpath, allow_pickle=False)
    return {
        "cluster_start": int(data["cluster_start"]),
        "cluster_end": int(data["cluster_end"]),
        "binder_sets": data["binder_sets"],
        "z_probs": data["z_probs"],
        "metadata": json.loads(str(data["metadata_json"])),
    }


# ---------------------------------------------------------------------------
# MODE: train
# ---------------------------------------------------------------------------

def run_train(args):
    """
    Train z_probs for the specified chunks.
    Each chunk saves an independent .npz checkpoint to <output_dir>/chunks/.
    """
    # Configure device FIRST (before any TF tensor creation)
    configure_device(args)

    import tensorflow as tf
    from utils import (
        SparseTCRModel, create_dataset, pad_list_to_array_without_max,
        PublicTcrHlaCsrReaderChunk,
    )

    # Reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    chunks_dir(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save config (idempotent: every job writes the same content)
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- Load donor matrix (small, fits in memory) ----
    print("Loading donor HLA matrix...")
    donor_hla_matrix = np.load(args.donor_matrix_path)["donor_hla_matrix"]
    num_donors, num_alleles = donor_hla_matrix.shape
    print(f"  Donor matrix: {num_donors} donors x {num_alleles} alleles")

    # ---- Compute chunk schedule ----
    print("Counting total clusters...")
    with PublicTcrHlaCsrReaderChunk(
        args.h5_data_path, include_counts=False, include_donors=False
    ) as reader:
        total_clusters = reader.num_clusters
    schedule = compute_chunk_schedule(total_clusters, args.chunk_size)
    total_chunks = len(schedule)
    print(f"  Total: {total_clusters:,} clusters -> {total_chunks} chunks")

    # ---- Determine target chunk IDs for this job ----
    if args.chunk_id is not None:
        if args.chunk_id < 0 or args.chunk_id >= total_chunks:
            raise ValueError(
                f"--chunk_id {args.chunk_id} out of range [0, {total_chunks})"
            )
        target_ids = [args.chunk_id]
    elif args.chunk_range is not None:
        cstart, cend = args.chunk_range
        cstart = max(0, cstart)
        cend = min(cend, total_chunks)
        target_ids = list(range(cstart, cend))
    else:
        target_ids = list(range(total_chunks))

    # ---- Resume: skip already-completed chunks ----
    if args.resume:
        completed = get_completed_chunk_ids(args.output_dir)
        before = len(target_ids)
        target_ids = [c for c in target_ids if c not in completed]
        skipped = before - len(target_ids)
        if skipped > 0:
            print(f"  Resume: skipping {skipped} already-completed chunks")

    if not target_ids:
        print("Nothing to train — all target chunks are already done!")
        return

    print(f"  Will train {len(target_ids)} chunks: "
          f"[{target_ids[0]} .. {target_ids[-1]}]")

    # Convert target_ids to a set for fast lookup during iteration
    target_set = set(target_ids)

    # ---- Optimized H5 read range ----
    # Only read the cluster range that covers our target chunks.
    min_cluster = schedule[target_ids[0]][0]
    max_cluster = schedule[target_ids[-1]][1]

    # ---- Training loop ----
    global_t0 = time.time()
    trained_count = 0

    with PublicTcrHlaCsrReaderChunk(
        args.h5_data_path,
        include_counts=True,
        include_pvals=False,
        include_donors=True,
    ) as reader:
        for chunk in reader.iter_cluster_chunks(
            chunk_rows=args.chunk_size,
            start=min_cluster,
            stop=max_cluster,
        ):
            # Map this chunk's cluster_start back to a chunk_id
            chunk_id = chunk.cluster_start // args.chunk_size

            # Skip chunks not assigned to this job
            if chunk_id not in target_set:
                continue

            chunk_t0 = time.time()
            n_clusters = chunk.n_clusters
            cluster_start = chunk.cluster_start
            cluster_end = chunk.cluster_end

            print(
                f"\n{'='*60}\n"
                f"Chunk {chunk_id}/{total_chunks}: "
                f"clusters [{cluster_start}, {cluster_end}) "
                f"({n_clusters:,} clusters)\n"
                f"{'='*60}"
            )

            # -- Donor indices: ragged CSR -> padded 2D array --
            donor_lists = split_ragged_to_list(
                chunk.raw_csr_donor_indices, chunk.raw_csr_donor_indptr
            )
            donor_indices_padded, max_donors = pad_list_to_array_without_max(
                donor_lists, args.pad_token
            )
            donor_indices_padded = donor_indices_padded.astype(np.int32)

            # -- Binder sets: nonzero allele indices from counts_dense --
            binder_sets, max_hlas = extract_binder_sets_from_counts(
                chunk.counts_dense, args.pad_token
            )
            binder_sets = binder_sets.astype(np.int32)
            print(f"  binder_sets: ({n_clusters}, {max_hlas}) | "
                  f"donor_indices: ({n_clusters}, {max_donors})")

            # Handle empty chunks
            if max_hlas == 0:
                print("  WARNING: no nonzero alleles — saving empty checkpoint.")
                save_chunk_checkpoint(
                    args.output_dir, chunk_id, cluster_start, cluster_end,
                    binder_sets=binder_sets,
                    z_probs=np.zeros_like(binder_sets, dtype=np.float32),
                    metadata={
                        "chunk_id": chunk_id, "skipped": True,
                        "reason": "no_nonzero_alleles",
                    },
                )
                trained_count += 1
                continue

            # -- TF dataset --
            train_dataset = create_dataset(donor_indices_padded, args.batch_size)

            # -- Model --
            steps_per_epoch = max(1, n_clusters // args.batch_size)
            lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=args.learning_rate,
                first_decay_steps=steps_per_epoch * 20,
                t_mul=2.0, m_mul=0.9, alpha=0.1,
            )
            model = SparseTCRModel(
                num_tcrs=n_clusters,
                max_hlas_per_tcr=max_hlas,
                donor_hla_matrix=donor_hla_matrix,
                binder_sets=binder_sets,
                beta=args.beta,
                mode="continuous",
                pad_token=args.pad_token,
                l2_reg_lambda=args.l2_reg,
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            model.compile(optimizer=optimizer)

            # -- Train --
            history = model.fit(
                train_dataset, epochs=args.epochs, verbose=args.verbose
            )

            # -- Extract z_probs --
            binder_mask = binder_sets != args.pad_token
            z_probs = extract_z_probs_from_model(model, binder_mask)

            # -- Save checkpoint (atomic write) --
            elapsed = time.time() - chunk_t0
            final_loss = float(history.history["final_loss"][-1])
            meta = {
                "chunk_id": chunk_id,
                "cluster_start": cluster_start,
                "cluster_end": cluster_end,
                "n_clusters": n_clusters,
                "max_hlas_per_tcr": max_hlas,
                "max_donors_per_cluster": max_donors,
                "final_loss": final_loss,
                "loss_history": [float(v) for v in history.history.get("loss", [])],
                "elapsed_seconds": round(elapsed, 2),
                "skipped": False,
            }
            fpath = save_chunk_checkpoint(
                args.output_dir, chunk_id, cluster_start, cluster_end,
                binder_sets=binder_sets, z_probs=z_probs, metadata=meta,
            )
            print(f"  Done in {elapsed:.1f}s | "
                  f"Loss: {final_loss:.4f} | Saved: {fpath.name}")

            # -- Free memory --
            del model, train_dataset, binder_sets, donor_indices_padded, z_probs
            tf.keras.backend.clear_session()
            trained_count += 1

    total_elapsed = time.time() - global_t0
    print(f"\nTraining finished: {trained_count} chunks "
          f"in {total_elapsed:.1f}s ({total_elapsed / 3600:.2f}h)")


# ---------------------------------------------------------------------------
# MODE: merge
# ---------------------------------------------------------------------------

def run_merge(args):
    """
    Merge all chunk .npz checkpoints into the final output H5.

    Steps:
      1. Copy original H5 to output dir (if not already there).
      2. Open copy in append mode via MleZprobsWriter.
      3. Load each chunk .npz in order and write its z_probs.
      4. Save a combined training log JSON.
    """
    from utils import PublicTcrHlaCsrReaderChunk, MleZprobsWriter, NumpyEncoder

    print("=" * 60)
    print("MERGE MODE: combining chunk checkpoints into output H5")
    print("=" * 60)

    # ---- Count total clusters ----
    with PublicTcrHlaCsrReaderChunk(
        args.h5_data_path, include_counts=False, include_donors=False
    ) as reader:
        total_clusters = reader.num_clusters
    schedule = compute_chunk_schedule(total_clusters, args.chunk_size)
    total_chunks = len(schedule)

    # ---- Check completeness ----
    completed = get_completed_chunk_ids(args.output_dir)
    missing = sorted(set(range(total_chunks)) - completed)
    print(f"\n  Total chunks: {total_chunks}")
    print(f"  Completed:    {len(completed)}")
    print(f"  Missing:      {len(missing)}")

    if missing:
        print(f"\n  WARNING: {len(missing)} chunks are missing!")
        ranges = _compress_ranges(missing)
        print(f"  Missing ranges: {ranges}")
        resp = input("  Continue? Missing chunks will have zero z_probs. [y/N]: ")
        if resp.strip().lower() != "y":
            print("  Aborted.")
            return

    # ---- Copy original H5 to output ----
    output_h5_path = Path(args.output_dir) / Path(args.h5_data_path).name
    if not output_h5_path.exists():
        print(f"\nCopying original H5...")
        print(f"  From: {args.h5_data_path}")
        print(f"  To:   {output_h5_path}")
        t0 = time.time()
        shutil.copy2(args.h5_data_path, str(output_h5_path))
        print(f"  Done in {time.time() - t0:.1f}s")
    else:
        print(f"\nOutput H5 already exists: {output_h5_path}")
        # Remove stale z_probs from a previous merge attempt
        import h5py
        with h5py.File(str(output_h5_path), "a") as f:
            if "z_probs" in f.get("clusters", {}):
                print("  Removing old z_probs group from previous merge...")
                del f["clusters"]["z_probs"]

    # ---- Write z_probs ----
    writer = MleZprobsWriter(
        str(output_h5_path),
        num_clusters=total_clusters,
        chunk_nnz=100_000,
        chunk_rows=10_000,
        compression={"compression": "gzip", "compression_opts": 4},
    )
    writer.open()

    all_metadata = []
    t0 = time.time()

    for chunk_id in range(total_chunks):
        fpath = chunk_path(args.output_dir, chunk_id)

        if not fpath.exists():
            # Missing chunk: indptr stays flat -> zero-length row
            print(f"  Chunk {chunk_id:6d}: MISSING (zeros)")
            all_metadata.append({
                "chunk_id": chunk_id, "skipped": True,
                "reason": "missing_checkpoint",
            })
            continue

        ckpt = load_chunk_checkpoint(fpath)
        writer.write_chunk(
            cluster_start=ckpt["cluster_start"],
            cluster_end=ckpt["cluster_end"],
            binder_sets=ckpt["binder_sets"],
            z_probs=ckpt["z_probs"],
            pad_token=args.pad_token,
        )
        meta = ckpt["metadata"]
        all_metadata.append(meta)
        loss = meta.get("final_loss", "N/A")
        print(f"  Chunk {chunk_id:6d}: "
              f"[{ckpt['cluster_start']}, {ckpt['cluster_end']}) | "
              f"loss={loss}")

    writer.close()
    merge_elapsed = time.time() - t0

    # ---- Save combined log ----
    log = {
        "total_clusters": total_clusters,
        "total_chunks": total_chunks,
        "completed_chunks": len(completed),
        "missing_chunks": len(missing),
        "merge_elapsed_seconds": round(merge_elapsed, 2),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "l2_reg": args.l2_reg,
            "chunk_size": args.chunk_size,
            "seed": args.seed,
        },
        "chunks": all_metadata,
    }
    log_path = os.path.join(args.output_dir, "mle_training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'='*60}")
    print(f"Merge complete!")
    print(f"  Output H5:     {output_h5_path}")
    print(f"  Training log:  {log_path}")
    print(f"  Merged:        {len(completed)}/{total_chunks} chunks")
    print(f"  Merge time:    {merge_elapsed:.1f}s")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# MODE: status
# ---------------------------------------------------------------------------

def run_status(args):
    """Print a summary of which chunks are completed vs missing."""
    from utils import PublicTcrHlaCsrReaderChunk

    with PublicTcrHlaCsrReaderChunk(
        args.h5_data_path, include_counts=False, include_donors=False
    ) as reader:
        total_clusters = reader.num_clusters

    schedule = compute_chunk_schedule(total_clusters, args.chunk_size)
    total_chunks = len(schedule)
    completed = get_completed_chunk_ids(args.output_dir)
    missing = sorted(set(range(total_chunks)) - completed)

    print(f"\n{'='*60}")
    print(f"MLE Pipeline Status")
    print(f"{'='*60}")
    print(f"  Input H5:        {args.h5_data_path}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Total clusters:  {total_clusters:,}")
    print(f"  Chunk size:      {args.chunk_size:,}")
    print(f"  Total chunks:    {total_chunks}")
    pct = 100 * len(completed) / total_chunks if total_chunks > 0 else 0
    print(f"  Completed:       {len(completed)}/{total_chunks} ({pct:.1f}%)")

    if missing:
        print(f"  Missing:         {len(missing)}")
        ranges = _compress_ranges(missing)
        print(f"  Missing ranges:  {ranges}")
        if len(missing) <= 30:
            print(f"  Missing IDs:     {missing}")

        # Print SLURM helper
        print(f"\n  SLURM array job command (one chunk per task):")
        print(f"    #SBATCH --array={missing[0]}-{missing[-1]}")
        print(f"    python mle_real_data.py \\")
        print(f"      --h5_data_path {args.h5_data_path} \\")
        print(f"      --donor_matrix_path {args.donor_matrix_path} \\")
        print(f"      --output_dir {args.output_dir} \\")
        print(f"      --chunk_size {args.chunk_size} \\")
        print(f"      --mode train --chunk_id $SLURM_ARRAY_TASK_ID")
    else:
        print(f"\n  All chunks complete! Ready to merge:")
        print(f"    python mle_real_data.py \\")
        print(f"      --h5_data_path {args.h5_data_path} \\")
        print(f"      --donor_matrix_path {args.donor_matrix_path} \\")
        print(f"      --output_dir {args.output_dir} \\")
        print(f"      --chunk_size {args.chunk_size} \\")
        print(f"      --mode merge")

    # ---- Aggregate statistics from completed chunks ----
    if completed:
        losses = []
        times = []
        for cid in sorted(completed):
            fpath = chunk_path(args.output_dir, cid)
            try:
                data = np.load(fpath, allow_pickle=False)
                meta = json.loads(str(data["metadata_json"]))
                if not meta.get("skipped", False):
                    losses.append(meta.get("final_loss", float("nan")))
                    times.append(meta.get("elapsed_seconds", 0))
            except Exception:
                pass

        if losses:
            print(f"\n  Loss stats (completed chunks):")
            print(f"    Mean:   {np.mean(losses):.4f}")
            print(f"    Median: {np.median(losses):.4f}")
            print(f"    Min:    {np.min(losses):.4f}")
            print(f"    Max:    {np.max(losses):.4f}")
        if times:
            print(f"  Timing stats:")
            print(f"    Mean per chunk: {np.mean(times):.1f}s")
            total_so_far = np.sum(times)
            print(f"    Total so far:   {total_so_far:.0f}s "
                  f"({total_so_far / 3600:.2f}h)")
            if missing:
                est = np.mean(times) * len(missing)
                print(f"    Est. remaining: {est:.0f}s ({est / 3600:.2f}h) "
                      f"for {len(missing)} chunks")

    print(f"{'='*60}\n")


def _compress_ranges(ids: list) -> str:
    """Compress [0,1,2,5,6,7,10] into '0-2, 5-7, 10'."""
    if not ids:
        return "none"
    ranges = []
    start = ids[0]
    end = ids[0]
    for i in ids[1:]:
        if i == end + 1:
            end = i
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = i
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ", ".join(ranges)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.mode == "train":
        run_train(args)
    elif args.mode == "merge":
        run_merge(args)
    elif args.mode == "status":
        run_status(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()