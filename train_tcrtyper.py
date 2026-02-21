#!/usr/bin/env python3
"""
TCRtyper: End-to-end Transformer Training Pipeline for TCR-HLA Binding
=======================================================================
Maximises the log-likelihood LL_B(theta,phi) = (1/|B|) sum_{i in B} LL_i(gamma_i(theta,phi))
where gamma_ia = sigmoid(NN_phi(r_a)^T . NN_theta(s_i)) predicts the probability
that TCR i binds HLA allotype a. Training uses minibatch SGD with Adam over
cluster-level chunks streamed from HDF5.
Modes:
  --mode train      : train from scratch or resume from checkpoint
  --mode inference   : load best checkpoint and write z_probs to output H5
  --mode export      : export SavedModel for serving
Usage:
  python train_tcrtyper.py --mode train --train_ds data/train.h5 --valid_ds data/valid.h5 \
      --donor_hla_matrix data/donor_hla_matrix.npz --output_dir runs/exp01
"""
from __future__ import annotations
# ── GPU memory config MUST run before any TF operation ───────────────
# Importing tensorflow or keras can initialise the GPU context, so we
# configure memory growth at the very top, before all other imports.
import os as _os
def _early_gpu_config():
    """Set GPU memory growth before TF context is locked.
    Also suppress noisy XLA autotuning and TF INFO logs."""
    import os as _os2
    _os2.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress INFO/WARNING
    _os2.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
    try:
        import tensorflow as _tf
        gpus = _tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            _tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        # Already initialised (e.g. in interactive session) — safe to ignore
        pass
_early_gpu_config()
# ── standard / third-party imports ───────────────────────────────────
import os, random
import argparse, json, sys, time, shutil, math, platform, subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# ── project imports ──────────────────────────────────────────────────
from src.utils import (
    PublicTcrHlaCsrReaderChunk,
    MleZprobsWriter,
    NumpyEncoder,
    match_hla_alleles,
    pad_and_mask_tcr,
    concatenate_cdrs_with_separator,
    pad_list_to_array_without_max,
    SequenceEncoderLayer,
    GatedTransformerLayer,
    TCRLikelihoodLoss,
    encode_msa_frequencies,
)
# ═════════════════════════════════════════════════════════════════════
# 1.  FLAGS / ARGUMENT PARSER
# ═════════════════════════════════════════════════════════════════════
def parse_args():
    """Parse all command-line flags for training, inference, and export."""
    p = argparse.ArgumentParser(
        description="TCRtyper: Likelihood-based TCR-HLA binding prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── mode ─────────────────────────────────────────────────────────
    p.add_argument("--mode", type=str, default="train",
                   choices=["train", "inference", "export"],
                   help="Pipeline mode: train | inference | export")
    # ── data paths ───────────────────────────────────────────────────
    p.add_argument("--train_ds", type=str, required=True,
                   help="Path to training HDF5 (PublicTcrHlaCsrReaderChunk)")
    p.add_argument("--valid_ds", type=str, default="",
                   help="Path to validation HDF5 (optional, skipped if empty)")
    p.add_argument("--inference_ds", type=str, default="",
                   help="Path to HDF5 for inference mode")
    p.add_argument("--donor_hla_matrix", type=str, required=True,
                   help="NPZ with key 'donor_hla_matrix' shape (N_donors, A)")
    p.add_argument("--idx_to_hla", type=str, required=True,
                   help="JSON mapping allele index -> HLA name")
    p.add_argument("--hla_embed", type=str, default="",
                   help="NPZ with HLA embeddings from ESM (keys=allele names)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Root directory for checkpoints, logs, outputs")
    # ── model architecture ───────────────────────────────────────────
    p.add_argument("--embed_dim", type=int, default=64,
                   help="Transformer embedding dimension")
    p.add_argument("--num_heads", type=int, default=2,
                   help="Number of attention heads per layer")
    p.add_argument("--num_layers", type=int, default=1,
                   help="Number of GatedTransformerLayer blocks")
    p.add_argument("--ff_dim", type=int, default=0,
                   help="FFN hidden dim (0 = 4*embed_dim)")
    p.add_argument("--hla_proj_dim", type=int, default=32,
                   help="Intermediate projection dim before HLA head")
    p.add_argument("--num_alleles", type=int, default=440,
                   help="Total number of HLA alleles (A)")
    p.add_argument("--max_seq_len", type=int, default=150,
                   help="Maximum concatenated CDR sequence length")
    p.add_argument("--encoding_mode", type=str, default="blosum",
                   choices=["blosum", "raw"],
                   help="Sequence encoder pre-projection mode")
    p.add_argument("--resnet", action="store_true", default=True,
                   help="Use residual connections in transformer")
    p.add_argument("--no_resnet", action="store_false", dest="resnet")
    p.add_argument("--train_hla_head", action="store_true", default=False,
                   help="Allow gradient flow through HLA embedding head")
    # ── loss / likelihood ────────────────────────────────────────────
    p.add_argument("--beta", type=float, default=4.0,
                   help="Beta-Binomial prior parameter")
    p.add_argument("--l2_reg", type=float, default=0.0,
                   help="L2 regularisation on z_logits")
    p.add_argument("--reduction", type=str, default="sum",
                   choices=["sum", "mean"],
                   help="Loss reduction mode")
    p.add_argument("--poisson_approx", action="store_true", default=False,
                   help="Use Poisson approximation for untyped HLAs")
    # ── training hyper-parameters ────────────────────────────────────
    p.add_argument("--batch_size", type=int, default=1024,
                   help="Cluster-level batch size (chunk_rows)")
    p.add_argument("--epochs", type=int, default=50,
                   help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Peak learning rate")
    p.add_argument("--min_lr", type=float, default=1e-6,
                   help="Minimum learning rate for cosine decay")
    p.add_argument("--warmup_steps", type=int, default=500,
                   help="Linear warmup steps before cosine decay")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="AdamW weight decay")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Global gradient norm clip (0 = disabled)")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout rate in encoder and transformer")
    p.add_argument("--masking_rate", type=float, default=0.0,
                   help="Random CDR masking rate for self-supervised signal")
    # ── tokens ───────────────────────────────────────────────────────
    p.add_argument("--pad_token", type=int, default=-1)
    p.add_argument("--mask_token", type=int, default=-2)
    p.add_argument("--sep_token", type=int, default=-3)
    p.add_argument("--normal_token", type=int, default=1)
    # ── logging / checkpointing ──────────────────────────────────────
    p.add_argument("--log_step", type=int, default=100,
                   help="Log metrics every N steps")
    p.add_argument("--save_every_epoch", type=int, default=1,
                   help="Save checkpoint every N epochs")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience (epochs w/o improvement)")
    p.add_argument("--resume", action="store_true", default=False,
                   help="Resume from latest checkpoint in output_dir")
    # ── hardware ─────────────────────────────────────────────────────
    p.add_argument("--mixed_precision", action="store_true", default=False,
                   help="Enable float16 mixed-precision training")
    p.add_argument("--gpu_memory_limit", type=int, default=0,
                   help="GPU memory limit in MB (0 = no limit)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    # ── TFRecord cache ───────────────────────────────────────────────
    p.add_argument("--use_tfrecord", action="store_true", default=False,
                   help="Convert H5 to TFRecords once, then use tf.data "
                        "pipeline for all epochs (much faster)")
    p.add_argument("--num_shards", type=int, default=16,
                   help="Number of TFRecord shard files for parallel reads")
    # ── donor filtering ──────────────────────────────────────────────
    # Only keep clusters whose n_donors >= this threshold during
    # TFRecord generation and HDF5 streaming.  Default 1 keeps all.
    p.add_argument("--keep_only_upperthan_n_donors", type=int, default=1,
                   help="Only store/use clusters with at least N donors. "
                        "Clusters with fewer donors are skipped during "
                        "TFRecord generation and HDF5 streaming (default: 1)")
    # ── custom TFRecord path ─────────────────────────────────────────
    # If set, the model reads/writes/resumes TFRecord cache files from
    # this path instead of the default <output_dir>/tfrecord_cache_*.
    p.add_argument("--tf_record_path", type=str, default="",
                   help="Custom base directory for TFRecord cache files. "
                        "If set, TFRecords are read/written/resumed from "
                        "this path instead of the default output_dir")
    return p.parse_args()
# ═════════════════════════════════════════════════════════════════════
# 2.  HARDWARE SETUP
# ═════════════════════════════════════════════════════════════════════
def setup_hardware(args):
    """Configure GPU memory, mixed precision, and random seeds.
    NOTE: memory growth is already configured at import time by
    _early_gpu_config(). This function handles the remaining setup
    (seeds, memory limits, mixed precision) and is safe to call
    after the TF runtime has been initialised.
    """
    # ── seed all RNGs for reproducibility ────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    # Enable deterministic ops (slight perf hit, full reproducibility)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    # GPU detection (memory growth was set at import time)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Apply explicit memory limit if requested (must be set before
        # context init — skip gracefully if already initialised)
        if args.gpu_memory_limit > 0:
            try:
                for gpu in gpus:
                    tf.config.set_logical_device_configuration(
                        gpu, [tf.config.LogicalDeviceConfiguration(
                            memory_limit=args.gpu_memory_limit)])
            except RuntimeError:
                print("[HW] GPU already initialised — "
                      "gpu_memory_limit flag ignored")
        print(f"[HW] {len(gpus)} GPU(s) detected: "
              f"{[g.name for g in gpus]}")
    else:
        print("[HW] No GPU detected — running on CPU")
    # Mixed precision
    if args.mixed_precision and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[HW] Mixed precision enabled (float16)")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
    return "GPU" if gpus else "CPU"
# ═════════════════════════════════════════════════════════════════════
# 3.  DATA LOADING HELPERS
# ═════════════════════════════════════════════════════════════════════
def split_ragged_to_list(flat_indices: np.ndarray,
                         indptr: np.ndarray) -> List[np.ndarray]:
    """Split flat CSR indices into per-cluster lists using indptr."""
    return np.split(flat_indices, indptr[1:-1])
def _pad_and_mask_numpy(ragged_arrays, pad_token=-1, mask_token=-2,
                        masking_rate=0.0):
    """Fast numpy-only pad+mask for a list of 2D arrays (L_i, 21).
    Avoids per-element tf.convert_to_tensor — builds one big numpy
    array then converts once.
    Args:
        ragged_arrays: list of np.ndarray each (L_i, 21)
        pad_token:     int value for padding positions in mask
        mask_token:    int value for masked positions in mask
        masking_rate:  fraction of valid positions to mask
    Returns:
        features: np.ndarray (B, max_L, 21) float32
        mask:     np.ndarray (B, max_L)     float32
    """
    lengths = np.array([a.shape[0] for a in ragged_arrays], dtype=np.int32)
    B = len(ragged_arrays)
    max_L = int(lengths.max()) if B > 0 else 0
    # Pre-allocate padded arrays
    features = np.zeros((B, max_L, 21), dtype=np.float32)
    mask = np.full((B, max_L), float(pad_token), dtype=np.float32)
    # Fill valid positions (vectorised per-row copy)
    for i, (arr, L) in enumerate(zip(ragged_arrays, lengths)):
        features[i, :L, :] = arr
        mask[i, :L] = 1.0  # NORMAL_TOKEN
    # Random masking
    if masking_rate > 0.0:
        rand = np.random.random((B, max_L)).astype(np.float32)
        # valid_mask: True where not pad
        valid = mask > 0.0
        is_masked = (rand < masking_rate) & valid
        mask[is_masked] = float(mask_token)
        features[is_masked] = 0.0
    return features, mask
def _concat_cdrs_with_sep_numpy(feat_list, mask_list, sep_mask_val=-3):
    """Concatenate CDR features+masks with separator columns, all in numpy.
    Much faster than calling the TF version which creates intermediate tensors.
    Args:
        feat_list: list of (B, L_k, 21) numpy arrays
        mask_list: list of (B, L_k) numpy arrays
        sep_mask_val: mask value for separator token
    Returns:
        combined_feat: (B, total_L, 21) numpy float32
        combined_mask: (B, total_L)     numpy float32
    """
    B = feat_list[0].shape[0]
    sep_feat = np.zeros((B, 1, 21), dtype=np.float32)
    sep_mask = np.full((B, 1), float(sep_mask_val), dtype=np.float32)
    parts_f, parts_m = [], []
    for i, (f, m) in enumerate(zip(feat_list, mask_list)):
        parts_f.append(f)
        parts_m.append(m)
        if i < len(feat_list) - 1:  # separator between regions
            parts_f.append(sep_feat)
            parts_m.append(sep_mask)
    return np.concatenate(parts_f, axis=1), np.concatenate(parts_m, axis=1)
def prepare_batch(chunk, args) -> Dict[str, tf.Tensor]:
    """
    Convert a PublicTcrHlaClusterChunk into padded tensors for one training step.
    All heavy lifting is done in numpy; tf.constant is called once per tensor
    at the end to minimise Python↔TF overhead.
    Returns a dict with keys:
      combined_cdr   (B, L, 21) float32
      combined_mask  (B, L)     int32
      binder_dense   (B, A)     float32
      donor_indices  (B, P)     int32
    """
    # ── donor indices (variable-length → padded) ─────────────────────
    donor_lists = split_ragged_to_list(
        chunk.raw_csr_donor_indices, chunk.raw_csr_donor_indptr)
    donor_pad, _ = pad_list_to_array_without_max(donor_lists, args.pad_token)
    donor_pad = donor_pad.astype(np.int32)
    # ── CDR frequency profiles → pad + mask each region (all numpy) ──
    cdr1, m1 = _pad_and_mask_numpy(
        chunk.cdr_freq["cdr1"][:], args.pad_token, args.mask_token,
        args.masking_rate)
    cdr2, m2 = _pad_and_mask_numpy(
        chunk.cdr_freq["cdr2"][:], args.pad_token, args.mask_token,
        args.masking_rate)
    cdr25, m25 = _pad_and_mask_numpy(
        chunk.cdr_freq["cdr25"][:], args.pad_token, args.mask_token,
        args.masking_rate)
    cdr3, m3 = _pad_and_mask_numpy(
        chunk.cdr_freq["cdr3"][:], args.pad_token, args.mask_token,
        args.masking_rate)
    # ── concatenate with separator tokens (all numpy) ────────────────
    combined_np, mask_np = _concat_cdrs_with_sep_numpy(
        [cdr1, cdr2, cdr25, cdr3], [m1, m2, m25, m3],
        sep_mask_val=args.sep_token)
    # ── binder dense set: binary mask over alleles ───────────────────
    binder_np = (chunk.counts_dense > 0).astype(np.float32)
    # ── single tf.constant call per tensor (minimal TF overhead) ─────
    return {
        "combined_cdr":  tf.constant(combined_np,  dtype=tf.float32),
        "combined_mask": tf.constant(mask_np.astype(np.int32), dtype=tf.int32),
        "binder_dense":  tf.constant(binder_np,    dtype=tf.float32),
        "donor_indices": tf.constant(donor_pad,    dtype=tf.int32),
    }
# ═════════════════════════════════════════════════════════════════════
# 3b. TFRECORD CACHE — one-time HDF5 → sharded TFRecords conversion
# ═════════════════════════════════════════════════════════════════════
# Storing pre-processed tensors as TFRecords lets us bypass HDF5/scipy/
# numpy padding on every epoch. tf.data reads TFRecords in pure C++
# with true multi-threaded parallelism and built-in prefetch.
#
# Layout per example (one cluster):
#   combined_cdr_flat  : float32 VarLen  (L*21 flattened)
#   combined_mask      : int64   VarLen  (L,)
#   binder_dense       : float32 FixedLen (A,)
#   donor_indices      : int64   VarLen  (P,)
#   seq_len            : int64   FixedLen scalar  (L, for reshaping)
# ═════════════════════════════════════════════════════════════════════
def _serialize_cluster(combined_cdr, combined_mask, binder_dense,
                       donor_indices):
    """Serialize one cluster into a tf.train.Example protobuf.
    Args:
        combined_cdr:   np.ndarray (L, 21) float32
        combined_mask:  np.ndarray (L,)    int32
        binder_dense:   np.ndarray (A,)    float32
        donor_indices:  np.ndarray (P,)    int32
    Returns:
        bytes: serialised Example
    """
    L = combined_cdr.shape[0]
    feat = {
        "combined_cdr_flat": tf.train.Feature(
            float_list=tf.train.FloatList(
                value=combined_cdr.flatten().tolist())),
        "combined_mask": tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=combined_mask.astype(np.int64).tolist())),
        "binder_dense": tf.train.Feature(
            float_list=tf.train.FloatList(
                value=binder_dense.tolist())),
        "donor_indices": tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=donor_indices.astype(np.int64).tolist())),
        "seq_len": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[L])),
    }
    return tf.train.Example(
        features=tf.train.Features(feature=feat)).SerializeToString()
def convert_h5_to_tfrecords(h5_path: str, output_dir: str, args,
                            num_shards: int = 16,
                            tag: str = "train") -> List[str]:
    """One-time conversion: stream HDF5 → sharded TFRecord files.
    Each shard stores individual clusters (not batches) so tf.data
    can shuffle and batch them freely.
    Clusters with n_donors < args.keep_only_upperthan_n_donors are
    skipped (filtered out) and not written to any shard.
    If args.tf_record_path is set, TFRecords are written there instead
    of the default <output_dir>/tfrecord_cache_<tag>/.
    Args:
        h5_path:    path to PublicTcrHlaCsrReaderChunk-compatible H5
        output_dir: directory to write shard files into (fallback)
        args:       parsed flags (pad/mask/sep tokens, masking_rate=0 for cache,
                    keep_only_upperthan_n_donors, tf_record_path)
        num_shards: number of output files for parallel reads
        tag:        prefix for shard filenames (train / valid)
    Returns:
        list of shard file paths
    """
    # ── resolve TFRecord base directory ──────────────────────────────
    # Use custom tf_record_path if provided, otherwise default to output_dir
    tfr_base = args.tf_record_path if args.tf_record_path else output_dir
    tfr_dir = os.path.join(tfr_base, f"tfrecord_cache_{tag}")
    os.makedirs(tfr_dir, exist_ok=True)
    # ── minimum donor threshold for filtering ────────────────────────
    min_donors = int(args.keep_only_upperthan_n_donors)
    # Check if already converted (idempotent)
    manifest_path = os.path.join(tfr_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        # Reuse only if source H5 AND min_donors threshold match
        if (manifest.get("source_h5") == h5_path and
                manifest.get("min_donors", 1) == min_donors):
            print(f"  [CACHE] Reusing existing TFRecords in {tfr_dir} "
                  f"({manifest['num_clusters']} clusters, "
                  f"{manifest['num_shards']} shards, "
                  f"min_donors={min_donors})")
            return manifest["shard_paths"]
    print(f"  [CACHE] Converting {h5_path} → TFRecords in {tfr_dir} ...")
    if min_donors > 1:
        print(f"  [CACHE] Filtering: keeping only clusters with "
              f"n_donors >= {min_donors}")
    t0 = time.time()
    # Open shard writers
    shard_paths = [os.path.join(tfr_dir, f"{tag}_{i:04d}.tfrecord")
                   for i in range(num_shards)]
    writers = [tf.io.TFRecordWriter(p) for p in shard_paths]
    cluster_count = 0   # clusters actually written (after filtering)
    skipped_count = 0   # clusters skipped due to donor threshold
    shard_idx = 0
    # Read HDF5 in large chunks for speed, write individual examples
    with PublicTcrHlaCsrReaderChunk(
            h5_path, include_counts=True, include_donors=True,
            include_pvals=False, include_cdr_freq=True) as reader:
        
        for chunk in reader.iter_cluster_chunks(chunk_rows=10000):
            if chunk.counts_dense is None or chunk.cdr_freq is None:
                continue
            
            B = chunk.counts_dense.shape[0]
            
            # 1. Vectorized Filtering
            keep_mask = chunk.n_donors >= min_donors
            valid_indices = np.where(keep_mask)[0]
            
            if len(valid_indices) == 0:
                skipped_count += B
                continue
            
            skipped_count += (B - len(valid_indices))
            
            # 2. Fast Vectorized Numpy Prep (Process the whole chunk at once)
            donor_lists = split_ragged_to_list(
                chunk.raw_csr_donor_indices, chunk.raw_csr_donor_indptr)
            donor_pad, _ = pad_list_to_array_without_max(donor_lists, args.pad_token)
            
            cdr1, m1 = _pad_and_mask_numpy(chunk.cdr_freq["cdr1"][:], args.pad_token, args.mask_token, 0.0)
            cdr2, m2 = _pad_and_mask_numpy(chunk.cdr_freq["cdr2"][:], args.pad_token, args.mask_token, 0.0)
            cdr25, m25 = _pad_and_mask_numpy(chunk.cdr_freq["cdr25"][:], args.pad_token, args.mask_token, 0.0)
            cdr3, m3 = _pad_and_mask_numpy(chunk.cdr_freq["cdr3"][:], args.pad_token, args.mask_token, 0.0)
            
            combined_np, mask_np = _concat_cdrs_with_sep_numpy(
                [cdr1, cdr2, cdr25, cdr3], [m1, m2, m25, m3], sep_mask_val=args.sep_token)
                
            binder_np = (chunk.counts_dense > 0).astype(np.float32)
            
            # 3. Apply the valid indices filter all at once
            combined_np = combined_np[valid_indices]
            mask_np = mask_np[valid_indices]
            binder_np = binder_np[valid_indices]
            donor_pad = donor_pad[valid_indices]
            
            # 4. Calculate actual unpadded lengths (Vectorized)
            seq_lens = np.sum(mask_np != args.pad_token, axis=1)
            donor_lens = np.sum(donor_pad != args.pad_token, axis=1)
            
            # 5. Fast Loop for Protobuf Serialization
            for i in range(len(valid_indices)):
                s_len = seq_lens[i]
                d_len = donor_lens[i]
                
                # Instantly strip padding using array slicing
                actual_cdr = combined_np[i, :s_len]
                actual_mask = mask_np[i, :s_len]
                actual_donors = donor_pad[i, :d_len]
                
                serialized = _serialize_cluster(
                    actual_cdr, actual_mask, binder_np[i], actual_donors)
                
                writers[shard_idx % num_shards].write(serialized)
                shard_idx += 1
                cluster_count += 1
    # Close writers
    for w in writers:
        w.close()
    elapsed = time.time() - t0
    # Write manifest (includes min_donors for idempotency check)
    manifest = {
        "source_h5": h5_path,
        "num_clusters": cluster_count,
        "num_clusters_skipped": skipped_count,
        "num_shards": num_shards,
        "num_alleles": args.num_alleles,
        "shard_paths": shard_paths,
        "pad_token": args.pad_token,
        "sep_token": args.sep_token,
        "min_donors": min_donors,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [CACHE] Wrote {cluster_count} clusters to {num_shards} shards "
          f"in {elapsed:.1f}s ({cluster_count/max(elapsed,1e-6):.0f} clusters/s)")
    if skipped_count > 0:
        print(f"  [CACHE] Skipped {skipped_count} clusters with "
              f"n_donors < {min_donors}")
    return shard_paths
def build_tfrecord_dataset(shard_paths: List[str], args,
                           num_alleles: int,
                           shuffle: bool = True,
                           drop_remainder: bool = False) -> tf.data.Dataset:
    """Build a tf.data pipeline from sharded TFRecords.
    The pipeline runs entirely in C++ (no Python GIL):
      file interleave → parse → optional masking → shuffle → batch → pad → prefetch
    Args:
        shard_paths:  list of .tfrecord file paths
        args:         parsed flags
        num_alleles:  A dimension for binder_dense
        shuffle:      whether to shuffle
        drop_remainder: drop last incomplete batch
    Returns:
        tf.data.Dataset yielding (combined_cdr, combined_mask,
                                   binder_dense, donor_indices)
    """
    # ── file-level interleave for parallel reads ─────────────────────
    files_ds = tf.data.Dataset.from_tensor_slices(shard_paths)
    if shuffle:
        files_ds = files_ds.shuffle(len(shard_paths))
    dataset = files_ds.interleave(
        lambda f: tf.data.TFRecordDataset(f),
        cycle_length=min(8, len(shard_paths)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle)
    # ── parse function ───────────────────────────────────────────────
    def _parse(example_proto):
        """Parse one Example → (cdr_flat, mask, binder, donors, seq_len)."""
        desc = {
            "combined_cdr_flat": tf.io.VarLenFeature(tf.float32),
            "combined_mask":     tf.io.VarLenFeature(tf.int64),
            "binder_dense":      tf.io.FixedLenFeature([num_alleles], tf.float32),
            "donor_indices":     tf.io.VarLenFeature(tf.int64),
            "seq_len":           tf.io.FixedLenFeature([], tf.int64),
        }
        parsed = tf.io.parse_single_example(example_proto, desc)
        seq_len = parsed["seq_len"]
        # Reconstruct 2D CDR tensor from flat storage
        cdr_flat = tf.sparse.to_dense(parsed["combined_cdr_flat"])
        cdr = tf.reshape(cdr_flat, [seq_len, 21])
        mask = tf.cast(tf.sparse.to_dense(parsed["combined_mask"]), tf.int32)
        binder = parsed["binder_dense"]
        donors = tf.cast(tf.sparse.to_dense(parsed["donor_indices"]), tf.int32)
        return cdr, mask, binder, donors
    dataset = dataset.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    # ── online random masking (only during training) ─────────────────
    if args.masking_rate > 0 and shuffle:
        @tf.function
        def _apply_masking(cdr, mask, binder, donors):
            """Apply random masking to CDR features at train time."""
            L = tf.shape(cdr)[0]
            valid = tf.not_equal(mask, args.pad_token)
            rand = tf.random.uniform([L], 0.0, 1.0)
            is_masked = tf.logical_and(rand < args.masking_rate, valid)
            # Zero out masked positions in features
            mask_expand = tf.expand_dims(
                tf.cast(tf.logical_not(is_masked), tf.float32), -1)
            cdr = cdr * mask_expand
            # Update mask values
            mask = tf.where(is_masked, args.mask_token, mask)
            return cdr, mask, binder, donors
        dataset = dataset.map(_apply_masking,
                              num_parallel_calls=tf.data.AUTOTUNE)
    # ── shuffle ──────────────────────────────────────────────────────
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(250000, 100 * args.batch_size), reshuffle_each_iteration=True)
    # ── batch with dynamic padding ───────────────────────────────────
    # Pad CDR and mask to max-in-batch length, donors to max-in-batch count
    pad_shapes = (
        [None, 21],    # combined_cdr:  (L, 21)
        [None],        # combined_mask: (L,)
        [num_alleles], # binder_dense:  (A,) — fixed
        [None],        # donor_indices: (P,)
    )
    pad_values = (
        tf.constant(0.0, tf.float32),         # CDR pad = zeros
        tf.constant(args.pad_token, tf.int32), # mask pad
        tf.constant(0.0, tf.float32),          # binder pad (unused — fixed)
        tf.constant(args.pad_token, tf.int32), # donor pad
    )
    dataset = dataset.padded_batch(
        args.batch_size, padded_shapes=pad_shapes, padding_values=pad_values,
        drop_remainder=drop_remainder)
    # ── prefetch into GPU memory ─────────────────────────────────────
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
def count_dataset_clusters(h5_path: str, batch_size: int) -> Tuple[int, int]:
    """Count total clusters and estimate number of steps per epoch."""
    with PublicTcrHlaCsrReaderChunk(h5_path, include_counts=False,
                                     include_donors=False) as r:
        n = r.num_clusters
    steps = max(1, math.ceil(n / batch_size))
    return n, steps
def _get_git_hash() -> str:
    """Return short git commit hash, or 'unknown' if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"
def _get_gpu_info() -> List[Dict[str, str]]:
    """Return list of dicts with GPU name and memory for each device."""
    gpus = tf.config.list_physical_devices("GPU")
    info = []
    for gpu in gpus:
        try:
            details = tf.config.experimental.get_device_details(gpu)
            info.append({
                "name": details.get("device_name", gpu.name),
                "compute_capability": str(
                    details.get("compute_capability", "?")),
            })
        except Exception:
            info.append({"name": gpu.name})
    return info
def save_config(args, output_dir: str, device_type: str,
                donor_shape: Tuple[int, ...],
                n_train_clusters: int = 0,
                n_valid_clusters: int = 0) -> str:
    """Write a comprehensive config.json capturing everything needed
    to reproduce the experiment.
    Sections:
      experiment  — timestamp, git hash, command line
      hardware    — device, GPU names, mixed precision
      seed        — random seed (Python, NumPy, TF)
      data        — paths, shapes, dataset sizes
      model       — architecture hyperparameters
      loss        — beta, l2, reduction, poisson approx
      training    — lr, schedule, batch size, epochs, clipping
      tokens      — pad / mask / sep / normal values
    Returns:
      path to the written config.json
    """
    config = {
        "experiment": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git_hash": _get_git_hash(),
            "command": " ".join(sys.argv),
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "keras_version": keras.__version__,
            "numpy_version": np.__version__,
            "hostname": platform.node(),
        },
        "hardware": {
            "device_type": device_type,
            "gpus": _get_gpu_info(),
            "mixed_precision": args.mixed_precision,
            "gpu_memory_limit_mb": args.gpu_memory_limit,
        },
        "seed": {
            "value": args.seed,
            "deterministic_ops": True,
            "note": "Applied to random, numpy, tensorflow, PYTHONHASHSEED",
        },
        "data": {
            "train_ds": args.train_ds,
            "valid_ds": args.valid_ds,
            "inference_ds": args.inference_ds,
            "donor_hla_matrix": args.donor_hla_matrix,
            "donor_hla_matrix_shape": list(donor_shape),
            "idx_to_hla": args.idx_to_hla,
            "hla_embed": args.hla_embed,
            "num_alleles": args.num_alleles,
            "n_train_clusters": n_train_clusters,
            "n_valid_clusters": n_valid_clusters,
        },
        "model": {
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "ff_dim": args.ff_dim if args.ff_dim > 0 else args.embed_dim * 4,
            "hla_proj_dim": args.hla_proj_dim,
            "max_seq_len": args.max_seq_len,
            "encoding_mode": args.encoding_mode,
            "resnet": args.resnet,
            "dropout": args.dropout,
            "train_hla_head": args.train_hla_head,
        },
        "loss": {
            "beta": args.beta,
            "l2_reg": args.l2_reg,
            "reduction": args.reduction,
            "poisson_approx": args.poisson_approx,
        },
        "training": {
            "mode": args.mode,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "masking_rate": args.masking_rate,
            "log_step": args.log_step,
            "save_every_epoch": args.save_every_epoch,
            "patience": args.patience,
            "resume": args.resume,
            "use_tfrecord": args.use_tfrecord,
            "num_shards": args.num_shards,
            "keep_only_upperthan_n_donors": args.keep_only_upperthan_n_donors,
            "tf_record_path": args.tf_record_path,
        },
        "tokens": {
            "pad": args.pad_token,
            "mask": args.mask_token,
            "sep": args.sep_token,
            "normal": args.normal_token,
        },
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, cls=NumpyEncoder)
    print(f"[CONFIG] Saved → {config_path}")
    return config_path
# ═════════════════════════════════════════════════════════════════════
# 4.  MODEL CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════
class MaskedGlobalAveragePooling(layers.Layer):
    """Global average pooling that ignores PAD positions.
    Input:  (B, L, D) features  +  (B, L) int mask
    Output: (B, D)  pooled representation
    """
    def __init__(self, pad_token: int = -1, **kw):
        super().__init__(**kw)
        self.pad_token = pad_token
    def call(self, inputs):
        x, mask = inputs  # (B,L,D), (B,L)
        # valid = 1 where not pad, 0 where pad
        valid = tf.cast(tf.not_equal(mask, self.pad_token), tf.float32)
        valid_3d = valid[:, :, tf.newaxis]  # (B,L,1)
        # zero-out pad positions and sum
        x_masked = x * valid_3d  # (B,L,D)
        summed = tf.reduce_sum(x_masked, axis=1)  # (B,D)
        # count valid positions per sample
        counts = tf.reduce_sum(valid, axis=1, keepdims=True)  # (B,1)
        counts = tf.maximum(counts, 1.0)  # avoid div-by-zero
        return summed / counts  # (B,D)
    def get_config(self):
        cfg = super().get_config()
        cfg["pad_token"] = self.pad_token
        return cfg
def build_model(args, hla_embed_matrix: Optional[np.ndarray] = None, hla_bias_init: Optional[np.ndarray] = None):
    """
    Build the TCRtyper Keras functional model.
    Architecture:
      CDR freq (B,L,21) + mask (B,L)
        → SequenceEncoderLayer  → (B,L,E)
        → N × GatedTransformerLayer  → (B,L,E)
        → MaskedGlobalAveragePooling → (B,E)
        → Dense(hla_proj_dim) → Dropout → Dense(A)  → z_logits (B,A)
    If hla_embed_matrix is provided the final Dense(A) is initialised
    with HLA pseudo-sequence embeddings (optionally frozen).
    """
    ff_dim = args.ff_dim if args.ff_dim > 0 else args.embed_dim * 4
    # ── inputs ───────────────────────────────────────────────────────
    inp_seq = keras.Input(shape=(None, 21), name="seq_input")
    inp_mask = keras.Input(shape=(None,), dtype=tf.int32, name="mask_input")
    # ── sequence encoder ─────────────────────────────────────────────
    h = SequenceEncoderLayer(
        embed_dim=args.embed_dim, max_len=args.max_seq_len,
        dropout_rate=args.dropout, encoding_mode=args.encoding_mode,
        pad_token=args.pad_token, mask_token=args.mask_token,
        sep_token=args.sep_token, normal_token=args.normal_token,
        name="seq_encoder",
    )([inp_seq, inp_mask])
    # ── transformer stack ────────────────────────────────────────────
    for i in range(args.num_layers):
        h = GatedTransformerLayer(
            embed_dim=args.embed_dim, num_heads=args.num_heads,
            ff_dim=ff_dim, resnet=args.resnet, dropout_rate=args.dropout,
            pad_token=args.pad_token, mask_token=args.mask_token,
            sep_token=args.sep_token, normal_token=args.normal_token,
            name=f"transformer_{i}",
        )([h, inp_mask])
    # ── masked global average pooling → (B, E) ──────────────────────
    h = MaskedGlobalAveragePooling(
        pad_token=args.pad_token, name="pool"
    )([h, inp_mask])
    # ── HLA projection head ──────────────────────────────────────────
    h = layers.Dense(args.hla_proj_dim, activation="gelu",
                     name="hla_proj")(h)
    h = layers.Dropout(args.dropout, name="hla_drop")(h)
    # Final layer: project to A alleles
    # Initialise with HLA embeddings if available
    if hla_embed_matrix is not None:
        # hla_embed_matrix shape expected: (hla_proj_dim, A)
        # Dense kernel shape: (hla_proj_dim, A)
        hla_init = tf.keras.initializers.Constant(hla_embed_matrix)
    else:
        hla_init = "glorot_uniform"
    # Apply the log-odds as the bias initializer
    if hla_bias_init is not None:
        bias_init = tf.keras.initializers.Constant(hla_bias_init)
    else:
        bias_init = "zeros"
    z_logits = layers.Dense(
        args.num_alleles, kernel_initializer=hla_init, bias_initializer=bias_init, name="hla_head")(h)
    # ── assemble model ───────────────────────────────────────────────
    model = keras.Model(inputs=[inp_seq, inp_mask], outputs=z_logits,
                        name="TCRtyper")
    # ── optionally freeze the HLA head ───────────────────────────────
    if not args.train_hla_head:
        model.get_layer("hla_head").trainable = False
    return model
# ═════════════════════════════════════════════════════════════════════
# 5.  LEARNING RATE SCHEDULE
# ═════════════════════════════════════════════════════════════════════
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay to min_lr.
    lr(step) =
      step < warmup : lr * step / warmup
      step >= warmup : min_lr + 0.5*(lr - min_lr)*(1 + cos(pi * progress))
    """
    def __init__(self, peak_lr, min_lr, warmup_steps, total_steps):
        super().__init__()
        self.peak_lr = tf.cast(peak_lr, tf.float32)
        self.min_lr = tf.cast(min_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps = tf.cast(max(total_steps, warmup_steps + 1), tf.float32)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # linear warmup phase
        warmup_lr = self.peak_lr * (step / tf.maximum(self.warmup_steps, 1.0))
        # cosine decay phase
        progress = (step - self.warmup_steps) / tf.maximum(
            self.total_steps - self.warmup_steps, 1.0)
        progress = tf.minimum(progress, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)
    def get_config(self):
        return {"peak_lr": float(self.peak_lr.numpy()),
                "min_lr": float(self.min_lr.numpy()),
                "warmup_steps": int(self.warmup_steps.numpy()),
                "total_steps": int(self.total_steps.numpy())}
# ═════════════════════════════════════════════════════════════════════
# 6.  TRAINING STEP (tf.function compiled)
# ═════════════════════════════════════════════════════════════════════
@tf.function(reduce_retracing=True)
def train_step(model, loss_fn, optimizer,
               combined_cdr, combined_mask, binder_dense, donor_indices,
               grad_clip):
    """
    Single training step: forward + backward + apply gradients.
    All tensor args have fixed rank but variable first/second dims,
    reduce_retracing=True lets TF generalise across shapes after
    one or two traces instead of retracing every batch.
    Args:
        combined_cdr:   (B, L, 21) float32
        combined_mask:  (B, L)     int32
        binder_dense:   (B, A)     float32
        donor_indices:  (B, P)     int32
        grad_clip:      scalar     float32
    Returns dict of scalar metrics.
    """
    with tf.GradientTape() as tape:
        z_logits = model([combined_cdr, combined_mask], training=True)
        nll, reg = loss_fn(z_logits, binder_dense, donor_indices)
        total_loss = nll + reg
        if hasattr(optimizer, "get_scaled_loss"):
            scaled_loss = optimizer.get_scaled_loss(total_loss)
        else:
            scaled_loss = total_loss
    trainable_vars = model.trainable_variables
    if hasattr(optimizer, "get_scaled_loss"):
        scaled_grads = tape.gradient(scaled_loss, trainable_vars)
        grads = optimizer.get_unscaled_gradients(scaled_grads)
    else:
        grads = tape.gradient(total_loss, trainable_vars)
    if grad_clip > 0:
        grads, grad_norm = tf.clip_by_global_norm(grads, grad_clip)
    else:
        grad_norm = tf.linalg.global_norm(grads)
    optimizer.apply_gradients(zip(grads, trainable_vars))
    diag = compute_diagnostics(z_logits, binder_dense)
    return {"total_loss": total_loss, "nll": nll, "reg": reg,
            "grad_norm": grad_norm, **diag}
# ═════════════════════════════════════════════════════════════════════
# 7.  VALIDATION STEP
# ═════════════════════════════════════════════════════════════════════
@tf.function(reduce_retracing=True)
def compute_diagnostics(z_logits, binder_dense):
    """Vectorised diagnostic metrics shared by train and eval steps.
    All ops are pure TF — no Python loops.
    Args:
        z_logits:     (B, A) raw model output (logits)
        binder_dense: (B, A) binary co-occurrence mask
    Returns:
        dict of scalar tf.Tensors
    """
    eps = 1e-7
    z_prob = tf.sigmoid(z_logits)
    active_mask = tf.cast(binder_dense > 0.5, tf.float32)
    n_active = tf.reduce_sum(active_mask)
    # mean gamma over active alleles
    mean_active_gamma = tf.reduce_sum(z_prob * active_mask) / tf.maximum(n_active, 1.0)
    # max gamma per sample (over active alleles), then average across batch
    max_gamma_per_sample = tf.reduce_max(
        z_prob * active_mask + (1.0 - active_mask) * -1e9, axis=-1)
    mean_max_gamma = tf.reduce_mean(max_gamma_per_sample)
    # binary entropy of gamma over active positions
    p_safe = tf.clip_by_value(z_prob, eps, 1.0 - eps)
    ent = -(p_safe * tf.math.log(p_safe) +
            (1.0 - p_safe) * tf.math.log(1.0 - p_safe))
    entropy_active = tf.reduce_sum(ent * active_mask) / tf.maximum(n_active, 1.0)
    # top-1 vs top-2 logit gap over active alleles
    masked_logits = z_logits * active_mask + (1.0 - active_mask) * -1e9
    top2_vals, _ = tf.math.top_k(masked_logits, k=2)
    logit_gap = tf.reduce_mean(top2_vals[:, 0] - top2_vals[:, 1])
    # min non-zero prob among active alleles per sample
    probs_for_min = z_prob + (1.0 - active_mask) * 1e9
    min_active_prob = tf.reduce_mean(tf.reduce_min(probs_for_min, axis=-1))
    # mean number of active alleles per sample
    mean_n_active = tf.reduce_mean(tf.reduce_sum(active_mask, axis=-1))
    return {
        "mean_active_gamma": mean_active_gamma,
        "mean_max_gamma": mean_max_gamma,
        "entropy_active": entropy_active,
        "logit_gap_top1_top2": logit_gap,
        "min_active_prob": min_active_prob,
        "mean_n_active_alleles": mean_n_active,
    }
@tf.function(reduce_retracing=True)
def eval_step(model, loss_fn,
              combined_cdr, combined_mask, binder_dense, donor_indices):
    """Forward-only evaluation step matching train_step's tensor signature."""
    z_logits = model([combined_cdr, combined_mask], training=False)
    nll, reg = loss_fn(z_logits, binder_dense, donor_indices)
    total_loss = nll + reg
    diag = compute_diagnostics(z_logits, binder_dense)
    return {"total_loss": total_loss, "nll": nll, "reg": reg, **diag}
# ═════════════════════════════════════════════════════════════════════
# 8.  METRIC LOGGER (CSV + JSON + TensorBoard)
# ═════════════════════════════════════════════════════════════════════
import csv
class MetricLogger:
    """Writes every epoch's averaged metrics to CSV, JSON, and TensorBoard.
    Args:
        output_dir: root experiment directory
        tb_writer:  tf.summary.FileWriter for TensorBoard
    """
    def __init__(self, output_dir: str, tb_writer):
        self.csv_path = os.path.join(output_dir, "metrics.csv")
        self.json_path = os.path.join(output_dir, "metrics.json")
        self.tb_writer = tb_writer
        self._history: List[Dict] = []
        self._csv_file = None
        self._csv_writer = None
        self._header_written = False
    def log_epoch(self, epoch: int, split: str, metrics: Dict,
                  extra: Optional[Dict] = None):
        """Log one epoch's metrics to CSV, JSON, and TensorBoard.
        Args:
            epoch:   0-based epoch index
            split:   'train' or 'valid'
            metrics: dict of metric_name -> float value
            extra:   optional extra scalars (e.g. lr, epoch_time)
        """
        row = {"epoch": epoch, "split": split, **metrics}
        if extra:
            row.update(extra)
        self._history.append(row)
        # ── CSV (append) ─────────────────────────────────────────────
        if not self._header_written:
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()
            self._header_written = True
        else:
            # Re-open in append if columns haven't changed
            if self._csv_file is None or self._csv_file.closed:
                self._csv_file = open(self.csv_path, "a", newline="")
                self._csv_writer = csv.DictWriter(
                    self._csv_file, fieldnames=list(row.keys()))
        self._csv_writer.writerow(row)
        self._csv_file.flush()
        # ── JSON (overwrite full history) ────────────────────────────
        with open(self.json_path, "w") as f:
            json.dump(self._history, f, indent=2, cls=NumpyEncoder)
        # ── TensorBoard ─────────────────────────────────────────────
        with self.tb_writer.as_default(step=epoch):
            for k, v in metrics.items():
                tf.summary.scalar(f"{split}_epoch/{k}", v)
            if extra:
                for k, v in extra.items():
                    tf.summary.scalar(f"{split}_epoch/{k}", v)
    def close(self):
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.close()
# ═════════════════════════════════════════════════════════════════════
# 9.  EPOCH RUNNERS  (with background prefetch)
# ═════════════════════════════════════════════════════════════════════
import threading, queue
def _prefetch_batches(reader, args, prefetch_q, max_prefetch=3):
    """Background thread: reads HDF5 chunks, prepares numpy→tf batches,
    and pushes them into a queue.  The training loop pops from the queue
    so GPU never waits for data prep.
    Clusters with n_donors < args.keep_only_upperthan_n_donors are
    filtered out via vectorised boolean indexing after batch preparation.
    Args:
        reader:       opened PublicTcrHlaCsrReaderChunk context
        args:         parsed flags (includes keep_only_upperthan_n_donors)
        prefetch_q:   queue.Queue shared with main thread
        max_prefetch: queue capacity (limits CPU memory usage)
    """
    # ── cache the donor threshold for fast access ────────────────────
    min_donors = int(args.keep_only_upperthan_n_donors)
    for chunk in reader.iter_cluster_chunks(chunk_rows=args.batch_size):
        if chunk.counts_dense is None or chunk.counts_dense.shape[0] == 0:
            continue
        if chunk.cdr_freq is None:
            continue
        # ── donor-count filtering (vectorised boolean mask) ──────────
        if min_donors > 1:
            keep_mask = chunk.n_donors >= min_donors
            # Skip entire chunk if no cluster passes the threshold
            if not np.any(keep_mask):
                continue
            # If all pass, no filtering needed — fall through
            if not np.all(keep_mask):
                # Build batch first, then gather only kept rows
                batch = prepare_batch(chunk, args)
                keep_idx = tf.constant(
                    np.where(keep_mask)[0], dtype=tf.int32)
                batch = {k: tf.gather(v, keep_idx)
                         for k, v in batch.items()}
                prefetch_q.put(batch)  # blocks if queue is full
                continue
        # No filtering needed (all clusters pass or threshold is 1)
        batch = prepare_batch(chunk, args)
        prefetch_q.put(batch)  # blocks if queue is full
    prefetch_q.put(None)  # sentinel: end of epoch
def run_epoch(model, loss_fn, optimizer, h5_path, args,
              tb_writer, global_step, epoch, is_train=True):
    """
    Stream through the H5 file with background prefetching.
    A separate thread reads HDF5 + runs numpy data prep while the main
    thread executes GPU forward/backward passes, so the two overlap.
    Returns (mean_metrics_dict, updated_global_step).
    """
    tag = "train" if is_train else "valid"
    accum = {}  # accumulate metric sums
    n_steps = 0
    grad_clip_t = tf.constant(args.grad_clip, tf.float32)  # create once
    prefetch_q = queue.Queue(maxsize=3)  # limit CPU-ahead to 3 batches
    reader = PublicTcrHlaCsrReaderChunk(
        h5_path, include_counts=True, include_donors=True,
        include_pvals=False, include_cdr_freq=True)
    reader.open()
    # Start background data-prep thread
    loader_thread = threading.Thread(
        target=_prefetch_batches,
        args=(reader, args, prefetch_q),
        daemon=True)
    loader_thread.start()
    # Main loop: pop pre-built batches from queue
    while True:
        batch = prefetch_q.get()
        if batch is None:  # sentinel — epoch done
            break
        cdr  = batch["combined_cdr"]
        mask = batch["combined_mask"]
        bind = batch["binder_dense"]
        dids = batch["donor_indices"]
        if is_train:
            metrics = train_step(
                model, loss_fn, optimizer,
                cdr, mask, bind, dids, grad_clip_t)
        else:
            metrics = eval_step(
                model, loss_fn, cdr, mask, bind, dids)
        n_steps += 1
        # Accumulate
        for k, v in metrics.items():
            val = float(v.numpy()) if hasattr(v, "numpy") else float(v)
            accum[k] = accum.get(k, 0.0) + val
        # TensorBoard per-step logging (train only)
        if is_train and n_steps % args.log_step == 0:
            with tb_writer.as_default(step=global_step):
                for k, v in metrics.items():
                    tf.summary.scalar(f"{tag}_step/{k}", v)
                current_lr = optimizer.learning_rate
                if callable(current_lr):
                    current_lr = current_lr(optimizer.iterations)
                tf.summary.scalar(f"{tag}_step/learning_rate", current_lr)
            print(f"  [{tag}] epoch {epoch} step {n_steps} | "
                  f"loss={float(metrics['total_loss']):.4f} "
                  f"nll={float(metrics['nll']):.4f} "
                  f"reg={float(metrics['reg']):.6f} "
                  f"gnorm={float(metrics.get('grad_norm', 0)):.3f}")
        if is_train:
            global_step += 1
    # Wait for loader thread to finish and close reader
    loader_thread.join()
    reader.close()
    # Epoch-level averages
    if n_steps == 0:
        print(f"  [{tag}] WARNING: 0 steps processed — check dataset")
        return {}, global_step
    means = {k: v / n_steps for k, v in accum.items()}
    print(f"  [{tag}] epoch {epoch} done | steps={n_steps} | "
          + " | ".join(f"{k}={v:.5f}" for k, v in means.items()))
    return means, global_step
def run_epoch_tfrecord(model, loss_fn, optimizer, dataset, args,
                       tb_writer, global_step, epoch, is_train=True):
    """Run one epoch from a pre-built tf.data.Dataset (TFRecord pipeline).
    Entire I/O + parsing + batching + prefetch runs in C++ threads —
    GPU is fed continuously with zero Python overhead in the data path.
    Returns (mean_metrics_dict, updated_global_step).
    """
    tag = "train" if is_train else "valid"
    accum = {}
    n_steps = 0
    grad_clip_t = tf.constant(args.grad_clip, tf.float32)
    for cdr, mask, binder, donors in dataset:
        if is_train:
            metrics = train_step(
                model, loss_fn, optimizer,
                cdr, mask, binder, donors, grad_clip_t)
        else:
            metrics = eval_step(model, loss_fn, cdr, mask, binder, donors)
        n_steps += 1
        for k, v in metrics.items():
            val = float(v.numpy()) if hasattr(v, "numpy") else float(v)
            accum[k] = accum.get(k, 0.0) + val
        if is_train and n_steps % args.log_step == 0:
            with tb_writer.as_default(step=global_step):
                for k, v in metrics.items():
                    tf.summary.scalar(f"{tag}_step/{k}", v)
                current_lr = optimizer.learning_rate
                if callable(current_lr):
                    current_lr = current_lr(optimizer.iterations)
                tf.summary.scalar(f"{tag}_step/learning_rate", current_lr)
            print(f"  [{tag}] epoch {epoch} step {n_steps} | "
                  f"loss={float(metrics['total_loss']):.4f} "
                  f"nll={float(metrics['nll']):.4f} "
                  f"reg={float(metrics['reg']):.6f} "
                  f"gnorm={float(metrics.get('grad_norm', 0)):.3f}")
        if is_train:
            global_step += 1
    if n_steps == 0:
        print(f"  [{tag}] WARNING: 0 steps processed — check TFRecords")
        return {}, global_step
    means = {k: v / n_steps for k, v in accum.items()}
    print(f"  [{tag}] epoch {epoch} done | steps={n_steps} | "
          + " | ".join(f"{k}={v:.5f}" for k, v in means.items()))
    return means, global_step
# ═════════════════════════════════════════════════════════════════════
# 9.  CHECKPOINTING HELPERS
# ═════════════════════════════════════════════════════════════════════
def save_checkpoint(model, optimizer, epoch, global_step, best_val_loss,
                    ckpt_dir, tag="latest"):
    """Save model weights + optimizer state + metadata."""
    path = os.path.join(ckpt_dir, tag)
    os.makedirs(path, exist_ok=True)
    model.save_weights(os.path.join(path, "model.weights.h5"))
    # Save optimizer config (weights are tied to model via checkpoint)
    meta = {"epoch": epoch, "global_step": global_step,
            "best_val_loss": best_val_loss}
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, cls=NumpyEncoder, indent=2)
    # Keras full-model checkpoint for optimizer state
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt.write(os.path.join(path, "ckpt"))
    print(f"  [CKPT] Saved {tag} checkpoint → {path}")
def load_checkpoint(model, optimizer, ckpt_dir, tag="latest"):
    """Load model weights + optimizer state + metadata. Returns meta dict."""
    path = os.path.join(ckpt_dir, tag)
    meta_path = os.path.join(path, "meta.json")
    if not os.path.exists(meta_path):
        print(f"  [CKPT] No checkpoint found at {path}")
        return None
    with open(meta_path, "r") as f:
        meta = json.load(f)
    # Restore via tf.train.Checkpoint for optimizer state
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt.read(os.path.join(path, "ckpt")).expect_partial()
    print(f"  [CKPT] Restored {tag}: epoch={meta['epoch']} "
          f"step={meta['global_step']} best_val={meta['best_val_loss']:.6f}")
    return meta
# ═════════════════════════════════════════════════════════════════════
# 10. INFERENCE
# ═════════════════════════════════════════════════════════════════════
def run_inference(model, h5_path, output_path, args):
    """
    Load best model and write z_probs into a copy of the input H5.
    Uses MleZprobsWriter to append sparse z_probs per cluster.
    """
    print(f"\n{'='*60}")
    print("INFERENCE MODE")
    print(f"{'='*60}")
    # Copy input H5 to output path
    shutil.copy2(h5_path, output_path)
    print(f"  Copied {h5_path} → {output_path}")
    # Count clusters
    with PublicTcrHlaCsrReaderChunk(output_path, include_counts=True,
                                     include_donors=True,
                                     include_cdr_freq=True) as reader:
        num_clusters = reader.num_clusters
    print(f"  Total clusters: {num_clusters}")
    # Open writer
    with MleZprobsWriter(output_path, num_clusters=num_clusters) as writer:
        with PublicTcrHlaCsrReaderChunk(
                output_path, include_counts=True, include_donors=True,
                include_pvals=False, include_cdr_freq=True) as reader:
            for chunk in reader.iter_cluster_chunks(chunk_rows=args.batch_size):
                if chunk.counts_dense is None or chunk.cdr_freq is None:
                    continue
                batch = prepare_batch(chunk, args)
                # Forward pass (no gradient)
                z_logits = model(
                    [batch["combined_cdr"], batch["combined_mask"]],
                    training=False)
                z_probs = tf.sigmoid(z_logits).numpy()
                # binder_sets: padded allele indices per cluster
                binder = chunk.counts_dense  # (B, A) dense
                # Convert dense to padded sparse indices for the writer
                binder_binary = (binder > 0).astype(np.float32)
                # Write z_probs aligned with binder indices
                # The writer expects binder_sets (padded indices) and z_probs (padded probs)
                # For dense z_probs we create index arrays
                n_chunk = z_probs.shape[0]
                max_hlas = int(binder_binary.sum(axis=1).max())
                binder_sets_padded = np.full(
                    (n_chunk, max_hlas), args.pad_token, dtype=np.float32)
                z_probs_padded = np.full(
                    (n_chunk, max_hlas), 0.0, dtype=np.float32)
                for i in range(n_chunk):
                    nz_idx = np.where(binder_binary[i] > 0)[0]
                    binder_sets_padded[i, :len(nz_idx)] = nz_idx
                    z_probs_padded[i, :len(nz_idx)] = z_probs[i, nz_idx]
                writer.write_chunk(
                    chunk.cluster_start, chunk.cluster_end,
                    binder_sets_padded, z_probs_padded,
                    pad_token=args.pad_token)
    print(f"  ✓ Inference complete → {output_path}")
# ═════════════════════════════════════════════════════════════════════
# 11. MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    device_type = setup_hardware(args)
    # ── create output directories ────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print("TCRtyper Training Pipeline")
    print(f"{'='*60}")
    print(f"  Mode:       {args.mode}")
    print(f"  Device:     {device_type}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Embed dim:  {args.embed_dim}")
    print(f"  Layers:     {args.num_layers}")
    print(f"  Heads:      {args.num_heads}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Epochs:     {args.epochs}")
    # ── print new flags ──────────────────────────────────────────────
    if args.keep_only_upperthan_n_donors > 1:
        print(f"  Min donors: {args.keep_only_upperthan_n_donors} "
              f"(clusters below this are skipped)")
    if args.tf_record_path:
        print(f"  TFRecord:   {args.tf_record_path} (custom path)")
    print(f"{'='*60}\n")
    # ── load donor HLA matrix ────────────────────────────────────────
    print("[DATA] Loading donor HLA matrix...")
    donor_data = np.load(args.donor_hla_matrix)
    donor_hla_matrix = donor_data["donor_hla_matrix"]
    print(f"  Donor HLA matrix shape: {donor_hla_matrix.shape}")
    # Compute empirical allele frequencies (f_a) and their log-odds
    # We use a small epsilon to prevent log(0) or division by zero for rare/missing alleles
    epsilon = 1e-7
    allele_freqs = np.clip(donor_hla_matrix.mean(axis=0), epsilon, 1.0 - epsilon)
    hla_log_odds = np.log(allele_freqs / (1.0 - allele_freqs)).astype(np.float32)
    print(f"  Computed log-odds bias initialization for {len(hla_log_odds)} alleles.")
    # ── load idx_to_hla ──────────────────────────────────────────────
    print("[DATA] Loading HLA index mapping...")
    with open(args.idx_to_hla, "r") as f:
        idx_to_hla = json.load(f)
    print(f"  {len(idx_to_hla)} HLA alleles")
    # Update num_alleles from actual data
    args.num_alleles = donor_hla_matrix.shape[1]
    print(f"  Updated num_alleles = {args.num_alleles}")
    # ── load and match HLA embeddings (optional) ─────────────────────
    hla_embed_matrix = None
    if args.hla_embed and os.path.exists(args.hla_embed):
        print("[DATA] Loading HLA embeddings...")
        hla_embed_raw = np.load(args.hla_embed)
        # Convert npz to dict
        hla_embed_dict = {k: hla_embed_raw[k] for k in hla_embed_raw.files}
        hla_matched, unmatched, embed_matrix = match_hla_alleles(
            hla_embed_dict, idx_to_hla)
        # embed_matrix: (A, embed_dim_esm)
        # Project to hla_proj_dim for kernel initialiser
        # Dense kernel shape is (hla_proj_dim, A), so we need (hla_proj_dim, A)
        # If embed_dim_esm != hla_proj_dim, use SVD truncation
        esm_dim = embed_matrix.shape[1]
        if esm_dim != args.hla_proj_dim:
            print(f"  Projecting HLA embeds {esm_dim}→{args.hla_proj_dim} via SVD")
            U, S, Vt = np.linalg.svd(embed_matrix, full_matrices=False)
            k = min(args.hla_proj_dim, esm_dim)
            embed_projected = U[:, :k] * S[:k][np.newaxis, :]
            if k < args.hla_proj_dim:
                pad_cols = args.hla_proj_dim - k
                embed_projected = np.hstack([
                    embed_projected,
                    np.zeros((embed_projected.shape[0], pad_cols))])
            hla_embed_matrix = embed_projected.T.astype(np.float32)
        else:
            hla_embed_matrix = embed_matrix.T.astype(np.float32)
        print(f"  HLA embed kernel shape: {hla_embed_matrix.shape}")
    else:
        print("[DATA] No HLA embeddings provided — random init")
    # ── build model ──────────────────────────────────────────────────
    print("\n[MODEL] Building TCRtyper...")
    model = build_model(args, hla_embed_matrix, hla_bias_init=hla_log_odds)
    model.summary(line_length=100)
    # ── loss function ────────────────────────────────────────────────
    loss_fn = TCRLikelihoodLoss(
        donor_hla_matrix, beta=args.beta, pad_token=args.pad_token,
        l2_reg_lambda=args.l2_reg, reduction=args.reduction,
        poisson_approx_untyped_hlas=args.poisson_approx,
        hla_bias_init=hla_log_odds)
    # ── count dataset size for LR schedule ───────────────────────────
    n_train = 0
    if args.mode == "train":
        n_train, steps_per_epoch = count_dataset_clusters(
            args.train_ds, args.batch_size)
        total_steps = steps_per_epoch * args.epochs
        print(f"\n[TRAIN] Dataset: {n_train} clusters, "
              f"~{steps_per_epoch} steps/epoch, "
              f"~{total_steps} total steps")
    else:
        total_steps = 1  # placeholder for inference
    # ── count validation clusters (for config) ───────────────────────
    n_valid = 0
    if args.valid_ds and os.path.exists(args.valid_ds):
        n_valid, _ = count_dataset_clusters(args.valid_ds, args.batch_size)
    # ── save config.json ─────────────────────────────────────────────
    save_config(args, args.output_dir, device_type,
                donor_shape=donor_hla_matrix.shape,
                n_train_clusters=n_train,
                n_valid_clusters=n_valid)
    # ── optimizer with LR schedule ───────────────────────────────────
    lr_schedule = WarmupCosineDecay(
        peak_lr=args.lr, min_lr=args.min_lr,
        warmup_steps=args.warmup_steps, total_steps=total_steps)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=args.weight_decay,
        clipnorm=None)  # we clip manually in train_step
    # ── warm up model with a dummy forward pass (build all weights) ──
    dummy_seq = tf.zeros((1, 10, 21), dtype=tf.float32)
    dummy_mask = tf.ones((1, 10), dtype=tf.int32)
    _ = model([dummy_seq, dummy_mask], training=False)
    # ════════════════════════════════════════════════════════════════
    # TRAIN MODE
    # ════════════════════════════════════════════════════════════════
    if args.mode == "train":
        tb_writer = tf.summary.create_file_writer(log_dir)
        metric_logger = MetricLogger(args.output_dir, tb_writer)
        # ── optional TFRecord cache conversion ───────────────────────
        # Resolve the base directory: use tf_record_path if set,
        # otherwise fall back to output_dir.
        tfr_base = args.tf_record_path if args.tf_record_path else args.output_dir
        train_tfr_ds, valid_tfr_ds = None, None
        if args.use_tfrecord:
            print("\n[CACHE] Converting datasets to TFRecords ...")
            if args.tf_record_path:
                print(f"  [CACHE] Using custom TFRecord path: {tfr_base}")
            train_shard_paths = convert_h5_to_tfrecords(
                args.train_ds, tfr_base, args,
                num_shards=args.num_shards, tag="train")
            train_tfr_ds = build_tfrecord_dataset(
                train_shard_paths, args, num_alleles=args.num_alleles,
                shuffle=True, drop_remainder=False)
            if args.valid_ds and os.path.exists(args.valid_ds):
                valid_shard_paths = convert_h5_to_tfrecords(
                    args.valid_ds, tfr_base, args,
                    num_shards=max(1, args.num_shards // 4), tag="valid")
                valid_tfr_ds = build_tfrecord_dataset(
                    valid_shard_paths, args, num_alleles=args.num_alleles,
                    shuffle=False, drop_remainder=False)
        # Resume from checkpoint if requested
        start_epoch = 0
        global_step = 0
        best_val_loss = float("inf")
        if args.resume:
            meta = load_checkpoint(model, optimizer, ckpt_dir, "latest")
            if meta is not None:
                start_epoch = meta["epoch"] + 1
                global_step = meta["global_step"]
                best_val_loss = meta["best_val_loss"]
        patience_counter = 0
        # ── epoch loop ───────────────────────────────────────────────
        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            print(f"\n{'─'*60}")
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f"{'─'*60}")
            # ── training ─────────────────────────────────────────────
            if args.use_tfrecord:
                train_metrics, global_step = run_epoch_tfrecord(
                    model, loss_fn, optimizer, train_tfr_ds, args,
                    tb_writer, global_step, epoch, is_train=True)
            else:
                train_metrics, global_step = run_epoch(
                    model, loss_fn, optimizer, args.train_ds, args,
                    tb_writer, global_step, epoch, is_train=True)
            train_loss = train_metrics.get("total_loss", 0.0)
            # Get current LR for logging
            current_lr = optimizer.learning_rate
            if callable(current_lr):
                current_lr = float(current_lr(optimizer.iterations).numpy())
            else:
                current_lr = float(current_lr)
            elapsed = time.time() - t0
            # Log train epoch to CSV/JSON/TB
            metric_logger.log_epoch(epoch, "train", train_metrics,
                                    extra={"learning_rate": current_lr,
                                           "epoch_time_s": elapsed})
            # ── validation ───────────────────────────────────────────
            val_loss = float("inf")
            if args.valid_ds and os.path.exists(args.valid_ds):
                if args.use_tfrecord and valid_tfr_ds is not None:
                    val_metrics, _ = run_epoch_tfrecord(
                        model, loss_fn, optimizer, valid_tfr_ds, args,
                        tb_writer, global_step, epoch, is_train=False)
                else:
                    val_metrics, _ = run_epoch(
                        model, loss_fn, optimizer, args.valid_ds, args,
                        tb_writer, global_step, epoch, is_train=False)
                val_loss = val_metrics.get("total_loss", float("inf"))
                metric_logger.log_epoch(epoch, "valid", val_metrics)
            # ── checkpointing ────────────────────────────────────────
            if (epoch + 1) % args.save_every_epoch == 0:
                save_checkpoint(model, optimizer, epoch, global_step,
                                best_val_loss, ckpt_dir, "latest")
            # ── best model tracking ──────────────────────────────────
            monitor_loss = val_loss if val_loss < float("inf") else train_loss
            if monitor_loss < best_val_loss:
                best_val_loss = monitor_loss
                save_checkpoint(model, optimizer, epoch, global_step,
                                best_val_loss, ckpt_dir, "best")
                patience_counter = 0
                print(f"  ★ New best model (loss={best_val_loss:.6f})")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{args.patience})")
            # Log epoch timing
            with tb_writer.as_default(step=epoch):
                tf.summary.scalar("epoch/time_seconds", elapsed)
                tf.summary.scalar("epoch/best_val_loss", best_val_loss)
            print(f"  Epoch time: {elapsed:.1f}s")
            # ── early stopping ───────────────────────────────────────
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch + 1}")
                break
        # Final save
        save_checkpoint(model, optimizer, epoch, global_step,
                        best_val_loss, ckpt_dir, "latest")
        tb_writer.flush()
        metric_logger.close()
        print(f"\n{'='*60}")
        print(f"Training complete — best loss: {best_val_loss:.6f}")
        print(f"Outputs: {args.output_dir}")
        print(f"  config.json  — full experiment config")
        print(f"  metrics.csv  — per-epoch metrics (train + valid)")
        print(f"  metrics.json — full history as JSON")
        print(f"  logs/        — TensorBoard events")
        print(f"  checkpoints/ — best + latest model weights")
        if args.use_tfrecord:
            tfr_loc = args.tf_record_path if args.tf_record_path else args.output_dir
            print(f"  {tfr_loc}/tfrecord_cache_train/ — cached TFRecord shards")
            if args.valid_ds:
                print(f"  {tfr_loc}/tfrecord_cache_valid/ — cached TFRecord shards")
        print(f"{'='*60}")
    # ════════════════════════════════════════════════════════════════
    # INFERENCE MODE
    # ════════════════════════════════════════════════════════════════
    elif args.mode == "inference":
        # Load best checkpoint
        meta = load_checkpoint(model, optimizer, ckpt_dir, "best")
        if meta is None:
            meta = load_checkpoint(model, optimizer, ckpt_dir, "latest")
        if meta is None:
            print("ERROR: No checkpoint found for inference")
            sys.exit(1)
        # Determine input and output paths
        infer_ds = args.inference_ds if args.inference_ds else args.train_ds
        output_h5 = os.path.join(args.output_dir, "predictions.h5")
        run_inference(model, infer_ds, output_h5, args)
    # ════════════════════════════════════════════════════════════════
    # EXPORT MODE
    # ════════════════════════════════════════════════════════════════
    elif args.mode == "export":
        # Load best checkpoint
        meta = load_checkpoint(model, optimizer, ckpt_dir, "best")
        if meta is None:
            meta = load_checkpoint(model, optimizer, ckpt_dir, "latest")
        if meta is None:
            print("ERROR: No checkpoint found for export")
            sys.exit(1)
        export_path = os.path.join(args.output_dir, "saved_model")
        model.save(export_path)
        print(f"  ✓ SavedModel exported → {export_path}")
if __name__ == "__main__":
    main()