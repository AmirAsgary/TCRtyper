#!/usr/bin/env python3
"""
TCRtyper: End-to-end Transformer Training Pipeline for TCR-HLA Binding
=======================================================================
Maximises the log-likelihood LL_B(theta,phi) = (1/|B|) sum_{i in B} LL_i(gamma_i(theta,phi))
where gamma_ia = sigmoid(NN_phi(r_a)^T . NN_theta(s_i)) predicts the probability
that TCR i binds HLA allotype a. Training uses minibatch SGD with Adam over
cluster-level chunks streamed from HDF5.
Modes:
  --mode train             : train from scratch or resume from checkpoint
  --mode train --mle_pretrain : Stage 1 pre-training on MLE pseudo-labels
  --mode inference         : load best checkpoint and write z_probs to output H5
  --mode export            : export SavedModel for serving
Usage:
  python train_tcrtyper.py --mode train --train_ds data/train.h5 --valid_ds data/valid.h5 \
      --donor_hla_matrix data/donor_hla_matrix.npz --output_dir runs/exp01
  python train_tcrtyper.py --mode train --mle_pretrain --train_ds data/train_with_zprobs.h5 \
      --donor_hla_matrix data/donor_hla_matrix.npz --output_dir runs/pretrain01
"""
from __future__ import annotations
# ── GPU memory config MUST run before any TF operation ───────────────
import os as _os
def _early_gpu_config():
    """Set GPU memory growth before TF context is locked.
    Also suppress noisy XLA autotuning and TF INFO logs."""
    import os as _os2
    _os2.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    _os2.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
    try:
        import tensorflow as _tf
        gpus = _tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            _tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
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
    p.add_argument("--scale_hla_embed", action="store_true", default=False,
                   help="If enabled, scales esm embedding.")
    p.add_argument("--train_bias", action="store_true", default=False,
                   help="If enabled, even bias of final pred is trainable.")
    # ── loss / likelihood ────────────────────────────────────────────
    p.add_argument("--beta", type=float, default=4.0,
                   help="Beta-Binomial prior parameter")
    p.add_argument("--l2_reg", type=float, default=0.0,
                   help="L2 regularisation on z_logits")
    p.add_argument("--invariant_lambda", type=float, default=0.0,
                   help="lambda of sparsity inducing regularization term")
    p.add_argument("--false_pos_lambda", type=float, default=0.0,
                   help="penalizes how prob for non occuring hlas.")
    p.add_argument("--reduction", type=str, default="mean",
                   choices=["sum", "mean"],
                   help="Loss reduction mode")
    p.add_argument("--poisson_approx", action="store_true", default=False,
                   help="Use Poisson approximation for untyped HLAs")
    # ── MLE pre-training ─────────────────────────────────────────────
    p.add_argument("--mle_pretrain", action="store_true", default=False,
                   help="Stage 1: pre-train on MLE pseudo-labels "
                        "instead of likelihood. Requires z_probs in H5.")
    p.add_argument("--mle_confidence_cap", type=float, default=20.0,
                   help="n_donors value at which confidence weight "
                        "saturates to 1.0 for MLE pre-training")
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
    p.add_argument("--lr_schedule_restart", action="store_true",
                   default=False, help="If enabled, restart cosine decay")
    p.add_argument("--weight_decay", type=float, default=1e-5,
                   help="AdamW weight decay")
    p.add_argument("--grad_clip", type=float, default=0.5,
                   help="Global gradient norm clip (0 = disabled)")
    p.add_argument("--dropout", type=float, default=0.3,
                   help="Dropout rate in encoder and transformer")
    p.add_argument("--masking_rate", type=float, default=0.1,
                   help="Random CDR masking rate for self-supervised signal")
    p.add_argument("--cold_start", action="store_true", default=False,
                   help="Shift hla_log_odds so most common allele starts "
                        "at gamma approx 0.01")
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
                   help="Early stopping patience")
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
                   help="Convert H5 to TFRecords once, then use tf.data")
    p.add_argument("--num_shards", type=int, default=16,
                   help="Number of TFRecord shard files for parallel reads")
    # ── donor filtering ──────────────────────────────────────────────
    p.add_argument("--keep_only_upperthan_n_donors", type=int, default=1,
                   help="Only use clusters with at least N donors")
    # ── custom TFRecord path ─────────────────────────────────────────
    p.add_argument("--tf_record_path", type=str, default="",
                   help="Custom base directory for TFRecord cache files")
    return p.parse_args()
# ═════════════════════════════════════════════════════════════════════
# 2.  HARDWARE SETUP
# ═════════════════════════════════════════════════════════════════════
def setup_hardware(args):
    """Configure GPU memory, mixed precision, and random seeds."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
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
        valid = mask > 0.0
        is_masked = (rand < masking_rate) & valid
        mask[is_masked] = float(mask_token)
        features[is_masked] = 0.0
    return features, mask
def _concat_cdrs_with_sep_numpy(feat_list, mask_list, sep_mask_val=-3):
    """Concatenate CDR features+masks with separator columns, all in numpy.
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
        if i < len(feat_list) - 1:
            parts_f.append(sep_feat)
            parts_m.append(sep_mask)
    return np.concatenate(parts_f, axis=1), np.concatenate(parts_m, axis=1)
# ═════════════════════════════════════════════════════════════════════
# 3a. MLE PRE-TRAINING HELPERS
# ═════════════════════════════════════════════════════════════════════
def _validate_z_probs_exist(h5_path: str) -> None:
    """Check that z_probs group exists in H5 file. Raises ValueError if not.
    Args:
        h5_path: path to HDF5 file to check.
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        if "clusters" not in f or "z_probs" not in f["clusters"]:
            raise ValueError(
                f"ERROR: --mle_pretrain requires z_probs in the HDF5 file "
                f"'{h5_path}'. Run MLE first to generate pseudo-labels.\n"
                f"  Available groups under 'clusters': "
                f"{list(f['clusters'].keys()) if 'clusters' in f else '(none)'}"
            )
def prepare_pretrain_targets(z_probs_dense, binder_dense, n_donors,
                             num_alleles, min_donors=3,
                             confidence_cap=20.0):
    """Convert dense MLE z_probs into logit targets + confidence weights.
    Fully vectorized numpy — no Python for-loops over the batch dimension.
    Args:
        z_probs_dense:   np.ndarray (B, A) float32, MLE binding probs (dense).
        binder_dense:    np.ndarray (B, A) float32, binary co-occurrence mask.
        n_donors:        np.ndarray (B,) int/float, donor count per TCR.
        num_alleles:     int, total number of HLA alleles (A).
        min_donors:      int, minimum donors to use a TCR's pseudo-labels.
        confidence_cap:  float, n_donors value at which confidence saturates.
    Returns:
        target_logits:       np.ndarray (B, A) float32, clipped logit targets.
        confidence_weights:  np.ndarray (B,) float32, per-TCR weight in [0,1].
    """
    # ── active allele mask (where this TCR co-occurs with alleles) ───
    active_mask = (binder_dense > 0.5).astype(np.float32)  # (B, A)
    # ── background logit for non-active alleles (strong negative) ────
    bg_prob = 1e-4
    bg_logit = np.float32(np.log(bg_prob / (1.0 - bg_prob)))  # ≈ -9.2
    # ── clip z_probs to avoid log(0) or log(inf) ────────────────────
    z_clipped = np.clip(z_probs_dense, 1e-4, 1.0 - 1e-4)  # (B, A)
    # ── convert to logits: logit(p) = log(p / (1-p)) ────────────────
    raw_logits = np.log(z_clipped / (1.0 - z_clipped))  # (B, A)
    # ── scatter: active positions get MLE logits, rest get bg_logit ──
    target_logits = np.where(active_mask > 0.5, raw_logits, bg_logit)
    # ── clip final logits for numerical safety ───────────────────────
    target_logits = np.clip(target_logits, -8.0, 8.0).astype(np.float32)
    # ── confidence weights (vectorized) ──────────────────────────────
    # Only consider active alleles for concentration calculation
    active_probs = z_probs_dense * active_mask  # (B, A)
    # max_prob: highest MLE probability among active alleles per TCR
    max_prob = np.max(active_probs, axis=1)  # (B,)
    # total_prob: sum of MLE probabilities among active alleles
    total_prob = np.sum(active_probs, axis=1)  # (B,)
    # concentration: how focused the MLE solution is (1.0 = single allele)
    concentration = max_prob / np.maximum(total_prob, 1e-7)  # (B,)
    # donor_weight: more donors = more reliable MLE estimate
    n_donors_f = n_donors.astype(np.float32)
    donor_weight = np.minimum(n_donors_f / confidence_cap, 1.0)  # (B,)
    # combined confidence = concentration * donor_weight
    confidence_weights = (concentration * donor_weight).astype(np.float32)
    # ── zero out TCRs below min_donors threshold ─────────────────────
    confidence_weights = np.where(
        n_donors_f >= min_donors, confidence_weights, 0.0
    ).astype(np.float32)
    return target_logits, confidence_weights
def _log_pretrain_stats(z_probs_dense, binder_dense, n_donors,
                        confidence_weights, tag="train"):
    """Log summary statistics about pseudo-labels at startup.
    Args:
        z_probs_dense:      (B, A) MLE probabilities.
        binder_dense:       (B, A) binary co-occurrence mask.
        n_donors:           (B,) donor counts.
        confidence_weights: (B,) computed confidence weights.
        tag:                dataset label for printing.
    """
    active = binder_dense > 0.5
    active_probs = z_probs_dense * active
    n_with_conf = int(np.sum(confidence_weights > 0))
    n_total = len(confidence_weights)
    mean_conf = float(np.mean(confidence_weights))
    max_per_tcr = np.max(active_probs, axis=1)
    mean_max_gamma = float(np.mean(max_per_tcr))
    # sparsity: mean number of alleles with gamma > 0.1 per TCR
    n_above_01 = np.sum(active_probs > 0.1, axis=1).astype(np.float32)
    mean_sparsity = float(np.mean(n_above_01))
    print(f"  [PRETRAIN-{tag}] Pseudo-label statistics:")
    print(f"    TCRs with confidence > 0: {n_with_conf:,} / {n_total:,}")
    print(f"    Mean confidence weight:   {mean_conf:.4f}")
    print(f"    Mean max target gamma:    {mean_max_gamma:.4f}")
    print(f"    Mean target sparsity (gamma > 0.1): "
          f"{mean_sparsity:.1f} alleles/TCR")
    print(f"    Mean n_donors: {np.mean(n_donors):.1f}  "
          f"min={np.min(n_donors)}  max={np.max(n_donors)}")
# ═════════════════════════════════════════════════════════════════════
# 3b. BATCH PREPARATION
# ═════════════════════════════════════════════════════════════════════
def prepare_batch(chunk, args, mle_pretrain=False) -> Dict[str, tf.Tensor]:
    """Convert a PublicTcrHlaClusterChunk into padded tensors for one step.
    All heavy lifting is done in numpy; tf.constant is called once per
    tensor at the end to minimise Python-to-TF overhead.
    Args:
        chunk:        PublicTcrHlaClusterChunk from the HDF5 reader.
        args:         parsed argument namespace.
        mle_pretrain: if True, also compute target_logits and
                      confidence_weights from chunk.z_probs_dense.
    Returns:
        dict with keys: combined_cdr, combined_mask, binder_dense,
        donor_indices. When mle_pretrain=True, also: target_logits,
        confidence_weights.
    """
    # ── donor indices (variable-length → padded) ─────────────────────
    donor_lists = split_ragged_to_list(
        chunk.raw_csr_donor_indices, chunk.raw_csr_donor_indptr)
    donor_pad, _ = pad_list_to_array_without_max(donor_lists, args.pad_token)
    donor_pad = donor_pad.astype(np.int32)
    # ── CDR frequency profiles → pad + mask each region (numpy) ──────
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
    # ── concatenate with separator tokens (numpy) ────────────────────
    combined_np, mask_np = _concat_cdrs_with_sep_numpy(
        [cdr1, cdr2, cdr25, cdr3], [m1, m2, m25, m3],
        sep_mask_val=args.sep_token)
    # ── binder dense set: binary mask over alleles ───────────────────
    binder_np = (chunk.counts_dense > 0).astype(np.float32)
    # ── build output dict with single tf.constant calls ──────────────
    batch = {
        "combined_cdr":  tf.constant(combined_np, dtype=tf.float32),
        "combined_mask": tf.constant(mask_np.astype(np.int32), dtype=tf.int32),
        "binder_dense":  tf.constant(binder_np, dtype=tf.float32),
        "donor_indices": tf.constant(donor_pad, dtype=tf.int32),
    }
    # ── MLE pretrain targets ─────────────────────────────────────────
    if mle_pretrain:
        if chunk.z_probs_dense is None:
            raise ValueError(
                "mle_pretrain=True but chunk.z_probs_dense is None. "
                "Ensure the HDF5 file contains z_probs and the reader "
                "was opened with include_z_probs=True."
            )
        target_logits_np, conf_np = prepare_pretrain_targets(
            z_probs_dense=chunk.z_probs_dense,
            binder_dense=binder_np,
            n_donors=chunk.n_donors,
            num_alleles=chunk.z_probs_dense.shape[1],
            min_donors=int(args.keep_only_upperthan_n_donors),
            confidence_cap=float(args.mle_confidence_cap),
        )
        batch["target_logits"] = tf.constant(
            target_logits_np, dtype=tf.float32)
        batch["confidence_weights"] = tf.constant(
            conf_np, dtype=tf.float32)
    return batch
# ═════════════════════════════════════════════════════════════════════
# 3c. TFRECORD CACHE — HDF5 → sharded TFRecords conversion
# ═════════════════════════════════════════════════════════════════════
def _serialize_cluster(combined_cdr, combined_mask, binder_dense,
                       donor_indices, target_logits=None,
                       confidence_weight=None):
    """Serialize one cluster into a tf.train.Example protobuf.
    Args:
        combined_cdr:      np.ndarray (L, 21) float32
        combined_mask:     np.ndarray (L,)    int32
        binder_dense:      np.ndarray (A,)    float32
        donor_indices:     np.ndarray (P,)    int32
        target_logits:     np.ndarray (A,) float32 or None (MLE pretrain)
        confidence_weight: float scalar or None (MLE pretrain)
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
    # ── optional MLE pretrain fields ─────────────────────────────────
    if target_logits is not None:
        feat["target_logits"] = tf.train.Feature(
            float_list=tf.train.FloatList(value=target_logits.tolist()))
    if confidence_weight is not None:
        feat["confidence_weight"] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[float(confidence_weight)]))
    return tf.train.Example(
        features=tf.train.Features(feature=feat)).SerializeToString()
def convert_h5_to_tfrecords(h5_path: str, output_dir: str, args,
                            num_shards: int = 16,
                            tag: str = "train") -> List[str]:
    """One-time conversion: stream HDF5 → sharded TFRecord files.
    When args.mle_pretrain is True, also computes and serializes
    target_logits and confidence_weight per cluster.
    Clusters with n_donors < args.keep_only_upperthan_n_donors are
    skipped. If args.tf_record_path is set, writes there instead.
    Args:
        h5_path:    path to H5
        output_dir: directory to write shard files (fallback)
        args:       parsed flags
        num_shards: number of output files
        tag:        prefix for shard filenames
    Returns:
        list of shard file paths
    """
    # ── resolve TFRecord base directory ──────────────────────────────
    tfr_base = args.tf_record_path if args.tf_record_path else output_dir
    tfr_dir = os.path.join(tfr_base, f"tfrecord_cache_{tag}")
    os.makedirs(tfr_dir, exist_ok=True)
    min_donors = int(args.keep_only_upperthan_n_donors)
    mle_pretrain = bool(args.mle_pretrain)
    # ── idempotency check via manifest ───────────────────────────────
    manifest_path = os.path.join(tfr_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        # Reuse only if source H5, min_donors, AND mle_pretrain match
        if (manifest.get("source_h5") == h5_path and
                manifest.get("min_donors", 1) == min_donors and
                manifest.get("mle_pretrain", False) == mle_pretrain):
            print(f"  [CACHE] Reusing existing TFRecords in {tfr_dir} "
                  f"({manifest['num_clusters']} clusters, "
                  f"{manifest['num_shards']} shards, "
                  f"min_donors={min_donors}, "
                  f"mle_pretrain={mle_pretrain})")
            return manifest["shard_paths"]
        else:
            # Manifest mismatch — force re-conversion
            print(f"  [CACHE] Manifest mismatch (mle_pretrain or "
                  f"min_donors changed) — re-converting ...")
            # Remove old manifest so we regenerate
            os.remove(manifest_path)
    print(f"  [CACHE] Converting {h5_path} → TFRecords in {tfr_dir} ...")
    if min_donors > 1:
        print(f"  [CACHE] Filtering: keeping clusters with "
              f"n_donors >= {min_donors}")
    if mle_pretrain:
        print(f"  [CACHE] MLE pretrain mode: storing target_logits + "
              f"confidence_weight per cluster")
    t0 = time.time()
    # ── open shard writers ───────────────────────────────────────────
    shard_paths = [os.path.join(tfr_dir, f"{tag}_{i:04d}.tfrecord")
                   for i in range(num_shards)]
    writers = [tf.io.TFRecordWriter(p) for p in shard_paths]
    cluster_count = 0
    skipped_count = 0
    shard_idx = 0
    # ── stats accumulators for pretrain logging ──────────────────────
    stats_conf_sum, stats_conf_count = 0.0, 0
    stats_conf_pos = 0
    # ── read HDF5 in large chunks ────────────────────────────────────
    with PublicTcrHlaCsrReaderChunk(
            h5_path, include_counts=True, include_donors=True,
            include_pvals=False, include_cdr_freq=True,
            include_z_probs=mle_pretrain) as reader:
        for chunk in reader.iter_cluster_chunks(chunk_rows=10000):
            if chunk.counts_dense is None or chunk.cdr_freq is None:
                continue
            B = chunk.counts_dense.shape[0]
            # 1. Vectorized donor-count filtering
            keep_mask = chunk.n_donors >= min_donors
            valid_indices = np.where(keep_mask)[0]
            if len(valid_indices) == 0:
                skipped_count += B
                continue
            skipped_count += (B - len(valid_indices))
            # 2. Fast vectorized numpy prep (whole chunk)
            donor_lists = split_ragged_to_list(
                chunk.raw_csr_donor_indices, chunk.raw_csr_donor_indptr)
            donor_pad, _ = pad_list_to_array_without_max(
                donor_lists, args.pad_token)
            cdr1, m1 = _pad_and_mask_numpy(
                chunk.cdr_freq["cdr1"][:], args.pad_token,
                args.mask_token, 0.0)
            cdr2, m2 = _pad_and_mask_numpy(
                chunk.cdr_freq["cdr2"][:], args.pad_token,
                args.mask_token, 0.0)
            cdr25, m25 = _pad_and_mask_numpy(
                chunk.cdr_freq["cdr25"][:], args.pad_token,
                args.mask_token, 0.0)
            cdr3, m3 = _pad_and_mask_numpy(
                chunk.cdr_freq["cdr3"][:], args.pad_token,
                args.mask_token, 0.0)
            combined_np, mask_np = _concat_cdrs_with_sep_numpy(
                [cdr1, cdr2, cdr25, cdr3], [m1, m2, m25, m3],
                sep_mask_val=args.sep_token)
            binder_np = (chunk.counts_dense > 0).astype(np.float32)
            # 3. MLE pretrain targets (vectorized for whole chunk)
            tgt_logits_np, conf_np = None, None
            if mle_pretrain:
                if chunk.z_probs_dense is None:
                    raise ValueError(
                        "mle_pretrain=True but z_probs_dense is None. "
                        "Ensure the H5 has z_probs (run MLE first).")
                tgt_logits_np, conf_np = prepare_pretrain_targets(
                    z_probs_dense=chunk.z_probs_dense,
                    binder_dense=binder_np,
                    n_donors=chunk.n_donors,
                    num_alleles=args.num_alleles,
                    min_donors=min_donors,
                    confidence_cap=float(args.mle_confidence_cap),
                )
            # 4. Apply valid_indices filter
            combined_np = combined_np[valid_indices]
            mask_np = mask_np[valid_indices]
            binder_np = binder_np[valid_indices]
            donor_pad = donor_pad[valid_indices]
            if mle_pretrain:
                tgt_logits_np = tgt_logits_np[valid_indices]
                conf_np = conf_np[valid_indices]
                # accumulate stats
                stats_conf_sum += float(conf_np.sum())
                stats_conf_count += len(conf_np)
                stats_conf_pos += int(np.sum(conf_np > 0))
            # 5. Compute unpadded lengths (vectorized)
            seq_lens = np.sum(mask_np != args.pad_token, axis=1)
            donor_lens = np.sum(donor_pad != args.pad_token, axis=1)
            # 6. Serialize each cluster
            for i in range(len(valid_indices)):
                s_len = seq_lens[i]
                d_len = donor_lens[i]
                actual_cdr = combined_np[i, :s_len]
                actual_mask = mask_np[i, :s_len]
                actual_donors = donor_pad[i, :d_len]
                tgt_i = tgt_logits_np[i] if mle_pretrain else None
                conf_i = float(conf_np[i]) if mle_pretrain else None
                serialized = _serialize_cluster(
                    actual_cdr, actual_mask, binder_np[i],
                    actual_donors, target_logits=tgt_i,
                    confidence_weight=conf_i)
                writers[shard_idx % num_shards].write(serialized)
                shard_idx += 1
                cluster_count += 1
    # ── close writers ────────────────────────────────────────────────
    for w in writers:
        w.close()
    elapsed = time.time() - t0
    # ── log pretrain stats ───────────────────────────────────────────
    if mle_pretrain and stats_conf_count > 0:
        print(f"  [CACHE-PRETRAIN] TCRs with confidence > 0: "
              f"{stats_conf_pos:,} / {stats_conf_count:,}")
        print(f"  [CACHE-PRETRAIN] Mean confidence: "
              f"{stats_conf_sum / stats_conf_count:.4f}")
    # ── write manifest ───────────────────────────────────────────────
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
        "mle_pretrain": mle_pretrain,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [CACHE] Wrote {cluster_count} clusters to {num_shards} "
          f"shards in {elapsed:.1f}s "
          f"({cluster_count / max(elapsed, 1e-6):.0f} clusters/s)")
    if skipped_count > 0:
        print(f"  [CACHE] Skipped {skipped_count} clusters with "
              f"n_donors < {min_donors}")
    return shard_paths
def build_tfrecord_dataset(shard_paths: List[str], args,
                           num_alleles: int,
                           shuffle: bool = True,
                           drop_remainder: bool = False,
                           mle_pretrain: bool = False) -> tf.data.Dataset:
    """Build a tf.data pipeline from sharded TFRecords.
    Runs entirely in C++ (no Python GIL):
      file interleave → parse → optional masking → shuffle → batch → pad → prefetch
    When mle_pretrain=True, the dataset yields 6-tuples including
    target_logits and confidence_weight.
    Args:
        shard_paths:    list of .tfrecord file paths
        args:           parsed flags
        num_alleles:    A dimension for binder_dense
        shuffle:        whether to shuffle
        drop_remainder: drop last incomplete batch
        mle_pretrain:   if True, parse additional pretrain features
    Returns:
        tf.data.Dataset yielding tuples of tensors.
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
        """Parse one Example into tensors."""
        desc = {
            "combined_cdr_flat": tf.io.VarLenFeature(tf.float32),
            "combined_mask":     tf.io.VarLenFeature(tf.int64),
            "binder_dense":      tf.io.FixedLenFeature(
                                     [num_alleles], tf.float32),
            "donor_indices":     tf.io.VarLenFeature(tf.int64),
            "seq_len":           tf.io.FixedLenFeature([], tf.int64),
        }
        if mle_pretrain:
            desc["target_logits"] = tf.io.FixedLenFeature(
                [num_alleles], tf.float32)
            desc["confidence_weight"] = tf.io.FixedLenFeature(
                [], tf.float32)
        parsed = tf.io.parse_single_example(example_proto, desc)
        seq_len = parsed["seq_len"]
        cdr_flat = tf.sparse.to_dense(parsed["combined_cdr_flat"])
        cdr = tf.reshape(cdr_flat, [seq_len, 21])
        mask = tf.cast(
            tf.sparse.to_dense(parsed["combined_mask"]), tf.int32)
        binder = parsed["binder_dense"]
        donors = tf.cast(
            tf.sparse.to_dense(parsed["donor_indices"]), tf.int32)
        if mle_pretrain:
            tgt = parsed["target_logits"]
            conf = parsed["confidence_weight"]
            return cdr, mask, binder, donors, tgt, conf
        return cdr, mask, binder, donors
    dataset = dataset.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    # ── online random masking (training only) ────────────────────────
    if args.masking_rate > 0 and shuffle:
        if mle_pretrain:
            @tf.function
            def _apply_masking_pt(cdr, mask, binder, donors, tgt, conf):
                """Apply random masking, pass through pretrain fields."""
                L = tf.shape(cdr)[0]
                valid = tf.not_equal(mask, args.pad_token)
                rand = tf.random.uniform([L], 0.0, 1.0)
                is_masked = tf.logical_and(
                    rand < args.masking_rate, valid)
                mask_expand = tf.expand_dims(
                    tf.cast(tf.logical_not(is_masked), tf.float32), -1)
                cdr = cdr * mask_expand
                mask = tf.where(is_masked, args.mask_token, mask)
                return cdr, mask, binder, donors, tgt, conf
            dataset = dataset.map(
                _apply_masking_pt,
                num_parallel_calls=tf.data.AUTOTUNE)
        else:
            @tf.function
            def _apply_masking(cdr, mask, binder, donors):
                """Apply random masking to CDR features."""
                L = tf.shape(cdr)[0]
                valid = tf.not_equal(mask, args.pad_token)
                rand = tf.random.uniform([L], 0.0, 1.0)
                is_masked = tf.logical_and(
                    rand < args.masking_rate, valid)
                mask_expand = tf.expand_dims(
                    tf.cast(tf.logical_not(is_masked), tf.float32), -1)
                cdr = cdr * mask_expand
                mask = tf.where(is_masked, args.mask_token, mask)
                return cdr, mask, binder, donors
            dataset = dataset.map(
                _apply_masking,
                num_parallel_calls=tf.data.AUTOTUNE)
    # ── shuffle ──────────────────────────────────────────────────────
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(250000, 100 * args.batch_size),
            reshuffle_each_iteration=True)
    # ── batch with dynamic padding ───────────────────────────────────
    if mle_pretrain:
        pad_shapes = (
            [None, 21], [None], [num_alleles], [None],
            [num_alleles], [],  # target_logits (fixed), conf (scalar)
        )
        pad_values = (
            tf.constant(0.0, tf.float32),
            tf.constant(args.pad_token, tf.int32),
            tf.constant(0.0, tf.float32),
            tf.constant(args.pad_token, tf.int32),
            tf.constant(0.0, tf.float32),   # target_logits pad (unused)
            tf.constant(0.0, tf.float32),   # confidence_weight pad
        )
    else:
        pad_shapes = (
            [None, 21], [None], [num_alleles], [None],
        )
        pad_values = (
            tf.constant(0.0, tf.float32),
            tf.constant(args.pad_token, tf.int32),
            tf.constant(0.0, tf.float32),
            tf.constant(args.pad_token, tf.int32),
        )
    dataset = dataset.padded_batch(
        args.batch_size, padded_shapes=pad_shapes,
        padding_values=pad_values, drop_remainder=drop_remainder)
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
    """Write a comprehensive config.json for experiment reproduction."""
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
            "invariant_lambda": args.invariant_lambda,
        },
        "pretrain": {
            "mle_pretrain": args.mle_pretrain,
            "mle_confidence_cap": args.mle_confidence_cap,
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
            "keep_only_upperthan_n_donors":
                args.keep_only_upperthan_n_donors,
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
    Output: (B, D) pooled representation
    """
    def __init__(self, pad_token: int = -1, **kw):
        super().__init__(**kw)
        self.pad_token = pad_token
    def call(self, inputs):
        x, mask = inputs
        valid = tf.cast(tf.not_equal(mask, self.pad_token), tf.float32)
        valid_3d = valid[:, :, tf.newaxis]
        x_masked = x * valid_3d
        summed = tf.reduce_sum(x_masked, axis=1)
        counts = tf.reduce_sum(valid, axis=1, keepdims=True)
        counts = tf.maximum(counts, 1.0)
        return summed / counts
    def get_config(self):
        cfg = super().get_config()
        cfg["pad_token"] = self.pad_token
        return cfg
def load_hla_embeddings_for_clip(args, idx_to_hla):
    """Load and match HLA embeddings for the CLIP head.
    Returns:
        hla_embed_matrix : np.ndarray (A, esm_dim) or None
    """
    if not args.hla_embed or not os.path.exists(args.hla_embed):
        print("[DATA] No HLA embeddings provided — CLIP head uses "
              "random init")
        return None
    print("[DATA] Loading HLA embeddings for CLIP head...")
    hla_embed_raw = np.load(args.hla_embed)
    hla_embed_dict = {k: hla_embed_raw[k] for k in hla_embed_raw.files}
    hla_matched, unmatched, embed_matrix = match_hla_alleles(
        hla_embed_dict, idx_to_hla)
    print(f"  HLA embeddings: {embed_matrix.shape}  "
          f"(matched={len(hla_matched)}, unmatched={len(unmatched)})")
    return embed_matrix.astype(np.float32)
class CLIPBindingHead(layers.Layer):
    """CLIP-style binding head: z_logits = tcr_embed @ hla_embed^T + bias.
    Args:
        num_alleles:     total HLA alleles (A).
        clip_dim:        shared embedding dimension (D).
        hla_input_dim:   dimensionality of raw HLA embeddings.
        hla_embed_init:  (A, hla_input_dim) or None.
        bias_init:       (A,) log-odds or None.
        train_hla_proj:  allow gradients through HLA projection.
        init_temperature: initial temperature for dot-product scaling.
        min_temperature:  lower bound for temperature.
        scale_hla_embed:  normalize and shrink ESM embeddings.
    """
    def __init__(self, num_alleles, clip_dim, hla_input_dim=32,
                 hla_embed_init=None, bias_init=None,
                 train_hla_proj=True, init_temperature=5.0,
                 min_temperature=0.5, scale_hla_embed=True,
                 train_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_alleles = num_alleles
        self.clip_dim = clip_dim
        self.hla_input_dim = hla_input_dim
        self.train_hla_proj = train_hla_proj
        self.scale_hla_embed = scale_hla_embed
        self.train_bias = train_bias
        # ── HLA raw embeddings ───────────────────────────────────────
        if hla_embed_init is not None:
            if scale_hla_embed:
                norms = np.linalg.norm(
                    hla_embed_init, axis=1, keepdims=True)
                scaled = hla_embed_init / np.clip(norms, 1e-6, None)
                scaled *= 0.1
                init = tf.keras.initializers.Constant(scaled)
            else:
                init = tf.keras.initializers.Constant(hla_embed_init)
        else:
            init = tf.keras.initializers.RandomNormal(stddev=0.02)
        self.hla_raw = self.add_weight(
            name="hla_raw_embeddings",
            shape=(num_alleles, hla_input_dim),
            initializer=init, trainable=train_hla_proj)
        # ── HLA projection ───────────────────────────────────────────
        self.hla_proj = layers.Dense(
            clip_dim, use_bias=True, name="hla_clip_proj",
            trainable=True,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer="zeros")
        # ── log-odds bias ────────────────────────────────────────────
        if bias_init is not None:
            b_init = tf.keras.initializers.Constant(bias_init)
        else:
            b_init = "zeros"
        self.allele_bias = self.add_weight(
            name="allele_bias", shape=(num_alleles,),
            initializer=b_init, trainable=self.train_bias)
        # ── learnable temperature ────────────────────────────────────
        self.log_temperature = self.add_weight(
            name="log_temperature", shape=(),
            initializer=tf.keras.initializers.Constant(
                np.log(init_temperature)),
            trainable=True)
        self.min_temperature = min_temperature
    def call(self, tcr_embed):
        """Forward: (B, clip_dim) → (B, A) raw logits."""
        hla_embed = self.hla_proj(self.hla_raw)
        # L2-normalize both embeddings → dot product becomes cosine similarity
        tcr_embed = tf.nn.l2_normalize(tcr_embed, axis=-1)   # (B, clip_dim)
        hla_embed = tf.nn.l2_normalize(hla_embed, axis=-1)    # (A, clip_dim)
        
        temperature = tf.maximum(tf.exp(self.log_temperature), self.min_temperature)
        z_logits = tf.matmul(tcr_embed, hla_embed, transpose_b=True) / temperature
        z_logits = z_logits + self.allele_bias
        return z_logits
    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_alleles": self.num_alleles,
            "clip_dim": self.clip_dim,
            "hla_input_dim": self.hla_input_dim,
            "train_hla_proj": self.train_hla_proj,
            "scale_hla_embed": self.scale_hla_embed,
        })
        return cfg
def build_model(args, hla_embed_matrix=None, hla_bias_init=None):
    """Build the TCRtyper Keras functional model with CLIP-style head.
    Args:
        args:             parsed argument namespace.
        hla_embed_matrix: (A, hla_input_dim) or None.
        hla_bias_init:    (A,) log-odds or None.
    Returns:
        keras.Model with inputs [seq_input, mask_input], output z_logits.
    """
    ff_dim = args.ff_dim if args.ff_dim > 0 else args.embed_dim * 4
    clip_dim = args.hla_proj_dim
    # ── inputs ───────────────────────────────────────────────────────
    inp_seq = tf.keras.Input(shape=(None, 21), name="seq_input")
    inp_mask = tf.keras.Input(
        shape=(None,), dtype=tf.int32, name="mask_input")
    # ── TCR encoder ──────────────────────────────────────────────────
    h = SequenceEncoderLayer(
        embed_dim=args.embed_dim, max_len=args.max_seq_len,
        dropout_rate=args.dropout, encoding_mode=args.encoding_mode,
        pad_token=args.pad_token, mask_token=args.mask_token,
        sep_token=args.sep_token, normal_token=args.normal_token,
        name="seq_encoder")([inp_seq, inp_mask])
    for i in range(args.num_layers):
        h = GatedTransformerLayer(
            embed_dim=args.embed_dim, num_heads=args.num_heads,
            ff_dim=ff_dim, resnet=args.resnet,
            dropout_rate=args.dropout,
            pad_token=args.pad_token, mask_token=args.mask_token,
            sep_token=args.sep_token, normal_token=args.normal_token,
            name=f"transformer_{i}")([h, inp_mask])
    # ── masked pooling → TCR embedding ───────────────────────────────
    h = MaskedGlobalAveragePooling(
        pad_token=args.pad_token, name="pool")([h, inp_mask])
    tcr_embed = layers.Dense(
        clip_dim, activation="gelu", name="tcr_clip_proj",
        kernel_initializer=tf.keras.initializers.GlorotUniform())(h)
    tcr_embed = layers.Dropout(
        args.dropout, name="tcr_clip_drop")(tcr_embed)
    # ── CLIP binding head ────────────────────────────────────────────
    if hla_embed_matrix is not None:
        hla_input_dim = hla_embed_matrix.shape[1]
    else:
        hla_input_dim = clip_dim
    z_logits = CLIPBindingHead(
        num_alleles=args.num_alleles, clip_dim=clip_dim,
        hla_input_dim=hla_input_dim,
        hla_embed_init=hla_embed_matrix, bias_init=hla_bias_init,
        train_hla_proj=args.train_hla_head,
        scale_hla_embed=args.scale_hla_embed,
        train_bias=args.train_bias,
        name="clip_binding_head")(tcr_embed)
    model = tf.keras.Model(
        inputs=[inp_seq, inp_mask], outputs=z_logits,
        name="TCRtyper")
    return model
# ═════════════════════════════════════════════════════════════════════
# 4b. MLE PRE-TRAINING LOSS
# ═════════════════════════════════════════════════════════════════════
class MLEPretrainLoss(tf.keras.layers.Layer):
    """Weighted binary cross-entropy loss for pre-training on MLE labels.
    Implements: L = mean_i( w_i * sum_a BCE(sigma(target_a), sigma(pred_a)) )
    Uses tf.nn.sigmoid_cross_entropy_with_logits for numerical stability
    (never computes sigmoid explicitly in the loss path).
    Returns (loss, 0.0) to match TCRLikelihoodLoss (nll, reg) signature.
    """
    def __init__(self, l2_reg_lambda=0.0, hla_prior_logits=0., **kwargs):
        super().__init__(**kwargs)
        self.l2_reg_lambda = l2_reg_lambda
        self.hla_prior_logits = tf.constant(hla_prior_logits, dtype=tf.float32)
    def call(self, z_logits, target_logits, confidence_weights):
        """Compute weighted BCE between predicted and target logits.
        Args:
            z_logits:           (B, A) model output logits.
            target_logits:      (B, A) MLE pseudo-label logits.
            confidence_weights: (B,) per-TCR reliability weights.
        Returns:
            (loss, tf.constant(0.0)) — weighted mean BCE and zero reg.
        """
        # ── convert target logits to probabilities for BCE labels ────
        target_probs = tf.sigmoid(target_logits)  # (B, A)
        # ── per-element BCE (numerically stable via logits) ──────────
        bce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=target_probs, logits=z_logits)  # (B, A)
        # ── sum over alleles per TCR ─────────────────────────────────
        bce_per_tcr = tf.reduce_mean(bce, axis=1)  # (B,)
        # ── weight by confidence and mean over batch ─────────────────
        weighted = bce_per_tcr * confidence_weights  # (B,)
        # Normalize by sum of weights (avoids dilution by zero-weight)
        weight_sum = tf.maximum(
            tf.reduce_sum(confidence_weights), 1e-7)
        loss = tf.reduce_sum(weighted) / weight_sum

        drift = tf.square(z_logits - self.hla_prior_logits[tf.newaxis, :])
        reg = self.l2_reg_lambda * tf.reduce_mean(drift)
        return loss, reg
# ═════════════════════════════════════════════════════════════════════
# 5.  LEARNING RATE SCHEDULE
# ═════════════════════════════════════════════════════════════════════
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay to min_lr."""
    def __init__(self, peak_lr, min_lr, warmup_steps, total_steps):
        super().__init__()
        self.peak_lr = tf.cast(peak_lr, tf.float32)
        self.min_lr = tf.cast(min_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps = tf.cast(
            max(total_steps, warmup_steps + 1), tf.float32)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.peak_lr * (
            step / tf.maximum(self.warmup_steps, 1.0))
        progress = (step - self.warmup_steps) / tf.maximum(
            self.total_steps - self.warmup_steps, 1.0)
        progress = tf.minimum(progress, 1.0)
        cosine_lr = self.min_lr + 0.5 * (
            self.peak_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)
    def get_config(self):
        return {"peak_lr": float(self.peak_lr.numpy()),
                "min_lr": float(self.min_lr.numpy()),
                "warmup_steps": int(self.warmup_steps.numpy()),
                "total_steps": int(self.total_steps.numpy())}
class WarmupCosineDecayRestarts(
        tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by Cosine Annealing with Warm Restarts."""
    def __init__(self, peak_lr, min_lr, warmup_steps, cycle_steps):
        super().__init__()
        self.peak_lr = tf.cast(peak_lr, tf.float32)
        self.min_lr = tf.cast(min_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.cycle_steps = tf.cast(max(cycle_steps, 1), tf.float32)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.peak_lr * (
            step / tf.maximum(self.warmup_steps, 1.0))
        step_after = tf.maximum(step - self.warmup_steps, 0.0)
        step_in_cycle = tf.math.mod(step_after, self.cycle_steps)
        progress = step_in_cycle / self.cycle_steps
        cosine_lr = self.min_lr + 0.5 * (
            self.peak_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)
    def get_config(self):
        return {"peak_lr": float(self.peak_lr.numpy()),
                "min_lr": float(self.min_lr.numpy()),
                "warmup_steps": int(self.warmup_steps.numpy()),
                "cycle_steps": int(self.cycle_steps.numpy())}
# ═════════════════════════════════════════════════════════════════════
# 6.  TRAINING STEPS — LIKELIHOOD (original)
# ═════════════════════════════════════════════════════════════════════
@tf.function(reduce_retracing=True)
def train_step_lean(model, loss_fn, optimizer,
                    combined_cdr, combined_mask, binder_dense,
                    donor_indices, grad_clip):
    """Fast likelihood training step without diagnostics."""
    with tf.GradientTape() as tape:
        z_logits = model(
            [combined_cdr, combined_mask], training=True)
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
    return {"total_loss": total_loss, "nll": nll, "reg": reg,
            "grad_norm": grad_norm}
@tf.function(reduce_retracing=True)
def train_step_with_diag(model, loss_fn, optimizer,
                         combined_cdr, combined_mask, binder_dense,
                         donor_indices, grad_clip):
    """Likelihood training step with full diagnostics."""
    with tf.GradientTape() as tape:
        z_logits = model(
            [combined_cdr, combined_mask], training=True)
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
    diag = compute_diagnostics(z_logits, binder_dense, model)
    return {"total_loss": total_loss, "nll": nll, "reg": reg,
            "grad_norm": grad_norm, **diag}
# ═════════════════════════════════════════════════════════════════════
# 6b. TRAINING STEPS — MLE PRE-TRAINING
# ═════════════════════════════════════════════════════════════════════
@tf.function(reduce_retracing=True)
def pretrain_step_lean(model, loss_fn, optimizer,
                       combined_cdr, combined_mask,
                       target_logits, confidence_weights, grad_clip):
    """Fast MLE pre-training step without diagnostics.
    Args:
        combined_cdr:       (B, L, 21) float32
        combined_mask:      (B, L)     int32
        target_logits:      (B, A)     float32 — MLE pseudo-label logits
        confidence_weights: (B,)       float32 — per-TCR weights
        grad_clip:          scalar     float32
    Returns:
        dict with total_loss, bce, reg (=0), grad_norm.
    """
    with tf.GradientTape() as tape:
        z_logits = model(
            [combined_cdr, combined_mask], training=True)
        bce, reg = loss_fn(z_logits, target_logits, confidence_weights)
        total_loss = bce + reg
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
    return {"total_loss": total_loss, "bce": bce, "reg": reg,
            "grad_norm": grad_norm}
@tf.function(reduce_retracing=True)
def pretrain_step_with_diag(model, loss_fn, optimizer,
                            combined_cdr, combined_mask,
                            target_logits, confidence_weights,
                            binder_dense, grad_clip):
    """MLE pre-training step with full diagnostics.
    Accepts binder_dense additionally for compute_diagnostics.
    Args:
        combined_cdr:       (B, L, 21) float32
        combined_mask:      (B, L)     int32
        target_logits:      (B, A)     float32
        confidence_weights: (B,)       float32
        binder_dense:       (B, A)     float32 — for diagnostics only
        grad_clip:          scalar     float32
    Returns:
        dict with total_loss, bce, reg, grad_norm, and diagnostics.
    """
    with tf.GradientTape() as tape:
        z_logits = model(
            [combined_cdr, combined_mask], training=True)
        bce, reg = loss_fn(z_logits, target_logits, confidence_weights)
        total_loss = bce + reg
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
    diag = compute_diagnostics(z_logits, binder_dense, model)
    return {"total_loss": total_loss, "bce": bce, "reg": reg,
            "grad_norm": grad_norm, **diag}
@tf.function(reduce_retracing=True)
def pretrain_eval_step(model, loss_fn,
                       combined_cdr, combined_mask,
                       target_logits, confidence_weights,
                       binder_dense):
    """Forward-only evaluation step for MLE pre-training.
    Args:
        combined_cdr:       (B, L, 21) float32
        combined_mask:      (B, L)     int32
        target_logits:      (B, A)     float32
        confidence_weights: (B,)       float32
        binder_dense:       (B, A)     float32 — for diagnostics
    Returns:
        dict with total_loss, bce, reg, and diagnostics.
    """
    z_logits = model(
        [combined_cdr, combined_mask], training=False)
    bce, reg = loss_fn(z_logits, target_logits, confidence_weights)
    total_loss = bce + reg
    diag = compute_diagnostics(z_logits, binder_dense, model)
    return {"total_loss": total_loss, "bce": bce, "reg": reg, **diag}
# ═════════════════════════════════════════════════════════════════════
# 7.  DIAGNOSTICS + EVAL STEP (likelihood)
# ═════════════════════════════════════════════════════════════════════
@tf.function(reduce_retracing=True)
def compute_diagnostics(z_logits, binder_dense, model=None):
    """Vectorised diagnostic metrics shared by train and eval steps."""
    eps = 1e-7
    z_prob = tf.sigmoid(z_logits)
    active_mask = tf.cast(binder_dense > 0.5, tf.float32)
    inactive_mask = 1.0 - active_mask
    n_active = tf.reduce_sum(active_mask)
    n_inactive = tf.reduce_sum(inactive_mask)
    # ── active allele metrics ────────────────────────────────────────
    mean_active_gamma = tf.reduce_sum(
        z_prob * active_mask) / tf.maximum(n_active, 1.0)
    masked_logits = z_logits * active_mask + (
        1.0 - active_mask) * -1e9
    max_gamma_per = tf.reduce_max(
        tf.sigmoid(masked_logits), axis=-1)
    mean_max_gamma = tf.reduce_mean(max_gamma_per)
    p_safe = tf.clip_by_value(z_prob, eps, 1.0 - eps)
    ent = -(p_safe * tf.math.log(p_safe) + (
        1.0 - p_safe) * tf.math.log(1.0 - p_safe))
    entropy_active = tf.reduce_sum(
        ent * active_mask) / tf.maximum(n_active, 1.0)
    top2_vals, _ = tf.math.top_k(masked_logits, k=2)
    logit_gap = tf.reduce_mean(top2_vals[:, 0] - top2_vals[:, 1])
    probs_for_min = z_prob + (1.0 - active_mask) * 1e9
    min_active_prob = tf.reduce_mean(
        tf.reduce_min(probs_for_min, axis=-1))
    mean_n_active = tf.reduce_mean(
        tf.reduce_sum(active_mask, axis=-1))
    # ── inactive allele metrics ──────────────────────────────────────
    mean_inactive_prob = tf.reduce_sum(
        z_prob * inactive_mask) / tf.maximum(n_inactive, 1.0)
    max_inact = tf.reduce_max(
        z_prob * inactive_mask + active_mask * -1e9, axis=-1)
    mean_max_inactive_prob = tf.reduce_mean(max_inact)
    min_inact = tf.reduce_min(
        z_prob + active_mask * 1e9, axis=-1)
    mean_min_inactive_prob = tf.reduce_mean(min_inact)
    # ── diversity tracking ───────────────────────────────────────────
    num_alleles = z_logits.shape[-1]
    batch_size_float = tf.cast(tf.shape(z_logits)[0], tf.float32)
    top1_active = tf.argmax(masked_logits, axis=-1)
    active_counts = tf.reduce_sum(
        tf.one_hot(top1_active, depth=num_alleles), axis=0)
    num_unique_top1_active = tf.reduce_sum(
        tf.cast(active_counts > 0, tf.float32))
    max_batch_frac_active = tf.reduce_max(
        active_counts) / tf.maximum(batch_size_float, 1.0)
    top1_global = tf.argmax(z_logits, axis=-1)
    global_counts = tf.reduce_sum(
        tf.one_hot(top1_global, depth=num_alleles), axis=0)
    num_unique_top1_global = tf.reduce_sum(
        tf.cast(global_counts > 0, tf.float32))
    max_batch_frac_global = tf.reduce_max(
        global_counts) / tf.maximum(batch_size_float, 1.0)
    # ── top-5 global alleles ─────────────────────────────────────────
    mean_prob_per = tf.reduce_mean(z_prob, axis=0)
    top5g_prob, top5g_idx = tf.math.top_k(mean_prob_per, k=5)
    active_frac_per = tf.reduce_mean(active_mask, axis=0)
    top5g_frac = tf.gather(active_frac_per, top5g_idx)
    # ── top-5 active alleles ─────────────────────────────────────────
    active_count = tf.reduce_sum(active_mask, axis=0)
    mean_prob_where_act = tf.reduce_sum(
        z_prob * active_mask, axis=0) / tf.maximum(active_count, 1.0)
    top5a_prob, top5a_idx = tf.math.top_k(mean_prob_where_act, k=5)
    top5a_frac = tf.gather(
        active_count, top5a_idx) / tf.maximum(batch_size_float, 1.0)
    # ── temperature ──────────────────────────────────────────────────
    temperature = tf.constant(0.0)
    if model is not None:
        temperature = tf.exp(
            model.get_layer("clip_binding_head").log_temperature)
    return {
        "mean_active_gamma": mean_active_gamma,
        "mean_max_gamma": mean_max_gamma,
        "entropy_active": entropy_active,
        "logit_gap_top1_top2": logit_gap,
        "min_active_prob": min_active_prob,
        "mean_n_active_alleles": mean_n_active,
        "mean_inactive_prob": mean_inactive_prob,
        "mean_max_inactive_prob": mean_max_inactive_prob,
        "mean_min_inactive_prob": mean_min_inactive_prob,
        "num_unique_top1_active": num_unique_top1_active,
        "max_batch_frac_active": max_batch_frac_active,
        "num_unique_top1_global": num_unique_top1_global,
        "max_batch_frac_global": max_batch_frac_global,
        "top5g_idx_0": tf.cast(top5g_idx[0], tf.float32),
        "top5g_idx_1": tf.cast(top5g_idx[1], tf.float32),
        "top5g_idx_2": tf.cast(top5g_idx[2], tf.float32),
        "top5g_idx_3": tf.cast(top5g_idx[3], tf.float32),
        "top5g_idx_4": tf.cast(top5g_idx[4], tf.float32),
        "top5g_prob_0": top5g_prob[0],
        "top5g_prob_1": top5g_prob[1],
        "top5g_prob_2": top5g_prob[2],
        "top5g_prob_3": top5g_prob[3],
        "top5g_prob_4": top5g_prob[4],
        "top5g_frac_0": top5g_frac[0],
        "top5g_frac_1": top5g_frac[1],
        "top5g_frac_2": top5g_frac[2],
        "top5g_frac_3": top5g_frac[3],
        "top5g_frac_4": top5g_frac[4],
        "top5a_idx_0": tf.cast(top5a_idx[0], tf.float32),
        "top5a_idx_1": tf.cast(top5a_idx[1], tf.float32),
        "top5a_idx_2": tf.cast(top5a_idx[2], tf.float32),
        "top5a_idx_3": tf.cast(top5a_idx[3], tf.float32),
        "top5a_idx_4": tf.cast(top5a_idx[4], tf.float32),
        "top5a_prob_0": top5a_prob[0],
        "top5a_prob_1": top5a_prob[1],
        "top5a_prob_2": top5a_prob[2],
        "top5a_prob_3": top5a_prob[3],
        "top5a_prob_4": top5a_prob[4],
        "top5a_frac_0": top5a_frac[0],
        "top5a_frac_1": top5a_frac[1],
        "top5a_frac_2": top5a_frac[2],
        "top5a_frac_3": top5a_frac[3],
        "top5a_frac_4": top5a_frac[4],
        "temperature": temperature,
    }
@tf.function(reduce_retracing=True)
def eval_step(model, loss_fn,
              combined_cdr, combined_mask, binder_dense, donor_indices):
    """Forward-only evaluation step for likelihood training."""
    z_logits = model(
        [combined_cdr, combined_mask], training=False)
    nll, reg = loss_fn(z_logits, binder_dense, donor_indices)
    total_loss = nll + reg
    diag = compute_diagnostics(z_logits, binder_dense, model)
    return {"total_loss": total_loss, "nll": nll, "reg": reg, **diag}
# ═════════════════════════════════════════════════════════════════════
# 8.  METRIC LOGGER
# ═════════════════════════════════════════════════════════════════════
import csv
class MetricLogger:
    """Writes epoch metrics to CSV, JSON, and TensorBoard."""
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
        row = {"epoch": epoch, "split": split, **metrics}
        if extra:
            row.update(extra)
        self._history.append(row)
        if not self._header_written:
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()
            self._header_written = True
        else:
            if self._csv_file is None or self._csv_file.closed:
                self._csv_file = open(self.csv_path, "a", newline="")
                self._csv_writer = csv.DictWriter(
                    self._csv_file, fieldnames=list(row.keys()))
        self._csv_writer.writerow(row)
        self._csv_file.flush()
        with open(self.json_path, "w") as f:
            json.dump(self._history, f, indent=2, cls=NumpyEncoder)
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
# 9.  EPOCH RUNNERS — LIKELIHOOD (original, with background prefetch)
# ═════════════════════════════════════════════════════════════════════
import threading, queue
def _prefetch_batches(reader, args, prefetch_q, max_prefetch=3,
                      mle_pretrain=False):
    """Background thread: reads HDF5 chunks, prepares tensors, pushes
    into queue. GPU never waits for data prep.
    Args:
        reader:       opened PublicTcrHlaCsrReaderChunk context.
        args:         parsed flags.
        prefetch_q:   queue.Queue shared with main thread.
        max_prefetch: queue capacity.
        mle_pretrain: if True, include pretrain targets in batch.
    """
    min_donors = int(args.keep_only_upperthan_n_donors)
    for chunk in reader.iter_cluster_chunks(chunk_rows=args.batch_size):
        if chunk.counts_dense is None or chunk.counts_dense.shape[0] == 0:
            continue
        if chunk.cdr_freq is None:
            continue
        # ── donor-count filtering (vectorised) ───────────────────────
        if min_donors > 1:
            keep_mask = chunk.n_donors >= min_donors
            if not np.any(keep_mask):
                continue
            if not np.all(keep_mask):
                batch = prepare_batch(chunk, args,
                                      mle_pretrain=mle_pretrain)
                keep_idx = tf.constant(
                    np.where(keep_mask)[0], dtype=tf.int32)
                batch = {k: tf.gather(v, keep_idx)
                         for k, v in batch.items()}
                prefetch_q.put(batch)
                continue
        batch = prepare_batch(chunk, args, mle_pretrain=mle_pretrain)
        prefetch_q.put(batch)
    prefetch_q.put(None)  # sentinel
def run_epoch(model, loss_fn, optimizer, h5_path, args,
              tb_writer, global_step, epoch, is_train=True):
    """Stream through H5 with background prefetching (likelihood mode).
    Returns (mean_metrics_dict, updated_global_step).
    """
    tag = "train" if is_train else "valid"
    accum = {}
    n_steps = 0
    n_diag_steps = 0
    grad_clip_t = tf.constant(args.grad_clip, tf.float32)
    prefetch_q = queue.Queue(maxsize=3)
    reader = PublicTcrHlaCsrReaderChunk(
        h5_path, include_counts=True, include_donors=True,
        include_pvals=False, include_cdr_freq=True)
    reader.open()
    loader_thread = threading.Thread(
        target=_prefetch_batches,
        args=(reader, args, prefetch_q, 3, False), daemon=True)
    loader_thread.start()
    while True:
        batch = prefetch_q.get()
        if batch is None:
            break
        cdr  = batch["combined_cdr"]
        mask = batch["combined_mask"]
        bind = batch["binder_dense"]
        dids = batch["donor_indices"]
        if is_train:
            if (n_steps + 1) % args.log_step == 0:
                metrics = train_step_with_diag(
                    model, loss_fn, optimizer,
                    cdr, mask, bind, dids, grad_clip_t)
            else:
                metrics = train_step_lean(
                    model, loss_fn, optimizer,
                    cdr, mask, bind, dids, grad_clip_t)
        else:
            metrics = eval_step(model, loss_fn, cdr, mask, bind, dids)
        n_steps += 1
        has_diag = "mean_active_gamma" in metrics
        if has_diag:
            n_diag_steps += 1
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + float(v)
        if is_train and n_steps % args.log_step == 0:
            with tb_writer.as_default(step=global_step):
                for k, v in metrics.items():
                    tf.summary.scalar(f"{tag}_step/{k}", v)
                current_lr = optimizer.learning_rate
                if callable(current_lr):
                    current_lr = current_lr(optimizer.iterations)
                tf.summary.scalar(f"{tag}_step/learning_rate", current_lr)
            parts = [f"  [{tag}] epoch {epoch} step {n_steps}"]
            for k, v in metrics.items():
                parts.append(f"{k}={float(v):.5f}")
            print(" | ".join(parts))
        if is_train:
            global_step += 1
    loader_thread.join()
    reader.close()
    if n_steps == 0:
        print(f"  [{tag}] WARNING: 0 steps — check dataset")
        return {}, global_step
    core_keys = {"total_loss", "nll", "reg", "grad_norm"}
    means = {}
    for k, v in accum.items():
        if k in core_keys:
            means[k] = v / max(n_steps, 1)
        else:
            means[k] = v / max(n_diag_steps, 1)
    print(f"  [{tag}] epoch {epoch} done | steps={n_steps} | "
          + " | ".join(f"{k}={v:.5f}" for k, v in
                       list(means.items())))
    return means, global_step
def run_epoch_tfrecord(model, loss_fn, optimizer, dataset, args,
                       tb_writer, global_step, epoch, is_train=True):
    """Run one epoch from tf.data.Dataset (TFRecord, likelihood mode)."""
    tag = "train" if is_train else "valid"
    accum = {}
    n_steps = 0
    n_diag_steps = 0
    grad_clip_t = tf.constant(args.grad_clip, tf.float32)
    for cdr, mask, binder, donors in dataset:
        if is_train:
            if (n_steps + 1) % args.log_step == 0:
                metrics = train_step_with_diag(
                    model, loss_fn, optimizer,
                    cdr, mask, binder, donors, grad_clip_t)
            else:
                metrics = train_step_lean(
                    model, loss_fn, optimizer,
                    cdr, mask, binder, donors, grad_clip_t)
        else:
            metrics = eval_step(
                model, loss_fn, cdr, mask, binder, donors)
        n_steps += 1
        has_diag = "mean_active_gamma" in metrics
        if has_diag:
            n_diag_steps += 1
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + float(v)
        if is_train and n_steps % args.log_step == 0:
            with tb_writer.as_default(step=global_step):
                for k, v in metrics.items():
                    tf.summary.scalar(f"{tag}_step/{k}", v)
                current_lr = optimizer.learning_rate
                if callable(current_lr):
                    current_lr = current_lr(optimizer.iterations)
                tf.summary.scalar(f"{tag}_step/learning_rate", current_lr)
            parts = [f"  [{tag}] epoch {epoch} step {n_steps}"]
            for k, v in metrics.items():
                parts.append(f"{k}={float(v):.5f}")
            print(" | ".join(parts))
        if is_train:
            global_step += 1
    if n_steps == 0:
        print(f"  [{tag}] WARNING: 0 steps — check TFRecords")
        return {}, global_step
    core_keys = {"total_loss", "nll", "reg", "grad_norm"}
    means = {}
    for k, v in accum.items():
        if k in core_keys:
            means[k] = v / max(n_steps, 1)
        else:
            means[k] = v / max(n_diag_steps, 1)
    print(f"  [{tag}] epoch {epoch} done | steps={n_steps} | "
          + " | ".join(f"{k}={v:.5f}" for k, v in
                       list(means.items())))
    return means, global_step
# ═════════════════════════════════════════════════════════════════════
# 9b. EPOCH RUNNERS — MLE PRE-TRAINING
# ═════════════════════════════════════════════════════════════════════
def run_epoch_pretrain(model, loss_fn, optimizer, h5_path, args,
                       tb_writer, global_step, epoch, is_train=True):
    """Stream through H5 with background prefetching (pretrain mode).
    Opens reader with include_z_probs=True for MLE pseudo-labels.
    Returns (mean_metrics_dict, updated_global_step).
    """
    tag = "train" if is_train else "valid"
    accum = {}
    n_steps = 0
    n_diag_steps = 0
    grad_clip_t = tf.constant(args.grad_clip, tf.float32)
    prefetch_q = queue.Queue(maxsize=3)
    # ── open reader with z_probs enabled ─────────────────────────────
    reader = PublicTcrHlaCsrReaderChunk(
        h5_path, include_counts=True, include_donors=True,
        include_pvals=False, include_cdr_freq=True,
        include_z_probs=True)
    reader.open()
    loader_thread = threading.Thread(
        target=_prefetch_batches,
        args=(reader, args, prefetch_q, 3, True),  # mle_pretrain=True
        daemon=True)
    loader_thread.start()
    while True:
        batch = prefetch_q.get()
        if batch is None:
            break
        cdr   = batch["combined_cdr"]
        mask  = batch["combined_mask"]
        bind  = batch["binder_dense"]
        tgt   = batch["target_logits"]
        conf  = batch["confidence_weights"]
        if is_train:
            if (n_steps + 1) % args.log_step == 0:
                metrics = pretrain_step_with_diag(
                    model, loss_fn, optimizer,
                    cdr, mask, tgt, conf, bind, grad_clip_t)
            else:
                metrics = pretrain_step_lean(
                    model, loss_fn, optimizer,
                    cdr, mask, tgt, conf, grad_clip_t)
        else:
            metrics = pretrain_eval_step(
                model, loss_fn, cdr, mask, tgt, conf, bind)
        n_steps += 1
        has_diag = "mean_active_gamma" in metrics
        if has_diag:
            n_diag_steps += 1
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + float(v)
        if is_train and n_steps % args.log_step == 0:
            with tb_writer.as_default(step=global_step):
                for k, v in metrics.items():
                    tf.summary.scalar(f"{tag}_step/{k}", v)
                current_lr = optimizer.learning_rate
                if callable(current_lr):
                    current_lr = current_lr(optimizer.iterations)
                tf.summary.scalar(f"{tag}_step/learning_rate", current_lr)

            parts = [f"  [{tag}] epoch {epoch} step {n_steps}"]
            for k, v in metrics.items():
                parts.append(f"{k}={float(v):.5f}")
            print(" | ".join(parts))
        if is_train:
            global_step += 1
    loader_thread.join()
    reader.close()
    if n_steps == 0:
        print(f"  [{tag}] WARNING: 0 steps — check dataset/z_probs")
        return {}, global_step
    core_keys = {"total_loss", "bce", "reg", "grad_norm"}
    means = {}
    for k, v in accum.items():
        if k in core_keys:
            means[k] = v / max(n_steps, 1)
        else:
            means[k] = v / max(n_diag_steps, 1)
    print(f"  [{tag}] epoch {epoch} done | steps={n_steps} | "
          + " | ".join(f"{k}={v:.5f}" for k, v in
                       list(means.items())))
    return means, global_step
def run_epoch_pretrain_tfrecord(model, loss_fn, optimizer, dataset,
                                args, tb_writer, global_step, epoch,
                                is_train=True):
    """Run one epoch from tf.data.Dataset (TFRecord, pretrain mode).
    Dataset yields 6-tuples: (cdr, mask, binder, donors, tgt, conf).
    Returns (mean_metrics_dict, updated_global_step).
    """
    tag = "train" if is_train else "valid"
    accum = {}
    n_steps = 0
    n_diag_steps = 0
    grad_clip_t = tf.constant(args.grad_clip, tf.float32)
    for cdr, mask, binder, donors, tgt, conf in dataset:
        if is_train:
            if (n_steps + 1) % args.log_step == 0:
                metrics = pretrain_step_with_diag(
                    model, loss_fn, optimizer,
                    cdr, mask, tgt, conf, binder, grad_clip_t)
            else:
                metrics = pretrain_step_lean(
                    model, loss_fn, optimizer,
                    cdr, mask, tgt, conf, grad_clip_t)
        else:
            metrics = pretrain_eval_step(
                model, loss_fn, cdr, mask, tgt, conf, binder)
        n_steps += 1
        has_diag = "mean_active_gamma" in metrics
        if has_diag:
            n_diag_steps += 1
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + float(v)
        if is_train and n_steps % args.log_step == 0:
            with tb_writer.as_default(step=global_step):
                for k, v in metrics.items():
                    tf.summary.scalar(f"{tag}_step/{k}", v)
                current_lr = optimizer.learning_rate
                if callable(current_lr):
                    current_lr = current_lr(optimizer.iterations)
                tf.summary.scalar(f"{tag}_step/learning_rate", current_lr)

            parts = [f"  [{tag}] epoch {epoch} step {n_steps}"]
            for k, v in metrics.items():
                parts.append(f"{k}={float(v):.5f}")
            print(" | ".join(parts))
        if is_train:
            global_step += 1
    if n_steps == 0:
        print(f"  [{tag}] WARNING: 0 steps — check TFRecords")
        return {}, global_step
    core_keys = {"total_loss", "bce", "reg", "grad_norm"}
    means = {}
    for k, v in accum.items():
        if k in core_keys:
            means[k] = v / max(n_steps, 1)
        else:
            means[k] = v / max(n_diag_steps, 1)
    print(f"  [{tag}] epoch {epoch} done | steps={n_steps} | "
          + " | ".join(f"{k}={v:.5f}" for k, v in
                       list(means.items())))
    return means, global_step
# ═════════════════════════════════════════════════════════════════════
# 10. CHECKPOINTING HELPERS
# ═════════════════════════════════════════════════════════════════════
def save_checkpoint(model, optimizer, epoch, global_step, best_val_loss,
                    ckpt_dir, tag="latest"):
    """Save model weights + optimizer state + metadata."""
    path = os.path.join(ckpt_dir, tag)
    os.makedirs(path, exist_ok=True)
    model.save_weights(os.path.join(path, "model.weights.h5"))
    meta = {"epoch": epoch, "global_step": global_step,
            "best_val_loss": best_val_loss}
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, cls=NumpyEncoder, indent=2)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt.write(os.path.join(path, "ckpt"))
    print(f"  [CKPT] Saved {tag} checkpoint → {path}")
def load_checkpoint(model, optimizer, ckpt_dir, tag="latest"):
    """Load model weights + optimizer state + metadata."""
    path = os.path.join(ckpt_dir, tag)
    meta_path = os.path.join(path, "meta.json")
    if not os.path.exists(meta_path):
        print(f"  [CKPT] No checkpoint found at {path}")
        return None
    with open(meta_path, "r") as f:
        meta = json.load(f)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt.read(os.path.join(path, "ckpt")).expect_partial()
    print(f"  [CKPT] Restored {tag}: epoch={meta['epoch']} "
          f"step={meta['global_step']} "
          f"best_val={meta['best_val_loss']:.6f}")
    return meta
# ═════════════════════════════════════════════════════════════════════
# 11. INFERENCE
# ═════════════════════════════════════════════════════════════════════
def run_inference(model, h5_path, output_path, args):
    """Load best model and write z_probs into a copy of the input H5."""
    print(f"\n{'='*60}")
    print("INFERENCE MODE")
    print(f"{'='*60}")
    shutil.copy2(h5_path, output_path)
    print(f"  Copied {h5_path} → {output_path}")
    with PublicTcrHlaCsrReaderChunk(
            output_path, include_counts=True, include_donors=True,
            include_cdr_freq=True) as reader:
        num_clusters = reader.num_clusters
    print(f"  Total clusters: {num_clusters}")
    from tqdm import tqdm
    total_steps = math.ceil(num_clusters / args.batch_size)
    with MleZprobsWriter(
            output_path, num_clusters=num_clusters) as writer:
        with PublicTcrHlaCsrReaderChunk(
                output_path, include_counts=True, include_donors=True,
                include_pvals=False, include_cdr_freq=True) as reader:
            for chunk in tqdm(
                    reader.iter_cluster_chunks(
                        chunk_rows=args.batch_size),
                    total=total_steps, desc="Inferring z_probs",
                    unit="batch"):
                if chunk.counts_dense is None or \
                        chunk.cdr_freq is None:
                    continue
                batch = prepare_batch(chunk, args)
                z_logits = model(
                    [batch["combined_cdr"], batch["combined_mask"]],
                    training=False)
                z_probs = tf.sigmoid(z_logits).numpy()
                binder = chunk.counts_dense
                binder_binary = (binder > 0).astype(np.float32)
                n_chunk = z_probs.shape[0]
                max_hlas = int(binder_binary.sum(axis=1).max())
                binder_sets_padded = np.full(
                    (n_chunk, max_hlas), args.pad_token,
                    dtype=np.float32)
                z_probs_padded = np.full(
                    (n_chunk, max_hlas), 0.0, dtype=np.float32)
                for i in range(n_chunk):
                    nz_idx = np.where(binder_binary[i] > 0)[0]
                    binder_sets_padded[i, :len(nz_idx)] = nz_idx
                    z_probs_padded[i, :len(nz_idx)] = z_probs[
                        i, nz_idx]
                writer.write_chunk(
                    chunk.cluster_start, chunk.cluster_end,
                    binder_sets_padded, z_probs_padded,
                    pad_token=args.pad_token)
    print(f"  ✓ Inference complete → {output_path}")
def run_inference_tfrecord(model, tfrecord_dir, output_path, args):
    """Run inference on TFRecord shards and save results to NPZ."""
    print(f"\n{'='*60}")
    print("INFERENCE MODE (TFRecord)")
    print(f"{'='*60}")
    shard_paths = sorted([
        os.path.join(tfrecord_dir, f)
        for f in os.listdir(tfrecord_dir)
        if f.endswith(".tfrecord")])
    if not shard_paths:
        print(f"ERROR: No .tfrecord files found in {tfrecord_dir}")
        sys.exit(1)
    print(f"  Found {len(shard_paths)} shard files in {tfrecord_dir}")
    manifest_path = os.path.join(tfrecord_dir, "manifest.json")
    n_clusters_est = None
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        n_clusters_est = manifest.get("num_clusters")
    num_alleles = args.num_alleles
    def _parse(example_proto):
        desc = {
            "combined_cdr_flat": tf.io.VarLenFeature(tf.float32),
            "combined_mask": tf.io.VarLenFeature(tf.int64),
            "binder_dense": tf.io.FixedLenFeature(
                [num_alleles], tf.float32),
            "donor_indices": tf.io.VarLenFeature(tf.int64),
            "seq_len": tf.io.FixedLenFeature([], tf.int64),
        }
        parsed = tf.io.parse_single_example(example_proto, desc)
        seq_len = parsed["seq_len"]
        cdr = tf.reshape(
            tf.sparse.to_dense(parsed["combined_cdr_flat"]),
            [seq_len, 21])
        mask = tf.cast(
            tf.sparse.to_dense(parsed["combined_mask"]), tf.int32)
        binder = parsed["binder_dense"]
        donors = tf.cast(
            tf.sparse.to_dense(parsed["donor_indices"]), tf.int32)
        return cdr, mask, binder, donors
    dataset = tf.data.TFRecordDataset(shard_paths)
    dataset = dataset.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    pad_shapes = ([None, 21], [None], [num_alleles], [None])
    pad_values = (
        tf.constant(0.0, tf.float32),
        tf.constant(args.pad_token, tf.int32),
        tf.constant(0.0, tf.float32),
        tf.constant(args.pad_token, tf.int32))
    dataset = dataset.padded_batch(
        args.batch_size, padded_shapes=pad_shapes,
        padding_values=pad_values, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    from tqdm import tqdm
    total_steps = (math.ceil(n_clusters_est / args.batch_size)
                   if n_clusters_est else None)
    all_logits, all_probs, all_binders, all_donors = [], [], [], []
    n_processed = 0
    max_donor_len = 0
    for cdr, mask, binder, donors in tqdm(
            dataset, total=total_steps, desc="Inferring", unit="batch"):
        z_logits = model([cdr, mask], training=False)
        z_probs = tf.sigmoid(z_logits)
        all_logits.append(z_logits.numpy())
        all_probs.append(z_probs.numpy())
        all_binders.append(binder.numpy())
        donors_np = donors.numpy()
        all_donors.append(donors_np)
        max_donor_len = max(max_donor_len, donors_np.shape[1])
        n_processed += z_logits.shape[0]
    if n_processed == 0:
        print("ERROR: 0 clusters processed")
        sys.exit(1)
    z_logits_all = np.concatenate(all_logits, axis=0)
    z_probs_all = np.concatenate(all_probs, axis=0)
    binder_all = np.concatenate(all_binders, axis=0)
    donor_unified = np.full(
        (n_processed, max_donor_len), args.pad_token, dtype=np.int32)
    offset = 0
    for d in all_donors:
        n, cols = d.shape
        donor_unified[offset:offset + n, :cols] = d
        offset += n
    n_donors_all = np.sum(
        donor_unified != args.pad_token, axis=1).astype(np.int32)
    np.savez_compressed(
        output_path, z_logits=z_logits_all, z_probs=z_probs_all,
        binder_dense=binder_all, donor_indices=donor_unified,
        n_donors=n_donors_all)
    print(f"  Processed {n_processed} clusters")
    print(f"  ✓ Inference complete → {output_path}")
# ═════════════════════════════════════════════════════════════════════
# 12. MAIN
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
    print(f"  Mode:       {args.mode}"
          f"{' [MLE PRETRAIN]' if args.mle_pretrain else ''}")
    print(f"  Device:     {device_type}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Embed dim:  {args.embed_dim}")
    print(f"  Layers:     {args.num_layers}")
    print(f"  Heads:      {args.num_heads}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Epochs:     {args.epochs}")
    if args.keep_only_upperthan_n_donors > 1:
        print(f"  Min donors: {args.keep_only_upperthan_n_donors}")
    if args.tf_record_path:
        print(f"  TFRecord:   {args.tf_record_path}")
    if args.mle_pretrain:
        print(f"  Confidence cap: {args.mle_confidence_cap}")
    print(f"{'='*60}\n")
    # ── validate z_probs exist when mle_pretrain is set ──────────────
    if args.mle_pretrain and args.mode == "train":
        _validate_z_probs_exist(args.train_ds)
        if args.valid_ds and os.path.exists(args.valid_ds):
            try:
                _validate_z_probs_exist(args.valid_ds)
            except ValueError:
                print("[WARN] Validation H5 does not contain z_probs. "
                      "Validation will be skipped in pretrain mode.")
                args.valid_ds = ""
    # ── load donor HLA matrix ────────────────────────────────────────
    print("[DATA] Loading donor HLA matrix...")
    donor_data = np.load(args.donor_hla_matrix)
    donor_hla_matrix = donor_data["donor_hla_matrix"]
    print(f"  Donor HLA matrix shape: {donor_hla_matrix.shape}")
    epsilon = 1e-7
    allele_freqs = np.clip(
        donor_hla_matrix.mean(axis=0), epsilon, 1.0 - epsilon)
    hla_log_odds = np.log(
        allele_freqs / (1.0 - allele_freqs)).astype(np.float32)
    if args.cold_start:
        shift = hla_log_odds.max() + 4.6
        hla_log_odds = hla_log_odds - shift
    print(f"  Computed log-odds bias for {len(hla_log_odds)} alleles.")
    # ── load idx_to_hla ──────────────────────────────────────────────
    print("[DATA] Loading HLA index mapping...")
    with open(args.idx_to_hla, "r") as f:
        idx_to_hla = json.load(f)
    print(f"  {len(idx_to_hla)} HLA alleles")
    args.num_alleles = donor_hla_matrix.shape[1]
    print(f"  Updated num_alleles = {args.num_alleles}")
    # ── load HLA embeddings (optional) ───────────────────────────────
    hla_embed_matrix = None
    if args.hla_embed and os.path.exists(args.hla_embed):
        print("[DATA] Loading HLA embeddings...")
        hla_embed_matrix = load_hla_embeddings_for_clip(
            args, idx_to_hla)
        print(f"  HLA embed shape: {hla_embed_matrix.shape}")
    else:
        print("[DATA] No HLA embeddings — random init")
    # ── build model ──────────────────────────────────────────────────
    print("\n[MODEL] Building TCRtyper...")
    model = build_model(
        args, hla_embed_matrix, hla_bias_init=hla_log_odds)
    model.summary(line_length=100)
    # ── loss function ────────────────────────────────────────────────
    if args.mle_pretrain:
        print("[LOSS] Using MLEPretrainLoss (weighted BCE)")
        loss_fn = MLEPretrainLoss(l2_reg_lambda=args.l2_reg, hla_prior_logits=hla_log_odds, name="mle_pretrain_loss")
    else:
        print("[LOSS] Using TCRLikelihoodLoss")
        loss_fn = TCRLikelihoodLoss(
            donor_hla_matrix, beta=args.beta,
            pad_token=args.pad_token, l2_reg_lambda=args.l2_reg,
            reduction=args.reduction,
            poisson_approx_untyped_hlas=args.poisson_approx,
            hla_bias_init=hla_log_odds,
            invariant_lambda=args.invariant_lambda,
            false_pos_lambda=args.false_pos_lambda)
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
        total_steps = 1
        steps_per_epoch = 1
    n_valid = 0
    if args.valid_ds and os.path.exists(args.valid_ds):
        n_valid, _ = count_dataset_clusters(
            args.valid_ds, args.batch_size)
    # ── save config ──────────────────────────────────────────────────
    save_config(args, args.output_dir, device_type,
                donor_shape=donor_hla_matrix.shape,
                n_train_clusters=n_train,
                n_valid_clusters=n_valid)
    # ── optimizer with LR schedule ───────────────────────────────────
    cycle_length_steps = steps_per_epoch * 10
    if args.lr_schedule_restart:
        lr_schedule = WarmupCosineDecayRestarts(
            peak_lr=args.lr, min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            cycle_steps=cycle_length_steps)
    else:
        lr_schedule = WarmupCosineDecay(
            peak_lr=args.lr, min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            total_steps=steps_per_epoch * args.epochs)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=args.weight_decay,
        clipnorm=None)
    # ── warm up model with dummy forward pass ────────────────────────
    dummy_seq = tf.zeros((1, 10, 21), dtype=tf.float32)
    dummy_mask = tf.ones((1, 10), dtype=tf.int32)
    _ = model([dummy_seq, dummy_mask], training=False)
    # ════════════════════════════════════════════════════════════════
    # TRAIN MODE
    # ════════════════════════════════════════════════════════════════
    if args.mode == "train":
        tb_writer = tf.summary.create_file_writer(log_dir)
        metric_logger = MetricLogger(args.output_dir, tb_writer)
        # ── TFRecord cache conversion ────────────────────────────────
        tfr_base = (args.tf_record_path
                    if args.tf_record_path else args.output_dir)
        train_tfr_ds, valid_tfr_ds = None, None
        if args.use_tfrecord:
            print("\n[CACHE] Converting datasets to TFRecords ...")
            if args.tf_record_path:
                print(f"  [CACHE] Custom TFRecord path: {tfr_base}")
            train_shard_paths = convert_h5_to_tfrecords(
                args.train_ds, tfr_base, args,
                num_shards=args.num_shards, tag="train")
            train_tfr_ds = build_tfrecord_dataset(
                train_shard_paths, args,
                num_alleles=args.num_alleles,
                shuffle=True, drop_remainder=False,
                mle_pretrain=args.mle_pretrain)
            if args.valid_ds and os.path.exists(args.valid_ds):
                valid_shard_paths = convert_h5_to_tfrecords(
                    args.valid_ds, tfr_base, args,
                    num_shards=max(1, args.num_shards // 4),
                    tag="valid")
                valid_tfr_ds = build_tfrecord_dataset(
                    valid_shard_paths, args,
                    num_alleles=args.num_alleles,
                    shuffle=False, drop_remainder=False,
                    mle_pretrain=args.mle_pretrain)
        # ── resume from checkpoint ───────────────────────────────────
        start_epoch = 0
        global_step = 0
        best_val_loss = float("inf")
        if args.resume:
            meta = load_checkpoint(
                model, optimizer, ckpt_dir, "latest")
            if meta is not None:
                start_epoch = meta["epoch"] + 1
                global_step = meta["global_step"]
                best_val_loss = meta["best_val_loss"]
        patience_counter = 0
        # ── select epoch runner based on mode ────────────────────────
        if args.mle_pretrain:
            _run_ep = run_epoch_pretrain
            _run_ep_tfr = run_epoch_pretrain_tfrecord
        else:
            _run_ep = run_epoch
            _run_ep_tfr = run_epoch_tfrecord
        # ── epoch loop ───────────────────────────────────────────────
        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            print(f"\n{'─'*60}")
            print(f"Epoch {epoch + 1}/{args.epochs}"
                  f"{'  [PRETRAIN]' if args.mle_pretrain else ''}")
            print(f"{'─'*60}")
            # ── training ─────────────────────────────────────────────
            if args.use_tfrecord:
                train_metrics, global_step = _run_ep_tfr(
                    model, loss_fn, optimizer, train_tfr_ds, args,
                    tb_writer, global_step, epoch, is_train=True)
            else:
                train_metrics, global_step = _run_ep(
                    model, loss_fn, optimizer, args.train_ds, args,
                    tb_writer, global_step, epoch, is_train=True)
            train_loss = train_metrics.get("total_loss", 0.0)
            current_lr = optimizer.learning_rate
            if callable(current_lr):
                current_lr = float(
                    current_lr(optimizer.iterations).numpy())
            else:
                current_lr = float(current_lr)
            elapsed = time.time() - t0
            metric_logger.log_epoch(
                epoch, "train", train_metrics,
                extra={"learning_rate": current_lr,
                       "epoch_time_s": elapsed})
            # ── validation ───────────────────────────────────────────
            val_loss = float("inf")
            if args.valid_ds and os.path.exists(args.valid_ds):
                if args.use_tfrecord and valid_tfr_ds is not None:
                    val_metrics, _ = _run_ep_tfr(
                        model, loss_fn, optimizer, valid_tfr_ds,
                        args, tb_writer, global_step, epoch,
                        is_train=False)
                else:
                    val_metrics, _ = _run_ep(
                        model, loss_fn, optimizer, args.valid_ds,
                        args, tb_writer, global_step, epoch,
                        is_train=False)
                val_loss = val_metrics.get(
                    "total_loss", float("inf"))
                metric_logger.log_epoch(epoch, "valid", val_metrics)
            # ── checkpointing ────────────────────────────────────────
            if (epoch + 1) % args.save_every_epoch == 0:
                save_checkpoint(model, optimizer, epoch, global_step,
                                best_val_loss, ckpt_dir, "latest")
            # ── best model tracking ──────────────────────────────────
            monitor_loss = (val_loss if val_loss < float("inf")
                           else train_loss)
            if monitor_loss < best_val_loss:
                best_val_loss = monitor_loss
                save_checkpoint(model, optimizer, epoch, global_step,
                                best_val_loss, ckpt_dir, "best")
                patience_counter = 0
                print(f"  ★ New best model "
                      f"(loss={best_val_loss:.6f})")
            else:
                patience_counter += 1
                print(f"  No improvement "
                      f"({patience_counter}/{args.patience})")
            with tb_writer.as_default(step=epoch):
                tf.summary.scalar("epoch/time_seconds", elapsed)
                tf.summary.scalar(
                    "epoch/best_val_loss", best_val_loss)
            print(f"  Epoch time: {elapsed:.1f}s")
            # ── early stopping ───────────────────────────────────────
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch + 1}")
                break
        # ── final save ───────────────────────────────────────────────
        save_checkpoint(model, optimizer, epoch, global_step,
                        best_val_loss, ckpt_dir, "latest")
        tb_writer.flush()
        metric_logger.close()
        print(f"\n{'='*60}")
        loss_label = "BCE" if args.mle_pretrain else "NLL"
        print(f"Training complete — best {loss_label}: "
              f"{best_val_loss:.6f}")
        print(f"Outputs: {args.output_dir}")
        print(f"{'='*60}")
    # ════════════════════════════════════════════════════════════════
    # INFERENCE MODE
    # ════════════════════════════════════════════════════════════════
    elif args.mode == "inference":
        meta = load_checkpoint(model, optimizer, ckpt_dir, "best")
        if meta is None:
            meta = load_checkpoint(
                model, optimizer, ckpt_dir, "latest")
        if meta is None:
            print("ERROR: No checkpoint found for inference")
            sys.exit(1)
        if args.use_tfrecord:
            if not args.tf_record_path:
                print("ERROR: --tf_record_path required for "
                      "TFRecord inference")
                sys.exit(1)
            if not os.path.isdir(args.tf_record_path):
                print(f"ERROR: {args.tf_record_path} is not a dir")
                sys.exit(1)
            output_npz = os.path.join(
                args.output_dir, "predictions.npz")
            run_inference_tfrecord(
                model, args.tf_record_path, output_npz, args)
        else:
            infer_ds = (args.inference_ds
                        if args.inference_ds else args.train_ds)
            output_h5 = os.path.join(
                args.output_dir, "predictions.h5")
            run_inference(model, infer_ds, output_h5, args)
    # ════════════════════════════════════════════════════════════════
    # EXPORT MODE
    # ════════════════════════════════════════════════════════════════
    elif args.mode == "export":
        meta = load_checkpoint(model, optimizer, ckpt_dir, "best")
        if meta is None:
            meta = load_checkpoint(
                model, optimizer, ckpt_dir, "latest")
        if meta is None:
            print("ERROR: No checkpoint found for export")
            sys.exit(1)
        export_path = os.path.join(args.output_dir, "saved_model")
        model.save(export_path)
        print(f"  ✓ SavedModel exported → {export_path}")
if __name__ == "__main__":
    main()