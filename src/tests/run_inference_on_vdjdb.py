#!/usr/bin/env python3
"""
VDJdb Inference + Analysis Script for TCRtyper
================================================
Loads a trained TCRtyper checkpoint and runs inference on ground-truth
TCR-HLA pairs stored in a VDJdb .npz file.  Produces:
  1. Binary binding metrics  (AUROC, AUPRC, F1, …)
  2. Per-TCR CSV             (entropy, gap, rank, neighbourhood probs, …)
  3. HLA-head analysis       (learned HLA similarity clustermap, dendrogram)
  4. Neighbourhood analysis  (do FN TCRs light up similar HLAs?)
CDR2/CDR2.5 swap
-----------------
In fast_msa_freq.py the column indices for CDR2 and CDR2.5 were swapped.
The model was trained with this swap, so we replicate it here:
    model slot "cdr2"  ← actual CDR2.5  = parts[3]
    model slot "cdr25" ← actual CDR2    = parts[1]
Usage:
    python -m src.tests.run_inference_on_vdjdb \
        --model_dir  runs/exp01 \
        --output_dir results/vdjdb \
        [--analyze_hla_head] \
        [--neighborhood_k 10] \
        [--batch_size 512]
"""
from __future__ import annotations
# ── GPU memory growth MUST run before any TF import ──────────────────
import os as _os
def _early_gpu_config():
    """Set GPU memory growth before TF context is locked."""
    _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    _os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
    try:
        import tensorflow as _tf
        for gpu in _tf.config.list_physical_devices("GPU"):
            _tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass
_early_gpu_config()
# ── standard imports ─────────────────────────────────────────────────
import os, sys, json, argparse, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster / headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils import (
    SequenceEncoderLayer,
    GatedTransformerLayer,
)
# ═════════════════════════════════════════════════════════════════════
# 1.  CONSTANTS
# ═════════════════════════════════════════════════════════════════════
BASE = str(Path(__file__).resolve().parents[2])
# -- amino-acid alphabet (20 AAs + gap) --
AA_CHARS = "ACDEFGHIKLMNPQRSTVWY."
N_SYM = 21
# -- fast char → index lookup table (ASCII-indexed, vectorisable) --
AA_TO_IDX = np.full(128, 20, dtype=np.int8)  # default = gap index
for _i, _c in enumerate(AA_CHARS):
    AA_TO_IDX[ord(_c)] = _i
# -- one-hot identity matrix for OHE look-up --
OHE_MATRIX = np.eye(N_SYM, dtype=np.float32)
# -- CDR column indices inside '-'-split string --
# npz format: "cdr3 - cdr2 - cdr1 - cdr2.5"  [0] [1] [2] [3]
# We intentionally swap cdr2↔cdr2.5 to match training:
CDR3_COL  = 0
CDR2_COL  = 1  # actual CDR2  → goes into model's "cdr25" slot (swapped)
CDR1_COL  = 2
CDR25_COL = 3  # actual CDR2.5 → goes into model's "cdr2" slot (swapped)
# -- token constants (must match train_tcrtyper.py defaults) --
PAD_TOKEN    = -1
MASK_TOKEN   = -2
SEP_TOKEN    = -3
NORMAL_TOKEN = 1
MASKING_RATE = 0.0  # no masking at inference
# ═════════════════════════════════════════════════════════════════════
# 2.  ARGUMENT PARSER
# ═════════════════════════════════════════════════════════════════════
def parse_args():
    """Parse command-line flags for VDJdb inference and analysis."""
    p = argparse.ArgumentParser(
        description="TCRtyper inference + analysis on VDJdb ground-truth data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── I/O paths ────────────────────────────────────────────────────
    p.add_argument("--model_dir", type=str, required=True,
                   help="Path to train_tcrtyper.py output dir (config.json + checkpoints/)")
    p.add_argument("--path_npz", type=str,
                   default=os.path.join(BASE, "data/autotcr/vdjdb/vdjdb_tcr_hla_pairs.npz"),
                   help="Path to VDJdb npz")
    p.add_argument("--path_hla_to_id", type=str,
                   default=os.path.join(BASE, "data/autotcr/hla_to_id.json"),
                   help="JSON mapping HLA allele → column index in model output")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory for all output files (CSVs, plots, npy)")
    # ── inference settings ───────────────────────────────────────────
    p.add_argument("--batch_size", type=int, default=512,
                   help="Inference batch size")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Sigmoid threshold for binary binding call")
    # ── HLA-head analysis ────────────────────────────────────────────
    p.add_argument("--analyze_hla_head", action="store_true", default=False,
                   help="Extract hla_head weights, compute HLA similarity clustermap")
    p.add_argument("--n_hla_clusters", type=int, default=12,
                   help="Number of HLA clusters for flat clustering from dendrogram")
    # ── neighbourhood analysis ───────────────────────────────────────
    p.add_argument("--neighborhood_k", type=int, nargs="+", default=[5, 10, 20],
                   help="K values for neighbourhood recall analysis")
    return p.parse_args()
# ═════════════════════════════════════════════════════════════════════
# 3.  VECTORISED DATA LOADING
# ═════════════════════════════════════════════════════════════════════
def seq_to_ohe(seq: str) -> np.ndarray:
    """Convert an amino-acid string to one-hot encoding via lookup table.
    Args:
        seq: amino-acid string (e.g. 'CASSF...')
    Returns:
        np.ndarray (L, 21) float32
    """
    idx = AA_TO_IDX[np.frombuffer(seq.encode('ascii'), dtype=np.uint8)]
    return OHE_MATRIX[idx]
def load_and_prepare_data(
    path_npz: str,
    path_hla_to_id: str,
) -> Dict[str, np.ndarray]:
    """Load VDJdb npz, one-hot encode CDRs, resolve HLA indices.
    Returns dict with cdr*_list, hla_indices, labels, valid_mask, etc.
    """
    arr = np.load(path_npz, allow_pickle=False)
    tcr_seq4   = arr["tcr_seq4"]
    hla_allele = arr["hla_allele"]
    labels     = arr["labels"]
    n_total = len(labels)
    print(f"[DATA] Loaded {n_total} pairs "
          f"(pos={int(arr['n_positive'])}, neg={int(arr['n_negative'])})")
    assert len(tcr_seq4) == len(hla_allele) == n_total
    # -- load HLA → index mapping --
    with open(path_hla_to_id, 'r') as f:
        hla_to_id = json.load(f)
    print(f"[DATA] HLA vocabulary: {len(hla_to_id)} alleles")
    # -- resolve HLA allele → model column index --
    hla_indices = np.full(n_total, -1, dtype=np.int32)
    for i, allele in enumerate(hla_allele):
        hla_indices[i] = hla_to_id.get(str(allele), -1)
    valid_mask = hla_indices >= 0
    n_valid = int(valid_mask.sum())
    n_missing = n_total - n_valid
    if n_missing > 0:
        missing_alleles = set(hla_allele[~valid_mask])
        print(f"[WARN] {n_missing} samples with unknown HLA (excluded): {missing_alleles}")
    print(f"[DATA] {n_valid} valid samples")
    # -- parse CDR strings → OHE arrays --
    cdr1_list, cdr2_list, cdr25_list, cdr3_list = [], [], [], []
    placeholder = np.zeros((1, N_SYM), dtype=np.float32)
    for i in range(n_total):
        if not valid_mask[i]:
            cdr1_list.append(placeholder)
            cdr2_list.append(placeholder)
            cdr25_list.append(placeholder)
            cdr3_list.append(placeholder)
            continue
        parts = tcr_seq4[i].split('-')
        # model "cdr1"=actual CDR1, "cdr2"=actual CDR2.5(swap), "cdr25"=actual CDR2(swap)
        cdr1_list.append(seq_to_ohe(parts[CDR1_COL]))
        cdr2_list.append(seq_to_ohe(parts[CDR25_COL]))   # swap
        cdr25_list.append(seq_to_ohe(parts[CDR2_COL]))    # swap
        cdr3_list.append(seq_to_ohe(parts[CDR3_COL]))
    return {
        "cdr1_list": cdr1_list, "cdr2_list": cdr2_list,
        "cdr25_list": cdr25_list, "cdr3_list": cdr3_list,
        "hla_indices": hla_indices, "labels": labels,
        "valid_mask": valid_mask, "hla_allele": hla_allele,
        "tcr_seq4": tcr_seq4,
    }
# ═════════════════════════════════════════════════════════════════════
# 4.  FAST NUMPY PAD + MASK + CONCAT (mirrors train_tcrtyper.py)
# ═════════════════════════════════════════════════════════════════════
def _pad_and_mask_numpy(ragged_arrays, pad_token=PAD_TOKEN,
                        mask_token=MASK_TOKEN, masking_rate=0.0):
    """Fast numpy pad+mask for a list of (L_i, 21) arrays.
    Args:
        ragged_arrays: list of np.ndarray each (L_i, 21)
        pad_token:     mask value for padding positions
        mask_token:    mask value for masked positions
        masking_rate:  fraction of valid positions to mask (0 at inference)
    Returns:
        features: (B, max_L, 21) float32
        mask:     (B, max_L)     float32
    """
    lengths = np.array([a.shape[0] for a in ragged_arrays], dtype=np.int32)
    B = len(ragged_arrays)
    max_L = int(lengths.max()) if B > 0 else 0
    features = np.zeros((B, max_L, N_SYM), dtype=np.float32)
    mask = np.full((B, max_L), float(pad_token), dtype=np.float32)
    for i, (arr, L) in enumerate(zip(ragged_arrays, lengths)):
        features[i, :L, :] = arr
        mask[i, :L] = 1.0
    if masking_rate > 0.0:
        rand = np.random.random((B, max_L)).astype(np.float32)
        valid = mask > 0.0
        is_masked = (rand < masking_rate) & valid
        mask[is_masked] = float(mask_token)
        features[is_masked] = 0.0
    return features, mask
def _concat_cdrs_with_sep_numpy(feat_list, mask_list, sep_mask_val=SEP_TOKEN):
    """Concatenate CDR features+masks with separator columns in numpy.
    Args:
        feat_list: list of (B, L_k, 21) arrays
        mask_list: list of (B, L_k) arrays
        sep_mask_val: mask value for separator token
    Returns:
        combined_feat: (B, total_L, 21) float32
        combined_mask: (B, total_L)     float32
    """
    B = feat_list[0].shape[0]
    sep_feat = np.zeros((B, 1, N_SYM), dtype=np.float32)
    sep_mask = np.full((B, 1), float(sep_mask_val), dtype=np.float32)
    parts_f, parts_m = [], []
    for i, (f, m) in enumerate(zip(feat_list, mask_list)):
        parts_f.append(f)
        parts_m.append(m)
        if i < len(feat_list) - 1:
            parts_f.append(sep_feat)
            parts_m.append(sep_mask)
    return np.concatenate(parts_f, axis=1), np.concatenate(parts_m, axis=1)
def prepare_inference_batch(c1, c2, c25, c3):
    """Pad, mask, concat CDR OHE arrays into model-ready numpy tensors.
    Order: [cdr1, cdr2(=actual_cdr25), cdr25(=actual_cdr2), cdr3]
    Returns:
        combined_feat: (B, total_L, 21) float32
        combined_mask: (B, total_L)     int32
    """
    f1, m1 = _pad_and_mask_numpy(c1, PAD_TOKEN, MASK_TOKEN, MASKING_RATE)
    f2, m2 = _pad_and_mask_numpy(c2, PAD_TOKEN, MASK_TOKEN, MASKING_RATE)
    f25, m25 = _pad_and_mask_numpy(c25, PAD_TOKEN, MASK_TOKEN, MASKING_RATE)
    f3, m3 = _pad_and_mask_numpy(c3, PAD_TOKEN, MASK_TOKEN, MASKING_RATE)
    cf, cm = _concat_cdrs_with_sep_numpy(
        [f1, f2, f25, f3], [m1, m2, m25, m3], sep_mask_val=SEP_TOKEN)
    return cf, cm.astype(np.int32)
# ═════════════════════════════════════════════════════════════════════
# 5.  MODEL RECONSTRUCTION + CHECKPOINT LOADING
# ═════════════════════════════════════════════════════════════════════
class MaskedGlobalAveragePooling(layers.Layer):
    """Global average pool ignoring PAD positions (mirrors train_tcrtyper.py)."""
    def __init__(self, pad_token: int = -1, **kw):
        super().__init__(**kw)
        self.pad_token = pad_token
    def call(self, inputs):
        x, mask = inputs
        valid = tf.cast(tf.not_equal(mask, self.pad_token), tf.float32)
        x_masked = x * valid[:, :, tf.newaxis]
        summed = tf.reduce_sum(x_masked, axis=1)
        counts = tf.maximum(tf.reduce_sum(valid, axis=1, keepdims=True), 1.0)
        return summed / counts
    def get_config(self):
        cfg = super().get_config()
        cfg["pad_token"] = self.pad_token
        return cfg
def build_model_from_config(config: dict) -> keras.Model:
    """Rebuild TCRtyper architecture from saved config.json.
    Args:
        config: dict loaded from {model_dir}/config.json
    Returns:
        keras.Model matching the trained architecture
    """
    m = config["model"]
    t = config["tokens"]
    num_alleles = config["data"]["num_alleles"]
    inp_seq  = keras.Input(shape=(None, N_SYM), name="seq_input")
    inp_mask = keras.Input(shape=(None,), dtype=tf.int32, name="mask_input")
    h = SequenceEncoderLayer(
        embed_dim=m["embed_dim"], max_len=m["max_seq_len"],
        dropout_rate=m["dropout"], encoding_mode=m["encoding_mode"],
        pad_token=t["pad"], mask_token=t["mask"],
        sep_token=t["sep"], normal_token=t["normal"],
        name="seq_encoder",
    )([inp_seq, inp_mask])
    for i in range(m["num_layers"]):
        h = GatedTransformerLayer(
            embed_dim=m["embed_dim"], num_heads=m["num_heads"],
            ff_dim=m["ff_dim"], resnet=m["resnet"], dropout_rate=m["dropout"],
            pad_token=t["pad"], mask_token=t["mask"],
            sep_token=t["sep"], normal_token=t["normal"],
            name=f"transformer_{i}",
        )([h, inp_mask])
    h = MaskedGlobalAveragePooling(pad_token=t["pad"], name="pool")([h, inp_mask])
    h = layers.Dense(m["hla_proj_dim"], activation="gelu", name="hla_proj")(h)
    h = layers.Dropout(m["dropout"], name="hla_drop")(h)
    z_logits = layers.Dense(num_alleles, name="hla_head")(h)
    return keras.Model(inputs=[inp_seq, inp_mask], outputs=z_logits, name="TCRtyper")
def load_model(model_dir: str) -> Tuple[keras.Model, dict]:
    """Load config, rebuild model, restore best checkpoint weights.
    Args:
        model_dir: path to train_tcrtyper.py output directory
    Returns:
        (model, config) tuple
    """
    config_path = os.path.join(model_dir, "config.json")
    assert os.path.exists(config_path), f"config.json not found at {config_path}"
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"[MODEL] Config: embed={config['model']['embed_dim']}  "
          f"layers={config['model']['num_layers']}  "
          f"heads={config['model']['num_heads']}  "
          f"alleles={config['data']['num_alleles']}")
    model = build_model_from_config(config)
    # build weights with dummy forward pass
    _ = model([tf.zeros((1, 10, N_SYM)), tf.ones((1, 10), dtype=tf.int32)], training=False)
    # restore model weights only (no optimizer needed for inference)
    # save_checkpoint writes both model.weights.h5 and tf.train.Checkpoint;
    # we use model.weights.h5 directly to avoid optimizer shape mismatches
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    restored = False
    for tag in ["best", "latest"]:
        w_path = os.path.join(ckpt_dir, tag, "model.weights.h5")
        meta_path = os.path.join(ckpt_dir, tag, "meta.json")
        if os.path.exists(w_path):
            model.load_weights(w_path)
            # print training metadata if available
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                print(f"[MODEL] Loaded '{tag}' weights: "
                      f"epoch={meta['epoch']} best_val={meta['best_val_loss']:.6f}")
            else:
                print(f"[MODEL] Loaded '{tag}' weights from {w_path}")
            restored = True
            break
    assert restored, f"No model.weights.h5 found under {ckpt_dir}/best or {ckpt_dir}/latest"
    model.summary(print_fn=lambda s: print(f"  {s}"))
    return model, config
# ═════════════════════════════════════════════════════════════════════
# 6.  BATCHED INFERENCE
# ═════════════════════════════════════════════════════════════════════
def run_inference(model, data, batch_size=512):
    """Run batched inference on valid samples, return raw logits + sigmoid probs.
    Args:
        model:      loaded TCRtyper keras model
        data:       dict from load_and_prepare_data
        batch_size: GPU/CPU batch size
    Returns:
        z_logits: (N_valid, A) float32 raw logits
        z_probs:  (N_valid, A) float32 sigmoid probabilities
    """
    valid_idx = np.where(data["valid_mask"])[0]
    N = len(valid_idx)
    cdr1  = [data["cdr1_list"][i]  for i in valid_idx]
    cdr2  = [data["cdr2_list"][i]  for i in valid_idx]
    cdr25 = [data["cdr25_list"][i] for i in valid_idx]
    cdr3  = [data["cdr3_list"][i]  for i in valid_idx]
    all_logits = []
    n_batches = (N + batch_size - 1) // batch_size
    t0 = time.time()
    for b in range(n_batches):
        s, e = b * batch_size, min((b + 1) * batch_size, N)
        cf, cm = prepare_inference_batch(cdr1[s:e], cdr2[s:e], cdr25[s:e], cdr3[s:e])
        logits = model(
            [tf.constant(cf, dtype=tf.float32), tf.constant(cm, dtype=tf.int32)],
            training=False,
        )
        all_logits.append(logits.numpy())
        if (b + 1) % 20 == 0 or (b + 1) == n_batches:
            print(f"  batch {b+1}/{n_batches}  ({e}/{N}, {time.time()-t0:.1f}s)")
    z_logits = np.concatenate(all_logits, axis=0)
    z_probs  = 1.0 / (1.0 + np.exp(-z_logits.astype(np.float64)))  # float64 for numerics
    z_probs  = z_probs.astype(np.float32)
    print(f"[INFER] {N} samples in {time.time()-t0:.1f}s")
    return z_logits, z_probs
# ═════════════════════════════════════════════════════════════════════
# 7.  PER-TCR ANALYSIS
# ═════════════════════════════════════════════════════════════════════
def compute_per_tcr_stats(
    z_probs: np.ndarray,
    hla_indices: np.ndarray,
    labels: np.ndarray,
    idx_to_hla: dict,
    hla_sim: Optional[np.ndarray] = None,
    neighborhood_ks: List[int] = [5, 10, 20],
) -> dict:
    """Compute per-TCR statistics: entropy, gap, rank, neighbourhood probs.
    Args:
        z_probs:       (N, A) sigmoid probabilities
        hla_indices:   (N,) target HLA column index per sample
        labels:        (N,) ground-truth 0/1
        idx_to_hla:    dict mapping str(idx)→HLA name
        hla_sim:       (A, A) cosine similarity matrix (optional, for neighbourhood)
        neighborhood_ks: list of K values for neighbourhood analysis
    Returns:
        dict of per-TCR arrays and aggregated neighbourhood metrics
    """
    N, A = z_probs.shape
    # ── 1. Normalised entropy ────────────────────────────────────────
    # Normalise raw sigmoid probs to a proper distribution per TCR
    row_sums = z_probs.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)  # avoid div-by-zero
    p_norm = z_probs / row_sums  # (N, A) sums to 1 per row
    # Shannon entropy: H = -sum(p * log2(p)), with 0*log(0)=0
    log_p = np.zeros_like(p_norm)
    nonzero = p_norm > 0
    log_p[nonzero] = np.log2(p_norm[nonzero])
    entropy = -np.sum(p_norm * log_p, axis=1)  # (N,)
    # Maximum possible entropy for reference
    max_entropy = np.log2(A)
    normalised_entropy = entropy / max_entropy  # 0→concentrated, 1→uniform
    # ── 2. Top-1 / Top-2 gap (raw probs, before normalisation) ──────
    # Vectorised: partition to get top-2 values per row
    top2_idx = np.argpartition(-z_probs, kth=2, axis=1)[:, :2]  # (N, 2)
    top2_vals = np.take_along_axis(z_probs, top2_idx, axis=1)   # (N, 2)
    # Sort the 2 values descending
    top2_sorted = np.sort(top2_vals, axis=1)[:, ::-1]           # (N, 2)
    top1_prob = top2_sorted[:, 0]
    top2_prob = top2_sorted[:, 1]
    gap = top1_prob - top2_prob  # (N,)
    # Full argsort for top-1 HLA identity
    top1_hla_idx = np.argmax(z_probs, axis=1)  # (N,)
    # ── 3. Rank of true HLA (1-based, lower=better) ─────────────────
    # ranks[i] = how many alleles have prob >= true HLA's prob
    true_probs = z_probs[np.arange(N), hla_indices]  # (N,)
    # Vectorised rank: count alleles with prob > true_prob + ties
    ranks = np.sum(z_probs > true_probs[:, None], axis=1) + 1  # (N,) 1-based
    # ── 4. Neighbourhood analysis (requires HLA similarity matrix) ───
    nbr_results = {}
    if hla_sim is not None:
        for K in neighborhood_ks:
            # For each sample's true HLA, find top-K most similar HLAs
            # hla_sim[hla_idx] gives similarity to all A alleles
            sim_rows = hla_sim[hla_indices]  # (N, A)
            # Set self-similarity to -inf so true HLA is not in its own neighbourhood
            sim_rows[np.arange(N), hla_indices] = -np.inf
            # Top-K neighbour indices per sample (vectorised argpartition)
            if K >= A:
                nbr_idx = np.tile(np.arange(A), (N, 1))
            else:
                nbr_idx = np.argpartition(-sim_rows, kth=K, axis=1)[:, :K]  # (N, K)
            # Sum predicted probability over the K neighbours
            nbr_probs = np.take_along_axis(z_probs, nbr_idx, axis=1)  # (N, K)
            nbr_prob_sum = nbr_probs.sum(axis=1)  # (N,)
            # Is the top-1 predicted HLA among the K neighbours of true HLA?
            # Build set membership check vectorised
            nbr_hit = np.any(nbr_idx == top1_hla_idx[:, None], axis=1)  # (N,) bool
            # Similarity-weighted score: sum_j(sim(true, j) * prob_j) / sum_j(sim(true, j))
            # Use all alleles, weighted by similarity to true HLA
            sim_to_true = hla_sim[hla_indices]  # (N, A)
            # Zero out self
            sim_to_true_clean = sim_to_true.copy()
            sim_to_true_clean[np.arange(N), hla_indices] = 0.0
            # Clip negatives (cosine sim can be negative)
            sim_to_true_clean = np.maximum(sim_to_true_clean, 0.0)
            sim_weighted_score = np.sum(sim_to_true_clean * z_probs, axis=1)
            sim_denom = np.sum(sim_to_true_clean, axis=1)
            sim_denom = np.maximum(sim_denom, 1e-12)
            sim_weighted_score = sim_weighted_score / sim_denom  # (N,)
            nbr_results[K] = {
                "nbr_prob_sum": nbr_prob_sum,
                "nbr_hit": nbr_hit,
                "sim_weighted_score": sim_weighted_score if K == neighborhood_ks[0] else None,
            }
    # ── 5. Collect HLA name strings ──────────────────────────────────
    top1_hla_name = np.array([idx_to_hla.get(str(i), "?") for i in top1_hla_idx])
    true_hla_name = np.array([idx_to_hla.get(str(i), "?") for i in hla_indices])
    return {
        "entropy": entropy,
        "normalised_entropy": normalised_entropy,
        "top1_prob": top1_prob,
        "top2_prob": top2_prob,
        "gap": gap,
        "top1_hla_idx": top1_hla_idx,
        "top1_hla_name": top1_hla_name,
        "true_hla_name": true_hla_name,
        "true_prob": true_probs,
        "rank": ranks,
        "labels": labels,
        "nbr": nbr_results,
    }
# ═════════════════════════════════════════════════════════════════════
# 8.  BINARY EVALUATION METRICS
# ═════════════════════════════════════════════════════════════════════
def _compute_auroc(scores, labels):
    """Trapezoidal AUROC (no sklearn)."""
    order = np.argsort(-scores)
    sl = labels[order]
    n_pos, n_neg = int(labels.sum()), len(labels) - int(labels.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    tpr = np.concatenate([[0.0], np.cumsum(sl) / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(1 - sl) / n_neg])
    return float(np.trapz(tpr, fpr))
def _compute_auprc(scores, labels):
    """Trapezoidal AUPRC (no sklearn)."""
    order = np.argsort(-scores)
    sl = labels[order]
    n_pos = int(labels.sum())
    if n_pos == 0:
        return 0.0
    tp = np.cumsum(sl)
    prec = np.concatenate([[1.0], tp / np.arange(1, len(labels) + 1)])
    rec  = np.concatenate([[0.0], tp / n_pos])
    return float(np.trapz(prec, rec))
def evaluate(z_probs, hla_indices, labels, threshold=0.5):
    """Compute binary evaluation metrics for TCR-HLA binding prediction.
    Returns dict of metric_name→value.
    """
    N = len(labels)
    pred_probs = z_probs[np.arange(N), hla_indices]
    pred_bin = (pred_probs >= threshold).astype(np.int32)
    lab = labels.astype(np.int32)
    tp = int(((pred_bin == 1) & (lab == 1)).sum())
    tn = int(((pred_bin == 0) & (lab == 0)).sum())
    fp = int(((pred_bin == 1) & (lab == 0)).sum())
    fn = int(((pred_bin == 0) & (lab == 1)).sum())
    acc = (tp + tn) / max(N, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    return {
        "n_samples": N, "n_pos": int(lab.sum()), "n_neg": int((1 - lab).sum()),
        "threshold": threshold, "accuracy": acc,
        "precision": prec, "recall": rec, "f1": f1,
        "auroc": _compute_auroc(pred_probs, lab),
        "auprc": _compute_auprc(pred_probs, lab),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "mean_pred_pos": float(pred_probs[lab == 1].mean()) if tp + fn > 0 else 0.0,
        "mean_pred_neg": float(pred_probs[lab == 0].mean()) if tn + fp > 0 else 0.0,
    }
# ═════════════════════════════════════════════════════════════════════
# 9.  HLA-HEAD WEIGHT EXTRACTION + SIMILARITY
# ═════════════════════════════════════════════════════════════════════
def extract_hla_similarity(model, idx_to_hla):
    """Extract hla_head learned weights, compute cosine similarity between HLAs.
    The hla_head Dense layer has:
        kernel: (hla_proj_dim, A) — each column is one HLA's learned embedding
        bias:   (A,) — per-allele log-odds prior
    Cosine similarity between HLA i and j is computed on the kernel columns.
    Args:
        model:      loaded TCRtyper keras model
        idx_to_hla: dict mapping str(idx)→HLA name
    Returns:
        hla_embeddings: (A, hla_proj_dim) float32 — row per HLA
        hla_bias:       (A,) float32
        cos_sim:        (A, A) float32 cosine similarity matrix
        hla_names:      list of A HLA name strings (ordered by index)
    """
    hla_layer = model.get_layer("hla_head")
    kernel, bias = hla_layer.get_weights()  # kernel: (D, A), bias: (A,)
    # transpose so each row is one HLA embedding
    hla_embeddings = kernel.T.astype(np.float32)  # (A, D)
    hla_bias = bias.astype(np.float32)             # (A,)
    # cosine similarity: normalise rows then dot product
    norms = np.linalg.norm(hla_embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    hla_normed = hla_embeddings / norms  # (A, D)
    cos_sim = hla_normed @ hla_normed.T  # (A, A)
    # ordered HLA names
    A = hla_embeddings.shape[0]
    hla_names = [idx_to_hla.get(str(i), f"HLA_{i}") for i in range(A)]
    print(f"[HLA] Extracted embeddings: ({A}, {hla_embeddings.shape[1]}), "
          f"bias: ({A},), similarity: ({A}, {A})")
    return hla_embeddings, hla_bias, cos_sim, hla_names
# ═════════════════════════════════════════════════════════════════════
# 10. VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════
def plot_hla_clustermap(cos_sim, hla_names, n_clusters, output_dir):
    """Plot HLA-HLA cosine similarity as a clustermap with dendrogram.
    Uses scipy hierarchical clustering + matplotlib (no seaborn dependency).
    Args:
        cos_sim:    (A, A) cosine similarity matrix
        hla_names:  list of A HLA name strings
        n_clusters: number of flat clusters to colour
        output_dir: directory to save figures
    """
    A = cos_sim.shape[0]
    # convert similarity → distance for linkage
    # clip to [0,1] range then distance = 1 - sim
    sim_clipped = np.clip(cos_sim, -1.0, 1.0)
    np.fill_diagonal(sim_clipped, 1.0)
    dist_matrix = 1.0 - sim_clipped
    # ensure symmetry and non-negative diagonal
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.maximum(dist_matrix, 0.0)
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method='ward')
    cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    # ── Figure 1: dendrogram ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(20, 6))
    dn = dendrogram(Z, labels=hla_names, leaf_rotation=90, leaf_font_size=3,
                    color_threshold=Z[-(n_clusters - 1), 2], ax=ax)
    ax.set_title(f"HLA Dendrogram (Ward linkage, {n_clusters} clusters)", fontsize=14)
    ax.set_ylabel("Distance (1 − cosine similarity)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "hla_dendrogram.png"), dpi=200)
    plt.close(fig)
    print(f"  → hla_dendrogram.png")
    # ── Figure 2: reordered heatmap ──────────────────────────────────
    leaf_order = dn["leaves"]
    reordered_sim = cos_sim[np.ix_(leaf_order, leaf_order)]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(reordered_sim, cmap="RdBu_r", vmin=-0.5, vmax=1.0, aspect="auto")
    ax.set_title("HLA-HLA Cosine Similarity (learned embeddings)", fontsize=14)
    fig.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.8)
    # sparse tick labels (show every Nth)
    n_ticks = min(50, A)
    tick_step = max(1, A // n_ticks)
    tick_pos = list(range(0, A, tick_step))
    reordered_names = [hla_names[i] for i in leaf_order]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([reordered_names[i] for i in tick_pos], rotation=90, fontsize=4)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([reordered_names[i] for i in tick_pos], fontsize=4)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "hla_similarity_heatmap.png"), dpi=200)
    plt.close(fig)
    print(f"  → hla_similarity_heatmap.png")
    return cluster_labels, leaf_order
def plot_per_tcr_diagnostics(stats, output_dir):
    """Plot per-TCR diagnostic visualisations.
    Args:
        stats:      dict from compute_per_tcr_stats
        output_dir: directory to save figures
    """
    labels = stats["labels"].astype(np.int32)
    pos = labels == 1
    neg = labels == 0
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # ── 1. Entropy distribution (pos vs neg) ─────────────────────────
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 50)
    ax.hist(stats["normalised_entropy"][pos], bins=bins, alpha=0.6, label="Positive", density=True, color="tab:blue")
    ax.hist(stats["normalised_entropy"][neg], bins=bins, alpha=0.6, label="Negative", density=True, color="tab:orange")
    ax.set_xlabel("Normalised entropy (0=focused, 1=uniform)")
    ax.set_ylabel("Density")
    ax.set_title("Entropy distribution")
    ax.legend()
    # ── 2. Gap distribution (pos vs neg) ─────────────────────────────
    ax = axes[0, 1]
    bins = np.linspace(0, 1, 50)
    ax.hist(stats["gap"][pos], bins=bins, alpha=0.6, label="Positive", density=True, color="tab:blue")
    ax.hist(stats["gap"][neg], bins=bins, alpha=0.6, label="Negative", density=True, color="tab:orange")
    ax.set_xlabel("Top-1 − Top-2 probability gap")
    ax.set_ylabel("Density")
    ax.set_title("Prediction confidence gap")
    ax.legend()
    # ── 3. Rank of true HLA (pos vs neg) ─────────────────────────────
    ax = axes[0, 2]
    max_rank = min(50, int(stats["rank"].max()))
    bins = np.arange(0.5, max_rank + 1.5, 1)
    ax.hist(stats["rank"][pos], bins=bins, alpha=0.6, label="Positive", density=True, color="tab:blue")
    ax.hist(stats["rank"][neg], bins=bins, alpha=0.6, label="Negative", density=True, color="tab:orange")
    ax.set_xlabel("Rank of true HLA (1=best)")
    ax.set_ylabel("Density")
    ax.set_title("True HLA rank distribution")
    ax.set_xlim(0.5, max_rank + 0.5)
    ax.legend()
    # ── 4. Entropy vs Rank scatter ───────────────────────────────────
    ax = axes[1, 0]
    ax.scatter(stats["rank"][pos], stats["normalised_entropy"][pos],
               alpha=0.3, s=8, label="Positive", c="tab:blue")
    ax.scatter(stats["rank"][neg], stats["normalised_entropy"][neg],
               alpha=0.3, s=8, label="Negative", c="tab:orange")
    ax.set_xlabel("Rank of true HLA")
    ax.set_ylabel("Normalised entropy")
    ax.set_title("Entropy vs Rank")
    ax.set_xlim(0, min(100, int(stats["rank"].max())))
    ax.legend()
    # ── 5. True HLA prob vs gap ──────────────────────────────────────
    ax = axes[1, 1]
    ax.scatter(stats["true_prob"][pos], stats["gap"][pos],
               alpha=0.3, s=8, label="Positive", c="tab:blue")
    ax.scatter(stats["true_prob"][neg], stats["gap"][neg],
               alpha=0.3, s=8, label="Negative", c="tab:orange")
    ax.set_xlabel("Predicted prob for true HLA")
    ax.set_ylabel("Top-1 − Top-2 gap")
    ax.set_title("True HLA probability vs Gap")
    ax.legend()
    # ── 6. Top-K accuracy curve ──────────────────────────────────────
    ax = axes[1, 2]
    ks = np.arange(1, min(51, stats["rank"].max() + 1))
    for lab_val, lab_name, col in [(1, "Positive", "tab:blue"), (0, "Negative", "tab:orange")]:
        mask = labels == lab_val
        if mask.sum() == 0:
            continue
        r = stats["rank"][mask]
        topk_acc = np.array([np.mean(r <= k) for k in ks])
        ax.plot(ks, topk_acc, label=lab_name, color=col, linewidth=2)
    ax.set_xlabel("K")
    ax.set_ylabel("Top-K accuracy")
    ax.set_title("True HLA in Top-K predictions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.suptitle("Per-TCR Diagnostic Plots", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_tcr_diagnostics.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → per_tcr_diagnostics.png")
def plot_neighbourhood_analysis(stats, output_dir):
    """Plot neighbourhood analysis: do FN TCRs light up similar HLAs?
    Args:
        stats:      dict from compute_per_tcr_stats (must have 'nbr' key)
        output_dir: directory to save figures
    """
    nbr = stats.get("nbr", {})
    if not nbr:
        print("[PLOT] Skipping neighbourhood plots (no HLA similarity data)")
        return
    labels = stats["labels"].astype(np.int32)
    true_prob = stats["true_prob"]
    threshold_mask = true_prob < 0.5  # "missed" predictions
    ks = sorted(nbr.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # ── 1. Neighbourhood prob sum at each K (pos vs neg) ─────────────
    ax = axes[0]
    for lab_val, lab_name, col in [(1, "Positive", "tab:blue"), (0, "Negative", "tab:orange")]:
        mask = labels == lab_val
        means = [nbr[k]["nbr_prob_sum"][mask].mean() for k in ks]
        stds  = [nbr[k]["nbr_prob_sum"][mask].std() / np.sqrt(mask.sum()) for k in ks]
        ax.errorbar(ks, means, yerr=stds, marker='o', label=lab_name, color=col, capsize=3)
    ax.set_xlabel("K (neighbourhood size)")
    ax.set_ylabel("Mean sum of probs in top-K similar HLAs")
    ax.set_title("Neighbourhood probability mass")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # ── 2. Neighbourhood hit rate at each K ──────────────────────────
    ax = axes[1]
    for lab_val, lab_name, col in [(1, "Positive", "tab:blue"), (0, "Negative", "tab:orange")]:
        mask = labels == lab_val
        hit_rates = [nbr[k]["nbr_hit"][mask].mean() for k in ks]
        ax.bar([k + (0.2 if lab_val == 1 else -0.2) for k in ks],
               hit_rates, width=0.35, label=lab_name, color=col, alpha=0.8)
    ax.set_xlabel("K (neighbourhood size)")
    ax.set_ylabel("Hit rate (top-1 pred in neighbourhood)")
    ax.set_title("Is top prediction a neighbour of true HLA?")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    # ── 3. For FN positives: true prob vs neighbourhood prob ─────────
    ax = axes[2]
    max_k = max(ks)
    fn_mask = (labels == 1) & threshold_mask  # false negatives
    tp_mask = (labels == 1) & (~threshold_mask)  # true positives
    if fn_mask.sum() > 0 and max_k in nbr:
        ax.scatter(true_prob[fn_mask], nbr[max_k]["nbr_prob_sum"][fn_mask],
                   alpha=0.4, s=15, c="tab:red", label=f"FN (n={fn_mask.sum()})")
    if tp_mask.sum() > 0 and max_k in nbr:
        ax.scatter(true_prob[tp_mask], nbr[max_k]["nbr_prob_sum"][tp_mask],
                   alpha=0.4, s=15, c="tab:green", label=f"TP (n={tp_mask.sum()})")
    ax.set_xlabel("Predicted prob for true HLA")
    ax.set_ylabel(f"Sum of probs in top-{max_k} similar HLAs")
    ax.set_title(f"FN rescue: neighbourhood probability (K={max_k})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.suptitle("Neighbourhood Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "neighbourhood_analysis.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → neighbourhood_analysis.png")
# ═════════════════════════════════════════════════════════════════════
# 11. SAVE RESULTS
# ═════════════════════════════════════════════════════════════════════
def save_per_tcr_csv(stats, data, z_probs, output_dir, neighborhood_ks):
    """Save per-TCR results to CSV with all computed statistics.
    Args:
        stats:           dict from compute_per_tcr_stats
        data:            dict from load_and_prepare_data
        z_probs:         (N_valid, A) sigmoid probs
        output_dir:      output directory
        neighborhood_ks: list of K values used
    """
    import csv
    valid_idx = np.where(data["valid_mask"])[0]
    N = len(valid_idx)
    csv_path = os.path.join(output_dir, "per_tcr_results.csv")
    # build header
    header = [
        "sample_idx", "tcr_seq4", "true_hla", "true_hla_idx", "label",
        "true_hla_prob", "rank_of_true_hla",
        "top1_hla", "top1_hla_idx", "top1_prob",
        "top2_prob", "gap_top1_top2",
        "entropy_bits", "normalised_entropy",
    ]
    # add neighbourhood columns
    for K in neighborhood_ks:
        header.append(f"nbr_prob_sum_K{K}")
        header.append(f"nbr_hit_K{K}")
    # first K's similarity-weighted score
    if stats["nbr"]:
        header.append("sim_weighted_score")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(N):
            row = [
                int(valid_idx[i]),
                str(data["tcr_seq4"][valid_idx[i]]),
                stats["true_hla_name"][i],
                int(data["hla_indices"][valid_idx[i]]),
                int(stats["labels"][i]),
                f"{stats['true_prob'][i]:.6f}",
                int(stats["rank"][i]),
                stats["top1_hla_name"][i],
                int(stats["top1_hla_idx"][i]),
                f"{stats['top1_prob'][i]:.6f}",
                f"{stats['top2_prob'][i]:.6f}",
                f"{stats['gap'][i]:.6f}",
                f"{stats['entropy'][i]:.4f}",
                f"{stats['normalised_entropy'][i]:.6f}",
            ]
            for K in neighborhood_ks:
                if K in stats["nbr"]:
                    row.append(f"{stats['nbr'][K]['nbr_prob_sum'][i]:.6f}")
                    row.append(int(stats["nbr"][K]["nbr_hit"][i]))
                else:
                    row.extend(["", ""])
            if stats["nbr"] and neighborhood_ks[0] in stats["nbr"]:
                sws = stats["nbr"][neighborhood_ks[0]].get("sim_weighted_score")
                row.append(f"{sws[i]:.6f}" if sws is not None else "")
            writer.writerow(row)
    print(f"  → per_tcr_results.csv  ({N} rows)")
    return csv_path
def save_all_outputs(output_dir, z_probs, data, metrics, stats,
                     hla_embeddings=None, hla_bias=None, cos_sim=None,
                     hla_names=None, cluster_labels=None):
    """Save all numpy arrays, metrics JSON, and HLA analysis outputs.
    Args:
        output_dir:      output directory
        z_probs:         (N_valid, A) predictions
        data:            raw data dict
        metrics:         evaluation metrics dict
        stats:           per-TCR stats dict
        hla_embeddings:  (A, D) learned HLA embeddings (optional)
        hla_bias:        (A,) learned HLA bias (optional)
        cos_sim:         (A, A) cosine similarity matrix (optional)
        hla_names:       list of HLA names (optional)
        cluster_labels:  (A,) cluster assignments (optional)
    """
    # -- metrics JSON --
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  → metrics.json")
    # -- z_probs npy (full prediction matrix) --
    np.save(os.path.join(output_dir, "z_probs.npy"), z_probs)
    print(f"  → z_probs.npy  {z_probs.shape}")
    # -- per-TCR stats npy --
    np.savez_compressed(
        os.path.join(output_dir, "per_tcr_stats.npz"),
        entropy=stats["entropy"],
        normalised_entropy=stats["normalised_entropy"],
        gap=stats["gap"],
        rank=stats["rank"],
        true_prob=stats["true_prob"],
        top1_prob=stats["top1_prob"],
        labels=stats["labels"],
    )
    print(f"  → per_tcr_stats.npz")
    # -- HLA head analysis outputs --
    if hla_embeddings is not None:
        np.save(os.path.join(output_dir, "hla_embeddings.npy"), hla_embeddings)
        np.save(os.path.join(output_dir, "hla_bias.npy"), hla_bias)
        np.save(os.path.join(output_dir, "hla_cosine_similarity.npy"), cos_sim)
        print(f"  → hla_embeddings.npy, hla_bias.npy, hla_cosine_similarity.npy")
    if cluster_labels is not None:
        # save cluster assignments as JSON: {hla_name: cluster_id}
        hla_clusters = {hla_names[i]: int(cluster_labels[i]) for i in range(len(hla_names))}
        with open(os.path.join(output_dir, "hla_clusters.json"), 'w') as f:
            json.dump(hla_clusters, f, indent=2)
        print(f"  → hla_clusters.json  ({max(cluster_labels)} clusters)")
    # -- neighbourhood summary --
    if stats["nbr"]:
        nbr_summary = {}
        labels_int = stats["labels"].astype(np.int32)
        for K, v in stats["nbr"].items():
            for lab_val, lab_name in [(1, "positive"), (0, "negative")]:
                mask = labels_int == lab_val
                if mask.sum() == 0:
                    continue
                nbr_summary[f"K={K}_{lab_name}_mean_nbr_prob"] = float(v["nbr_prob_sum"][mask].mean())
                nbr_summary[f"K={K}_{lab_name}_hit_rate"] = float(v["nbr_hit"][mask].mean())
        with open(os.path.join(output_dir, "neighbourhood_summary.json"), 'w') as f:
            json.dump(nbr_summary, f, indent=2)
        print(f"  → neighbourhood_summary.json")
# ═════════════════════════════════════════════════════════════════════
# 12. MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    """Entry point: load data → load model → inference → analysis → save."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 60)
    print("TCRtyper — VDJdb Inference + Analysis")
    print("=" * 60)
    # ── 1. Load and prepare data ─────────────────────────────────────
    data = load_and_prepare_data(args.path_npz, args.path_hla_to_id)
    # ── 2. Load trained model ────────────────────────────────────────
    model, config = load_model(args.model_dir)
    # ── 3. Load idx_to_hla from model config ─────────────────────────
    idx_to_hla_path = config["data"].get("idx_to_hla", "")
    if not idx_to_hla_path or not os.path.exists(idx_to_hla_path):
        # fallback: try relative to model_dir
        idx_to_hla_path = os.path.join(args.model_dir, "idx_to_hla.json")
    if not os.path.exists(idx_to_hla_path):
        # try from BASE
        idx_to_hla_path = os.path.join(BASE, config["data"].get("idx_to_hla", ""))
    assert os.path.exists(idx_to_hla_path), (
        f"idx_to_hla JSON not found. Tried: {idx_to_hla_path}. "
        f"Set the path in config['data']['idx_to_hla'] or place it in model_dir.")
    with open(idx_to_hla_path, 'r') as f:
        idx_to_hla = json.load(f)
    print(f"[DATA] idx_to_hla: {len(idx_to_hla)} alleles from {idx_to_hla_path}")
    # ── 4. HLA-head analysis (optional) ──────────────────────────────
    hla_embeddings, hla_bias, cos_sim, hla_names = None, None, None, None
    cluster_labels, leaf_order = None, None
    if args.analyze_hla_head:
        print("\n[HLA HEAD ANALYSIS]")
        hla_embeddings, hla_bias, cos_sim, hla_names = extract_hla_similarity(model, idx_to_hla)
        cluster_labels, leaf_order = plot_hla_clustermap(
            cos_sim, hla_names, args.n_hla_clusters, args.output_dir)
    # ── 5. Run batched inference ─────────────────────────────────────
    print("\n[INFERENCE]")
    z_logits, z_probs = run_inference(model, data, batch_size=args.batch_size)
    # ── 6. Binary evaluation ─────────────────────────────────────────
    valid_mask = data["valid_mask"]
    valid_hla = data["hla_indices"][valid_mask]
    valid_labels = data["labels"][valid_mask]
    metrics = evaluate(z_probs, valid_hla, valid_labels, threshold=args.threshold)
    # ── 7. Per-TCR analysis ──────────────────────────────────────────
    print("\n[PER-TCR ANALYSIS]")
    stats = compute_per_tcr_stats(
        z_probs, valid_hla, valid_labels, idx_to_hla,
        hla_sim=cos_sim,
        neighborhood_ks=args.neighborhood_k,
    )
    # ── 8. Print summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples:   {metrics['n_samples']}  "
          f"(pos={metrics['n_pos']}, neg={metrics['n_neg']})")
    print(f"  Threshold: {metrics['threshold']}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"  AUPRC:     {metrics['auprc']:.4f}")
    print(f"  TP={metrics['tp']}  TN={metrics['tn']}  "
          f"FP={metrics['fp']}  FN={metrics['fn']}")
    print(f"  Mean pred (pos): {metrics['mean_pred_pos']:.4f}")
    print(f"  Mean pred (neg): {metrics['mean_pred_neg']:.4f}")
    # rank stats
    pos_mask = valid_labels == 1
    neg_mask = valid_labels == 0
    print(f"\n  Rank stats (positive): "
          f"median={np.median(stats['rank'][pos_mask]):.0f}  "
          f"mean={stats['rank'][pos_mask].mean():.1f}")
    print(f"  Rank stats (negative): "
          f"median={np.median(stats['rank'][neg_mask]):.0f}  "
          f"mean={stats['rank'][neg_mask].mean():.1f}")
    print(f"  Entropy (positive): mean={stats['normalised_entropy'][pos_mask].mean():.4f}")
    print(f"  Entropy (negative): mean={stats['normalised_entropy'][neg_mask].mean():.4f}")
    print(f"  Gap (positive):     mean={stats['gap'][pos_mask].mean():.4f}")
    print(f"  Gap (negative):     mean={stats['gap'][neg_mask].mean():.4f}")
    # neighbourhood summary
    if stats["nbr"]:
        print(f"\n  Neighbourhood analysis:")
        for K in args.neighborhood_k:
            if K not in stats["nbr"]:
                continue
            v = stats["nbr"][K]
            print(f"    K={K}:")
            print(f"      Pos — nbr_prob_mean={v['nbr_prob_sum'][pos_mask].mean():.4f}  "
                  f"hit_rate={v['nbr_hit'][pos_mask].mean():.4f}")
            print(f"      Neg — nbr_prob_mean={v['nbr_prob_sum'][neg_mask].mean():.4f}  "
                  f"hit_rate={v['nbr_hit'][neg_mask].mean():.4f}")
    print("=" * 60)
    # ── 9. Save everything ───────────────────────────────────────────
    print("\n[SAVING]")
    save_per_tcr_csv(stats, data, z_probs, args.output_dir, args.neighborhood_k)
    save_all_outputs(
        args.output_dir, z_probs, data, metrics, stats,
        hla_embeddings=hla_embeddings, hla_bias=hla_bias,
        cos_sim=cos_sim, hla_names=hla_names,
        cluster_labels=cluster_labels,
    )
    # ── 10. Visualisations ───────────────────────────────────────────
    print("\n[PLOTS]")
    plot_per_tcr_diagnostics(stats, args.output_dir)
    plot_neighbourhood_analysis(stats, args.output_dir)
    print(f"\nAll outputs saved to: {args.output_dir}")
    print("Done.")
if __name__ == "__main__":
    main()