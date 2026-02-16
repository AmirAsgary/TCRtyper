#!/usr/bin/env python3
"""
Uncertainty-Aware Inference of TCR Binder Set Size from Synthetic Training Data.
================================================================================
Two-mode pipeline:
  train      – Train a multiclass classifier on synthetic TCR-HLA data using
               distance-weighted cross-entropy (ordinal-aware), perform k-fold
               cross-validated temperature calibration with tolerance-band error,
               and derive error-controlled uncertainty thresholds.
  inference  – Apply the trained model to real TCR data (streamed in chunks),
               calibrate probabilities, compute continuous expected binder size,
               compute uncertainty, and produce diagnostic plots.
All heavy lifting uses TensorFlow / Keras with vectorised NumPy;
designed for CPU or single-GPU execution within 40 GB RAM / 16 GB VRAM.
"""
import argparse
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster / headless runs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# ── TensorFlow setup ────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
# ── Custom JSON encoder for numpy types ────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types and arrays."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
# ── Device detection ───────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    log.info("TensorFlow %s | GPUs detected: %d", tf.__version__, len(gpus))
    for g in gpus:
        log.info("  GPU: %s", g.name)
    # allow memory growth to avoid grabbing all VRAM
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
else:
    log.info("TensorFlow %s | No GPU detected, running on CPU", tf.__version__)
# ── Binder-set bins (fixed, ordered) ───────────────────────────────────────
BINDER_BINS = np.array([3, 5, 10, 15, 25, 35], dtype=np.float32)
NUM_CLASSES = len(BINDER_BINS)
# ============================================================================
#  Argument parser
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser with sensible defaults."""
    p = argparse.ArgumentParser(
        description="Train / infer TCR binder-set size classifier."
    )
    p.add_argument("--mode", choices=["train", "inference"], default="train",
                   help="Pipeline mode: 'train' on synthetic or 'inference' on real data.")
    # ── paths ───────────────────────────────────────────────────────────────
    p.add_argument("--input_train", type=str,
                   default="output/synthetic_analysis_with_reg",
                   help="Directory with bX_nY sub-folders (train mode).")
    p.add_argument("--input_inference", type=str,
                   default="outputs/real_data_2/analysis/metrics.h5",
                   help="HDF5 file for inference mode.")
    p.add_argument("--output_dir", type=str,
                   default="output/synthetic_analysis_with_reg/models",
                   help="Directory for model checkpoints, metadata, and plots.")
    # ── training hyper-parameters ───────────────────────────────────────────
    p.add_argument("--val_split", type=float, default=0.05,
                   help="Fraction held out for validation *and* test each.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=10000)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--convergence", action="store_true", default=True,
                   help="Enable early-stopping on validation loss.")
    p.add_argument("--resume", action="store_true", default=False,
                   help="Resume from best checkpoint if available.")
    p.add_argument("--k_crossval", type=int, default=5,
                   help="Number of cross-validation folds for calibration.")
    # ── architecture ────────────────────────────────────────────────────────
    p.add_argument("--num_neurons", type=int, default=128,
                   help="Hidden-layer width.")
    p.add_argument("--num_layers", type=int, default=1,
                   help="Number of hidden Dense layers.")
    p.add_argument("--dropout_rate", type=float, default=0.3)
    p.add_argument("--reg_lambda", type=float, default=1e-6,
                   help="L1 + L2 regularisation strength.")
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--lr_schedule", choices=["constant", "cosine"], default="cosine",
                   help="Learning-rate schedule type.")
    # ── ordinal loss ────────────────────────────────────────────────────────
    p.add_argument("--ordinal_lambda", type=float, default=1.0,
                   help="Sharpness parameter lambda for distance-weighted soft targets.")
    # ── thresholds ──────────────────────────────────────────────────────────
    p.add_argument("--max_error_rate", type=float, default=0.05,
                   help="Maximum acceptable error rate (epsilon) for selective prediction.")
    p.add_argument("--delta", type=float, default=5.0,
                   help="Tolerance band delta for continuous error indicator |b_hat - y| > delta.")
    # ── inference ────────────────────────────────────────────────────────────
    p.add_argument("--infer_chunk", type=int, default=500_000,
                   help="Chunk size for streaming inference on large H5.")
    return p
# ============================================================================
#  Utility functions
# ============================================================================
def compute_entropy(z_probs: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy (base-2) row-wise for an unnormalised probability matrix.
    Args:
        z_probs: (N, C) non-negative array (need not sum to 1).
    Returns:
        (N,) entropy values in bits.
    """
    z_sum = z_probs.sum(axis=1, keepdims=True)
    z_sum_safe = np.maximum(z_sum, 1e-10)
    z_norm = z_probs / z_sum_safe
    log_z = np.log2(np.maximum(z_norm, 1e-10))
    entropy = -(z_norm * log_z).sum(axis=1).astype(np.float32)
    entropy[z_sum.ravel() == 0] = 0.0
    return entropy
def compute_predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """Predictive entropy H(x) = -sum p log p  (natural log, per the paper).
    Args:
        probs: (N, C) calibrated probability matrix (rows sum to 1).
    Returns:
        (N,) entropy values (nats).
    """
    safe = np.maximum(probs, 1e-12)
    return -(probs * np.log(safe)).sum(axis=1).astype(np.float32)
def compute_margin(probs: np.ndarray) -> np.ndarray:
    """Margin = p_(1) - p_(2), gap between top-two class probabilities.
    Args:
        probs: (N, C) probability matrix.
    Returns:
        (N,) margin values in [0, 1].
    """
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    return (top2.max(axis=1) - top2.min(axis=1)).astype(np.float32)
def compute_expected_binder(probs: np.ndarray, bins: np.ndarray = BINDER_BINS) -> np.ndarray:
    """Compute continuous expected binder size: b_hat = sum_b p(b|x) * b.
    Args:
        probs: (N, C) calibrated probabilities over discrete bins.
        bins : (C,) the binder-size bin values [3, 5, 10, 15, 25, 35].
    Returns:
        (N,) continuous expected binder sizes.
    """
    return (probs * bins[np.newaxis, :]).sum(axis=1).astype(np.float32)
def set_global_seed(seed: int):
    """Set random seeds for reproducibility across numpy, tf, and python."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    log.info("Global seed set to %d (numpy, tf, python hash)", seed)
# ============================================================================
#  Distance-weighted soft targets & custom loss
# ============================================================================
def make_soft_targets(labels: np.ndarray, bins: np.ndarray = BINDER_BINS,
                      lam: float = 1.0) -> np.ndarray:
    """Build distance-weighted soft target distributions for ordinal loss.
    For true label y, the target for bin b is:
        p_tilde(b|y) = exp(-lambda * |b - y|) / Z
    Args:
        labels: (N,) integer class indices (0..C-1).
        bins  : (C,) actual binder-size values.
        lam   : sharpness parameter lambda.
    Returns:
        (N, C) soft target matrix (rows sum to 1, float32).
    """
    y_vals = bins[labels]  # (N,) true binder-size values
    # |b - y| for all (sample, bin) pairs: (N, C)
    dists = np.abs(bins[np.newaxis, :] - y_vals[:, np.newaxis])
    logits = -lam * dists
    # stable softmax
    logits -= logits.max(axis=1, keepdims=True)
    exp_l = np.exp(logits)
    soft = exp_l / exp_l.sum(axis=1, keepdims=True)
    log.info("  soft targets built: shape=%s  lambda=%.2f  "
             "max_prob=[%.4f, %.4f]  min_prob=[%.6f, %.6f]",
             soft.shape, lam,
             soft.max(axis=1).min(), soft.max(axis=1).max(),
             soft.min(axis=1).min(), soft.min(axis=1).max())
    return soft.astype(np.float32)
class DistanceWeightedCELoss(keras.losses.Loss):
    """Distance-Weighted Cross-Entropy loss for ordinal classification.
    Expects y_true as soft target distributions (N, C) and y_pred as raw logits (N, C).
    Computes: -sum_b p_tilde(b|y) * log softmax(z)_b
    """
    def __init__(self, name="distance_weighted_ce", **kwargs):
        super().__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        """
        Args:
            y_true: (batch, C) soft target distributions.
            y_pred: (batch, C) raw logits from the model.
        Returns:
            Scalar mean loss.
        """
        # log-softmax for numerical stability
        log_probs = y_pred - tf.reduce_logsumexp(y_pred, axis=-1, keepdims=True)
        # cross-entropy against soft targets
        per_sample = -tf.reduce_sum(y_true * log_probs, axis=-1)
        return tf.reduce_mean(per_sample)
# ============================================================================
#  Data loading (training)
# ============================================================================
def binder_to_class_index(binder_size: int) -> int:
    """Map a binder-set size to its fixed class index (0..5).
    The 6 classes correspond to BINDER_BINS = [3, 5, 10, 15, 25, 35].
    Args:
        binder_size: one of {3, 5, 10, 15, 25, 35}.
    Returns:
        Integer class index.
    """
    mapping = {int(b): i for i, b in enumerate(BINDER_BINS)}
    assert binder_size in mapping, f"Unknown binder size {binder_size}; expected one of {list(mapping.keys())}"
    return mapping[binder_size]
def load_synthetic_data(input_dir: str):
    """Load and concatenate feature matrices from all bX_nY sub-folders.
    Labels are mapped to the 6 unique binder-size classes (NOT per-folder).
    Multiple folders with the same binder size but different donor counts all
    receive the same class label.
    Returns:
        X         : (N_all, 21) feature matrix [entropy, donor_size, 19 explanation fracs].
        Y         : (N_all,)    integer class labels in {0, 1, 2, 3, 4, 5}.
        label_map : dict  {class_index: binder_set_size}.
    """
    pattern = re.compile(r"^b\d+_n\d+$")
    folder_names = sorted(
        [f for f in os.listdir(input_dir) if pattern.match(f)]
    )
    assert len(folder_names) > 0, f"No bX_nY folders found in {input_dir}"
    folders = [os.path.join(input_dir, f) for f in folder_names]
    log.info("  found %d bX_nY folders to load", len(folders))
    # canonical label map: class index -> binder size (always 6 classes)
    label_map = {i: int(b) for i, b in enumerate(BINDER_BINS)}
    log.info("  label mapping: %s", label_map)
    X_parts, Y_parts = [], []
    thrs = np.arange(0.05, 0.95 + 0.05, 0.05).round(2)  # 19 thresholds
    t0_load = time.time()
    for fi, folder in enumerate(folders):
        t_folder = time.time()
        ds_path = os.path.join(folder, "figures", "donor_scores_matrix.npz")
        ar_path = os.path.join(folder, "figures", "analysis_arrays.npz")
        ds = np.load(ds_path)
        ar = np.load(ar_path)
        # extract binder size from folder name (e.g. b10_n100 -> binder=10)
        binder_size = int(os.path.basename(folder).split("_")[0].replace("b", ""))
        class_idx = binder_to_class_index(binder_size)
        n_tcr = ar["analysis_probs"].shape[0]
        # explanation fractions: vectorised (N, D, 1) > (1, 1, 19) -> (N, 19)
        donor_scores = ds["donor_scores"]          # (N, D)
        total_donors = ds["total_donors_per_tcr"]   # (N,)
        explanation_fractions = (
            (donor_scores[:, :, None] > thrs[None, None, :]).sum(axis=1)
            / np.maximum(total_donors[:, None], 1)
        ).astype(np.float32)  # (N, 19)
        donor_size = total_donors[:, np.newaxis].astype(np.float32)  # (N, 1)
        ent = compute_entropy(ar["analysis_probs"])[:, np.newaxis]    # (N, 1)
        x = np.concatenate([ent, donor_size, explanation_fractions], axis=-1)  # (N, 21)
        y = np.full(n_tcr, class_idx, dtype=np.int32)  # (N,) all same class
        X_parts.append(x)
        Y_parts.append(y)
        log.info("  [%2d/%2d] loaded %-12s  binder=%2d -> class %d  N=%d  "
                 "donor_scores=%s  ent=[%.3f, %.3f]  (%.1fs)",
                 fi + 1, len(folders), os.path.basename(folder), binder_size,
                 class_idx, n_tcr, donor_scores.shape,
                 float(ent.min()), float(ent.max()), time.time() - t_folder)
    X = np.concatenate(X_parts, axis=0)
    Y = np.concatenate(Y_parts, axis=0)
    elapsed = time.time() - t0_load
    mem_mb = (X.nbytes + Y.nbytes) / (1024 ** 2)
    log.info("Total synthetic samples: %d  |  Features: %d  |  Classes: %d",
             X.shape[0], X.shape[1], NUM_CLASSES)
    log.info("  X shape: %s  dtype: %s", X.shape, X.dtype)
    log.info("  Y shape: %s  dtype: %s  unique: %s", Y.shape, Y.dtype, np.unique(Y))
    # class balance
    for ci in range(NUM_CLASSES):
        count_ci = (Y == ci).sum()
        log.info("    class %d (binder=%d): %d samples (%.1f%%)",
                 ci, label_map[ci], count_ci, 100.0 * count_ci / len(Y))
    log.info("  data memory: %.1f MB  |  load time: %.1fs", mem_mb, elapsed)
    # quick feature sanity check
    log.info("  feature stats:  min=%.4f  max=%.4f  mean=%.4f  std=%.4f",
             X.min(), X.max(), X.mean(), X.std())
    return X, Y, label_map
def split_data(X, Y, val_frac, seed):
    """Random split into train / val / test (val_frac each for val and test).
    Returns:
        (X_train, Y_train, X_val, Y_val, X_test, Y_test)
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    indices = rng.permutation(n)
    n_val = int(n * val_frac)
    n_test = int(n * val_frac)
    idx_test = indices[:n_test]
    idx_val = indices[n_test:n_test + n_val]
    idx_train = indices[n_test + n_val:]
    log.info("  split sizes -> train: %d (%.1f%%)  val: %d (%.1f%%)  test: %d (%.1f%%)",
             len(idx_train), 100.0 * len(idx_train) / n,
             len(idx_val),   100.0 * len(idx_val) / n,
             len(idx_test),  100.0 * len(idx_test) / n)
    # log class distribution per split for sanity check
    for name, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        unique, counts = np.unique(Y[idx], return_counts=True)
        dist_str = "  ".join([f"c{u}:{c}" for u, c in zip(unique, counts)])
        log.info("    %s class dist: %s", name, dist_str)
    return (X[idx_train], Y[idx_train],
            X[idx_val],   Y[idx_val],
            X[idx_test],  Y[idx_test])
def make_tf_dataset(X, Y_soft, batch_size, shuffle=True, seed=42):
    """Create tf.data.Dataset from features and soft target distributions.
    Args:
        X      : (N, F) feature matrix.
        Y_soft : (N, C) soft target distributions (float32).
    Returns:
        Batched, optionally shuffled, prefetched tf.data.Dataset.
    """
    ds = tf.data.Dataset.from_tensor_slices(
        (X.astype(np.float32), Y_soft.astype(np.float32))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(X.shape[0], 500_000), seed=seed)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    n_batches = int(np.ceil(X.shape[0] / batch_size))
    log.info("  tf.data.Dataset: %d samples -> %d batches (batch_size=%d, shuffle=%s)",
             X.shape[0], n_batches, batch_size, shuffle)
    return ds
# ============================================================================
#  Accuracy metric compatible with soft targets
# ============================================================================
class SoftTargetAccuracy(keras.metrics.Metric):
    """Accuracy metric that extracts the hard label from soft targets via argmax.
    Works with distance-weighted soft targets as y_true.
    """
    def __init__(self, name="accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Args:
            y_true: (batch, C) soft target distributions.
            y_pred: (batch, C) raw logits.
        """
        true_class = tf.argmax(y_true, axis=-1)
        pred_class = tf.argmax(y_pred, axis=-1)
        matches = tf.cast(tf.equal(true_class, pred_class), tf.float32)
        self.correct.assign_add(tf.reduce_sum(matches))
        self.total.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    def result(self):
        return self.correct / tf.maximum(self.total, 1.0)
    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)
# ============================================================================
#  Model definition
# ============================================================================
def build_model(input_dim: int, num_classes: int, num_neurons: int,
                num_layers: int, dropout_rate: float, reg_lambda: float,
                learning_rate: float, lr_schedule: str, epochs: int,
                train_steps: int) -> keras.Model:
    """Build feedforward classifier: Input -> Dropout -> (Dense -> Dropout)*L -> Dense(logits).
    Uses L1+L2 regularisation. Output is raw logits (no softmax) for temperature scaling.
    Args:
        input_dim    : number of input features (21).
        num_classes  : number of binder-set bins (6).
        num_neurons  : hidden-layer width.
        num_layers   : number of hidden Dense layers.
        dropout_rate : dropout fraction.
        reg_lambda   : combined L1/L2 penalty weight.
        learning_rate: peak learning rate for Adam.
        lr_schedule  : 'constant' or 'cosine'.
        epochs       : total training epochs (for cosine schedule).
        train_steps  : steps per epoch (for cosine schedule).
    Returns:
        Compiled keras.Model outputting (batch, num_classes) raw logits.
    """
    reg = regularizers.L1L2(l1=reg_lambda, l2=reg_lambda)
    inp = keras.Input(shape=(input_dim,), name="features")
    x = layers.Dropout(dropout_rate, name="input_dropout")(inp)
    for i in range(num_layers):
        x = layers.Dense(num_neurons, activation="relu",
                         kernel_regularizer=reg, name=f"hidden_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)
    # raw logits, no softmax (needed for temperature scaling)
    logits = layers.Dense(num_classes, activation=None,
                          kernel_regularizer=reg, name="logits")(x)
    model = keras.Model(inputs=inp, outputs=logits, name="binder_classifier")
    total_params = model.count_params()
    log.info("  model built: %d total parameters", total_params)
    log.info("    input_dim=%d  num_classes=%d  layers=%d  neurons=%d  dropout=%.2f",
             input_dim, num_classes, num_layers, num_neurons, dropout_rate)
    log.info("    reg: L1=%.1e  L2=%.1e", reg_lambda, reg_lambda)
    # learning-rate schedule
    if lr_schedule == "cosine":
        total_steps = epochs * train_steps
        lr = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=total_steps,
            alpha=1e-6,
        )
        log.info("    lr schedule: cosine decay, init=%.1e, total_steps=%d", learning_rate, total_steps)
    else:
        lr = learning_rate
        log.info("    lr schedule: constant at %.1e", learning_rate)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=DistanceWeightedCELoss(),
        metrics=[SoftTargetAccuracy()],
    )
    return model
# ============================================================================
#  Temperature scaling
# ============================================================================
def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    lr: float = 0.01, max_iter: int = 500) -> float:
    """Fit a single scalar temperature T minimising NLL on (logits, hard labels).
    Temperature scaling uses standard NLL against hard labels for calibration.
    Args:
        logits: (N, C) raw model outputs.
        labels: (N,) integer ground-truth class indices.
    Returns:
        Optimal temperature T* > 0.
    """
    log_T = tf.Variable(0.0, dtype=tf.float32)  # T = exp(log_T) ensures T > 0
    logits_t = tf.constant(logits, dtype=tf.float32)
    labels_t = tf.constant(labels, dtype=tf.int32)
    opt = tf.optimizers.Adam(learning_rate=lr)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    best_loss, patience_ctr, patience = float("inf"), 0, 30
    for step in range(max_iter):
        with tf.GradientTape() as tape:
            T = tf.exp(log_T)
            scaled = logits_t / T
            loss = loss_fn(labels_t, scaled)
        grads = tape.gradient(loss, [log_T])
        opt.apply_gradients(zip(grads, [log_T]))
        lv = float(loss)
        if lv < best_loss - 1e-7:
            best_loss = lv
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            break
    T_star = float(tf.exp(log_T))
    log.info("    temperature fit: T*=%.4f  NLL=%.4f  steps=%d", T_star, best_loss, step + 1)
    return T_star
def calibrate_logits(logits: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling and return calibrated probabilities.
    Args:
        logits: (N, C) raw logits.
        T     : temperature scalar > 0.
    Returns:
        (N, C) calibrated probability matrix (rows sum to 1).
    """
    scaled = logits / T
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    return (exp_s / exp_s.sum(axis=1, keepdims=True)).astype(np.float32)
# ============================================================================
#  Error-controlled thresholds (tolerance-band)
# ============================================================================
def tolerance_band_errors(probs: np.ndarray, labels: np.ndarray,
                          delta: float, bins: np.ndarray = BINDER_BINS) -> np.ndarray:
    """Compute tolerance-band error indicator: E(x) = I[|b_hat(x) - y| > delta].
    Args:
        probs : (N, C) calibrated probabilities.
        labels: (N,) integer class indices.
        delta : absolute tolerance margin.
        bins  : (C,) binder-size values.
    Returns:
        (N,) binary error array (1 = error, 0 = acceptable).
    """
    b_hat = compute_expected_binder(probs, bins)
    y_true = bins[labels]
    return (np.abs(b_hat - y_true) > delta).astype(np.float64)
def find_threshold(values: np.ndarray, errors: np.ndarray, epsilon: float,
                   direction: str = "leq") -> float:
    """Find the tightest threshold on *values* keeping conditional error <= epsilon.
    Args:
        values   : (N,) uncertainty metric (entropy or margin).
        errors   : (N,) binary error indicators.
        epsilon  : maximum acceptable error rate.
        direction: 'leq' -> accept if value <= threshold (entropy).
                   'geq' -> accept if value >= threshold (margin).
    Returns:
        Threshold value.
    """
    order = np.argsort(values)
    sorted_vals = values[order]
    sorted_errs = errors[order]
    n = len(values)
    if direction == "leq":
        cum_err = np.cumsum(sorted_errs)
        cum_n = np.arange(1, n + 1, dtype=np.float64)
        err_rate = cum_err / cum_n
        valid = np.where(err_rate <= epsilon)[0]
        if len(valid) == 0:
            return float(sorted_vals[0])
        return float(sorted_vals[valid[-1]])
    else:  # "geq" -> accept high margin
        cum_err = np.cumsum(sorted_errs[::-1])
        cum_n = np.arange(1, n + 1, dtype=np.float64)
        err_rate = cum_err / cum_n
        valid = np.where(err_rate <= epsilon)[0]
        if len(valid) == 0:
            return float(sorted_vals[-1])
        idx = valid[-1]
        return float(sorted_vals[n - 1 - idx])
# ============================================================================
#  Evaluation helpers
# ============================================================================
def evaluate_and_plot(probs: np.ndarray, labels: np.ndarray, label_map: dict,
                      split_name: str, output_dir: str, delta: float):
    """Compute metrics and save confusion matrix plot.
    Metrics: categorical accuracy, MAE of continuous estimate, tolerance-band
    accuracy, macro AUC (OvR), macro PRAUC.
    Args:
        probs      : (N, C) calibrated probabilities.
        labels     : (N,) integer labels (0..5).
        label_map  : {int: binder_size}.
        split_name : e.g. 'train', 'val', 'test'.
        output_dir : path for saving figures.
        delta      : tolerance band for reporting tolerance-band accuracy.
    Returns:
        dict of metric values.
    """
    n_classes = probs.shape[1]
    preds = probs.argmax(axis=1)
    acc = (preds == labels).mean()
    # continuous expected binder size
    b_hat = compute_expected_binder(probs)
    y_true_vals = BINDER_BINS[labels]
    mae = np.mean(np.abs(b_hat - y_true_vals))
    tol_acc = np.mean(np.abs(b_hat - y_true_vals) <= delta)
    # one-hot for sklearn
    y_onehot = np.zeros_like(probs, dtype=np.int32)
    y_onehot[np.arange(len(labels)), labels] = 1
    try:
        auc_ovr = roc_auc_score(y_onehot, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc_ovr = float("nan")
    try:
        prauc = average_precision_score(y_onehot, probs, average="macro")
    except ValueError:
        prauc = float("nan")
    log.info("  [%s]  Acc=%.4f  MAE=%.2f  TolAcc(d=%.1f)=%.4f  AUC=%.4f  PRAUC=%.4f",
             split_name, acc, mae, delta, tol_acc, auc_ovr, prauc)
    # per-class accuracy
    for ci in range(n_classes):
        mask_ci = labels == ci
        if mask_ci.sum() > 0:
            acc_ci = (preds[mask_ci] == ci).mean()
            mae_ci = np.mean(np.abs(b_hat[mask_ci] - y_true_vals[mask_ci]))
            log.info("    class %d (binder=%s): acc=%.4f  mae=%.2f  n=%d",
                     ci, label_map.get(ci, "?"), acc_ci, mae_ci, mask_ci.sum())
    # confusion matrix plot
    cm = confusion_matrix(labels, preds)
    class_names = [str(label_map[i]) for i in range(n_classes)]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(n_classes), yticks=np.arange(n_classes),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True",
           title=f"Confusion - {split_name} (acc={acc:.3f})")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"confusion_{split_name}.png"), dpi=150)
    plt.close(fig)
    log.info("    confusion matrix saved to %s", os.path.join(output_dir, f"confusion_{split_name}.png"))
    return {"accuracy": float(acc), "mae": float(mae),
            "tolerance_accuracy": float(tol_acc),
            "auc_ovr": float(auc_ovr), "prauc": float(prauc)}
# ============================================================================
#  TRAIN mode
# ============================================================================
def run_train(args):
    """Full training pipeline: load -> ordinal train -> calibrate -> thresholds."""
    set_global_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    log.info("Output directories:")
    log.info("  output_dir:    %s", args.output_dir)
    log.info("  checkpoint:    %s", checkpoint_dir)
    log.info("  figures:       %s", figures_dir)
    meta = {}
    # ── 1. Load data ────────────────────────────────────────────────────────
    log.info("Loading synthetic data from %s ...", args.input_train)
    X, Y, label_map = load_synthetic_data(args.input_train)
    meta["label_map"] = {str(k): int(v) for k, v in label_map.items()}
    meta["num_classes"] = NUM_CLASSES
    meta["feature_dim"] = int(X.shape[1])
    meta["binder_bins"] = [int(b) for b in BINDER_BINS]
    meta["ordinal_lambda"] = args.ordinal_lambda
    meta["delta"] = args.delta
    # ── 2. Split ────────────────────────────────────────────────────────────
    log.info("Splitting data (val=%.2f, test=%.2f) ...", args.val_split, args.val_split)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(
        X, Y, args.val_split, args.seed
    )
    log.info("  train=%d  val=%d  test=%d", len(Y_train), len(Y_val), len(Y_test))
    meta["split_sizes"] = {
        "train": int(len(Y_train)),
        "val": int(len(Y_val)),
        "test": int(len(Y_test)),
    }
    # ── 3. Build soft targets ──────────────────────────────────────────────
    log.info("Building distance-weighted soft targets (lambda=%.2f) ...", args.ordinal_lambda)
    Y_train_soft = make_soft_targets(Y_train, BINDER_BINS, args.ordinal_lambda)
    Y_val_soft = make_soft_targets(Y_val, BINDER_BINS, args.ordinal_lambda)
    Y_test_soft = make_soft_targets(Y_test, BINDER_BINS, args.ordinal_lambda)
    log.info("  example soft target for class 0 (binder=%d): %s",
             label_map[0], np.round(Y_train_soft[Y_train == 0][0], 4))
    # log all class soft target shapes for verification
    for ci in range(NUM_CLASSES):
        mask_ci = Y_train == ci
        if mask_ci.sum() > 0:
            example = Y_train_soft[mask_ci][0]
            log.info("    class %d (binder=%2d): soft=[%s]  argmax=%d  max_p=%.4f",
                     ci, label_map[ci],
                     ", ".join([f"{v:.3f}" for v in example]),
                     example.argmax(), example.max())
    # ── 4. tf.data datasets ────────────────────────────────────────────────
    train_ds = make_tf_dataset(X_train, Y_train_soft, args.batch_size, shuffle=True, seed=args.seed)
    val_ds = make_tf_dataset(X_val, Y_val_soft, args.batch_size, shuffle=False)
    train_steps = int(np.ceil(len(Y_train) / args.batch_size))
    # ── 5. Build or resume model ────────────────────────────────────────────
    best_model_path = os.path.join(checkpoint_dir, "model_best.keras")
    if args.resume and os.path.exists(best_model_path):
        log.info("Resuming from %s", best_model_path)
        model = keras.models.load_model(
            best_model_path,
            custom_objects={
                "DistanceWeightedCELoss": DistanceWeightedCELoss,
                "SoftTargetAccuracy": SoftTargetAccuracy,
            },
        )
    else:
        log.info("Building new model (classes=%d, neurons=%d, layers=%d) ...",
                 NUM_CLASSES, args.num_neurons, args.num_layers)
        model = build_model(
            input_dim=X.shape[1],
            num_classes=NUM_CLASSES,
            num_neurons=args.num_neurons,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
            reg_lambda=args.reg_lambda,
            learning_rate=args.learning_rate,
            lr_schedule=args.lr_schedule,
            epochs=args.epochs,
            train_steps=train_steps,
        )
    model.summary(print_fn=log.info)
    # ── 6. Callbacks ────────────────────────────────────────────────────────
    cb_list = [
        callbacks.ModelCheckpoint(
            best_model_path, monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
        callbacks.CSVLogger(
            os.path.join(args.output_dir, "training_log.csv"), append=args.resume
        ),
    ]
    if args.convergence:
        cb_list.append(
            callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True, verbose=1,
            )
        )
    log.info("Callbacks: %s", [type(c).__name__ for c in cb_list])
    # ── 7. Train ────────────────────────────────────────────────────────────
    log.info("Training for up to %d epochs ...", args.epochs)
    log.info("  batch_size=%d  steps/epoch=%d  early_stop=%s (patience=15)",
             args.batch_size, train_steps, args.convergence)
    t0_train = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, callbacks=cb_list, verbose=2,
    )
    train_time = time.time() - t0_train
    n_epochs_ran = len(history.history["loss"])
    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    best_val_loss = min(history.history["val_loss"])
    best_epoch = np.argmin(history.history["val_loss"]) + 1
    log.info("Training complete: %d epochs in %.1fs (%.2fs/epoch)",
             n_epochs_ran, train_time, train_time / max(n_epochs_ran, 1))
    log.info("  final  -> train_loss=%.4f  val_loss=%.4f  train_acc=%.4f  val_acc=%.4f",
             final_train_loss, final_val_loss, final_train_acc, final_val_acc)
    log.info("  best   -> val_loss=%.4f at epoch %d", best_val_loss, best_epoch)
    meta["history"] = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    meta["training_time_sec"] = train_time
    meta["epochs_ran"] = n_epochs_ran
    meta["best_epoch"] = best_epoch
    # reload best weights
    if os.path.exists(best_model_path):
        log.info("Reloading best model from %s", best_model_path)
        model = keras.models.load_model(
            best_model_path,
            custom_objects={
                "DistanceWeightedCELoss": DistanceWeightedCELoss,
                "SoftTargetAccuracy": SoftTargetAccuracy,
            },
        )
        log.info("  best model reloaded successfully")
    # ── 8. Extract logits ──────────────────────────────────────────────────
    log.info("Extracting logits on train/val/test ...")
    t0_logits = time.time()
    logits_train = model.predict(X_train.astype(np.float32), batch_size=args.batch_size, verbose=0)
    log.info("  train logits: %s  range=[%.3f, %.3f]", logits_train.shape,
             logits_train.min(), logits_train.max())
    logits_val = model.predict(X_val.astype(np.float32), batch_size=args.batch_size, verbose=0)
    log.info("  val logits:   %s  range=[%.3f, %.3f]", logits_val.shape,
             logits_val.min(), logits_val.max())
    logits_test = model.predict(X_test.astype(np.float32), batch_size=args.batch_size, verbose=0)
    log.info("  test logits:  %s  range=[%.3f, %.3f]", logits_test.shape,
             logits_test.min(), logits_test.max())
    log.info("  logit extraction: %.1fs", time.time() - t0_logits)
    # ── 9. K-fold cross-validated calibration & thresholds ─────────────────
    log.info("=" * 60)
    log.info("K-fold calibration (K=%d) on val+test ...", args.k_crossval)
    logits_cal = np.concatenate([logits_val, logits_test], axis=0)
    labels_cal = np.concatenate([Y_val, Y_test], axis=0)
    n_cal = len(labels_cal)
    log.info("  calibration set: %d samples (val=%d + test=%d)", n_cal, len(Y_val), len(Y_test))
    log.info("  tolerance band delta=%.1f  |  max error rate epsilon=%.4f",
             args.delta, args.max_error_rate)
    K = args.k_crossval
    rng = np.random.RandomState(args.seed + 1)
    perm = rng.permutation(n_cal)
    fold_ids = np.array_split(perm, K)
    fold_temps, fold_H0, fold_D0 = [], [], []
    t0_cal = time.time()
    for k in range(K):
        log.info("  --- fold %d/%d ---", k + 1, K)
        heldout_idx = fold_ids[k]
        train_idx = np.concatenate([fold_ids[j] for j in range(K) if j != k])
        log.info("    train: %d samples  |  heldout: %d samples", len(train_idx), len(heldout_idx))
        # fit temperature on K-1 folds (hard labels for standard NLL calibration)
        T_k = fit_temperature(logits_cal[train_idx], labels_cal[train_idx])
        fold_temps.append(T_k)
        # calibrate held-out fold
        probs_k = calibrate_logits(logits_cal[heldout_idx], T_k)
        log.info("    calibrated probs: mean_max_p=%.4f  mean_entropy=%.4f",
                 probs_k.max(axis=1).mean(), compute_predictive_entropy(probs_k).mean())
        # tolerance-band errors using continuous expected binder size
        errors_k = tolerance_band_errors(probs_k, labels_cal[heldout_idx], args.delta)
        n_errors = int(errors_k.sum())
        ent_k = compute_predictive_entropy(probs_k)
        margin_k = compute_margin(probs_k)
        log.info("    tolerance-band errors: %d / %d (%.2f%%)",
                 n_errors, len(errors_k), 100.0 * errors_k.mean())
        log.info("    entropy: min=%.4f  max=%.4f  mean=%.4f",
                 ent_k.min(), ent_k.max(), ent_k.mean())
        log.info("    margin:  min=%.4f  max=%.4f  mean=%.4f",
                 margin_k.min(), margin_k.max(), margin_k.mean())
        # find thresholds on this fold
        H0_k = find_threshold(ent_k, errors_k, args.max_error_rate, direction="leq")
        D0_k = find_threshold(margin_k, errors_k, args.max_error_rate, direction="geq")
        fold_H0.append(H0_k)
        fold_D0.append(D0_k)
        tol_acc_k = 1.0 - errors_k.mean()
        # how many samples pass each threshold
        pass_H = (ent_k <= H0_k).sum()
        pass_D = (margin_k >= D0_k).sum()
        pass_both = ((ent_k <= H0_k) & (margin_k >= D0_k)).sum()
        log.info("    T=%.4f  H0=%.4f  D0=%.4f  tol_acc=%.4f",
                 T_k, H0_k, D0_k, tol_acc_k)
        log.info("    pass H0: %d (%.1f%%)  pass D0: %d (%.1f%%)  pass both: %d (%.1f%%)",
                 pass_H, 100.0 * pass_H / len(ent_k),
                 pass_D, 100.0 * pass_D / len(margin_k),
                 pass_both, 100.0 * pass_both / len(ent_k))
    cal_time = time.time() - t0_cal
    # aggregate with median (robust to outlier folds)
    T_star = float(np.median(fold_temps))
    H0 = float(np.median(fold_H0))
    D0 = float(np.median(fold_D0))
    log.info("=" * 60)
    log.info("Calibration summary (%.1fs):", cal_time)
    log.info("  per-fold T:  %s", [f"{t:.4f}" for t in fold_temps])
    log.info("  per-fold H0: %s", [f"{h:.4f}" for h in fold_H0])
    log.info("  per-fold D0: %s", [f"{d:.4f}" for d in fold_D0])
    log.info("  AGGREGATED (median):  T*=%.4f  H0=%.4f  D0=%.4f", T_star, H0, D0)
    meta["calibration"] = {
        "fold_temperatures": [float(t) for t in fold_temps],
        "fold_H0": [float(h) for h in fold_H0],
        "fold_D0": [float(d) for d in fold_D0],
        "T_star": T_star, "H0": H0, "D0": D0,
        "max_error_rate": args.max_error_rate,
        "delta": args.delta,
    }
    # ── 10. Evaluate calibrated probs on all splits ─────────────────────────
    log.info("=" * 60)
    log.info("Evaluating calibrated predictions (T*=%.4f) ...", T_star)
    perf = {}
    for name, logits_s, labels_s in [
        ("train", logits_train, Y_train),
        ("val",   logits_val,   Y_val),
        ("test",  logits_test,  Y_test),
    ]:
        probs_s = calibrate_logits(logits_s, T_star)
        log.info("  %s: calibrated prob max=%.4f  min=%.6f", name,
                 probs_s.max(), probs_s.min())
        perf[name] = evaluate_and_plot(probs_s, labels_s, label_map, name,
                                       figures_dir, args.delta)
    meta["performance"] = perf
    log.info("Performance summary:")
    for name in ["train", "val", "test"]:
        p = perf[name]
        log.info("  %-5s  Acc=%.4f  MAE=%.2f  TolAcc=%.4f  AUC=%.4f  PRAUC=%.4f",
                 name, p["accuracy"], p["mae"], p["tolerance_accuracy"],
                 p["auc_ovr"], p["prauc"])
    # ── 11. Training curves ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(meta["history"]["loss"], label="train")
    axes[0].plot(meta["history"]["val_loss"], label="val")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Distance-Weighted CE")
    axes[0].legend()
    axes[1].plot(meta["history"]["accuracy"], label="train")
    axes[1].plot(meta["history"]["val_accuracy"], label="val")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    log.info("Training curves saved to %s", os.path.join(figures_dir, "training_curves.png"))
    # ── 12. Save metadata ──────────────────────────────────────────────────
    meta_path = os.path.join(args.output_dir, "metadata_train.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, cls=NumpyEncoder)
    log.info("Metadata saved to %s (%.1f KB)", meta_path, os.path.getsize(meta_path) / 1024)
    log.info("=" * 60)
    log.info("TRAINING PIPELINE COMPLETE")
    log.info("  output_dir:   %s", args.output_dir)
    log.info("  best model:   %s", best_model_path)
    log.info("  metadata:     %s", meta_path)
    log.info("  figures:      %s", figures_dir)
    log.info("  T*=%.4f  H0=%.4f  D0=%.4f  (eps=%.4f, delta=%.1f)",
             T_star, H0, D0, args.max_error_rate, args.delta)
    log.info("=" * 60)
# ============================================================================
#  INFERENCE mode
# ============================================================================
def run_inference(args):
    """Stream-inference on real TCR data: calibrate, continuous estimate, uncertainty, plots."""
    set_global_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures_inference")
    os.makedirs(figures_dir, exist_ok=True)
    # ── 1. Load metadata ───────────────────────────────────────────────────
    meta_path = os.path.join(args.output_dir, "metadata_train.json")
    assert os.path.exists(meta_path), f"Metadata not found at {meta_path}"
    with open(meta_path) as f:
        meta = json.load(f)
    label_map = {int(k): int(v) for k, v in meta["label_map"].items()}
    binder_bins = np.array(meta["binder_bins"], dtype=np.float32)
    T_star = meta["calibration"]["T_star"]
    H0 = meta["calibration"]["H0"]
    D0 = meta["calibration"]["D0"]
    eps = meta["calibration"]["max_error_rate"]
    delta = meta["calibration"]["delta"]
    num_classes = meta["num_classes"]
    log.info("Calibration: T*=%.4f  H0=%.4f  D0=%.4f  eps=%.4f  delta=%.1f",
             T_star, H0, D0, eps, delta)
    log.info("  label_map: %s", label_map)
    log.info("  binder_bins: %s", binder_bins.tolist())
    # ── 2. Load model ──────────────────────────────────────────────────────
    model_path = os.path.join(args.output_dir, "checkpoint", "model_best.keras")
    assert os.path.exists(model_path), f"Model not found at {model_path}"
    log.info("Loading model from %s ...", model_path)
    model = keras.models.load_model(
        model_path,
        custom_objects={
            "DistanceWeightedCELoss": DistanceWeightedCELoss,
            "SoftTargetAccuracy": SoftTargetAccuracy,
        },
    )
    log.info("  model loaded: %d parameters", model.count_params())
    model.summary(print_fn=log.info)
    # ── 3. Open H5 ────────────────────────────────────────────────────────
    assert os.path.exists(args.input_inference), f"H5 not found: {args.input_inference}"
    h5_in = h5py.File(args.input_inference, "r")
    log.info("H5 file opened: %s", args.input_inference)
    log.info("  available keys: %s", list(h5_in.keys()))
    for key in ["entropy", "n_donors", "explanation_fractions"]:
        assert key in h5_in, f"Missing key '{key}' in H5"
        log.info("  '%s': shape=%s  dtype=%s", key, h5_in[key].shape, h5_in[key].dtype)
    N_total = h5_in["entropy"].shape[0]
    log.info("Total TCRs for inference: %d", N_total)
    n_chunks = int(np.ceil(N_total / args.infer_chunk))
    log.info("  chunk_size=%d  -> %d chunks", args.infer_chunk, n_chunks)
    chunk = args.infer_chunk
    # ── 4. Chunked inference ───────────────────────────────────────────────
    all_probs_raw = np.empty((N_total, num_classes), dtype=np.float32)
    all_probs_cal = np.empty((N_total, num_classes), dtype=np.float32)
    log.info("Pre-allocated output arrays: %.1f MB",
             (all_probs_raw.nbytes + all_probs_cal.nbytes) / (1024**2))
    t0_infer = time.time()
    chunk_i = 0
    for start in range(0, N_total, chunk):
        chunk_i += 1
        end = min(start + chunk, N_total)
        t_chunk = time.time()
        # assemble features: [entropy(1), n_donors(1), explanation_fractions(19)] = 21
        ent_chunk = np.array(h5_in["entropy"][start:end], dtype=np.float32).reshape(-1, 1)
        nd_chunk = np.array(h5_in["n_donors"][start:end], dtype=np.float32).reshape(-1, 1)
        ef_chunk = np.array(h5_in["explanation_fractions"][start:end, 1:-1], dtype=np.float32)
        x_chunk = np.concatenate([ent_chunk, nd_chunk, ef_chunk], axis=-1)  # (chunk, 21)
        logits_chunk = model.predict(x_chunk, batch_size=args.batch_size, verbose=0)
        all_probs_raw[start:end] = calibrate_logits(logits_chunk, 1.0)
        all_probs_cal[start:end] = calibrate_logits(logits_chunk, T_star)
        elapsed_chunk = time.time() - t_chunk
        pct = 100.0 * end / N_total
        log.info("  chunk %d/%d [%d, %d)  n=%d  x_shape=%s  %.1fs  (%.1f%% done)",
                 chunk_i, n_chunks, start, end, end - start, x_chunk.shape,
                 elapsed_chunk, pct)
    h5_in.close()
    total_infer_time = time.time() - t0_infer
    log.info("Inference complete: %d TCRs in %.1fs (%.0f TCR/s)",
             N_total, total_infer_time, N_total / max(total_infer_time, 1e-6))
    # ── 5. Derived quantities ──────────────────────────────────────────────
    log.info("Computing derived quantities ...")
    # continuous expected binder size (main output)
    pred_expected = compute_expected_binder(all_probs_cal, binder_bins)
    log.info("  expected binder size: min=%.2f  max=%.2f  mean=%.2f  median=%.2f",
             pred_expected.min(), pred_expected.max(),
             pred_expected.mean(), np.median(pred_expected))
    # argmax mapped to discrete binder label
    pred_best = np.array([label_map[c] for c in all_probs_cal.argmax(axis=1)], dtype=np.int32)
    unique_preds, pred_counts = np.unique(pred_best, return_counts=True)
    log.info("  discrete predictions distribution:")
    for bp, bc in zip(unique_preds, pred_counts):
        log.info("    binder=%d: %d (%.1f%%)", bp, bc, 100.0 * bc / N_total)
    # uncertainty metrics
    pred_entropy = compute_predictive_entropy(all_probs_cal)
    pred_margin = compute_margin(all_probs_cal)
    log.info("  entropy:  min=%.4f  max=%.4f  mean=%.4f  std=%.4f",
             pred_entropy.min(), pred_entropy.max(), pred_entropy.mean(), pred_entropy.std())
    log.info("  margin:   min=%.4f  max=%.4f  mean=%.4f  std=%.4f",
             pred_margin.min(), pred_margin.max(), pred_margin.mean(), pred_margin.std())
    # reliable = passes BOTH entropy and margin thresholds
    pass_entropy = (pred_entropy <= H0).sum()
    pass_margin = (pred_margin >= D0).sum()
    reliable = (pred_entropy <= H0) & (pred_margin >= D0)
    log.info("  threshold check:")
    log.info("    pass entropy (H <= %.4f): %d / %d (%.2f%%)",
             H0, pass_entropy, N_total, 100.0 * pass_entropy / N_total)
    log.info("    pass margin  (D >= %.4f): %d / %d (%.2f%%)",
             D0, pass_margin, N_total, 100.0 * pass_margin / N_total)
    log.info("    pass BOTH (reliable):     %d / %d (%.2f%%)",
             reliable.sum(), N_total, 100.0 * reliable.mean())
    # ── 6. Write results to H5 ────────────────────────────────────────────
    log.info("Writing results to %s ...", args.input_inference)
    h5_out = h5py.File(args.input_inference, "a")
    def _write(key, data):
        """Overwrite dataset if it already exists, then create with gzip compression."""
        if key in h5_out:
            log.info("    overwriting existing key '%s'", key)
            del h5_out[key]
        h5_out.create_dataset(key, data=data, compression="gzip", compression_opts=4)
        log.info("    wrote '%s': shape=%s  dtype=%s  (%.1f MB)",
                 key, data.shape, data.dtype, data.nbytes / (1024**2))
    _write("predicted_binderset_probs", all_probs_raw)
    _write("predicted_binderset_probs_calibrated", all_probs_cal)
    _write("predicted_binderset_best_calibrated", pred_best)
    _write("predicted_binderset_expected", pred_expected)
    _write("predicted_binderset_entropy", pred_entropy)
    _write("predicted_binderset_margin", pred_margin)
    _write("predicted_binderset_reliable", reliable.astype(np.uint8))
    h5_out.flush()
    log.info("  H5 flushed, all keys written")
    # also save as npz alongside the h5
    npz_path = os.path.splitext(args.input_inference)[0] + "_binderset_probs.npz"
    np.savez_compressed(npz_path,
                        probs_raw=all_probs_raw,
                        probs_calibrated=all_probs_cal,
                        best_calibrated=pred_best,
                        expected=pred_expected,
                        entropy=pred_entropy,
                        margin=pred_margin,
                        reliable=reliable)
    log.info("Probabilities also saved to %s", npz_path)
    # ── 7. Read donor counts and extra H5 metrics for plotting ──────────────
    log.info("Reading additional H5 keys for plotting ...")
    h5_rd = h5py.File(args.input_inference, "r")
    n_donors_all = np.array(h5_rd["n_donors"][:], dtype=np.int32)
    h5_n_active = np.array(h5_rd["n_active_alleles"][:], dtype=np.float32)
    h5_expl_auc = np.array(h5_rd["explanation_auc"][:], dtype=np.float32)
    h5_entropy = np.array(h5_rd["entropy"][:], dtype=np.float32)
    h5_max_zprob = np.array(h5_rd["max_z_prob"][:], dtype=np.float32)
    h5_mean_zprob_nz = np.array(h5_rd["mean_z_prob_nonzero"][:], dtype=np.float32)
    h5_rd.close()
    h5_out.close()
    log.info("  n_donors:            min=%d  max=%d  mean=%.1f  median=%d  unique=%d",
             n_donors_all.min(), n_donors_all.max(), n_donors_all.mean(),
             int(np.median(n_donors_all)), len(np.unique(n_donors_all)))
    log.info("  n_active_alleles:    min=%.0f  max=%.0f  mean=%.1f",
             h5_n_active.min(), h5_n_active.max(), h5_n_active.mean())
    log.info("  explanation_auc:     min=%.4f  max=%.4f  mean=%.4f",
             h5_expl_auc.min(), h5_expl_auc.max(), h5_expl_auc.mean())
    log.info("  entropy (MLE):       min=%.4f  max=%.4f  mean=%.4f",
             h5_entropy.min(), h5_entropy.max(), h5_entropy.mean())
    log.info("  max_z_prob:          min=%.4f  max=%.4f  mean=%.4f",
             h5_max_zprob.min(), h5_max_zprob.max(), h5_max_zprob.mean())
    log.info("  mean_z_prob_nonzero: min=%.4f  max=%.4f  mean=%.4f",
             h5_mean_zprob_nz.min(), h5_mean_zprob_nz.max(), h5_mean_zprob_nz.mean())
    # ── 8. Bar plot: # reliable TCRs, cumulative for 35+ ──────────────────
    log.info("Generating bar plot (individual 2-34, cumulative 35+) ...")
    bar_labels = [str(d) for d in range(2, 35)] + ["35+"]
    bar_counts = np.zeros(len(bar_labels), dtype=np.int64)
    for i, d in enumerate(range(2, 35)):
        bar_counts[i] = int(reliable[n_donors_all == d].sum())
    bar_counts[-1] = int(reliable[n_donors_all >= 35].sum())
    log.info("  bar bins: %d categories  |  total reliable in plot: %d",
             len(bar_labels), bar_counts.sum())
    log.info("  reliable 35+: %d", bar_counts[-1])
    fig, ax = plt.subplots(figsize=(14, 5))
    x_pos = np.arange(len(bar_labels))
    ax.bar(x_pos, bar_counts, color="steelblue", edgecolor="none")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_labels, rotation=90, fontsize=7)
    ax.set_xlabel("Number of donors (n)")
    ax.set_ylabel(f"# reliable TCRs (eps<={eps}, delta<={delta})")
    ax.set_title("Reliable predictions per donor count (35+ cumulative)")
    fig.tight_layout()
    barplot_path = os.path.join(figures_dir, "barplot_reliable_per_donor.png")
    fig.savefig(barplot_path, dpi=150)
    plt.close(fig)
    log.info("  bar plot saved to %s", barplot_path)
    # ── 9. Multi-panel heatmaps (6 panels in one figure) ──────────────────
    # donor bins: [2, 3, 4, 5, 6-10, 11-20, 21-35, 35+]
    DONOR_BIN_LABELS = ["2", "3", "4", "5", "6-10", "11-20", "21-35", "35+"]
    N_DONOR_BINS = len(DONOR_BIN_LABELS)
    def assign_donor_bin(donors):
        """Assign each donor count to one of 8 bins. Returns int bin index 0..7."""
        out = np.full(len(donors), -1, dtype=np.int32)
        out[donors == 2] = 0
        out[donors == 3] = 1
        out[donors == 4] = 2
        out[donors == 5] = 3
        out[(donors >= 6) & (donors <= 10)] = 4
        out[(donors >= 11) & (donors <= 20)] = 5
        out[(donors >= 21) & (donors <= 35)] = 6
        out[donors > 35] = 7
        return out
    donor_bin_all = assign_donor_bin(n_donors_all)
    binder_classes = sorted(label_map.values())  # [3, 5, 10, 15, 25, 35]
    n_binder = len(binder_classes)
    binder_labels = [str(b) for b in binder_classes]
    log.info("Generating 6-panel heatmap figure ...")
    log.info("  donor bins: %s", DONOR_BIN_LABELS)
    log.info("  binder classes: %s", binder_classes)
    # filter to reliable only
    rel_mask = reliable
    rel_donor_bin = donor_bin_all[rel_mask]
    rel_binder = pred_best[rel_mask]
    rel_n_active = h5_n_active[rel_mask]
    rel_expl_auc = h5_expl_auc[rel_mask]
    rel_entropy = h5_entropy[rel_mask]
    rel_max_zprob = h5_max_zprob[rel_mask]
    rel_mean_zprob_nz = h5_mean_zprob_nz[rel_mask]
    total_reliable = int(rel_mask.sum())
    log.info("  reliable TCRs for heatmaps: %d", total_reliable)
    # count matrix (shared across all panels for annotation)
    hm_counts = np.zeros((n_binder, N_DONOR_BINS), dtype=np.int64)
    for ri, bval in enumerate(binder_classes):
        mask_b = rel_binder == bval
        for ci in range(N_DONOR_BINS):
            hm_counts[ri, ci] = (mask_b & (rel_donor_bin == ci)).sum()
    def build_mean_heatmap(values):
        """Build (n_binder x N_DONOR_BINS) heatmap of mean values per cell.
        Cells with no data are NaN.
        """
        hmap = np.full((n_binder, N_DONOR_BINS), np.nan, dtype=np.float64)
        for ri, bval in enumerate(binder_classes):
            mask_b = rel_binder == bval
            for ci in range(N_DONOR_BINS):
                cell = mask_b & (rel_donor_bin == ci)
                if cell.sum() > 0:
                    hmap[ri, ci] = float(values[cell].mean())
        return hmap
    def build_frac_heatmap():
        """Build (n_binder x N_DONOR_BINS) heatmap of fraction of total reliable."""
        return hm_counts.astype(np.float64) / max(total_reliable, 1)
    # A. fraction of reliable
    hm_frac = build_frac_heatmap()
    # B. avg n_active_alleles
    hm_nactive = build_mean_heatmap(rel_n_active)
    # C. avg explanation_auc
    hm_explauc = build_mean_heatmap(rel_expl_auc)
    # D. avg entropy (MLE, from H5)
    hm_ent = build_mean_heatmap(rel_entropy)
    # E. avg max_z_prob
    hm_maxzp = build_mean_heatmap(rel_max_zprob)
    # F. avg mean_z_prob_nonzero
    hm_meanzpnz = build_mean_heatmap(rel_mean_zprob_nz)
    # panel definitions: (data, title, fmt_string, colorbar_label)
    panels = [
        (hm_frac,    f"A. Fraction of reliable (eps<={eps})", ".4f", "Fraction"),
        (hm_nactive, "B. Avg n_active_alleles",               ".1f", "# alleles"),
        (hm_explauc, "C. Avg explanation_auc",                 ".3f", "AUC"),
        (hm_ent,     "D. Avg entropy (MLE)",                   ".3f", "Entropy"),
        (hm_maxzp,   "E. Avg max_z_prob",                      ".3f", "Prob"),
        (hm_meanzpnz,"F. Avg mean_z_prob_nonzero",             ".4f", "Prob"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    axes_flat = axes.ravel()
    for pidx, (hm_data, title, fmt, cbar_label) in enumerate(panels):
        ax = axes_flat[pidx]
        # display: NaN -> 0 for colour mapping, but show "-" in text
        hm_display = np.nan_to_num(hm_data, nan=0.0)
        im = ax.imshow(hm_display, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_xticks(np.arange(N_DONOR_BINS))
        ax.set_xticklabels(DONOR_BIN_LABELS, fontsize=9)
        ax.set_yticks(np.arange(n_binder))
        ax.set_yticklabels(binder_labels, fontsize=10)
        ax.set_xlabel("Donor count bin", fontsize=9)
        ax.set_ylabel("Predicted binder set", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8, label=cbar_label)
        # annotate every cell with its value
        vmin, vmax = hm_display.min(), hm_display.max()
        vrange = max(vmax - vmin, 1e-10)
        for ri in range(n_binder):
            for ci in range(N_DONOR_BINS):
                val = hm_data[ri, ci]
                if np.isnan(val) or hm_counts[ri, ci] == 0:
                    txt = "-"
                else:
                    txt = f"{val:{fmt}}"
                # text colour: white on dark (low), black on bright (high)
                norm_val = (hm_display[ri, ci] - vmin) / vrange
                text_color = "white" if norm_val < 0.5 else "black"
                ax.text(ci, ri, txt, ha="center", va="center",
                        fontsize=7, color=text_color, fontweight="bold")
    fig.suptitle(f"Reliable TCR heatmaps  (n={total_reliable}, eps<={eps}, delta<={delta})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    heatmap_path = os.path.join(figures_dir, "heatmap_reliable_donor_binder.png")
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    log.info("  heatmap figure saved to %s", heatmap_path)
    # log summary per panel
    for hm_data, title, fmt, _ in panels:
        valid = hm_data[~np.isnan(hm_data)]
        if len(valid) > 0:
            log.info("    %-40s  min=%s  max=%s  mean=%s",
                     title,
                     f"{valid.min():{fmt}}",
                     f"{valid.max():{fmt}}",
                     f"{valid.mean():{fmt}}")
        else:
            log.info("    %-40s  all NaN (no reliable data)", title)
    log.info("=" * 60)
    log.info("INFERENCE PIPELINE COMPLETE")
    log.info("  input:      %s (%d TCRs)", args.input_inference, N_total)
    log.info("  reliable:   %d / %d (%.2f%%)", reliable.sum(), N_total, 100.0 * reliable.mean())
    log.info("  outputs:    %s", args.input_inference)
    log.info("  npz:        %s", npz_path)
    log.info("  figures:    %s", figures_dir)
    log.info("=" * 60)
# ============================================================================
#  Main
# ============================================================================
def main():
    parser = build_parser()
    args = parser.parse_args()
    log.info("=" * 60)
    log.info("TCR Binder Set Size Classifier")
    log.info("=" * 60)
    log.info("Mode: %s", args.mode)
    log.info("Args: %s", vars(args))
    t0_total = time.time()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "inference":
        run_inference(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    log.info("Total wall time: %.1fs", time.time() - t0_total)
if __name__ == "__main__":
    main()