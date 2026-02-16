#!/usr/bin/env python3
"""
Uncertainty-Aware Inference of TCR Binder Set Size from Synthetic Training Data.
================================================================================
Two-mode pipeline:
  train      – Train a multiclass classifier on synthetic TCR-HLA data,
               perform k-fold cross-validated temperature calibration,
               and derive error-controlled uncertainty thresholds.
  inference  – Apply the trained model to real TCR data (streamed in chunks),
               calibrate probabilities, compute uncertainty, and produce
               diagnostic plots.
All heavy lifting uses TensorFlow / Keras with vectorised NumPy;
designed for CPU or single-GPU execution within 40 GB RAM / 16 GB VRAM.
"""
import argparse
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster / headless runs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# ── TensorFlow setup ────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress INFO/WARNING
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
# ── Binder-set bins (fixed) ────────────────────────────────────────────────
BINDER_BINS = [3, 5, 10, 15, 25, 35]
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
    # ── thresholds ──────────────────────────────────────────────────────────
    p.add_argument("--max_error_rate", type=float, default=0.005,
                   help="Maximum acceptable error rate (epsilon) for selective prediction.")
    # ── inference ────────────────────────────────────────────────────────────
    p.add_argument("--infer_chunk", type=int, default=500_000,
                   help="Chunk size for streaming inference on large H5.")
    return p
# ============================================================================
#  Utility functions
# ============================================================================
def compute_entropy(z_probs: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy (base-2) row-wise for a probability matrix.
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
    """Predictive entropy H(x) = -sum p log p  (natural log, as in the paper).
    Args:
        probs: (N, C) calibrated probability matrix (rows sum to 1).
    Returns:
        (N,) entropy values (nats).
    """
    safe = np.maximum(probs, 1e-12)
    return -(probs * np.log(safe)).sum(axis=1).astype(np.float32)
def compute_margin(probs: np.ndarray) -> np.ndarray:
    """Margin = p_(1) - p_(2), the gap between top-two class probabilities.
    Args:
        probs: (N, C) probability matrix.
    Returns:
        (N,) margin values in [0, 1].
    """
    # partition is O(N) vs full sort O(N log C)
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    return (top2.max(axis=1) - top2.min(axis=1)).astype(np.float32)
def set_global_seed(seed: int):
    """Set random seeds for reproducibility across numpy, tf, and python."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
# ============================================================================
#  Data loading (training)
# ============================================================================
def load_synthetic_data(input_dir: str):
    """Load and concatenate feature matrices from all bX_nY sub-folders.
    Returns:
        X      : (N_all, 21) feature matrix [entropy, donor_size, 19 explanation fractions].
        Y      : (N_all, 1)  integer class labels.
        label_map : dict  {int_label: binder_set_size}.
    """
    pattern = re.compile(r"^b\d+_n\d+$")
    # sort folders so label assignment is deterministic
    folder_names = sorted(
        [f for f in os.listdir(input_dir) if pattern.match(f)],
        key=lambda s: int(s.split("_")[0].replace("b", ""))
    )
    assert len(folder_names) > 0, f"No bX_nY folders found in {input_dir}"
    folders = [os.path.join(input_dir, f) for f in folder_names]
    X_parts, Y_parts = [], []
    label_map = {}
    thrs = np.arange(0.05, 0.95 + 0.05, 0.05).round(2)  # 19 thresholds
    for num, folder in enumerate(folders):
        ds_path = os.path.join(folder, "figures", "donor_scores_matrix.npz")
        ar_path = os.path.join(folder, "figures", "analysis_arrays.npz")
        ds = np.load(ds_path)
        ar = np.load(ar_path)
        binder_size = int(os.path.basename(folder).split("_")[0].replace("b", ""))
        label_map[int(num)] = int(binder_size)
        n_tcr = ar["analysis_probs"].shape[0]
        # explanation fractions: vectorised with broadcasting (N, D, 1) > (1, 1, 19)
        donor_scores = ds["donor_scores"]          # (N, D)
        total_donors = ds["total_donors_per_tcr"]   # (N,)
        # expand for broadcasting:  (N, D, 1) > (19,) → (N, D, 19) → sum over D → (N, 19)
        explanation_fractions = (
            (donor_scores[:, :, None] > thrs[None, None, :]).sum(axis=1)
            / np.maximum(total_donors[:, None], 1)
        ).astype(np.float32)  # (N, 19)
        donor_size = total_donors[:, np.newaxis].astype(np.float32)  # (N, 1)
        ent = compute_entropy(ar["analysis_probs"])[:, np.newaxis]    # (N, 1)
        x = np.concatenate([ent, donor_size, explanation_fractions], axis=-1)  # (N, 21)
        y = np.full((n_tcr, 1), num, dtype=np.int32)
        X_parts.append(x)
        Y_parts.append(y)
        log.info("  loaded %-12s  binder=%2d  N=%d", os.path.basename(folder), binder_size, n_tcr)
    X = np.concatenate(X_parts, axis=0)
    Y = np.concatenate(Y_parts, axis=0)
    log.info("Total synthetic samples: %d  |  Features: %d  |  Classes: %d",
             X.shape[0], X.shape[1], len(label_map))
    return X, Y, label_map
def split_data(X, Y, val_frac, seed):
    """Stratified random split into train / val / test (val_frac each for val and test).
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
    return (X[idx_train], Y[idx_train],
            X[idx_val],   Y[idx_val],
            X[idx_test],  Y[idx_test])
def make_tf_dataset(X, Y, batch_size, shuffle=True, seed=42):
    """Create a tf.data.Dataset from numpy arrays with optional shuffle.
    Args:
        X: feature matrix (N, F).
        Y: label matrix (N, 1) – will be squeezed.
    Returns:
        Batched, optionally shuffled, prefetched tf.data.Dataset.
    """
    ds = tf.data.Dataset.from_tensor_slices(
        (X.astype(np.float32), Y.ravel().astype(np.int32))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(X.shape[0], 500_000), seed=seed)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
# ============================================================================
#  Model definition
# ============================================================================
def build_model(input_dim: int, num_classes: int, num_neurons: int,
                num_layers: int, dropout_rate: float, reg_lambda: float,
                learning_rate: float, lr_schedule: str, epochs: int,
                train_steps: int) -> keras.Model:
    """Build a small feedforward classifier: Input → Dropout → (Dense → Dropout)*L → Dense(softmax).
    Uses L1+L2 regularisation on every Dense layer.  Returns compiled model.
    Args:
        input_dim    : number of input features (21).
        num_classes  : number of binder-set bins.
        num_neurons  : hidden-layer width.
        num_layers   : number of hidden Dense layers.
        dropout_rate : dropout fraction after input and each hidden layer.
        reg_lambda   : combined L1/L2 penalty weight.
        learning_rate: peak learning rate for Adam.
        lr_schedule  : 'constant' or 'cosine'.
        epochs       : total training epochs (for cosine schedule).
        train_steps  : steps per epoch (for cosine schedule).
    Returns:
        Compiled keras.Model whose last layer outputs logits (no softmax)
        so we can extract logits for temperature scaling.
    """
    reg = regularizers.L1L2(l1=reg_lambda, l2=reg_lambda)
    inp = keras.Input(shape=(input_dim,), name="features")
    x = layers.Dropout(dropout_rate, name="input_dropout")(inp)
    for i in range(num_layers):
        x = layers.Dense(num_neurons, activation="relu",
                         kernel_regularizer=reg,
                         name=f"hidden_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)
    # output logits (no activation) – needed for temperature scaling later
    logits = layers.Dense(num_classes, activation=None,
                          kernel_regularizer=reg, name="logits")(x)
    model = keras.Model(inputs=inp, outputs=logits, name="binder_classifier")
    # learning-rate schedule
    if lr_schedule == "cosine":
        total_steps = epochs * train_steps
        lr = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=total_steps,
            alpha=1e-6,
        )
    else:
        lr = learning_rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model
# ============================================================================
#  Temperature scaling
# ============================================================================
def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    lr: float = 0.01, max_iter: int = 500) -> float:
    """Fit a single scalar temperature T that minimises NLL on (logits, labels).
    Uses TensorFlow gradient tape – runs on GPU if available.
    Args:
        logits: (N, C) raw model outputs.
        labels: (N,) integer ground-truth class indices.
    Returns:
        Optimal temperature T* > 0.
    """
    log_T = tf.Variable(0.0, dtype=tf.float32)  # T = exp(log_T) keeps T > 0
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
    log.info("    temperature fitting done – T*=%.4f  NLL=%.4f  steps=%d", T_star, best_loss, step + 1)
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
    # numerically stable softmax
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    return (exp_s / exp_s.sum(axis=1, keepdims=True)).astype(np.float32)
# ============================================================================
#  Error-controlled thresholds
# ============================================================================
def find_threshold(values: np.ndarray, errors: np.ndarray, epsilon: float,
                   direction: str = "leq") -> float:
    """Find the tightest threshold on *values* such that the conditional error
    rate among accepted predictions is ≤ epsilon.
    Args:
        values   : (N,) uncertainty metric (entropy or margin).
        errors   : (N,) binary error indicators (1=wrong, 0=correct).
        epsilon  : maximum acceptable error rate.
        direction: 'leq' → accept if value ≤ threshold (entropy).
                   'geq' → accept if value ≥ threshold (margin).
    Returns:
        Threshold value (float).  If no threshold achieves the target,
        returns the most restrictive value that minimises accepted errors.
    """
    order = np.argsort(values)
    sorted_vals = values[order]
    sorted_errs = errors[order]
    n = len(values)
    if direction == "leq":
        # cumulative error rate as we include samples from low → high entropy
        cum_err = np.cumsum(sorted_errs)
        cum_n = np.arange(1, n + 1, dtype=np.float64)
        err_rate = cum_err / cum_n
        # find largest index where error rate ≤ epsilon
        valid = np.where(err_rate <= epsilon)[0]
        if len(valid) == 0:
            return float(sorted_vals[0])  # most restrictive
        return float(sorted_vals[valid[-1]])
    else:  # direction == "geq" → accept high margin
        # iterate from high to low margin
        cum_err = np.cumsum(sorted_errs[::-1])
        cum_n = np.arange(1, n + 1, dtype=np.float64)
        err_rate = cum_err / cum_n
        valid = np.where(err_rate <= epsilon)[0]
        if len(valid) == 0:
            return float(sorted_vals[-1])  # most restrictive
        idx = valid[-1]
        return float(sorted_vals[n - 1 - idx])
# ============================================================================
#  Evaluation helpers
# ============================================================================
def evaluate_and_plot(probs: np.ndarray, labels: np.ndarray, label_map: dict,
                      split_name: str, output_dir: str):
    """Compute multi-class AUC, PRAUC, accuracy and save a confusion-matrix plot.
    Args:
        probs      : (N, C) calibrated probabilities.
        labels     : (N,) integer labels.
        label_map  : {int: binder_size}.
        split_name : e.g. 'train', 'val', 'test'.
        output_dir : path for saving figures.
    """
    preds = probs.argmax(axis=1)
    acc = (preds == labels).mean()
    n_classes = probs.shape[1]
    # one-hot for sklearn metrics
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
    log.info("  [%s]  Acc=%.4f  AUC(OvR)=%.4f  PRAUC=%.4f", split_name, acc, auc_ovr, prauc)
    # confusion matrix
    cm = confusion_matrix(labels, preds)
    class_names = [str(label_map[i]) for i in range(n_classes)]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(n_classes), yticks=np.arange(n_classes),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True",
           title=f"Confusion matrix – {split_name}")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"confusion_{split_name}.png"), dpi=150)
    plt.close(fig)
    return {"accuracy": float(acc), "auc_ovr": float(auc_ovr), "prauc": float(prauc)}
# ============================================================================
#  TRAIN mode
# ============================================================================
def run_train(args):
    """Full training pipeline: data loading → model training → calibration → thresholds."""
    set_global_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    meta = {}
    # ── 1. Load data ────────────────────────────────────────────────────────
    log.info("Loading synthetic data from %s …", args.input_train)
    X, Y, label_map = load_synthetic_data(args.input_train)
    num_classes = len(label_map)
    meta["label_map"] = label_map
    meta["num_classes"] = num_classes
    meta["feature_dim"] = int(X.shape[1])
    # ── 2. Split ────────────────────────────────────────────────────────────
    log.info("Splitting data (val=%.2f, test=%.2f) …", args.val_split, args.val_split)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(
        X, Y, args.val_split, args.seed
    )
    log.info("  train=%d  val=%d  test=%d", len(Y_train), len(Y_val), len(Y_test))
    meta["split_sizes"] = {
        "train": int(len(Y_train)),
        "val": int(len(Y_val)),
        "test": int(len(Y_test)),
    }
    # ── 3. tf.data datasets ────────────────────────────────────────────────
    train_ds = make_tf_dataset(X_train, Y_train, args.batch_size, shuffle=True, seed=args.seed)
    val_ds = make_tf_dataset(X_val, Y_val, args.batch_size, shuffle=False)
    test_ds = make_tf_dataset(X_test, Y_test, args.batch_size, shuffle=False)
    train_steps = int(np.ceil(len(Y_train) / args.batch_size))
    # ── 4. Build or resume model ────────────────────────────────────────────
    best_model_path = os.path.join(checkpoint_dir, "model_best.keras")
    if args.resume and os.path.exists(best_model_path):
        log.info("Resuming from %s", best_model_path)
        model = keras.models.load_model(best_model_path)
    else:
        log.info("Building new model …")
        model = build_model(
            input_dim=X.shape[1],
            num_classes=num_classes,
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
    # ── 5. Callbacks ────────────────────────────────────────────────────────
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
    # ── 6. Train ────────────────────────────────────────────────────────────
    log.info("Training for up to %d epochs …", args.epochs)
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, callbacks=cb_list, verbose=2,
    )
    meta["history"] = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    # ensure best weights are loaded
    if os.path.exists(best_model_path):
        model = keras.models.load_model(best_model_path)
    # ── 7. Extract logits on each split ────────────────────────────────────
    log.info("Extracting logits on train / val / test …")
    logits_train = model.predict(X_train.astype(np.float32), batch_size=args.batch_size, verbose=0)
    logits_val = model.predict(X_val.astype(np.float32), batch_size=args.batch_size, verbose=0)
    logits_test = model.predict(X_test.astype(np.float32), batch_size=args.batch_size, verbose=0)
    # ── 8. K-fold cross-validated temperature scaling & thresholds ─────────
    log.info("K-fold calibration (K=%d) on val+test combined …", args.k_crossval)
    # combine val+test for cross-validated calibration (as described in the paper)
    logits_cal = np.concatenate([logits_val, logits_test], axis=0)
    labels_cal = np.concatenate([Y_val.ravel(), Y_test.ravel()], axis=0)
    n_cal = len(labels_cal)
    K = args.k_crossval
    rng = np.random.RandomState(args.seed + 1)
    perm = rng.permutation(n_cal)
    fold_ids = np.array_split(perm, K)
    fold_temps, fold_H0, fold_D0 = [], [], []
    for k in range(K):
        # held-out fold
        heldout_idx = fold_ids[k]
        train_idx = np.concatenate([fold_ids[j] for j in range(K) if j != k])
        # fit temperature on training folds
        T_k = fit_temperature(logits_cal[train_idx], labels_cal[train_idx])
        fold_temps.append(T_k)
        # calibrate held-out
        probs_k = calibrate_logits(logits_cal[heldout_idx], T_k)
        preds_k = probs_k.argmax(axis=1)
        errors_k = (preds_k != labels_cal[heldout_idx]).astype(np.float64)
        ent_k = compute_predictive_entropy(probs_k)
        margin_k = compute_margin(probs_k)
        # error-controlled thresholds on this fold
        H0_k = find_threshold(ent_k, errors_k, args.max_error_rate, direction="leq")
        D0_k = find_threshold(margin_k, errors_k, args.max_error_rate, direction="geq")
        fold_H0.append(H0_k)
        fold_D0.append(D0_k)
        acc_k = 1.0 - errors_k.mean()
        log.info("  fold %d/%d  T=%.4f  H0=%.4f  D0=%.4f  acc=%.4f",
                 k + 1, K, T_k, H0_k, D0_k, acc_k)
    # aggregate with median (robust)
    T_star = float(np.median(fold_temps))
    H0 = float(np.median(fold_H0))
    D0 = float(np.median(fold_D0))
    log.info("Aggregated  T*=%.4f  H0=%.4f  D0=%.4f", T_star, H0, D0)
    meta["calibration"] = {
        "fold_temperatures": [float(t) for t in fold_temps],
        "fold_H0": [float(h) for h in fold_H0],
        "fold_D0": [float(d) for d in fold_D0],
        "T_star": T_star,
        "H0": H0,
        "D0": D0,
        "max_error_rate": args.max_error_rate,
    }
    # ── 9. Evaluate calibrated probabilities on all splits ─────────────────
    log.info("Evaluating calibrated predictions …")
    perf = {}
    for name, logits_s, labels_s in [
        ("train", logits_train, Y_train.ravel()),
        ("val",   logits_val,   Y_val.ravel()),
        ("test",  logits_test,  Y_test.ravel()),
    ]:
        probs_s = calibrate_logits(logits_s, T_star)
        perf[name] = evaluate_and_plot(probs_s, labels_s, label_map, name, figures_dir)
    meta["performance"] = perf
    # ── 10. Plot training curves ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(meta["history"]["loss"], label="train")
    axes[0].plot(meta["history"]["val_loss"], label="val")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Categorical Cross-Entropy")
    axes[0].legend()
    axes[1].plot(meta["history"]["accuracy"], label="train")
    axes[1].plot(meta["history"]["val_accuracy"], label="val")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    # ── 11. Save metadata ──────────────────────────────────────────────────
    meta_path = os.path.join(args.output_dir, "metadata_train.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata saved to %s", meta_path)
    log.info("Training pipeline complete.")
# ============================================================================
#  INFERENCE mode
# ============================================================================
def run_inference(args):
    """Stream-inference on real TCR data: calibrate, compute uncertainty, and plot."""
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
    T_star = meta["calibration"]["T_star"]
    H0 = meta["calibration"]["H0"]
    D0 = meta["calibration"]["D0"]
    eps = meta["calibration"]["max_error_rate"]
    num_classes = meta["num_classes"]
    log.info("Loaded calibration: T*=%.4f  H0=%.4f  D0=%.4f  eps=%.4f", T_star, H0, D0, eps)
    # ── 2. Load model ──────────────────────────────────────────────────────
    model_path = os.path.join(args.output_dir, "checkpoint", "model_best.keras")
    assert os.path.exists(model_path), f"Model not found at {model_path}"
    model = keras.models.load_model(model_path)
    log.info("Model loaded from %s", model_path)
    # ── 3. Open H5 and determine sizes ────────────────────────────────────
    assert os.path.exists(args.input_inference), f"H5 not found: {args.input_inference}"
    h5_in = h5py.File(args.input_inference, "r")
    # validate required keys
    for key in ["entropy", "n_donors", "explanation_fractions"]:
        assert key in h5_in, f"Missing key '{key}' in H5"
    N_total = h5_in["entropy"].shape[0]
    log.info("Total TCRs for inference: %d", N_total)
    chunk = args.infer_chunk
    # ── 4. Chunked inference ───────────────────────────────────────────────
    # pre-allocate output arrays on disk via a temporary npz
    all_probs_raw = np.empty((N_total, num_classes), dtype=np.float32)
    all_probs_cal = np.empty((N_total, num_classes), dtype=np.float32)
    for start in range(0, N_total, chunk):
        end = min(start + chunk, N_total)
        log.info("  inference chunk [%d, %d) …", start, end)
        # assemble features: [entropy(N,1), n_donors(N,1), explanation_fractions(N,19)]
        ent_chunk = np.array(h5_in["entropy"][start:end], dtype=np.float32).reshape(-1, 1)
        nd_chunk = np.array(h5_in["n_donors"][start:end], dtype=np.float32).reshape(-1, 1)
        # explanation_fractions stored as (N, 21) with first and last columns excluded → 19 cols
        ef_chunk = np.array(h5_in["explanation_fractions"][start:end, 1:-1], dtype=np.float32)
        x_chunk = np.concatenate([ent_chunk, nd_chunk, ef_chunk], axis=-1)  # (chunk, 21)
        # raw logits → probabilities
        logits_chunk = model.predict(x_chunk, batch_size=args.batch_size, verbose=0)
        all_probs_raw[start:end] = calibrate_logits(logits_chunk, 1.0)  # softmax only
        all_probs_cal[start:end] = calibrate_logits(logits_chunk, T_star)
    h5_in.close()
    # ── 5. Derived quantities ──────────────────────────────────────────────
    pred_best = np.array([label_map[c] for c in all_probs_cal.argmax(axis=1)], dtype=np.int32)
    pred_entropy = compute_predictive_entropy(all_probs_cal)
    pred_margin = compute_margin(all_probs_cal)
    # reliable = passes BOTH entropy and margin thresholds
    reliable = (pred_entropy <= H0) & (pred_margin >= D0)
    log.info("Reliable predictions: %d / %d (%.2f%%)",
             reliable.sum(), N_total, 100.0 * reliable.mean())
    # ── 6. Write results to H5 ────────────────────────────────────────────
    log.info("Writing results to %s …", args.input_inference)
    h5_out = h5py.File(args.input_inference, "a")
    def _write(key, data):
        """Overwrite dataset if it already exists."""
        if key in h5_out:
            del h5_out[key]
        h5_out.create_dataset(key, data=data, compression="gzip", compression_opts=4)
    _write("predicted_binderset_probs", all_probs_raw)
    _write("predicted_binderset_probs_calibrated", all_probs_cal)
    _write("predicted_binderset_best_calibrated", pred_best)
    _write("predicted_binderset_entropy", pred_entropy)
    _write("predicted_binderset_margin", pred_margin)
    _write("predicted_binderset_reliable", reliable.astype(np.uint8))
    h5_out.flush()
    # also save raw probs as npz alongside the h5
    npz_path = os.path.splitext(args.input_inference)[0] + "_binderset_probs.npz"
    np.savez_compressed(npz_path,
                        probs_raw=all_probs_raw,
                        probs_calibrated=all_probs_cal,
                        best_calibrated=pred_best,
                        entropy=pred_entropy,
                        margin=pred_margin,
                        reliable=reliable)
    log.info("Probabilities also saved to %s", npz_path)
    # ── 7. Read donor counts for plotting ──────────────────────────────────
    h5_rd = h5py.File(args.input_inference, "r")
    n_donors_all = np.array(h5_rd["n_donors"][:], dtype=np.int32)
    h5_rd.close()
    h5_out.close()
    # ── 8. Bar plot: # reliable TCRs per discrete donor size ───────────────
    log.info("Generating bar plot …")
    unique_donors = np.unique(n_donors_all)
    counts_reliable = np.array(
        [reliable[n_donors_all == d].sum() for d in unique_donors], dtype=np.int64
    )
    fig, ax = plt.subplots(figsize=(max(10, len(unique_donors) * 0.15), 5))
    ax.bar(unique_donors.astype(str), counts_reliable, color="steelblue", edgecolor="none")
    ax.set_xlabel("Number of donors (n)")
    ax.set_ylabel(f"# TCRs passing error-rate ≤ {eps}")
    ax.set_title("Reliable predictions per donor count")
    # thin x-tick labels if too many
    if len(unique_donors) > 60:
        for idx, lbl in enumerate(ax.get_xticklabels()):
            if idx % 10 != 0:
                lbl.set_visible(False)
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "barplot_reliable_per_donor.png"), dpi=150)
    plt.close(fig)
    # ── 9. Heatmap: fraction of reliable TCRs (donor-bin × binder-set) ────
    log.info("Generating heatmap …")
    # bin donors into [10, 20, …, 500, 500+]
    bin_edges = np.arange(10, 510, 10)  # 10,20,...,500
    donor_bin_idx = np.digitize(n_donors_all, bin_edges)  # 0 = <10, 1 = 10-19, …
    n_bins = len(bin_edges)  # index 0..len(bin_edges) inclusive
    binder_classes = sorted(label_map.values())
    class_to_row = {b: i for i, b in enumerate(binder_classes)}
    # only reliable TCRs
    rel_mask = reliable
    heatmap = np.zeros((len(binder_classes), n_bins + 1), dtype=np.float64)
    total_reliable = rel_mask.sum()
    if total_reliable > 0:
        rel_donors_bin = donor_bin_idx[rel_mask]
        rel_binder = pred_best[rel_mask]
        for row_b, binder_val in enumerate(binder_classes):
            mask_b = rel_binder == binder_val
            for col_d in range(n_bins + 1):
                mask_d = rel_donors_bin == col_d
                heatmap[row_b, col_d] = (mask_b & mask_d).sum() / total_reliable
    # x-tick labels
    xtick_labels = [f"<10"] + [f"{e}" for e in bin_edges] 
    ytick_labels = [str(b) for b in binder_classes]
    fig, ax = plt.subplots(figsize=(max(14, (n_bins + 1) * 0.3), 4))
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(np.arange(heatmap.shape[1]))
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(binder_classes)))
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel("Donor count bin")
    ax.set_ylabel("Predicted binder set")
    ax.set_title(f"Fraction of reliable TCRs (ε ≤ {eps})")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Fraction of total reliable")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "heatmap_reliable_donor_binder.png"), dpi=150)
    plt.close(fig)
    log.info("Inference pipeline complete.")
# ============================================================================
#  Main
# ============================================================================
def main():
    parser = build_parser()
    args = parser.parse_args()
    log.info("Mode: %s", args.mode)
    log.info("Args: %s", vars(args))
    if args.mode == "train":
        run_train(args)
    elif args.mode == "inference":
        run_inference(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
if __name__ == "__main__":
    main()