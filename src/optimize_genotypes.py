#!/usr/bin/env python3
"""
optimize_genotypes.py — Batched MAP genotype optimization over all donors.
=========================================================================
Loads per-donor bundles from build_per_donor_data.py and runs batched MAP
optimization on GPU. Memory-safe design: ALL tensors are pre-allocated once
as tf.Variable with fixed shapes, then .assign()'d per batch. The graph is
traced exactly once. Manual Adam (no keras optimizer) avoids slot-variable
leaks.
Two parameterizations for x (donor genotype probabilities):
  1. --use_softmax (recommended): x_a = 2 * softmax(theta_a within gene).
     Sum per gene is exactly 2 by construction. No regularizer needed.
  2. without --use_softmax: x_a = sigmoid(theta_a) * valid_mask_a.
     Soft gene regularizer: lambda * max(0, sum_gene(x) - 2).

     python src/optimize_genotypes.py \
  --meta outputs/.../per_donor_data/meta.npz \
  --per_donor_dir outputs/.../per_donor_data/per_donor \
  --donor_matrix_path data/autotcr/donor_hla_matrix.npz \
  --hla_to_id data/autotcr/hla_to_id.json \
  --output outputs/.../predictions/predictions_softmax.json \
  --donor_batch_size 8 \
  --max_tcrs_per_donor 15000 \
  --n_iters 300 \
  --lr 0.1 \
  --use_softmax \
  --gpu --force
=========================================================================
"""
import os
import sys
import re
import json
import time
import argparse
import numpy as np
from pathlib import Path


def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Batched MAP genotype optimization.")
    p.add_argument("--meta", required=True)
    p.add_argument("--per_donor_dir", required=True)
    p.add_argument("--donor_matrix_path", required=True)
    p.add_argument("--hla_to_id", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rare_threshold", type=int, default=5)
    p.add_argument("--lambda_reg", type=float, default=10.0,
                   help="Soft reg strength (ignored with --use_softmax).")
    p.add_argument("--n_iters", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--donor_batch_size", type=int, default=8)
    p.add_argument("--max_tcrs_per_donor", type=int, default=15000)
    p.add_argument("--use_softmax", action="store_true",
                   help="Hard sum=2 per gene via softmax.")
    p.add_argument("--add_absent", action="store_true")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def maybe_setup_gpu(use_gpu):
    """Configure GPU memory growth."""
    if not use_gpu:
        return None
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            print("[GPU] No GPU, CPU fallback.")
            return None
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"[GPU] Using {len(gpus)} GPU(s)")
        return tf
    except Exception as e:
        print(f"[GPU] init failed ({e}), CPU fallback.")
        return None


def parse_hla_gene(name):
    """Extract gene symbol from HLA name."""
    m = re.match(r"HLA-([ABC]|D[PQR][AB]\d?)\*", name)
    return m.group(1) if m else None


def build_gene_mapping(hla_to_id_path):
    """Build gene -> allele list and id -> name mappings."""
    with open(hla_to_id_path, "r") as f:
        hla_to_id = json.load(f)
    A = max(hla_to_id.values()) + 1
    id_to_name = {int(v): k for k, v in hla_to_id.items()}
    id_to_gene = np.array([""] * A, dtype=object)
    gene_to_alleles = {}
    for name, idx in hla_to_id.items():
        g = parse_hla_gene(name)
        if g is None:
            continue
        id_to_gene[idx] = g
        gene_to_alleles.setdefault(g, []).append(int(idx))
    return gene_to_alleles, id_to_gene, id_to_name


def load_donor_bundle(per_donor_dir, donor_id, max_tcrs):
    """Load one donor bundle, capping TCRs by confidence if over limit."""
    fp = Path(per_donor_dir) / f"donor_{int(donor_id):06d}.npz"
    d = np.load(fp)
    tcr_pos = d["tcr_pos"]
    conf = d["confidence"]
    n_t = int(d["n_tcrs"])
    if n_t > max_tcrs:
        order = np.argsort(-conf)[:max_tcrs]
        tcr_pos = tcr_pos[order]
        conf = conf[order]
        n_t = max_tcrs
    return (tcr_pos.astype(np.int32, copy=False),
            conf.astype(np.float32, copy=False),
            n_t)


def build_step_fn(tf, state, consts, cfg):
    """Build the compiled @tf.function Adam step. Called ONCE per run.
    All tensors are captured from Variables in `state` (assigned per batch)
    and tf.constants in `consts` (static for whole run).
    Returns:
        step_fn: @tf.function that runs one Adam step and returns loss.
        reset_fn: zeros m, v, step_counter and re-inits theta for new batch.
    """
    theta_var = state["theta"]
    m_var = state["m"]
    v_var = state["v"]
    step_var = state["step_counter"]
    tcr_pos_var = state["tcr_pos"]
    conf_var = state["conf"]
    mask_var = state["mask"]
    gamma_pool_tf = consts["gamma_pool"]
    q_pool_tf = consts["q_pool"]
    gene_tf = consts["gene"]
    allele_to_gene_tf = consts["allele_to_gene"]
    valid_tf = consts["valid"]
    lr = float(cfg["lr"])
    beta1 = float(cfg["beta1"])
    beta2 = float(cfg["beta2"])
    eps_adam = float(cfg["eps_adam"])
    use_softmax = bool(cfg["use_softmax"])
    lambda_reg = float(cfg["lambda_reg"])
    add_absent = bool(cfg["add_absent"])
    def softmax_per_gene(theta):
        """x_a = softmax over alleles within each gene. Sums to 1 per gene.
        Rare alleles get exp = 0 via the valid mask → contribute nothing.
        Shape: (D, A) -> (D, A).
        """
        neg_inf = tf.constant(-1e9, dtype=tf.float32)
        masked_theta = tf.where(
            valid_tf[None, :] > 0.5, theta,
            tf.fill(tf.shape(theta), neg_inf))
        # Per-gene max for numerical stability.
        # Build (D, G, A) via broadcast, mask non-gene-members to -inf.
        D = tf.shape(masked_theta)[0]
        G = tf.shape(gene_tf)[0]
        A_dim = tf.shape(gene_tf)[1]
        theta_expanded = tf.broadcast_to(
            masked_theta[:, None, :], (D, G, A_dim))
        gene_bool = gene_tf[None, :, :] > 0.5
        theta_gene = tf.where(gene_bool, theta_expanded, neg_inf)
        per_gene_max = tf.reduce_max(theta_gene, axis=2)  # (D, G)
        max_per_allele = tf.gather(per_gene_max, allele_to_gene_tf, axis=1)
        shifted = masked_theta - max_per_allele
        exp_theta = tf.exp(shifted)
        per_gene_sum = tf.linalg.matmul(
            exp_theta, gene_tf, transpose_b=True)  # (D, G)
        sum_per_allele = tf.gather(
            per_gene_sum, allele_to_gene_tf, axis=1)
        return exp_theta / (sum_per_allele + 1e-12)
    @tf.function
    def step_fn():
        """One Adam step. Returns loss scalar."""
        with tf.GradientTape() as tape:
            # Gather per-batch gammas + q_i INSIDE the graph — buffers
            # are transient and freed after the step.
            gamma_batch = tf.gather(gamma_pool_tf, tcr_pos_var)
            q_batch = tf.gather(q_pool_tf, tcr_pos_var)
            if use_softmax:
                x = 2.0 * softmax_per_gene(theta_var)
            else:
                x = tf.sigmoid(theta_var) * valid_tf[None, :]
            eps_log = 1e-12
            one_minus = 1.0 - x[:, None, :] * gamma_batch
            one_minus = tf.maximum(one_minus, eps_log)
            log_1_minus_p = tf.reduce_sum(tf.math.log(one_minus), axis=2)
            p = 1.0 - tf.exp(log_1_minus_p)
            qp = tf.clip_by_value(q_batch * p, eps_log, 1.0 - eps_log)
            term_obs = tf.math.log(qp)
            if add_absent:
                ll = term_obs + tf.math.log(1.0 - qp)
            else:
                ll = term_obs
            ll = ll * mask_var * conf_var
            weighted_ll = tf.reduce_sum(ll, axis=1)
            if use_softmax:
                loss_per_donor = -weighted_ll
            else:
                gene_sums = tf.linalg.matmul(
                    x, gene_tf, transpose_b=True)
                reg = lambda_reg * tf.reduce_sum(
                    tf.maximum(0.0, gene_sums - 2.0), axis=1)
                loss_per_donor = -weighted_ll + reg
            loss = tf.reduce_mean(loss_per_donor)
        grads = tape.gradient(loss, theta_var)
        step_var.assign_add(1.0)
        t = step_var
        m_new = beta1 * m_var + (1.0 - beta1) * grads
        v_new = beta2 * v_var + (1.0 - beta2) * grads * grads
        m_hat = m_new / (1.0 - tf.pow(beta1, t))
        v_hat = v_new / (1.0 - tf.pow(beta2, t))
        theta_var.assign_sub(lr * m_hat / (tf.sqrt(v_hat) + eps_adam))
        m_var.assign(m_new)
        v_var.assign(v_new)
        return loss
    def reset_fn(init_theta_value):
        """Zero Adam state and re-init theta."""
        theta_var.assign(tf.fill(theta_var.shape, float(init_theta_value)))
        m_var.assign(tf.zeros_like(m_var))
        v_var.assign(tf.zeros_like(v_var))
        step_var.assign(0.0)
    return step_fn, reset_fn


def _sigmoid_numpy(x):
    """Numerically stable sigmoid for result extraction."""
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _softmax_per_gene_numpy(theta, gene_group_mat, allele_to_gene, valid_mask):
    """Numpy version for extraction (not hot path)."""
    D, A = theta.shape
    G = gene_group_mat.shape[0]
    neg_inf = -1e9
    masked = np.where(valid_mask[None, :] > 0.5, theta, neg_inf)
    per_gene_max = np.full((D, G), neg_inf, dtype=np.float32)
    for g in range(G):
        mask_g = gene_group_mat[g] > 0.5
        if mask_g.any():
            per_gene_max[:, g] = masked[:, mask_g].max(axis=1)
    max_per_allele = per_gene_max[:, allele_to_gene]
    shifted = masked - max_per_allele
    exp_theta = np.exp(shifted)
    per_gene_sum = exp_theta @ gene_group_mat.T
    sum_per_allele = per_gene_sum[:, allele_to_gene]
    return exp_theta / (sum_per_allele + 1e-12)


def run_batched_optimization(tf, meta, donor_ids, per_donor_dir, args,
                               gene_group_mat, valid_mask, allele_to_gene):
    """Run MAP optimization across all donors with memory-safe batching.
    Everything pre-allocated ONCE, .assign()'d per batch. No accumulation.
    """
    gamma_used = meta["gamma_used"]
    q_used = meta["q_used"]
    A = int(meta["n_alleles"])
    D_max = args.donor_batch_size
    T_max = args.max_tcrs_per_donor
    print(f"\n  Pre-allocating GPU state:")
    print(f"    shape invariants: D={D_max}, T={T_max}, A={A}")
    # Static constants (never change)
    consts = {
        "gamma_pool": tf.constant(gamma_used, dtype=tf.float32),
        "q_pool": tf.constant(q_used, dtype=tf.float32),
        "gene": tf.constant(gene_group_mat, dtype=tf.float32),
        "allele_to_gene": tf.constant(allele_to_gene, dtype=tf.int32),
        "valid": tf.constant(valid_mask, dtype=tf.float32),
    }
    pool_gb = (gamma_used.nbytes + q_used.nbytes) / 1e9
    print(f"    shared pool: gamma_pool={gamma_used.shape} "
          f"+ q_pool={q_used.shape} = {pool_gb:.2f} GB")
    # State variables (pre-allocated once, .assign per batch)
    state = {
        "theta": tf.Variable(tf.zeros((D_max, A), dtype=tf.float32)),
        "m": tf.Variable(tf.zeros((D_max, A), dtype=tf.float32)),
        "v": tf.Variable(tf.zeros((D_max, A), dtype=tf.float32)),
        "step_counter": tf.Variable(0.0, dtype=tf.float32),
        "tcr_pos": tf.Variable(tf.zeros((D_max, T_max), dtype=tf.int32)),
        "conf": tf.Variable(tf.zeros((D_max, T_max), dtype=tf.float32)),
        "mask": tf.Variable(tf.zeros((D_max, T_max), dtype=tf.float32)),
    }
    state_gb = (
        D_max * A * 4 * 3 +             # theta, m, v
        D_max * T_max * 4 * 2 +         # conf, mask
        D_max * T_max * 4                # tcr_pos (int32)
    ) / 1e9
    print(f"    per-batch state variables: {state_gb:.2f} GB")
    print(f"    batch gamma intermediate (transient): "
          f"{D_max * T_max * A * 4 / 1e9:.2f} GB")
    # Build compiled step ONCE
    cfg = {
        "use_softmax": args.use_softmax,
        "lambda_reg": args.lambda_reg,
        "add_absent": args.add_absent,
        "lr": args.lr,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps_adam": 1e-8,
    }
    step_fn, reset_fn = build_step_fn(tf, state, consts, cfg)
    # Warm up graph (trace once)
    init_theta = 0.0 if args.use_softmax else -3.0
    reset_fn(init_theta_value=init_theta)
    _ = step_fn()
    reset_fn(init_theta_value=init_theta)
    print(f"  Graph traced. Starting optimization loop...")
    # Main loop
    predictions = {}
    n_donors = len(donor_ids)
    # Host-side scratch buffers (reused every batch)
    tcr_pos_buf = np.zeros((D_max, T_max), dtype=np.int32)
    conf_buf = np.zeros((D_max, T_max), dtype=np.float32)
    mask_buf = np.zeros((D_max, T_max), dtype=np.float32)
    t_opt = time.time()
    last_log = t_opt
    for batch_start in range(0, n_donors, D_max):
        batch_end = min(batch_start + D_max, n_donors)
        batch_donor_ids = donor_ids[batch_start:batch_end]
        n_real = len(batch_donor_ids)
        tcr_pos_buf.fill(0)
        conf_buf.fill(0.0)
        mask_buf.fill(0.0)
        real_n_tcrs = []
        real_mean_conf = []
        for i, d in enumerate(batch_donor_ids):
            tpos, cf, n_t = load_donor_bundle(
                per_donor_dir, d, args.max_tcrs_per_donor)
            tcr_pos_buf[i, :n_t] = tpos
            conf_buf[i, :n_t] = cf
            mask_buf[i, :n_t] = 1.0
            real_n_tcrs.append(n_t)
            real_mean_conf.append(float(cf.mean()) if n_t > 0 else 0.0)
        # Push to GPU via .assign (reuses same GPU memory — no alloc)
        state["tcr_pos"].assign(tcr_pos_buf)
        state["conf"].assign(conf_buf)
        state["mask"].assign(mask_buf)
        # Reset theta + Adam state (no new tensors allocated)
        reset_fn(init_theta_value=init_theta)
        # Run n_iters steps (graph already traced)
        for _ in range(args.n_iters):
            step_fn()
        # Extract final x for real donors only
        theta_np = state["theta"].numpy()
        if args.use_softmax:
            x_np = _softmax_per_gene_numpy(
                theta_np, gene_group_mat, allele_to_gene, valid_mask)
            x_np = 2.0 * x_np
        else:
            x_np = _sigmoid_numpy(theta_np) * valid_mask[None, :]
        for i, d in enumerate(batch_donor_ids):
            predictions[int(d)] = {
                "n_matched_tcrs": int(real_n_tcrs[i]),
                "mean_confidence": float(real_mean_conf[i]),
                "x_scores": x_np[i].tolist(),
            }
        # Progress log (every 5 seconds)
        now = time.time()
        if now - last_log > 5.0 or batch_end == n_donors:
            elapsed = now - t_opt
            rate = batch_end / elapsed if elapsed > 0 else 0
            eta = (n_donors - batch_end) / rate if rate > 0 else 0
            print(f"    [{batch_end}/{n_donors}] | {rate:.1f} donors/s | "
                  f"ETA {eta/60:.1f}min")
            last_log = now
    return predictions


def enumerate_topk_pairs(x_scores, gene_to_alleles, id_to_name, topk,
                          top_candidates=4):
    """Enumerate top-K candidate allele pairs per gene."""
    out = {}
    for gene, allele_ids in gene_to_alleles.items():
        ids = np.array(allele_ids, dtype=np.int64)
        scores = x_scores[ids]
        order = np.argsort(-scores)[:top_candidates]
        top_ids = ids[order]
        top_scores = scores[order]
        pairs = []
        for i in range(len(top_ids)):
            for j in range(i, len(top_ids)):
                a, b = int(top_ids[i]), int(top_ids[j])
                pairs.append({
                    "allele_a": id_to_name.get(a, f"idx_{a}"),
                    "allele_b": id_to_name.get(b, f"idx_{b}"),
                    "allele_a_id": a,
                    "allele_b_id": b,
                    "score_a": float(top_scores[i]),
                    "score_b": float(top_scores[j]),
                    "pair_score": float(top_scores[i] * top_scores[j]),
                })
        pairs.sort(key=lambda x: -x["pair_score"])
        out[gene] = pairs[:topk]
    return out


def main():
    """Run the optimization pipeline."""
    args = parse_args()
    if Path(args.output).exists() and not args.force:
        print(f"[SKIP] {args.output} exists (use --force)")
        return
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Batched MAP genotype optimization")
    print("=" * 60)
    print(f"  Meta:             {args.meta}")
    print(f"  Per-donor dir:    {args.per_donor_dir}")
    print(f"  Output:           {args.output}")
    print(f"  Parameterization: "
          f"{'softmax (hard sum=2 per gene)' if args.use_softmax else 'sigmoid + lambda reg'}")
    print(f"  donor_batch_size: {args.donor_batch_size}")
    print(f"  max_tcrs/donor:   {args.max_tcrs_per_donor}")
    print(f"  n_iters:          {args.n_iters}")
    print(f"  lr:               {args.lr}")
    if not args.use_softmax:
        print(f"  lambda_reg:       {args.lambda_reg}")
    tf = maybe_setup_gpu(args.gpu)
    if tf is None:
        print("ERROR: TF required.")
        sys.exit(1)
    # Load meta
    meta = np.load(args.meta)
    donor_ids = meta["donor_ids"].tolist()
    A = int(meta["n_alleles"])
    print(f"\n  n_donors:   {len(donor_ids)}")
    print(f"  n_alleles:  {A}")
    print(f"  gamma_used: {meta['gamma_used'].shape} "
          f"({meta['gamma_used'].nbytes/1e9:.2f} GB)")
    # Gene mapping
    gene_to_alleles, id_to_gene, id_to_name = build_gene_mapping(args.hla_to_id)
    genes = sorted(gene_to_alleles.keys())
    G = len(genes)
    print(f"  Genes:      {genes}")
    gene_group_mat = np.zeros((G, A), dtype=np.float32)
    for gi, g in enumerate(genes):
        for a in gene_to_alleles[g]:
            if a < A:
                gene_group_mat[gi, a] = 1.0
    # allele_to_gene: gene index per allele (0 for non-assigned, zeroed via mask)
    no_gene = gene_group_mat.sum(axis=0) == 0
    allele_to_gene = np.zeros(A, dtype=np.int32)
    for gi in range(G):
        in_g = gene_group_mat[gi] > 0.5
        allele_to_gene[in_g] = gi
    # Rare HLA mask
    donor_hla = np.load(args.donor_matrix_path)["donor_hla_matrix"]
    n_donors_per_allele = donor_hla.sum(axis=0)
    rare_mask = n_donors_per_allele < args.rare_threshold
    valid_mask = (~rare_mask).astype(np.float32) * (~no_gene).astype(np.float32)
    print(f"  Rare alleles masked: {int(rare_mask.sum())}/{A}")
    print(f"  Final valid alleles: {int(valid_mask.sum())}/{A}")
    predictions_raw = run_batched_optimization(
        tf, meta, donor_ids, args.per_donor_dir, args,
        gene_group_mat, valid_mask, allele_to_gene)
    print(f"\n  Enumerating top-K pairs per donor...")
    predictions = {}
    for d, pr in predictions_raw.items():
        x = np.array(pr["x_scores"], dtype=np.float32)
        topk = enumerate_topk_pairs(
            x, gene_to_alleles, id_to_name, args.topk)
        predictions[d] = {
            "n_matched_tcrs": pr["n_matched_tcrs"],
            "mean_confidence": pr["mean_confidence"],
            "x_scores": pr["x_scores"],
            "topk_pairs_per_gene": topk,
        }
    results = {
        "config": {
            "meta": str(args.meta),
            "rare_threshold": args.rare_threshold,
            "use_softmax": args.use_softmax,
            "lambda_reg": args.lambda_reg,
            "n_iters": args.n_iters,
            "lr": args.lr,
            "add_absent": args.add_absent,
            "donor_batch_size": args.donor_batch_size,
            "max_tcrs_per_donor": args.max_tcrs_per_donor,
        },
        "gene_list": genes,
        "num_alleles": A,
        "valid_allele_mask": valid_mask.tolist(),
        "rare_allele_count": int(rare_mask.sum()),
        "predictions": predictions,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {args.output}")


if __name__ == "__main__":
    main()