#!/usr/bin/env python3
"""
evaluate_genotype_predictions.py — Evaluate donor genotype predictions.
=========================================================================
Reads predictions.json produced by predict_donor_genotypes.py and compares
against ground truth HLA types from donor_hla_matrix.npz. Computes and
plots:
  - Per-donor per-gene accuracy (both/one/none of 2 alleles correct)
  - Per-allele AUROC and AUPRC across donors
  - Per-gene average AUROC
  - Confusion matrix across all (donor, allele) pairs
  - Score distributions stratified by true label
  - Exact match rate (all 18 class I/II alleles correct)
=========================================================================
USAGE:
    python evaluate_genotype_predictions.py \\
        --predictions_json /path/to/out/predictions.json \\
        --donor_matrix_path /path/to/donor_hla_matrix.npz \\
        --hla_to_id /path/to/hla_to_id.json \\
        --output_dir /path/to/eval_out
=========================================================================
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Evaluate donor genotype predictions.")
    p.add_argument("--predictions_json", required=True,
                   help="predictions.json from predict_donor_genotypes.py.")
    p.add_argument("--donor_matrix_path", required=True,
                   help="donor_hla_matrix.npz for ground truth.")
    p.add_argument("--hla_to_id", required=True, help="hla_to_id.json.")
    p.add_argument("--output_dir", required=True, help="Output directory.")
    p.add_argument("--topk_values", type=str, default="1,2,3,4,5",
                   help="Comma-separated K values for top-K precision/recall "
                        "(default: 1,2,3,4,5).")
    return p.parse_args()


def save_plot_data(output_dir, name, data):
    """Save the raw data backing a plot as sidecar JSON next to the PNG.
    Saves as <name>.json (without the .png extension).
    All numpy arrays are converted to lists.
    """
    def _clean(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {str(k): _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        return str(obj)
    base = Path(output_dir) / (name + ".json")
    with open(base, "w") as f:
        json.dump(_clean(data), f, indent=2)
    return base


def load_predictions(path):
    """Load predictions JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def build_gene_mapping(hla_to_id_path):
    """Parse HLA name map and group alleles by gene."""
    import re
    with open(hla_to_id_path, "r") as f:
        hla_to_id = json.load(f)
    A = max(hla_to_id.values()) + 1
    id_to_name = {int(v): k for k, v in hla_to_id.items()}
    id_to_gene = np.array([""] * A, dtype=object)
    gene_to_alleles = {}
    for name, idx in hla_to_id.items():
        m = re.match(r"HLA-([ABC]|D[PQR][AB]\d?)\*", name)
        if m:
            g = m.group(1)
            id_to_gene[idx] = g
            gene_to_alleles.setdefault(g, []).append(int(idx))
    return gene_to_alleles, id_to_gene, id_to_name


def evaluate_per_gene_accuracy(predictions, donor_hla, gene_to_alleles,
                                valid_mask):
    """For each (donor, gene), check how many of the 2 true alleles are in
    the top-2 predicted scores for that gene.
    Returns:
        per_donor_gene: dict[donor_id][gene] = {correct: 0/1/2, total: 2}
    """
    out = {}
    for donor_id_str, pred in predictions.items():
        donor_id = int(donor_id_str)
        if donor_id >= donor_hla.shape[0]:
            continue
        x = np.array(pred["x_scores"])
        true_vec = donor_hla[donor_id].astype(bool)
        gene_result = {}
        for gene, allele_ids in gene_to_alleles.items():
            ids = np.array(allele_ids)
            # only predictable (non-rare) alleles
            mask = valid_mask[ids] > 0.5
            ids_v = ids[mask]
            if len(ids_v) == 0:
                continue
            true_gene = true_vec[ids_v]
            true_count = int(true_gene.sum())
            if true_count == 0:
                continue
            # Top-2 by score in this gene
            scores = x[ids_v]
            top2 = np.argsort(-scores)[:2]
            top2_mask = np.zeros(len(ids_v), dtype=bool)
            top2_mask[top2] = True
            correct = int(np.logical_and(top2_mask, true_gene).sum())
            gene_result[gene] = {
                "correct": correct,
                "true_count": true_count,
                "max_correct": min(2, true_count),
            }
        out[donor_id] = gene_result
    return out


def evaluate_per_allele_auc(predictions, donor_hla, valid_mask):
    """Compute per-allele AUROC / AUPRC across donors.
    Returns:
        per_allele: dict[allele_id] = {auroc, auprc, n_pos, n_neg}
    """
    donor_ids = sorted(int(d) for d in predictions.keys())
    x_mat = np.array([
        predictions[str(d)]["x_scores"] for d in donor_ids
    ])  # (N, A)
    y_mat = donor_hla[donor_ids].astype(np.int8)  # (N, A)
    A = x_mat.shape[1]
    result = {}
    for a in range(A):
        if valid_mask[a] < 0.5:
            continue
        y = y_mat[:, a]
        n_pos = int(y.sum())
        n_neg = int((1 - y).sum())
        if n_pos == 0 or n_neg == 0:
            result[a] = {"auroc": None, "auprc": None,
                         "n_pos": n_pos, "n_neg": n_neg}
            continue
        scores = x_mat[:, a]
        try:
            auroc = float(roc_auc_score(y, scores))
            auprc = float(average_precision_score(y, scores))
        except Exception:
            auroc, auprc = None, None
        result[a] = {"auroc": auroc, "auprc": auprc,
                     "n_pos": n_pos, "n_neg": n_neg}
    return result


def plot_per_gene_accuracy(per_donor_gene, gene_list, output_dir):
    """Box plot of accuracy fractions per gene across donors."""
    accs = {g: [] for g in gene_list}
    for donor_id, gene_res in per_donor_gene.items():
        for g in gene_list:
            if g in gene_res:
                mc = gene_res[g]["max_correct"]
                if mc > 0:
                    accs[g].append(gene_res[g]["correct"] / mc)
    data = [accs[g] for g in gene_list if len(accs[g]) > 0]
    labels = [g for g in gene_list if len(accs[g]) > 0]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 6))
    ax.boxplot(data, labels=labels, patch_artist=True,
               boxprops=dict(facecolor="lightblue"))
    ax.set_ylabel("Accuracy (fraction of true alleles in top-2)")
    ax.set_xlabel("HLA gene")
    ax.set_title("Per-gene genotype accuracy across donors")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = Path(output_dir) / "per_gene_accuracy.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    # Sidecar data
    save_plot_data(output_dir, "per_gene_accuracy", {
        "gene_labels": labels,
        "accuracy_per_donor_per_gene": {
            labels[i]: data[i] for i in range(len(labels))
        },
        "note": "Each value = (correct top-2 picks) / min(2, true_count) "
                "for one donor in one gene.",
    })
    return out


def plot_per_allele_auc(per_allele, id_to_name, id_to_gene, output_dir):
    """Bar plot of per-allele AUROC, colored by gene."""
    items = [(a, r) for a, r in per_allele.items() if r.get("auroc") is not None]
    items.sort(key=lambda kv: (id_to_gene[kv[0]], -kv[1]["auroc"]))
    allele_ids = [kv[0] for kv in items]
    aurocs = [kv[1]["auroc"] for kv in items]
    auprcs = [kv[1]["auprc"] for kv in items]
    genes = [id_to_gene[a] for a in allele_ids]
    unique_genes = sorted(set(genes))
    color_map = {g: plt.cm.tab20(i / max(1, len(unique_genes)))
                 for i, g in enumerate(unique_genes)}
    colors = [color_map[g] for g in genes]
    fig, axes = plt.subplots(2, 1, figsize=(max(20, len(allele_ids) * 0.12), 10))
    x = np.arange(len(allele_ids))
    axes[0].bar(x, aurocs, color=colors, width=0.8)
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("Per-allele AUROC across donors")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x, auprcs, color=colors, width=0.8)
    axes[1].set_ylabel("AUPRC")
    axes[1].set_xlabel("HLA allele (grouped by gene)")
    axes[1].set_title("Per-allele AUPRC across donors")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=0.3)
    # Legend for genes
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=color_map[g], label=g) for g in unique_genes]
    axes[0].legend(handles=legend_elems, loc="lower right", fontsize=8,
                    ncol=len(unique_genes))
    fig.tight_layout()
    out = Path(output_dir) / "per_allele_auc.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    save_plot_data(output_dir, "per_allele_auc", {
        "allele_id": list(allele_ids),
        "allele_name": [id_to_name.get(a, f"idx_{a}") for a in allele_ids],
        "gene": genes,
        "auroc": aurocs,
        "auprc": auprcs,
        "n_pos": [per_allele[a].get("n_pos") for a in allele_ids],
        "n_neg": [per_allele[a].get("n_neg") for a in allele_ids],
    })
    return out


def plot_score_distributions(predictions, donor_hla, valid_mask, output_dir):
    """Histograms of x scores stratified by true label across all (donor, allele)."""
    donor_ids = sorted(int(d) for d in predictions.keys())
    x_mat = np.array([predictions[str(d)]["x_scores"] for d in donor_ids])
    y_mat = donor_hla[donor_ids]
    vmask = valid_mask > 0.5
    scores_pos = x_mat[:, vmask][y_mat[:, vmask] > 0.5]
    scores_neg = x_mat[:, vmask][y_mat[:, vmask] < 0.5]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(scores_neg, bins=50, alpha=0.6, label=f"true 0 (n={len(scores_neg)})",
            color="steelblue", density=True)
    ax.hist(scores_pos, bins=50, alpha=0.6, label=f"true 1 (n={len(scores_pos)})",
            color="red", density=True)
    ax.set_xlabel("Predicted x score")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.set_title("Score distributions by true label")
    ax.legend()
    fig.tight_layout()
    out = Path(output_dir) / "score_distributions.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    # Sidecar: save histogram bins, not raw millions of scores
    hist_neg, bin_edges = np.histogram(scores_neg, bins=50)
    hist_pos, _ = np.histogram(scores_pos, bins=bin_edges)
    save_plot_data(output_dir, "score_distributions", {
        "bin_edges": bin_edges.tolist(),
        "hist_true_0": hist_neg.tolist(),
        "hist_true_1": hist_pos.tolist(),
        "n_pos_total": int(len(scores_pos)),
        "n_neg_total": int(len(scores_neg)),
        "score_min": float(min(scores_pos.min() if len(scores_pos) else 0,
                                scores_neg.min() if len(scores_neg) else 0)),
        "score_max": float(max(scores_pos.max() if len(scores_pos) else 0,
                                scores_neg.max() if len(scores_neg) else 0)),
    })
    return out


def plot_per_donor_summary(per_donor_gene, gene_list, output_dir):
    """Heatmap of per-donor per-gene accuracy."""
    donor_ids = sorted(per_donor_gene.keys())
    mat = np.full((len(donor_ids), len(gene_list)), np.nan)
    for i, d in enumerate(donor_ids):
        for j, g in enumerate(gene_list):
            if g in per_donor_gene[d]:
                r = per_donor_gene[d][g]
                if r["max_correct"] > 0:
                    mat[i, j] = r["correct"] / r["max_correct"]
    fig, ax = plt.subplots(figsize=(max(8, len(gene_list) * 0.8),
                                     max(6, len(donor_ids) * 0.05)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(gene_list)))
    ax.set_xticklabels(gene_list, rotation=45, ha="right")
    ax.set_ylabel("Donor")
    ax.set_xlabel("Gene")
    ax.set_title("Per-donor per-gene accuracy")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = Path(output_dir) / "per_donor_per_gene_heatmap.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    # Sidecar: save matrix + axis labels. Keep NaN as null in JSON.
    mat_clean = np.where(np.isnan(mat), None, mat).tolist()
    save_plot_data(output_dir, "per_donor_per_gene_heatmap", {
        "donor_ids": [int(d) for d in donor_ids],
        "gene_labels": list(gene_list),
        "accuracy_matrix": mat_clean,
    })
    return out


def diagnose_allele_popularity(predictions, gene_to_alleles, valid_mask,
                                id_to_name):
    """Detect allele popularity bias: same alleles picked for many donors.
    For each gene, count how often each allele appears in the top-2 picks
    across donors. If a few alleles dominate, the optimizer is collapsing
    to "popular" alleles rather than using donor-specific signal.
    Returns:
        dict[gene] = {
            "top2_counts": {allele_name: count},
            "n_donors": int,
            "max_frac": float,  # fraction of donors where the most-picked
                                # allele was in top-2
            "entropy": float,    # entropy of top-2 distribution (nats)
        }
    """
    out = {}
    donor_ids = sorted(int(d) for d in predictions.keys())
    for gene, allele_ids in gene_to_alleles.items():
        ids = np.array(allele_ids, dtype=np.int64)
        ids_v = ids[valid_mask[ids] > 0.5]
        if len(ids_v) < 2:
            continue
        # Count how often each allele is picked in top-2 across donors
        counts = np.zeros(len(ids_v), dtype=np.int64)
        n = 0
        for d in donor_ids:
            scores = np.array(
                predictions[str(d)]["x_scores"], dtype=np.float32)[ids_v]
            top2 = np.argsort(-scores)[:2]
            counts[top2] += 1
            n += 1
        if n == 0:
            continue
        # Sort by frequency
        order = np.argsort(-counts)
        top2_counts = {
            id_to_name.get(int(ids_v[i]), f"idx_{int(ids_v[i])}"):
                int(counts[i])
            for i in order if counts[i] > 0
        }
        # Entropy of the pick distribution
        probs = counts / max(1, counts.sum())
        probs_nz = probs[probs > 0]
        ent = float(-np.sum(probs_nz * np.log(probs_nz)))
        max_ent = float(np.log(len(ids_v)))  # max possible entropy
        out[gene] = {
            "top2_counts": top2_counts,
            "n_donors": int(n),
            "max_frac_picked": float(counts.max() / n),
            "entropy": ent,
            "max_entropy": max_ent,
            "entropy_ratio": float(ent / max_ent) if max_ent > 0 else 0.0,
            "n_valid_alleles_in_gene": int(len(ids_v)),
        }
    return out


def diagnose_predictability_ceiling(donor_hla, predictions, gene_to_alleles,
                                      valid_mask):
    """Compute the max possible top-2 accuracy given rare-allele masking.
    For each donor and gene, check how many of the true alleles are
    actually predictable (non-rare). If a donor has 2 true alleles in gene
    A but only 1 is predictable, the max possible accuracy is 0.5.
    Returns:
        dict[gene] = {
            "mean_ceiling": float,  # average max-possible per-donor accuracy
            "n_donors_affected": int,  # donors whose ceiling < 1.0
        }
    """
    out = {}
    donor_ids = sorted(int(d) for d in predictions.keys())
    for gene, allele_ids in gene_to_alleles.items():
        ids = np.array(allele_ids, dtype=np.int64)
        predictable = valid_mask[ids] > 0.5
        ceilings = []
        n_limited = 0
        for d in donor_ids:
            if d >= donor_hla.shape[0]:
                continue
            true_vec = donor_hla[d, ids].astype(bool)
            true_count = int(true_vec.sum())
            if true_count == 0:
                continue
            # How many true alleles are predictable?
            true_pred = int(np.logical_and(true_vec, predictable).sum())
            max_corr = min(2, true_count)
            max_pred = min(2, true_pred)
            if max_corr > 0:
                ceiling = max_pred / max_corr
                ceilings.append(ceiling)
                if ceiling < 1.0:
                    n_limited += 1
        if ceilings:
            out[gene] = {
                "mean_ceiling": float(np.mean(ceilings)),
                "median_ceiling": float(np.median(ceilings)),
                "n_donors_affected": int(n_limited),
                "n_donors_total": int(len(ceilings)),
            }
    return out


def diagnose_score_gap(predictions, donor_hla, gene_to_alleles, valid_mask):
    """For wrong top-2 picks, compute the score gap between chosen and
    true allele. A small gap → close call (model was almost right).
    A big gap → model was confident in the wrong answer.
    Returns:
        dict[gene] = {
            "mean_gap_wrong": float,  # mean (chosen - true) gap when wrong
            "frac_close_calls": float,  # fraction of wrong picks with gap < 0.05
        }
    """
    out = {}
    donor_ids = sorted(int(d) for d in predictions.keys())
    for gene, allele_ids in gene_to_alleles.items():
        ids = np.array(allele_ids, dtype=np.int64)
        ids_v = ids[valid_mask[ids] > 0.5]
        if len(ids_v) < 2:
            continue
        gaps = []
        for d in donor_ids:
            if d >= donor_hla.shape[0]:
                continue
            true_vec_full = donor_hla[d].astype(bool)
            true_in_gene = np.where(
                np.logical_and(true_vec_full, valid_mask > 0.5))[0]
            # Filter to this gene
            true_in_gene = np.intersect1d(true_in_gene, ids_v)
            if len(true_in_gene) == 0:
                continue
            scores_full = np.array(
                predictions[str(d)]["x_scores"], dtype=np.float32)
            scores_g = scores_full[ids_v]
            top2_idx = np.argsort(-scores_g)[:2]
            top2_ids = ids_v[top2_idx]
            missed = np.setdiff1d(true_in_gene, top2_ids)
            if len(missed) > 0:
                # For each missed true allele, gap to the LOWEST top-2 score
                lowest_top2 = scores_full[top2_ids].min()
                for a in missed:
                    gap = float(lowest_top2 - scores_full[a])
                    gaps.append(gap)
        if gaps:
            gaps_arr = np.array(gaps)
            out[gene] = {
                "mean_gap": float(gaps_arr.mean()),
                "median_gap": float(np.median(gaps_arr)),
                "frac_close_calls": float((gaps_arr < 0.05).mean()),
                "n_wrong_picks": int(len(gaps)),
            }
    return out


def diagnose_accuracy_vs_evidence(predictions, per_donor_gene):
    """Correlate per-donor accuracy with n_matched_tcrs (amount of evidence).
    If correlation is strong, the problem is data (insufficient signal) not
    model. If weak, the problem is model (biased/collapsed optimization).
    Returns:
        {
            "pearson_corr": float,
            "spearman_corr": float,
            "bins": [
                {"n_tcrs_range": ..., "mean_acc": ..., "n_donors": ...}
                for 5 bins by n_matched_tcrs
            ]
        }
    """
    ns = []
    accs = []
    for d_str, pred in predictions.items():
        d = int(d_str)
        if d not in per_donor_gene:
            continue
        # Average accuracy across genes for this donor
        gr = per_donor_gene[d]
        donor_vals = [
            v["correct"] / v["max_correct"]
            for v in gr.values()
            if v["max_correct"] > 0
        ]
        if not donor_vals:
            continue
        ns.append(pred["n_matched_tcrs"])
        accs.append(float(np.mean(donor_vals)))
    if len(ns) < 2:
        return {}
    ns_arr = np.array(ns, dtype=np.float64)
    accs_arr = np.array(accs, dtype=np.float64)
    # Pearson
    mn = ns_arr.mean()
    ma = accs_arr.mean()
    num = ((ns_arr - mn) * (accs_arr - ma)).sum()
    den = np.sqrt(((ns_arr - mn) ** 2).sum() * ((accs_arr - ma) ** 2).sum())
    pearson = float(num / max(den, 1e-12))
    # Spearman: correlation of ranks
    rn = np.argsort(np.argsort(ns_arr))
    ra = np.argsort(np.argsort(accs_arr))
    mrn = rn.mean()
    mra = ra.mean()
    num_s = ((rn - mrn) * (ra - mra)).sum()
    den_s = np.sqrt(((rn - mrn) ** 2).sum() * ((ra - mra) ** 2).sum())
    spearman = float(num_s / max(den_s, 1e-12))
    # Bin donors by n_matched_tcrs quintile
    order = np.argsort(ns_arr)
    n_per_bin = max(1, len(order) // 5)
    bins = []
    for b in range(5):
        lo = b * n_per_bin
        hi = (b + 1) * n_per_bin if b < 4 else len(order)
        idx = order[lo:hi]
        if len(idx) == 0:
            continue
        bins.append({
            "n_tcrs_min": float(ns_arr[idx].min()),
            "n_tcrs_max": float(ns_arr[idx].max()),
            "mean_acc": float(accs_arr[idx].mean()),
            "n_donors": int(len(idx)),
        })
    return {
        "pearson_corr": pearson,
        "spearman_corr": spearman,
        "bins": bins,
    }


def plot_diagnostics(popularity, ceiling, gaps, evidence, gene_list,
                      output_dir):
    """Multi-panel diagnostic plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # Panel 1: allele popularity (entropy ratio per gene)
    ax = axes[0, 0]
    genes_p = [g for g in gene_list if g in popularity]
    entropy_ratios = [popularity[g]["entropy_ratio"] for g in genes_p]
    max_fracs = [popularity[g]["max_frac_picked"] for g in genes_p]
    x = np.arange(len(genes_p))
    w = 0.35
    ax.bar(x - w/2, entropy_ratios, w, label="entropy ratio (1 = uniform)",
           color="steelblue")
    ax.bar(x + w/2, max_fracs, w,
           label="max pick fraction (1 = same allele always)",
           color="salmon")
    ax.set_xticks(x)
    ax.set_xticklabels(genes_p, rotation=45)
    ax.set_ylabel("Value")
    ax.set_title("Allele popularity bias\n(low entropy + high max_frac = collapse)")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    # Panel 2: predictability ceiling per gene
    ax = axes[0, 1]
    genes_c = [g for g in gene_list if g in ceiling]
    mean_ceils = [ceiling[g]["mean_ceiling"] for g in genes_c]
    frac_limited = [
        ceiling[g]["n_donors_affected"] / max(1, ceiling[g]["n_donors_total"])
        for g in genes_c
    ]
    x = np.arange(len(genes_c))
    ax.bar(x - w/2, mean_ceils, w, label="mean max possible accuracy",
           color="seagreen")
    ax.bar(x + w/2, frac_limited, w, label="fraction of donors limited",
           color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(genes_c, rotation=45)
    ax.set_ylabel("Value")
    ax.set_title("Predictability ceiling\n(ceiling<1 = rare-allele masking limits donor)")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    # Panel 3: score gap for wrong picks per gene
    ax = axes[1, 0]
    genes_g = [g for g in gene_list if g in gaps]
    mean_gaps = [gaps[g]["mean_gap"] for g in genes_g]
    frac_close = [gaps[g]["frac_close_calls"] for g in genes_g]
    x = np.arange(len(genes_g))
    ax.bar(x - w/2, mean_gaps, w, label="mean score gap (chosen - true)",
           color="purple")
    ax.bar(x + w/2, frac_close, w, label="frac close calls (gap < 0.05)",
           color="gold")
    ax.set_xticks(x)
    ax.set_xticklabels(genes_g, rotation=45)
    ax.set_ylabel("Value")
    ax.set_title("Score gap for wrong picks\n(big gap = confidently wrong)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    # Panel 4: accuracy vs evidence (n_matched_tcrs)
    ax = axes[1, 1]
    if evidence and "bins" in evidence:
        bins = evidence["bins"]
        xs = [(b["n_tcrs_min"] + b["n_tcrs_max"]) / 2 for b in bins]
        ys = [b["mean_acc"] for b in bins]
        ns = [b["n_donors"] for b in bins]
        ax.plot(xs, ys, "o-", color="darkblue", markersize=10)
        for xi, yi, ni in zip(xs, ys, ns):
            ax.annotate(f"n={ni}", (xi, yi), textcoords="offset points",
                        xytext=(5, 5), fontsize=9)
        ax.set_xlabel("n_matched_tcrs (donor evidence)")
        ax.set_ylabel("Mean per-donor accuracy (across genes)")
        pr = evidence.get("pearson_corr", 0.0)
        sp = evidence.get("spearman_corr", 0.0)
        ax.set_title(f"Accuracy vs evidence\n"
                     f"Pearson={pr:.3f}, Spearman={sp:.3f}")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No evidence data", ha="center", va="center",
                transform=ax.transAxes)
    fig.suptitle("Genotype prediction diagnostics", fontsize=14, y=1.00)
    fig.tight_layout()
    out = Path(output_dir) / "diagnostics.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    save_plot_data(output_dir, "diagnostics", {
        "allele_popularity": popularity,
        "predictability_ceiling": ceiling,
        "score_gap_on_wrong_picks": gaps,
        "accuracy_vs_evidence": evidence,
    })
    return out


def plot_auc_vs_donors(per_allele, id_to_name, id_to_gene, output_dir):
    """Scatter: per-allele AUROC and AUPRC vs. n_donors_per_allele (log-x).
    Color by gene. Matches style of Fig 1D in paper.
    """
    items = [(a, r) for a, r in per_allele.items()
             if r.get("auroc") is not None]
    allele_ids = [kv[0] for kv in items]
    aurocs = np.array([kv[1]["auroc"] for kv in items])
    auprcs = np.array([kv[1]["auprc"] for kv in items])
    n_pos = np.array([kv[1]["n_pos"] for kv in items])
    genes = [id_to_gene[a] for a in allele_ids]
    unique_genes = sorted(set(genes))
    color_map = {g: plt.cm.tab10(i % 10) for i, g in enumerate(unique_genes)}
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for g in unique_genes:
        mask = np.array([gg == g for gg in genes])
        axes[0].scatter(n_pos[mask], aurocs[mask], color=color_map[g],
                        label=g, alpha=0.75, s=40, edgecolors="k",
                        linewidths=0.5)
        axes[1].scatter(n_pos[mask], auprcs[mask], color=color_map[g],
                        label=g, alpha=0.75, s=40, edgecolors="k",
                        linewidths=0.5)
    for ax, metric in zip(axes, ["AUROC", "AUPRC"]):
        ax.set_xscale("log")
        ax.set_xlabel("Number of donors carrying allele (log scale)")
        ax.set_ylabel(metric)
        ax.set_title(f"Per-allele {metric} vs allele prevalence")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5 if metric == "AUROC" else 0.0, color="gray",
                   linestyle="--", alpha=0.5)
        ax.grid(alpha=0.3)
        ax.legend(title="Gene", fontsize=9, loc="lower right", ncol=2)
    fig.tight_layout()
    out = Path(output_dir) / "auc_vs_donors.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    save_plot_data(output_dir, "auc_vs_donors", {
        "allele_id": [int(a) for a in allele_ids],
        "allele_name": [id_to_name.get(a, f"idx_{a}") for a in allele_ids],
        "gene": genes,
        "n_donors_positive": n_pos.tolist(),
        "auroc": aurocs.tolist(),
        "auprc": auprcs.tolist(),
    })
    return out


def plot_auc_vs_donors_per_gene(per_allele, id_to_gene, output_dir):
    """Scatter: per-gene mean AUROC/AUPRC vs mean allele prevalence in gene.
    Each dot = one gene, x = mean n_donors_per_allele in that gene, y = mean
    AUROC (or AUPRC) across alleles in that gene.
    """
    by_gene_auroc = {}
    by_gene_auprc = {}
    by_gene_n = {}
    for a, r in per_allele.items():
        if r.get("auroc") is None:
            continue
        g = id_to_gene[a]
        by_gene_auroc.setdefault(g, []).append(r["auroc"])
        by_gene_auprc.setdefault(g, []).append(r["auprc"])
        by_gene_n.setdefault(g, []).append(r["n_pos"])
    genes = sorted(by_gene_auroc.keys())
    mean_roc = [float(np.mean(by_gene_auroc[g])) for g in genes]
    mean_prc = [float(np.mean(by_gene_auprc[g])) for g in genes]
    mean_n = [float(np.mean(by_gene_n[g])) for g in genes]
    n_alleles = [len(by_gene_auroc[g]) for g in genes]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, ys, metric in zip(axes, [mean_roc, mean_prc], ["AUROC", "AUPRC"]):
        for i, g in enumerate(genes):
            ax.scatter(mean_n[i], ys[i], s=200, edgecolors="k",
                       linewidths=1.0)
            ax.annotate(f"{g} (n={n_alleles[i]})",
                        (mean_n[i], ys[i]),
                        textcoords="offset points", xytext=(8, 4),
                        fontsize=10)
        ax.set_xscale("log")
        ax.set_xlabel("Mean donors per allele in gene (log scale)")
        ax.set_ylabel(f"Mean {metric} across alleles in gene")
        ax.set_title(f"Per-gene mean {metric} vs prevalence")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(output_dir) / "auc_vs_donors_per_gene.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    save_plot_data(output_dir, "auc_vs_donors_per_gene", {
        "gene": genes,
        "n_alleles_in_gene": n_alleles,
        "mean_donors_per_allele": mean_n,
        "mean_auroc": mean_roc,
        "mean_auprc": mean_prc,
    })
    return out


def compute_topk_metrics(predictions, donor_hla, gene_to_alleles, valid_mask,
                           id_to_name, id_to_gene, k_values):
    """Compute per-donor top-K precision and recall within each gene.
    For each (donor, gene) and each K in k_values:
        top_k_ids = alleles in gene ranked by score, top K
        correct = |top_k_ids ∩ true_alleles_for_donor_in_gene|
        precision@K = correct / K_effective
        recall@K    = correct / min(K_effective, n_true_in_gene)
    NOTE on recall definition: the min(K, n_true) in the denominator means
    that at K=1, a donor with 2 true alleles who got 1 correct is credited
    with recall=1.0 (not 0.5). This matches the task: "did we find AT LEAST
    one of the donor's true alleles in the top-K?" when K < n_true.
    For K >= n_true (which, for HLA, means K >= 2 always), this formula is
    identical to the standard recall = correct / n_true.
    Also aggregates per-allele metrics:
        allele_recall_at_K[a]:  among donors who TRULY carry a,
            fraction of those donors for whom a was in their top-K for its gene.
        allele_precision_at_K[a]: among donors where a was in their top-K,
            fraction where a was truly present.
    Returns:
        per_donor: dict[donor_id][gene][K] = {precision, recall,
                                               n_correct, n_true, top_k_names}
        per_gene_summary: dict[gene][K] = {mean_precision, mean_recall,
                                           n_donors}
        per_allele_summary: dict[allele_id][K] = {allele_recall, allele_precision,
                                                  n_true, n_picked}
    """
    donor_ids = sorted(int(d) for d in predictions.keys())
    per_donor = {}
    per_allele_counts = {}
    per_gene_agg = {g: {K: {"p": [], "r": []} for K in k_values}
                    for g in gene_to_alleles}
    for donor_id in donor_ids:
        if donor_id >= donor_hla.shape[0]:
            continue
        pred = predictions[str(donor_id)]
        scores = np.array(pred["x_scores"], dtype=np.float32)
        true_vec = donor_hla[donor_id].astype(bool)
        per_donor[donor_id] = {}
        for gene, allele_ids in gene_to_alleles.items():
            ids = np.array(allele_ids, dtype=np.int64)
            ids_v = ids[valid_mask[ids] > 0.5]
            if len(ids_v) == 0:
                continue
            true_in_gene = set(int(a) for a in ids_v if true_vec[a])
            n_true = len(true_in_gene)
            if n_true == 0:
                continue
            gene_scores = scores[ids_v]
            order = np.argsort(-gene_scores)
            ranked_ids = ids_v[order]
            per_donor[donor_id][gene] = {}
            for K in k_values:
                K_eff = min(K, len(ranked_ids))
                top_ids = ranked_ids[:K_eff]
                picked_set = set(int(a) for a in top_ids)
                correct = len(picked_set & true_in_gene)
                precision = correct / K_eff
                # Recall capped at min(K, n_true): at K=1 with n_true=2,
                # getting 1 right = 1.0 (not 0.5).
                recall = correct / min(K_eff, n_true)
                per_donor[donor_id][gene][K] = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "n_correct": int(correct),
                    "n_true": int(n_true),
                    "K_effective": int(K_eff),
                    "top_k_names": [
                        id_to_name.get(int(a), f"idx_{int(a)}")
                        for a in top_ids
                    ],
                }
                per_gene_agg[gene][K]["p"].append(precision)
                per_gene_agg[gene][K]["r"].append(recall)
                for a in picked_set:
                    per_allele_counts.setdefault(
                        a, {K_: {"true_and_picked": 0, "picked": 0, "true": 0}
                            for K_ in k_values})
                    per_allele_counts[a][K]["picked"] += 1
                    if a in true_in_gene:
                        per_allele_counts[a][K]["true_and_picked"] += 1
                for a in true_in_gene:
                    per_allele_counts.setdefault(
                        a, {K_: {"true_and_picked": 0, "picked": 0, "true": 0}
                            for K_ in k_values})
                    per_allele_counts[a][K]["true"] += 1
    per_gene_summary = {}
    for g, kmap in per_gene_agg.items():
        per_gene_summary[g] = {}
        for K, lists in kmap.items():
            if lists["p"]:
                per_gene_summary[g][K] = {
                    "mean_precision": float(np.mean(lists["p"])),
                    "mean_recall": float(np.mean(lists["r"])),
                    "n_donors": len(lists["p"]),
                }
    per_allele_summary = {}
    for a, kmap in per_allele_counts.items():
        per_allele_summary[a] = {}
        for K, c in kmap.items():
            n_true = c["true"]
            n_picked = c["picked"]
            tap = c["true_and_picked"]
            per_allele_summary[a][K] = {
                "allele_recall": float(tap / n_true) if n_true > 0 else None,
                "allele_precision": (float(tap / n_picked)
                                     if n_picked > 0 else None),
                "n_true": int(n_true),
                "n_picked": int(n_picked),
                "n_true_and_picked": int(tap),
            }
    return per_donor, per_gene_summary, per_allele_summary


def plot_topk_per_gene(per_gene_summary, output_dir, k_values):
    """Grouped bar plot: precision@K and recall@K per gene across K values.
    Note: recall@K uses min(K, n_true) in denominator — at K=1 with a donor
    having 2 true alleles, getting 1 correct counts as recall=1.0.
    """
    genes = sorted(per_gene_summary.keys())
    K_list = sorted(k_values)
    n_genes = len(genes)
    n_K = len(K_list)
    prec_mat = np.zeros((n_genes, n_K))
    rec_mat = np.zeros((n_genes, n_K))
    for i, g in enumerate(genes):
        for j, K in enumerate(K_list):
            if K in per_gene_summary[g]:
                prec_mat[i, j] = per_gene_summary[g][K]["mean_precision"]
                rec_mat[i, j] = per_gene_summary[g][K]["mean_recall"]
    fig, axes = plt.subplots(2, 1, figsize=(max(10, n_genes * 1.2), 9))
    x = np.arange(n_genes)
    w = 0.8 / n_K
    for j, K in enumerate(K_list):
        offs = (j - (n_K - 1) / 2) * w
        axes[0].bar(x + offs, prec_mat[:, j], width=w, label=f"K={K}")
        axes[1].bar(x + offs, rec_mat[:, j], width=w, label=f"K={K}")
    for ax, title, ylab in zip(
            axes,
            ["Per-gene mean Precision@K across donors",
             "Per-gene mean Recall@K across donors  "
             "[correct / min(K, n_true)]"],
            ["Precision", "Recall"]):
        ax.set_xticks(x)
        ax.set_xticklabels(genes, rotation=45)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9, ncol=n_K)
    fig.tight_layout()
    out = Path(output_dir) / "topk_per_gene.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    save_plot_data(output_dir, "topk_per_gene", {
        "genes": genes,
        "K_values": K_list,
        "precision_gene_x_K": prec_mat.tolist(),
        "recall_gene_x_K": rec_mat.tolist(),
        "n_donors_gene_x_K": [
            [per_gene_summary[g].get(K, {}).get("n_donors", 0)
             for K in K_list]
            for g in genes
        ],
        "note": "recall@K = correct / min(K, n_true). For K=1 with n_true=2, "
                "picking 1 correct = recall 1.0.",
    })
    return out


def plot_topk_per_allele(per_allele_summary, id_to_name, id_to_gene,
                          output_dir, k_values):
    """Two bar plots: allele-recall@K (faceted by K) and allele-precision@K.
    Alleles grouped by gene for visual clarity.
    """
    K_list = sorted(k_values)
    # Sort alleles by (gene, name)
    alleles = sorted(per_allele_summary.keys(),
                     key=lambda a: (id_to_gene[a],
                                     id_to_name.get(a, f"idx_{a}")))
    names = [id_to_name.get(a, f"idx_{a}") for a in alleles]
    genes = [id_to_gene[a] for a in alleles]
    unique_genes = sorted(set(genes))
    color_map = {g: plt.cm.tab10(i % 10) for i, g in enumerate(unique_genes)}
    colors = [color_map[g] for g in genes]
    # Build (n_alleles, n_K) matrices
    recall_mat = np.full((len(alleles), len(K_list)), np.nan)
    precision_mat = np.full((len(alleles), len(K_list)), np.nan)
    for i, a in enumerate(alleles):
        for j, K in enumerate(K_list):
            v = per_allele_summary[a].get(K, {})
            if v.get("allele_recall") is not None:
                recall_mat[i, j] = v["allele_recall"]
            if v.get("allele_precision") is not None:
                precision_mat[i, j] = v["allele_precision"]
    n_al = len(alleles)
    fig_recall, axes = plt.subplots(len(K_list), 1,
                                     figsize=(max(20, n_al * 0.1),
                                              3 * len(K_list)))
    if len(K_list) == 1:
        axes = [axes]
    x = np.arange(n_al)
    for j, K in enumerate(K_list):
        axes[j].bar(x, recall_mat[:, j], color=colors, width=0.9)
        axes[j].set_ylabel(f"Recall@K={K}")
        axes[j].set_ylim(0, 1.05)
        axes[j].grid(axis="y", alpha=0.3)
        if j == 0:
            from matplotlib.patches import Patch
            legend_elems = [Patch(facecolor=color_map[g], label=g)
                            for g in unique_genes]
            axes[j].legend(handles=legend_elems, fontsize=8,
                           loc="lower right", ncol=len(unique_genes))
    axes[0].set_title("Allele recall@K (among donors who truly carry this "
                      "allele, fraction where allele is in top-K for gene)")
    axes[-1].set_xlabel("HLA allele (grouped by gene)")
    fig_recall.tight_layout()
    out_recall = Path(output_dir) / "topk_per_allele_recall.png"
    fig_recall.savefig(out_recall, dpi=200)
    plt.close(fig_recall)
    fig_prec, axes_p = plt.subplots(len(K_list), 1,
                                      figsize=(max(20, n_al * 0.1),
                                               3 * len(K_list)))
    if len(K_list) == 1:
        axes_p = [axes_p]
    for j, K in enumerate(K_list):
        axes_p[j].bar(x, precision_mat[:, j], color=colors, width=0.9)
        axes_p[j].set_ylabel(f"Precision@K={K}")
        axes_p[j].set_ylim(0, 1.05)
        axes_p[j].grid(axis="y", alpha=0.3)
    axes_p[0].set_title("Allele precision@K (among donors where allele was "
                        "in top-K, fraction where allele was truly present)")
    axes_p[-1].set_xlabel("HLA allele (grouped by gene)")
    fig_prec.tight_layout()
    out_prec = Path(output_dir) / "topk_per_allele_precision.png"
    fig_prec.savefig(out_prec, dpi=200)
    plt.close(fig_prec)
    save_plot_data(output_dir, "topk_per_allele", {
        "allele_id": [int(a) for a in alleles],
        "allele_name": names,
        "gene": genes,
        "K_values": K_list,
        "recall_allele_x_K": [
            [None if np.isnan(recall_mat[i, j]) else float(recall_mat[i, j])
             for j in range(len(K_list))]
            for i in range(n_al)
        ],
        "precision_allele_x_K": [
            [None if np.isnan(precision_mat[i, j])
             else float(precision_mat[i, j])
             for j in range(len(K_list))]
            for i in range(n_al)
        ],
        "n_true_allele_x_K": [
            [per_allele_summary[a].get(K, {}).get("n_true", 0)
             for K in K_list]
            for a in alleles
        ],
        "n_picked_allele_x_K": [
            [per_allele_summary[a].get(K, {}).get("n_picked", 0)
             for K in K_list]
            for a in alleles
        ],
    })
    return out_recall, out_prec


def main():
    """Run evaluation pipeline."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 60)
    print("Evaluate donor genotype predictions")
    print("=" * 60)
    print(f"  Predictions: {args.predictions_json}")
    print(f"  Donor matrix: {args.donor_matrix_path}")
    # ── load data ───────────────────────────────────────────────────
    results = load_predictions(args.predictions_json)
    predictions = results["predictions"]
    valid_mask = np.array(results["valid_allele_mask"], dtype=np.float32)
    gene_to_alleles, id_to_gene, id_to_name = build_gene_mapping(args.hla_to_id)
    donor_hla = np.load(args.donor_matrix_path)["donor_hla_matrix"]
    print(f"  Donors in predictions: {len(predictions)}")
    print(f"  Valid alleles:         {int(valid_mask.sum())}/{len(valid_mask)}")
    # ── per-gene accuracy ───────────────────────────────────────────
    print(f"\n  Computing per-gene accuracy...")
    per_donor_gene = evaluate_per_gene_accuracy(
        predictions, donor_hla, gene_to_alleles, valid_mask)
    # Aggregate
    gene_summary = {}
    for gene in gene_to_alleles.keys():
        vals = []
        for d, gr in per_donor_gene.items():
            if gene in gr and gr[gene]["max_correct"] > 0:
                vals.append(gr[gene]["correct"] / gr[gene]["max_correct"])
        if vals:
            gene_summary[gene] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "n_donors": len(vals),
            }
    print(f"\n  Per-gene accuracy summary:")
    for g in sorted(gene_summary.keys()):
        s = gene_summary[g]
        print(f"    {g:>6s}: mean={s['mean']:.3f} median={s['median']:.3f} "
              f"n={s['n_donors']}")
    # ── per-allele AUC ──────────────────────────────────────────────
    print(f"\n  Computing per-allele AUROC/AUPRC...")
    per_allele = evaluate_per_allele_auc(predictions, donor_hla, valid_mask)
    aurocs_valid = [r["auroc"] for r in per_allele.values()
                    if r.get("auroc") is not None]
    auprcs_valid = [r["auprc"] for r in per_allele.values()
                    if r.get("auprc") is not None]
    print(f"  Mean AUROC: {np.mean(aurocs_valid):.3f} "
          f"(median {np.median(aurocs_valid):.3f}, n={len(aurocs_valid)})")
    print(f"  Mean AUPRC: {np.mean(auprcs_valid):.3f} "
          f"(median {np.median(auprcs_valid):.3f})")
    # Per-gene mean AUROC
    gene_auc = {}
    for a, r in per_allele.items():
        if r.get("auroc") is None:
            continue
        g = id_to_gene[a]
        gene_auc.setdefault(g, []).append(r["auroc"])
    print(f"\n  Per-gene mean AUROC:")
    for g in sorted(gene_auc.keys()):
        print(f"    {g:>6s}: {np.mean(gene_auc[g]):.3f}  "
              f"(n_alleles={len(gene_auc[g])})")
    # ── plots ───────────────────────────────────────────────────────
    print(f"\n  Generating plots...")
    gene_list = sorted(gene_to_alleles.keys())
    p1 = plot_per_gene_accuracy(per_donor_gene, gene_list, args.output_dir)
    print(f"    {p1}")
    p2 = plot_per_allele_auc(per_allele, id_to_name, id_to_gene, args.output_dir)
    print(f"    {p2}")
    p3 = plot_score_distributions(predictions, donor_hla, valid_mask, args.output_dir)
    print(f"    {p3}")
    p4 = plot_per_donor_summary(per_donor_gene, gene_list, args.output_dir)
    print(f"    {p4}")
    # ── DIAGNOSTICS: why is accuracy low? ───────────────────────────
    print(f"\n  Running diagnostics...")
    popularity = diagnose_allele_popularity(
        predictions, gene_to_alleles, valid_mask, id_to_name)
    ceiling = diagnose_predictability_ceiling(
        donor_hla, predictions, gene_to_alleles, valid_mask)
    gaps = diagnose_score_gap(
        predictions, donor_hla, gene_to_alleles, valid_mask)
    evidence = diagnose_accuracy_vs_evidence(predictions, per_donor_gene)
    print(f"\n  [DIAG 1] Allele popularity bias (are same alleles picked "
          f"for everyone?):")
    print(f"    Gene | entropy_ratio | max_pick_frac | top 3 picked alleles")
    for g in sorted(popularity.keys()):
        p = popularity[g]
        top3 = list(p["top2_counts"].items())[:3]
        top3_str = ", ".join([f"{k} ({v})" for k, v in top3])
        print(f"    {g:>6s} | {p['entropy_ratio']:.3f}         | "
              f"{p['max_frac_picked']:.3f}         | {top3_str}")
    print(f"\n  [DIAG 2] Predictability ceiling (max possible accuracy given "
          f"rare-HLA masking):")
    for g in sorted(ceiling.keys()):
        c = ceiling[g]
        print(f"    {g:>6s}: mean_ceiling={c['mean_ceiling']:.3f} | "
              f"donors limited: {c['n_donors_affected']}/{c['n_donors_total']} "
              f"({100*c['n_donors_affected']/max(1, c['n_donors_total']):.1f}%)")
    print(f"\n  [DIAG 3] Score gap on wrong picks (how close were we?):")
    for g in sorted(gaps.keys()):
        gp = gaps[g]
        print(f"    {g:>6s}: mean_gap={gp['mean_gap']:.4f} | "
              f"close_calls (gap<0.05): {100*gp['frac_close_calls']:.1f}% | "
              f"n_wrong={gp['n_wrong_picks']}")
    print(f"\n  [DIAG 4] Accuracy vs evidence:")
    if evidence:
        print(f"    Pearson correlation:  {evidence['pearson_corr']:.3f}")
        print(f"    Spearman correlation: {evidence['spearman_corr']:.3f}")
        print(f"    Binned by n_matched_tcrs:")
        for b in evidence["bins"]:
            print(f"      [{int(b['n_tcrs_min']):>6d}-"
                  f"{int(b['n_tcrs_max']):>6d}]: "
                  f"mean_acc={b['mean_acc']:.3f} (n_donors={b['n_donors']})")
    p5 = plot_diagnostics(popularity, ceiling, gaps, evidence, gene_list,
                           args.output_dir)
    print(f"    {p5}")
    # ── NEW: AUC vs donors scatters (paper Fig 1D style) ────────────
    print(f"\n  Generating AUC-vs-donors scatter plots...")
    p6 = plot_auc_vs_donors(per_allele, id_to_name, id_to_gene,
                              args.output_dir)
    print(f"    {p6}")
    p7 = plot_auc_vs_donors_per_gene(per_allele, id_to_gene, args.output_dir)
    print(f"    {p7}")
    # ── NEW: Top-K precision/recall metrics ─────────────────────────
    k_values = [int(k.strip()) for k in args.topk_values.split(",")
                if k.strip()]
    print(f"\n  Computing top-K metrics for K = {k_values}...")
    per_donor_topk, per_gene_topk, per_allele_topk = compute_topk_metrics(
        predictions, donor_hla, gene_to_alleles, valid_mask,
        id_to_name, id_to_gene, k_values)
    print(f"  Per-gene top-K summary:")
    for g in sorted(per_gene_topk.keys()):
        print(f"    {g}:")
        for K in sorted(per_gene_topk[g].keys()):
            s = per_gene_topk[g][K]
            print(f"      K={K}: precision={s['mean_precision']:.3f}  "
                  f"recall={s['mean_recall']:.3f}  "
                  f"(n_donors={s['n_donors']})")
    # Save per-donor per-gene per-K JSON (separate file for detailed access)
    topk_per_donor_path = Path(args.output_dir) / "topk_per_donor_per_gene.json"
    with open(topk_per_donor_path, "w") as f:
        json.dump(
            {str(d): {g: {str(K): v for K, v in kmap.items()}
                      for g, kmap in gmap.items()}
             for d, gmap in per_donor_topk.items()},
            f, indent=2)
    print(f"  Saved: {topk_per_donor_path}")
    p8 = plot_topk_per_gene(per_gene_topk, args.output_dir, k_values)
    print(f"    {p8}")
    p9_r, p9_p = plot_topk_per_allele(per_allele_topk, id_to_name, id_to_gene,
                                         args.output_dir, k_values)
    print(f"    {p9_r}")
    print(f"    {p9_p}")
    # ── save full metrics JSON ──────────────────────────────────────
    full_report = {
        "predictions_file": str(args.predictions_json),
        "n_donors_evaluated": len(predictions),
        "gene_summary": gene_summary,
        "gene_auc_mean": {g: float(np.mean(v)) for g, v in gene_auc.items()},
        "overall_mean_auroc": float(np.mean(aurocs_valid)),
        "overall_median_auroc": float(np.median(aurocs_valid)),
        "overall_mean_auprc": float(np.mean(auprcs_valid)),
        "overall_median_auprc": float(np.median(auprcs_valid)),
        "per_allele": {
            int(a): {
                "name": id_to_name.get(a, f"idx_{a}"),
                "gene": id_to_gene[a],
                **r,
            }
            for a, r in per_allele.items()
        },
        "per_donor_per_gene": {
            int(d): {g: r for g, r in gene_res.items()}
            for d, gene_res in per_donor_gene.items()
        },
        "diagnostics": {
            "allele_popularity": popularity,
            "predictability_ceiling": ceiling,
            "score_gap_on_wrong_picks": gaps,
            "accuracy_vs_evidence": evidence,
        },
        "topk": {
            "k_values": k_values,
            "per_gene": {
                g: {str(K): v for K, v in kmap.items()}
                for g, kmap in per_gene_topk.items()
            },
            "per_allele": {
                int(a): {
                    "name": id_to_name.get(a, f"idx_{a}"),
                    "gene": id_to_gene[a],
                    **{str(K): v for K, v in kmap.items()},
                }
                for a, kmap in per_allele_topk.items()
            },
        },
    }
    out_path = Path(args.output_dir) / "evaluation_report.json"
    with open(out_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\n  Report saved: {out_path}")


if __name__ == "__main__":
    main()