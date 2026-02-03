#!/usr/bin/env python3
"""
Quantify linkage disequilibrium (LD) structure from a donor×allele presence matrix.

Input:
  - donor_hla_matrix.npz or donor_hla_matrix.npy : shape (N_donors, A_alleles), binary (0/1 or bool)
  - id_to_hla.json (optional) : mapping allele_id -> allele_name

Outputs (written into --out-dir):
  - ld_global_summary.json         : global LD metrics + gene-pair summaries
  - ld_allele_summary.json         : global per-allele LD stats (max posD r2)
  - ld_per_allele.tsv              : per-allele LD summary (freq, max r2, etc.)
  - ld_clusters.json               : LD blocks (connected components) + stats
  - ld_edges.tsv                   : edges (pairs) above --edge-r2-min (optionally top-N)
  - ld_gene_pair_summary.tsv       : gene×gene LD summaries

Metrics computed (2×2, per allele pair):
  - phi (Pearson correlation) and r^2 (=phi^2)
  - D and D' (Lewontin)
  - odds ratio (with pseudocount)
  - chi-square statistic
  - mutual information (MI) and normalized MI (NMI)
  - Jaccard index
  - conditional probabilities P(b|a), P(a|b)

Clustering:
  - Build an LD graph with edges where r^2 >= --cluster-r2-threshold
  - By default, only positive-association edges are used (D>0) to avoid
    connecting mutually exclusive alleles of the same gene.
  - Connected components are reported as LD clusters.

Usage example:
  python quantify_ld.py \
    --matrix export_train_dataset/donor_hla_matrix.npz \
    --id-to-hla export_train_dataset/id_to_hla.json \
    --out-dir ld_report \
    --cluster-r2-threshold 0.8 \
    --edge-r2-min 0.2
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Quantify LD in donor×allele matrix.")
    ap.add_argument(
        "--matrix",
        required=True,
        help="Path to donor_hla_matrix.npz or donor_hla_matrix.npy",
    )
    ap.add_argument(
        "--id-to-hla",
        default=None,
        help="Optional JSON mapping allele id -> name (e.g. export_train_dataset/id_to_hla.json)",
    )
    ap.add_argument("--out-dir", default="ld_report", help="Output directory.")
    ap.add_argument(
        "--edge-r2-min",
        type=float,
        default=0.20,
        help="Write pair edges with r^2 >= this threshold to ld_edges.tsv (default: 0.20).",
    )
    ap.add_argument(
        "--top-edges",
        type=int,
        default=200000,
        help="Max edges to write (sorted by r^2 desc). Default 200k; set 0 for all.",
    )
    ap.add_argument(
        "--cluster-r2-threshold",
        type=float,
        default=0.80,
        help="r^2 threshold to form LD clusters (default: 0.80).",
    )
    ap.add_argument(
        "--cluster-require-positive-d",
        action="store_true",
        default=True,
        help="Require D>0 for clustering edges (default: True).",
    )
    ap.add_argument(
        "--no-cluster-require-positive-d",
        dest="cluster_require_positive_d",
        action="store_false",
        help="Allow negative-D edges in clustering (usually not recommended).",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Small epsilon to avoid log/div-by-zero (default: 1e-12).",
    )
    ap.add_argument(
        "--or-pseudocount",
        type=float,
        default=0.5,
        help="Pseudocount added to 2x2 cells for odds ratio (default: 0.5).",
    )
    return ap.parse_args()


def load_id_to_hla(path: Optional[str], A: int) -> Dict[int, str]:
    if not path:
        return {i: f"allele_{i}" for i in range(A)}
    p = Path(path)
    with open(p, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    out: Dict[int, str] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            continue
    for i in range(A):
        out.setdefault(i, f"allele_{i}")
    return out


def parse_gene(allele_name: str) -> str:
    s = allele_name.strip()
    if s.startswith("HLA-"):
        s = s[4:]
    if "*" in s:
        return s.split("*", 1)[0]
    if ":" in s:
        return s.split(":", 1)[0]
    return s


@dataclass
class PairMetricArrays:
    n10: np.ndarray
    n01: np.ndarray
    n00: np.ndarray
    pa: np.ndarray
    pb: np.ndarray
    p11: np.ndarray
    D: np.ndarray
    Dprime: np.ndarray
    phi: np.ndarray
    r2: np.ndarray
    jaccard: np.ndarray
    p_b_given_a: np.ndarray
    p_a_given_b: np.ndarray
    oratio: np.ndarray
    chi2: np.ndarray
    mi: np.ndarray
    nmi: np.ndarray


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _entropy(p: np.ndarray, eps: float) -> np.ndarray:
    return -np.where(p > 0.0, p * np.log(p + eps), 0.0)


def compute_pair_metrics(
    N: int,
    Na: np.ndarray,
    Nb: np.ndarray,
    n11: np.ndarray,
    eps: float,
    or_pseudocount: float,
) -> PairMetricArrays:
    Na_f = Na.astype(np.float64)
    Nb_f = Nb.astype(np.float64)
    n11_f = n11.astype(np.float64)

    n10 = Na_f - n11_f
    n01 = Nb_f - n11_f
    n00 = float(N) - n11_f - n10 - n01

    pa = Na_f / float(N)
    pb = Nb_f / float(N)
    p11 = n11_f / float(N)

    D = p11 - pa * pb

    Dmax_pos = np.minimum(pa * (1.0 - pb), (1.0 - pa) * pb)
    Dmax_neg = np.minimum(pa * pb, (1.0 - pa) * (1.0 - pb))
    Dmax = np.where(D >= 0.0, Dmax_pos, Dmax_neg)
    Dprime = np.where(Dmax > eps, D / (Dmax + eps), 0.0)

    denom = np.sqrt(pa * (1.0 - pa) * pb * (1.0 - pb)) + eps
    phi = D / denom
    r2 = phi * phi

    jacc = n11_f / (Na_f + Nb_f - n11_f + eps)

    p_b_given_a = n11_f / (Na_f + eps)
    p_a_given_b = n11_f / (Nb_f + eps)

    a = n11_f + or_pseudocount
    b = n10 + or_pseudocount
    c = n01 + or_pseudocount
    d = n00 + or_pseudocount
    oratio = (a * d) / (b * c + eps)

    ad_bc = (n11_f * n00) - (n10 * n01)
    row1 = n11_f + n10
    row0 = n01 + n00
    col1 = n11_f + n01
    col0 = n10 + n00
    chi2 = (float(N) * (ad_bc ** 2)) / ((row1 * row0 * col1 * col0) + eps)

    p00 = n00 / float(N)
    p01 = n01 / float(N)
    p10 = n10 / float(N)

    p1a = pa
    p0a = 1.0 - pa
    p1b = pb
    p0b = 1.0 - pb

    def mi_term(pxy, px, py):
        return np.where(pxy > 0.0, pxy * np.log((pxy + eps) / (px * py + eps)), 0.0)

    mi = (
        mi_term(p11, p1a, p1b)
        + mi_term(p10, p1a, p0b)
        + mi_term(p01, p0a, p1b)
        + mi_term(p00, p0a, p0b)
    )

    Ha = _entropy(p1a, eps) + _entropy(p0a, eps)
    Hb = _entropy(p1b, eps) + _entropy(p0b, eps)
    nmi = np.where((Ha * Hb) > eps, mi / (np.sqrt(Ha * Hb) + eps), 0.0)

    return PairMetricArrays(
        n10=n10,
        n01=n01,
        n00=n00,
        pa=pa,
        pb=pb,
        p11=p11,
        D=D,
        Dprime=Dprime,
        phi=phi,
        r2=r2,
        jaccard=jacc,
        p_b_given_a=p_b_given_a,
        p_a_given_b=p_a_given_b,
        oratio=oratio,
        chi2=chi2,
        mi=mi,
        nmi=nmi,
    )


def quantiles(x: np.ndarray, qs=(0.25, 0.5, 0.9, 0.95, 0.99)) -> Dict[str, float]:
    if x.size == 0:
        return {f"q{int(q*100)}": float("nan") for q in qs}
    vals = np.quantile(x, qs)
    return {f"q{int(q*100)}": float(v) for q, v in zip(qs, vals)}


def summarize(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            **quantiles(x),
        }
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        **quantiles(x),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = Path(args.matrix)
    if matrix_path.suffix.lower() == ".npz":
        with np.load(matrix_path) as data:
            keys = list(data.keys())
            preferred = ["donor_hla_matrix", "donor_matrix", "matrix", "X", "data"]
            for key in preferred:
                if key in data:
                    X = data[key]
                    break
            else:
                if len(keys) == 1:
                    X = data[keys[0]]
                else:
                    raise KeyError(f"npz contains multiple arrays; available keys: {keys}")
    else:
        X = np.load(matrix_path, mmap_mode="r")
    if X.ndim != 2:
        raise SystemExit(f"Matrix must be 2D, got shape {X.shape}")
    N, A = X.shape

    Xb = (X != 0).astype(np.uint8, copy=False) if X.dtype != np.bool_ else X.astype(np.uint8, copy=False)

    id_to_hla = load_id_to_hla(args.id_to_hla, A)
    allele_names = [id_to_hla[i] for i in range(A)]
    genes = [parse_gene(name) for name in allele_names]

    row_sums = Xb.sum(axis=1).astype(np.int32)
    col_sums = Xb.sum(axis=0).astype(np.int32)

    cooc = (Xb.T @ Xb).astype(np.int32)

    iu, ju = np.triu_indices(A, k=1)
    n11 = cooc[iu, ju]
    Na = col_sums[iu]
    Nb = col_sums[ju]

    m = compute_pair_metrics(
        N=N, Na=Na, Nb=Nb, n11=n11, eps=args.eps, or_pseudocount=args.or_pseudocount
    )

    D = m.D
    r2 = m.r2
    phi = m.phi

    posD = D > 0
    negD = D < 0

    global_summary = {
        "matrix": str(Path(args.matrix).resolve()),
        "N_donors": int(N),
        "A_alleles": int(A),
        "row_sum_mean": float(row_sums.mean()),
        "row_sum_std": float(row_sums.std()),
        "row_sum_min": int(row_sums.min()),
        "row_sum_max": int(row_sums.max()),
        "allele_freq_min": float((col_sums / max(N, 1)).min()),
        "allele_freq_median": float(np.median(col_sums / max(N, 1))),
        "allele_freq_max": float((col_sums / max(N, 1)).max()),
        "pairs_total": int(r2.size),
        "pairs_posD": int(posD.sum()),
        "pairs_negD": int(negD.sum()),
        "r2_all": {"mean": float(r2.mean()), "std": float(r2.std()), **quantiles(r2)},
        "r2_posD": {
            "mean": float(r2[posD].mean()) if posD.any() else float("nan"),
            "std": float(r2[posD].std()) if posD.any() else float("nan"),
            **(quantiles(r2[posD]) if posD.any() else {k: float("nan") for k in quantiles(np.array([0.0])).keys()}),
        },
        "abs_phi_all": {
            "mean": float(np.abs(phi).mean()),
            "std": float(np.abs(phi).std()),
            **quantiles(np.abs(phi)),
        },
        "threshold_counts": {},
    }
    for thr in (0.2, 0.5, 0.8, 0.9, 0.95):
        global_summary["threshold_counts"][f"r2>={thr}"] = int((r2 >= thr).sum())
        global_summary["threshold_counts"][f"r2>={thr} & posD"] = int(((r2 >= thr) & posD).sum())

    # Gene-pair summaries (posD only)
    gene_pair_vals: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    gene_pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    for idx in range(r2.size):
        a = int(iu[idx]); b = int(ju[idx])
        ga = genes[a]; gb = genes[b]
        key = (ga, gb) if ga <= gb else (gb, ga)
        gene_pair_counts[key] += 1
        if posD[idx]:
            gene_pair_vals[key].append(float(r2[idx]))

    gene_pair_summary_rows = []
    for (ga, gb), cnt in sorted(gene_pair_counts.items(), key=lambda x: (-x[1], x[0])):
        vals = np.array(gene_pair_vals.get((ga, gb), []), dtype=np.float64)
        gene_pair_summary_rows.append({
            "gene_a": ga,
            "gene_b": gb,
            "pairs_total": int(cnt),
            "pairs_posD": int(vals.size),
            "r2_posD_mean": float(vals.mean()) if vals.size else float("nan"),
            "r2_posD_median": float(np.median(vals)) if vals.size else float("nan"),
            "r2_posD_q90": float(np.quantile(vals, 0.9)) if vals.size else float("nan"),
            "r2_posD_q99": float(np.quantile(vals, 0.99)) if vals.size else float("nan"),
        })

    # Per-allele LD summary
    max_r2_partner = np.full(A, -1.0, dtype=np.float64)
    max_partner_idx = np.full(A, -1, dtype=np.int32)
    max_phi_partner = np.full(A, 0.0, dtype=np.float64)

    thresholds = (0.2, 0.5, 0.8, 0.9)
    count_r2_ge = {thr: np.zeros(A, dtype=np.int32) for thr in thresholds}
    count_posD_r2_ge = {thr: np.zeros(A, dtype=np.int32) for thr in thresholds}

    for idx in range(r2.size):
        a = int(iu[idx]); b = int(ju[idx])
        r2v = float(r2[idx])
        is_pos = bool(posD[idx])

        if is_pos and r2v > max_r2_partner[a]:
            max_r2_partner[a] = r2v
            max_partner_idx[a] = b
            max_phi_partner[a] = float(phi[idx])
        if is_pos and r2v > max_r2_partner[b]:
            max_r2_partner[b] = r2v
            max_partner_idx[b] = a
            max_phi_partner[b] = float(phi[idx])

        for thr in thresholds:
            if r2v >= thr:
                count_r2_ge[thr][a] += 1
                count_r2_ge[thr][b] += 1
                if is_pos:
                    count_posD_r2_ge[thr][a] += 1
                    count_posD_r2_ge[thr][b] += 1

    # Global per-allele LD summary wrt max_posD_r2
    max_posD_r2 = np.where(max_r2_partner >= 0.0, max_r2_partner, np.nan)
    has_posD_partner = np.isfinite(max_posD_r2)
    posD_count = int(has_posD_partner.sum())
    posD_share = float(posD_count / A) if A else float("nan")
    max_posD_vals = max_posD_r2[has_posD_partner]
    allele_r2_thresholds = (0.0, 0.2, 0.5, 0.8, 0.9)
    allele_threshold_counts = {}
    for thr in allele_r2_thresholds:
        cnt = int((max_posD_r2 >= thr).sum()) if np.isfinite(max_posD_r2).any() else 0
        allele_threshold_counts[f"max_posD_r2>={thr}"] = {
            "count": cnt,
            "share": float(cnt / A) if A else float("nan"),
        }
    allele_summary = {
        "matrix": str(Path(args.matrix).resolve()),
        "N_donors": int(N),
        "A_alleles": int(A),
        "alleles_with_posD_partner": {"count": posD_count, "share": posD_share},
        "max_posD_r2_stats": summarize(max_posD_vals),
        "max_posD_r2_threshold_counts": allele_threshold_counts,
    }

    # Clustering by r^2
    uf = UnionFind(A)
    cluster_thr = float(args.cluster_r2_threshold)
    for idx in range(r2.size):
        if r2[idx] < cluster_thr:
            continue
        if args.cluster_require_positive_d and not posD[idx]:
            continue
        uf.union(int(iu[idx]), int(ju[idx]))

    clusters: Dict[int, List[int]] = defaultdict(list)
    for a in range(A):
        clusters[uf.find(a)].append(a)

    # Per-cluster stats: use only edges that satisfy clustering predicate
    deg = np.zeros(A, dtype=np.int32)
    internal_edges = defaultdict(int)
    internal_r2_vals = defaultdict(list)

    for idx in range(r2.size):
        if r2[idx] < cluster_thr:
            continue
        if args.cluster_require_positive_d and not posD[idx]:
            continue
        a = int(iu[idx]); b = int(ju[idx])
        root = uf.find(a)
        if root != uf.find(b):
            continue
        deg[a] += 1; deg[b] += 1
        internal_edges[root] += 1
        internal_r2_vals[root].append(float(r2[idx]))

    cluster_reports = []
    for root, members in sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        k = len(members)
        edges = internal_edges.get(root, 0)
        possible = k * (k - 1) // 2
        density = (edges / possible) if possible else 0.0
        vals = np.array(internal_r2_vals.get(root, []), dtype=np.float64)
        member_degs = deg[members]
        hub_deg = int(member_degs.max()) if k else 0
        hub_idx = int(members[int(member_degs.argmax())]) if k else -1
        cluster_reports.append({
            "cluster_id": int(root),
            "size": int(k),
            "alleles": [allele_names[m] for m in members],
            "genes": sorted({genes[m] for m in members}),
            "edges_r2_ge_threshold": int(edges),
            "edge_density": float(density),
            "r2_mean": float(vals.mean()) if vals.size else float("nan"),
            "r2_median": float(np.median(vals)) if vals.size else float("nan"),
            "r2_q90": float(np.quantile(vals, 0.9)) if vals.size else float("nan"),
            "r2_max": float(vals.max()) if vals.size else float("nan"),
            "hub_allele": allele_names[hub_idx] if hub_idx >= 0 else None,
            "hub_degree": int(hub_deg),
        })

    # Edges file (pairs above edge threshold)
    edge_min = float(args.edge_r2_min)
    keep = r2 >= edge_min
    keep_idx = np.where(keep)[0]
    if keep_idx.size:
        order = np.argsort(r2[keep_idx])[::-1]
        keep_idx = keep_idx[order]
        if args.top_edges and keep_idx.size > args.top_edges:
            keep_idx = keep_idx[: args.top_edges]

    # Write outputs
    with open(out_dir / "ld_global_summary.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "global": global_summary,
                "cluster_params": {
                    "cluster_r2_threshold": cluster_thr,
                    "cluster_require_positive_d": bool(args.cluster_require_positive_d),
                },
            },
            fh,
            indent=2,
        )

    with open(out_dir / "ld_allele_summary.json", "w", encoding="utf-8") as fh:
        json.dump(allele_summary, fh, indent=2)

    with open(out_dir / "ld_gene_pair_summary.tsv", "w", encoding="utf-8") as fh:
        header = ["gene_a","gene_b","pairs_total","pairs_posD","r2_posD_mean","r2_posD_median","r2_posD_q90","r2_posD_q99"]
        fh.write("\t".join(header) + "\n")
        for row in gene_pair_summary_rows:
            fh.write("\t".join(str(row[h]) for h in header) + "\n")

    # Map allele -> cluster info
    root_of = np.array([uf.find(a) for a in range(A)], dtype=np.int32)
    cluster_size = {root: len(members) for root, members in clusters.items()}

    with open(out_dir / "ld_per_allele.tsv", "w", encoding="utf-8") as fh:
        header = [
            "allele_id","allele","gene","freq","count",
            "max_posD_r2","max_posD_partner","max_posD_phi",
            "deg_r2_ge_0.2","deg_r2_ge_0.5","deg_r2_ge_0.8","deg_r2_ge_0.9",
            "deg_posD_r2_ge_0.2","deg_posD_r2_ge_0.5","deg_posD_r2_ge_0.8","deg_posD_r2_ge_0.9",
            "cluster_id","cluster_size",
        ]
        fh.write("\t".join(header) + "\n")
        for a in range(A):
            partner = int(max_partner_idx[a])
            fh.write("\t".join([
                str(a),
                allele_names[a],
                genes[a],
                str(float(col_sums[a] / max(N, 1))),
                str(int(col_sums[a])),
                str(float(max_r2_partner[a]) if max_r2_partner[a] >= 0 else float("nan")),
                (allele_names[partner] if partner >= 0 else ""),
                str(float(max_phi_partner[a]) if max_r2_partner[a] >= 0 else float("nan")),
                str(int(count_r2_ge[0.2][a])),
                str(int(count_r2_ge[0.5][a])),
                str(int(count_r2_ge[0.8][a])),
                str(int(count_r2_ge[0.9][a])),
                str(int(count_posD_r2_ge[0.2][a])),
                str(int(count_posD_r2_ge[0.5][a])),
                str(int(count_posD_r2_ge[0.8][a])),
                str(int(count_posD_r2_ge[0.9][a])),
                str(int(root_of[a])),
                str(int(cluster_size[int(root_of[a])])) ,
            ]) + "\n")

    with open(out_dir / "ld_clusters.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "cluster_params": {
                    "cluster_r2_threshold": cluster_thr,
                    "cluster_require_positive_d": bool(args.cluster_require_positive_d),
                },
                "num_clusters_total": int(len(cluster_reports)),
                "num_singletons": int(sum(1 for c in cluster_reports if c["size"] == 1)),
                "clusters": cluster_reports,
            },
            fh,
            indent=2,
        )

    with open(out_dir / "ld_edges.tsv", "w", encoding="utf-8") as fh:
        header = [
            "allele_a_id","allele_a","gene_a",
            "allele_b_id","allele_b","gene_b",
            "count_a","count_b","n11",
            "p_a","p_b","D","Dprime","phi","r2",
            "jaccard","p_b_given_a","p_a_given_b",
            "odds_ratio","chi2","mi","nmi","posD",
        ]
        fh.write("\t".join(header) + "\n")
        for idx in keep_idx:
            a = int(iu[idx]); b = int(ju[idx])
            fh.write("\t".join(map(str, [
                a, allele_names[a], genes[a],
                b, allele_names[b], genes[b],
                int(col_sums[a]), int(col_sums[b]), int(n11[idx]),
                float(m.pa[idx]), float(m.pb[idx]),
                float(D[idx]), float(m.Dprime[idx]),
                float(phi[idx]), float(r2[idx]),
                float(m.jaccard[idx]), float(m.p_b_given_a[idx]), float(m.p_a_given_b[idx]),
                float(m.oratio[idx]), float(m.chi2[idx]), float(m.mi[idx]), float(m.nmi[idx]),
                bool(posD[idx]),
            ])) + "\n")

    print("Wrote LD report to:", str(out_dir.resolve()))
    print("Files:")
    for fn in [
        "ld_global_summary.json",
        "ld_allele_summary.json",
        "ld_gene_pair_summary.tsv",
        "ld_per_allele.tsv",
        "ld_clusters.json",
        "ld_edges.tsv",
    ]:
        print("  -", fn)


if __name__ == "__main__":
    main()
