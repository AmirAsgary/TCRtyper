#!/usr/bin/env python3
# build_public_tcrs_from_clusters.py
#
# Input:  cluster_members_full_by_rep.tsv (no header), columns (NEW):
#   0 rep_id
#   1 seq_id
#   2 dataset
#   3 sample_file
#   4 cdr3aa
#   5 cdr2aa_gapped
#   6 cdr1aa_gapped
#   7 cdr2.5aa_gapped
#   8 v_b
#   9 j_b
#
# Output:
#   public_tcrs.json       { "<4-loop-key>": ["dataset/file.tsv", ...], ... }
#   public_tcrs_meta.json  { "<4-loop-key>": {"v_b": <int|str|"" >, "j_genes": [...]}, ... }
#   public_tcrs_stats.json summary stats

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build public_tcrs.json from cluster_members_full_by_rep.tsv."
    )
    ap.add_argument("--base", required=True, help="Base dir containing the processed root.")
    ap.add_argument(
        "--processed-root",
        default="export_train_dataset",
        help="Name of processed output folder under base (default: export_train_dataset).",
    )
    ap.add_argument(
        "--cluster-full",
        default="mmseqs/cluster_members_full_by_rep.tsv",
        help="Sorted-by-rep cluster+metadata TSV (relative to processed root).",
    )
    ap.add_argument(
        "--out",
        default="public_tcrs.json",
        help="Output JSON filename (within processed root).",
    )
    ap.add_argument(
        "--min-patients",
        type=int,
        default=2,
        help="Minimum donors for a TCR to be considered public (default: 2).",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return ap.parse_args()


def norm(s):
    if s is None:
        return ""
    x = str(s).strip()
    return "" if x.lower() in {"", "nan", "none", "na", "n/a"} else x


def _coerce_int_if_possible(x: str):
    x = (x or "").strip()
    if x == "":
        return ""
    try:
        return int(x)
    except Exception:
        return x


def main():
    args = parse_args()
    base = Path(args.base).resolve()
    processed_root = base / args.processed_root
    if not processed_root.is_dir():
        raise SystemExit(f"Processed root not found: {processed_root}")

    cluster_path = processed_root / args.cluster_full
    if not cluster_path.is_file():
        raise SystemExit(f"Cluster-full TSV not found: {cluster_path}")

    out_json = processed_root / args.out
    meta_json = processed_root / "public_tcrs_meta.json"
    stats_json = processed_root / "public_tcrs_stats.json"

    out_fh = out_json.open("w", encoding="utf-8")
    meta_fh = meta_json.open("w", encoding="utf-8")

    out_fh.write("{\n")
    meta_fh.write("{\n")
    first_entry = True

    rows_total = 0
    clusters_total = 0
    public_tcrs = 0
    v_b_conflicts = 0
    v_b_conflict_examples = []

    current_rep = None
    current_key = None
    donors = set()
    j_genes = set()
    v_b_counts = Counter()  # v_b values per cluster

    def flush_cluster():
        nonlocal first_entry, public_tcrs, v_b_conflicts, v_b_conflict_examples
        if current_rep is None:
            return

        donor_count = len(donors)
        if donor_count < args.min_patients:
            return

        # v_b is expected to be a cluster property; pick dominant if conflicting
        v_nonempty = [v for v in v_b_counts.keys() if v != ""]
        v_b_val = ""
        if len(v_nonempty) >= 1:
            if len(v_nonempty) > 1:
                v_b_conflicts += 1
                if len(v_b_conflict_examples) < 5:
                    v_b_conflict_examples.append(
                        {
                            "rep_id": current_rep,
                            "v_b_values": sorted(v_nonempty),
                        }
                    )
            v_b_choice = max(
                v_nonempty,
                key=lambda v: (v_b_counts.get(v, 0), str(v)),
            )
            v_b_val = _coerce_int_if_possible(v_b_choice)

        if not first_entry:
            out_fh.write(",\n")
            meta_fh.write(",\n")
        else:
            first_entry = False

        donors_sorted = sorted(donors)
        json.dump(current_key, out_fh)
        out_fh.write(": ")
        json.dump(donors_sorted, out_fh)

        meta_fh.write(json.dumps(current_key))
        meta_fh.write(": ")
        json.dump({"v_b": v_b_val, "j_genes": sorted(j_genes)}, meta_fh)

        public_tcrs += 1

    with cluster_path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.reader(fin, delimiter="\t")
        iterator = tqdm(reader, desc="Reducing clusters", unit="rows", disable=args.no_progress)

        for row in iterator:
            if not row:
                continue

            # NEW strict schema: require v_b and j_b
            if len(row) < 10:
                raise SystemExit(
                    f"Expected >=10 columns (including v_b and j_b). "
                    f"Got {len(row)} columns. First bad row: {row!r}"
                )

            rep_id = row[0]
            dataset = row[2]
            sample_file = row[3]
            cdr3aa = norm(row[4])
            cdr2 = norm(row[5])
            cdr1 = norm(row[6])
            cdr25 = norm(row[7])
            v_b = norm(row[8])
            j_b = norm(row[9])

            rows_total += 1

            if current_rep is None:
                current_rep = rep_id
                current_key = f"{cdr3aa},{cdr2},{cdr1},{cdr25}"
                donors = {f"{dataset}/{sample_file}"}
                j_genes = {j_b} if j_b else set()
                v_b_counts = Counter()
                if v_b:
                    v_b_counts[v_b] += 1
                clusters_total = 1
                continue

            if rep_id == current_rep:
                donors.add(f"{dataset}/{sample_file}")
                if j_b:
                    j_genes.add(j_b)
                if v_b:
                    v_b_counts[v_b] += 1
            else:
                flush_cluster()

                clusters_total += 1
                current_rep = rep_id
                current_key = f"{cdr3aa},{cdr2},{cdr1},{cdr25}"
                donors = {f"{dataset}/{sample_file}"}
                j_genes = {j_b} if j_b else set()
                v_b_counts = Counter()
                if v_b:
                    v_b_counts[v_b] += 1

        # flush last cluster
        if current_rep is not None:
            flush_cluster()

    out_fh.write("\n}\n")
    meta_fh.write("\n}\n")
    out_fh.close()
    meta_fh.close()

    stats = {
        "base": str(base),
        "processed_root": str(processed_root),
        "rows_total": rows_total,
        "clusters_total": clusters_total,
        "public_tcrs": public_tcrs,
        "min_patients": args.min_patients,
        "v_b_conflicts": v_b_conflicts,
        "v_b_conflict_examples": v_b_conflict_examples,
        "output_file": str(out_json),
        "metadata_file": str(meta_json),
        "cluster_full_file": str(cluster_path),
        "key_fields": ["cdr3aa", "cdr2aa_gapped", "cdr1aa_gapped", "cdr2.5aa_gapped"],
        "v_metadata_field": "v_b",
        "j_metadata_field": "j_b",
    }
    with stats_json.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(f"\nPublic TCRs: {public_tcrs} of {clusters_total} clusters (rows={rows_total})")
    print(f"Wrote: {out_json}")
    print(f"Wrote metadata: {meta_json}")
    print(f"Wrote stats: {stats_json}")


if __name__ == "__main__":
    main()
