#!/usr/bin/env python3
"""
Build public TCR HLA counts dataset from a sorted cluster_members_full_by_rep.tsv.

Inputs:
  1) export_root (export_train_dataset) with:
       - patients_index.tsv
       - id_to_hla.json (for num_alleles)
       - donor_hla_matrix.npz (preferred) or donor_hla_matrix.npy + donor_hla_matrix_donors.json
  2) mmseqs/cluster_members_full_by_rep.tsv (no header), columns:
       0 rep_id
       1 seq_id
       2 dataset
       3 sample_file
       4 cdr3aa
       5 cdr2aa_gapped
       6 cdr1aa_gapped
       7 cdr2.5aa_gapped
       8 v_b
       9 j_b

Outputs (HDF5 via h5py):
  - datasets/loops/*: vlen ASCII strings for loop sequences
  - datasets/n_donors: donors per cluster
  - datasets/cluster_id: cluster index (0-based) for each entry
  - datasets/n_identical_sequences: count of identical loop sequences in the cluster
  - datasets/v_gene_ids (optional)
  - datasets/y_counts/{indptr,indices,data}: CSR sparse counts matrix
  - public_tcrs.json / public_tcrs_meta.json / public_tcrs_stats.json (optional)

Notes:
  - Uses chunked, compressed datasets and sequential appends.
  - Public JSONs are written by default (cheap and useful).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tcrtyper.dataset_processing.utils import PublicTcrHlaCsrWriter


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build public TCR HLA counts HDF5 from cluster_members_full_by_rep.tsv."
        )
    )
    ap.add_argument(
        "--export-root",
        required=True,
        help="export_train_dataset root with patients_index.tsv and donor HLA matrix.",
    )
    ap.add_argument(
        "--cluster-full-by-rep",
        default=None,
        help=(
            "Path to cluster_members_full_by_rep.tsv. "
            "Default: <export_root>/mmseqs/cluster_members_full_by_rep.tsv"
        ),
    )
    ap.add_argument(
        "--public-tcrs-json",
        default=None,
        help="Output public_tcrs.json path (default: <export_root>/public_tcrs.json).",
    )
    ap.add_argument(
        "--public-tcrs-meta-json",
        default=None,
        help="Output public_tcrs_meta.json path (default: <export_root>/public_tcrs_meta.json).",
    )
    ap.add_argument(
        "--public-tcrs-stats-json",
        default=None,
        help="Output public_tcrs_stats.json path (default: <export_root>/public_tcrs_stats.json).",
    )
    ap.add_argument(
        "--no-public-json",
        action="store_true",
        help="Disable writing public_tcrs.json and metadata.",
    )
    ap.add_argument(
        "--out-h5",
        default=None,
        help="Output HDF5 path (recommended).",
    )
    ap.add_argument(
        "--out-npz",
        default=None,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--min-donors",
        type=int,
        default=1,
        help="Minimum unique donors per cluster to keep (default: 1).",
    )
    ap.add_argument(
        "--dtype",
        choices=["uint16", "uint32"],
        default="uint16",
        help="Integer dtype for counts and n_donors (default: uint16).",
    )
    ap.add_argument(
        "--no-v-genes",
        action="store_true",
        help="Disable v_gene_ids output.",
    )
    ap.add_argument(
        "--require-v-genes",
        action="store_true",
        help="Drop clusters with missing v_b if set.",
    )
    ap.add_argument(
        "--donor-hla-matrix",
        default=None,
        help=(
            "Donor HLA matrix (.npz or .npy). "
            "Default: <export_root>/donor_hla_matrix.npz."
        ),
    )
    ap.add_argument(
        "--donor-hla-donors",
        default=None,
        help=(
            "Donor keys JSON (optional if .npz contains donor_keys). "
            "Default: <export_root>/donor_hla_matrix_donors.json."
        ),
    )
    ap.add_argument(
        "--compression",
        choices=["gzip", "lzf", "none"],
        default="gzip",
        help="Compression for datasets (default: gzip).",
    )
    ap.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="Gzip compression level (default: 4).",
    )
    ap.add_argument(
        "--chunk-rows",
        type=int,
        default=4096,
        help="Chunk size for row-wise datasets (default: 4096).",
    )
    ap.add_argument(
        "--chunk-nnz",
        type=int,
        default=1_000_000,
        help="Chunk size for CSR indices/data (default: 1,000,000).",
    )
    ap.add_argument(
        "--flush-rows",
        type=int,
        default=10_000,
        help="Flush buffer every N clusters (default: 10000).",
    )
    ap.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm progress bars.",
    )
    return ap.parse_args()


def _compression_kwargs(args: argparse.Namespace) -> dict:
    if args.compression == "none":
        return {}
    if args.compression == "lzf":
        return {"compression": "lzf"}
    return {"compression": "gzip", "compression_opts": int(args.compression_level)}


def load_id_to_hla(export_root: Path) -> dict[int, str]:
    p = export_root / "id_to_hla.json"
    if not p.exists():
        alt = export_root / "hla_id_to_name.json"
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(f"id_to_hla.json not found under {export_root}")
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    out: dict[int, str] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            continue
    if not out:
        raise SystemExit(f"{p} is empty or has no valid integer keys.")
    return out


def _load_donor_keys_json(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"donor_hla_matrix_donors.json not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        donor_keys = json.load(fh)
    if not isinstance(donor_keys, list) or not donor_keys:
        raise SystemExit("donor_hla_matrix_donors.json must be a non-empty list.")
    return [str(k) for k in donor_keys]


def _load_matrix_from_npz(data, npz_key: str | None = None) -> np.ndarray:
    keys = list(data.keys())
    if npz_key:
        if npz_key not in data:
            raise KeyError(f"npz key {npz_key!r} not found. Available keys: {keys}")
        return data[npz_key]
    if len(keys) == 1:
        return data[keys[0]]
    preferred = ["donor_hla_matrix", "donor_matrix", "matrix", "X", "data"]
    for key in preferred:
        if key in data:
            return data[key]
    raise KeyError(f"npz contains multiple arrays; available keys: {keys}")


def _load_donor_keys_from_npz(data) -> list[str] | None:
    for key in ["donor_keys", "donors", "donor_hla_matrix_donors"]:
        if key in data:
            vals = data[key]
            return [str(v) for v in vals.tolist()]
    return None


def load_donor_hla_matrix(matrix_path: Path, donors_path: Path | None):
    if not matrix_path.exists():
        raise FileNotFoundError(f"donor_hla_matrix not found: {matrix_path}")

    if matrix_path.suffix.lower() == ".npz":
        with np.load(matrix_path) as data:
            matrix = _load_matrix_from_npz(data)
            donor_keys = _load_donor_keys_from_npz(data)
        if donor_keys is None and donors_path is not None:
            donor_keys = _load_donor_keys_json(donors_path)
        if donor_keys is None:
            raise SystemExit("donor keys not found in .npz and no JSON provided.")
    else:
        if donors_path is None:
            raise SystemExit("donor_hla_matrix_donors.json is required for .npy matrices.")
        donor_keys = _load_donor_keys_json(donors_path)
        matrix = np.load(matrix_path, mmap_mode="r")

    if matrix.ndim != 2:
        raise SystemExit(f"donor_hla_matrix has ndim={matrix.ndim}, expected 2.")
    if matrix.shape[0] != len(donor_keys):
        raise SystemExit(
            "donor_hla_matrix rows do not match donors list length: "
            f"{matrix.shape[0]} vs {len(donor_keys)}"
        )

    donor_index = {k: i for i, k in enumerate(donor_keys)}
    return matrix, donor_index


def norm(val: str | None) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() in {"", "nan", "none", "na", "n/a"} else s


def coerce_int(val: str) -> int | None:
    if val == "":
        return None
    try:
        return int(val)
    except Exception:
        return None


def open_json_writers(out_json: Path, meta_json: Path):
    out_fh = out_json.open("w", encoding="utf-8")
    meta_fh = meta_json.open("w", encoding="utf-8")
    out_fh.write("{\n")
    meta_fh.write("{\n")
    return out_fh, meta_fh


def close_json_writers(out_fh, meta_fh):
    out_fh.write("\n}\n")
    meta_fh.write("\n}\n")
    out_fh.close()
    meta_fh.close()


def main() -> None:
    args = parse_args()

    export_root = Path(args.export_root).resolve()
    out_path = args.out_h5 or args.out_npz
    if not out_path:
        raise SystemExit("Provide --out-h5 (or legacy --out-npz).")
    out_h5 = Path(out_path).resolve()
    if args.out_npz and not args.out_h5:
        print("Warning: --out-npz is deprecated; writing HDF5 output.")

    cluster_path = (
        Path(args.cluster_full_by_rep).resolve()
        if args.cluster_full_by_rep
        else (export_root / "mmseqs" / "cluster_members_full_by_rep.tsv")
    )

    if not export_root.exists():
        raise SystemExit(f"export_root not found: {export_root}")
    if not cluster_path.exists():
        raise SystemExit(f"cluster_members_full_by_rep.tsv not found: {cluster_path}")

    public_json = (
        Path(args.public_tcrs_json).resolve()
        if args.public_tcrs_json
        else (export_root / "public_tcrs.json")
    )
    meta_json = (
        Path(args.public_tcrs_meta_json).resolve()
        if args.public_tcrs_meta_json
        else (export_root / "public_tcrs_meta.json")
    )
    stats_json = (
        Path(args.public_tcrs_stats_json).resolve()
        if args.public_tcrs_stats_json
        else (export_root / "public_tcrs_stats.json")
    )

    id_to_hla = load_id_to_hla(export_root)
    num_alleles = len(id_to_hla)
    counts_dtype = np.uint16 if args.dtype == "uint16" else np.uint32

    if args.donor_hla_matrix:
        matrix_path = Path(args.donor_hla_matrix).resolve()
    else:
        candidates = [
            export_root / "donor_hla_matrix.npz",
            export_root / "donor_hla_matrix.npy",
        ]
        matrix_path = next((p for p in candidates if p.exists()), candidates[0])
    donors_path = Path(args.donor_hla_donors).resolve() if args.donor_hla_donors else None
    donor_matrix, donor_index = load_donor_hla_matrix(matrix_path, donors_path)

    if donor_matrix.shape[1] != num_alleles:
        raise SystemExit(
            f"donor_hla_matrix columns {donor_matrix.shape[1]} != num_alleles {num_alleles}"
        )

    out_h5.parent.mkdir(parents=True, exist_ok=True)

    comp = _compression_kwargs(args)
    chunk_rows = max(1, int(args.chunk_rows))
    chunk_nnz = max(1, int(args.chunk_nnz))
    flush_rows = max(1, int(args.flush_rows))

    indices_dtype = np.uint16 if num_alleles <= np.iinfo(np.uint16).max else np.uint32

    write_public_json = not args.no_public_json
    out_fh = meta_fh = None
    first_entry = True

    missing_donors = set()
    public_tcrs = 0
    clusters_seen = 0
    clusters_written = 0
    rows_total = 0
    v_b_conflicts = 0
    v_b_conflict_examples = []

    current_rep = None
    current_cluster_id = None
    donors = set()
    loop_groups = {}
    loop_order = []

    def add_loop(loops, v_b, j_b):
        group = loop_groups.get(loops)
        if group is None:
            group = {"count": 0, "v_b_counts": Counter(), "j_genes": set()}
            loop_groups[loops] = group
            loop_order.append(loops)
        group["count"] += 1
        if v_b:
            group["v_b_counts"][v_b] += 1
        if j_b:
            group["j_genes"].add(j_b)

    def flush_cluster():
        nonlocal first_entry, public_tcrs, v_b_conflicts, v_b_conflict_examples
        nonlocal current_rep, current_cluster_id, donors, loop_groups, loop_order
        nonlocal clusters_written

        if current_rep is None:
            return

        donors_unique = sorted(donors)
        if len(donors_unique) < args.min_donors:
            return

        donor_indices = []
        for d in donors_unique:
            di = donor_index.get(d)
            if di is None:
                missing_donors.add(d)
                continue
            donor_indices.append(di)

        if not donor_indices:
            return

        row_counts = np.zeros(num_alleles, dtype=counts_dtype)
        for di in donor_indices:
            row_counts += donor_matrix[di]

        if not np.any(row_counts):
            return

        wrote_any = False

        for loops in loop_order:
            group = loop_groups[loops]
            n_identical = group["count"]

            v_b_val = ""
            v_b_id = None
            v_nonempty = [v for v in group["v_b_counts"].keys() if v != ""]
            if v_nonempty:
                if len(v_nonempty) > 1:
                    v_b_conflicts += 1
                    if len(v_b_conflict_examples) < 5:
                        v_b_conflict_examples.append(
                            {"rep_id": current_rep, "v_b_values": sorted(v_nonempty)}
                        )
                v_b_choice = max(
                    v_nonempty,
                    key=lambda v: (group["v_b_counts"].get(v, 0), str(v)),
                )
                v_b_val = v_b_choice
                v_b_id = coerce_int(v_b_choice)

            if args.require_v_genes and (v_b_id is None):
                continue

            if write_public_json:
                key = f"{loops[0]},{loops[1]},{loops[2]},{loops[3]}"
                if not first_entry:
                    out_fh.write(",\n")
                    meta_fh.write(",\n")
                else:
                    first_entry = False

                json.dump(key, out_fh)
                out_fh.write(": ")
                json.dump(donors_unique, out_fh)

                meta_fh.write(json.dumps(key))
                meta_fh.write(": ")
                json.dump(
                    {"v_b": v_b_val, "j_genes": sorted(group["j_genes"])},
                    meta_fh,
                )

            writer.add_row(
                loops=loops,
                n_donors=len(donor_indices),
                cluster_id=current_cluster_id,
                n_identical=n_identical,
                counts=row_counts,
                v_gene_id=v_b_id,
            )
            public_tcrs += 1
            wrote_any = True

        if wrote_any:
            clusters_written += 1

    if write_public_json:
        public_json.parent.mkdir(parents=True, exist_ok=True)
        meta_json.parent.mkdir(parents=True, exist_ok=True)
        out_fh, meta_fh = open_json_writers(public_json, meta_json)

    writer_attrs = {
        "export_root": str(export_root),
        "cluster_full_file": str(cluster_path),
        "min_donors": int(args.min_donors),
    }

    with PublicTcrHlaCsrWriter(
        out_h5,
        num_alleles=num_alleles,
        counts_dtype=counts_dtype,
        indices_dtype=indices_dtype,
        chunk_rows=chunk_rows,
        chunk_nnz=chunk_nnz,
        flush_rows=flush_rows,
        compression=comp,
        include_v_genes=not args.no_v_genes,
        attrs=writer_attrs,
    ) as writer:
        iterator = None
        with cluster_path.open("r", encoding="utf-8", newline="") as fin:
            reader = csv.reader(fin, delimiter="\t")
            iterator = reader
            if args.progress:
                iterator = tqdm(reader, desc="Reducing clusters", unit="rows")

            for row in iterator:
                if not row:
                    continue
                if len(row) < 10:
                    raise SystemExit(
                        f"Expected >=10 columns (including v_b and j_b). Got {len(row)}. Row: {row!r}"
                    )

                rep_id = row[0]
                dataset = row[2]
                sample_file = row[3]
                cdr3 = norm(row[4])
                cdr2 = norm(row[5])
                cdr1 = norm(row[6])
                cdr25 = norm(row[7])
                v_b = norm(row[8])
                j_b = norm(row[9])
                loops = (cdr3, cdr2, cdr1, cdr25)
                donor_key = f"{dataset}/{sample_file}"

                rows_total += 1

                if current_rep is None:
                    current_rep = rep_id
                    clusters_seen = 1
                    current_cluster_id = clusters_seen - 1
                    donors = {donor_key}
                    loop_groups = {}
                    loop_order = []
                    add_loop(loops, v_b, j_b)
                    continue

                if rep_id == current_rep:
                    donors.add(donor_key)
                    add_loop(loops, v_b, j_b)
                else:
                    flush_cluster()

                    clusters_seen += 1
                    current_rep = rep_id
                    current_cluster_id = clusters_seen - 1
                    donors = {donor_key}
                    loop_groups = {}
                    loop_order = []
                    add_loop(loops, v_b, j_b)

            if current_rep is not None:
                flush_cluster()

    entries_written = writer.rows_written

    if write_public_json:
        close_json_writers(out_fh, meta_fh)

    if entries_written == 0:
        raise SystemExit("No entries retained after min_donors/missing-mask filtering.")

    stats = {
        "export_root": str(export_root),
        "cluster_full_file": str(cluster_path),
        "rows_total": rows_total,
        "clusters_seen": clusters_seen,
        "clusters_written": clusters_written,
        "entries_written": entries_written,
        "public_tcrs": public_tcrs,
        "min_donors": args.min_donors,
        "v_b_conflicts": v_b_conflicts,
        "v_b_conflict_examples": v_b_conflict_examples,
        "missing_donors": len(missing_donors),
        "output_h5": str(out_h5),
        "public_tcrs_json": str(public_json) if write_public_json else None,
        "public_tcrs_meta_json": str(meta_json) if write_public_json else None,
    }

    stats_json.parent.mkdir(parents=True, exist_ok=True)
    with stats_json.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    if missing_donors:
        print(f"Warning: {len(missing_donors):,} donors missing in donor_hla_matrix.")

    print(f"Wrote HDF5 dataset to: {out_h5}")


if __name__ == "__main__":
    main()
