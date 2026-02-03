#!/usr/bin/env bash
#
# # TODO snakemake
#
#  1) Export bio-identity FASTA + metadata:
#       src/export_bioidentity_fasta.py
#  2) Run mmseqs createdb + linclust (100% identity).
#  3) Filter clusters by size (>= min-size), keep >=2 by default.
#  4) Build cluster_members_by_seq.sorted.tsv from filtered clusters.
#  5) Sort all_bioid_metadata.tsv by bioid and join clusters+metadata:
#       src/join_clusters_with_metadata_simple.py
#  6) Sort cluster_members_full.tsv by representative:
#       cluster_members_full_by_rep.tsv
#  7) Build public TCR × HLA count matrix:
#       src/build_public_tcr_hla_numpy.py
#
# Default layout (matching assemble_train_dataset):
#   <BASE>/<PROCESSED_ROOT>/
#     mmseqs/
#       all_bioid.faa
#       all_bioid_metadata.tsv
#       linclust_100id_bioid.DB / .CLU
#       linclust_100id_bioid_clusters_all.tsv
#       linclust_100id_bioid_clusters_ge<N>.tsv
#       cluster_members_by_seq.sorted.tsv
#       all_bioid_metadata.sorted.tsv
#       cluster_members_full.tsv
#       cluster_members_full_by_rep.tsv
#
#   <BASE>/<PROCESSED_ROOT>/public_tcrs.json
#   <BASE>/<PROCESSED_ROOT>/public_tcr_hla_counts.h5
#
# Usage (typical):
#   ./run_mmseqs_public_tcr_pipeline.sh \
#       --base /path/to/base \
#       --processed-root export_train_dataset
#
# Environment overrides:
#   MMSEQS_THREADS  – default mmseqs threads if --threads not given
#   SORT_THREADS    – default sort --parallel if not given
#   SORT_MEM        – default sort -S memory (e.g. 60G) if not given

set -euo pipefail


BASE="."
PROCESSED_ROOT_NAME="export_train_dataset"

MMSEQS_PREFIX="linclust_100id_bioid"
MMSEQS_THREADS="${MMSEQS_THREADS:-8}"

SORT_THREADS_DEFAULT="${SORT_THREADS:-8}"
SORT_MEM_DEFAULT="${SORT_MEM:-60G}"

MIN_CLUSTER_SIZE=2
MIN_PUBLIC_DONORS=2
PUBLIC_DTYPE="uint16"

PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --base PATH             Base directory containing <processed-root>/ (default: ${BASE})
  --processed-root NAME   Processed export root directory name under base
                          (default: ${PROCESSED_ROOT_NAME})

  --threads N             Threads for mmseqs (default: MMSEQS_THREADS env or ${MMSEQS_THREADS})
  --sort-threads N        Threads for sort --parallel (default: SORT_THREADS env or ${SORT_THREADS_DEFAULT})
  --sort-mem SIZE         Memory for sort -S (default: SORT_MEM env or ${SORT_MEM_DEFAULT})

  --min-cluster-size N    Minimum cluster size to keep (default: ${MIN_CLUSTER_SIZE})
  --min-public-donors N   Minimum donors for a TCR to be considered public (default: ${MIN_PUBLIC_DONORS})

  --dtype TYPE            Dtype for public_tcr_hla_counts.h5 (default: ${PUBLIC_DTYPE})

  -h, --help              Show this help and exit
EOF
}

SORT_THREADS="${SORT_THREADS_DEFAULT}"
SORT_MEM="${SORT_MEM_DEFAULT}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)
      BASE="$2"; shift 2 ;;
    --processed-root)
      PROCESSED_ROOT_NAME="$2"; shift 2 ;;
    --threads)
      MMSEQS_THREADS="$2"; shift 2 ;;
    --sort-threads)
      SORT_THREADS="$2"; shift 2 ;;
    --sort-mem)
      SORT_MEM="$2"; shift 2 ;;
    --min-cluster-size)
      MIN_CLUSTER_SIZE="$2"; shift 2 ;;
    --min-public-donors)
      MIN_PUBLIC_DONORS="$2"; shift 2 ;;
    --dtype)
      PUBLIC_DTYPE="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

BASE="$(realpath "${BASE}")"
PROCESSED_ROOT="${BASE}/${PROCESSED_ROOT_NAME}"
MMSEQS_DIR="${PROCESSED_ROOT}/mmseqs"

INFA="${MMSEQS_DIR}/all_bioid.faa"
META_TSV="${MMSEQS_DIR}/all_bioid_metadata.tsv"
OUT_PREFIX="${MMSEQS_DIR}/${MMSEQS_PREFIX}"
TMP_DIR="${MMSEQS_DIR}/tmp_${MMSEQS_PREFIX}"

CLUSTERS_TSV="${OUT_PREFIX}_clusters_all.tsv"
CLUSTERS_GE_TSV="${OUT_PREFIX}_clusters_ge${MIN_CLUSTER_SIZE}.tsv"

CLUSTER_MEMBERS_BY_SEQ_SORTED="${MMSEQS_DIR}/cluster_members_by_seq.sorted.tsv"
META_SORTED="${MMSEQS_DIR}/all_bioid_metadata.sorted.tsv"
CLUSTER_MEMBERS_FULL="${MMSEQS_DIR}/cluster_members_full.tsv"
CLUSTER_MEMBERS_FULL_BY_REP="${MMSEQS_DIR}/cluster_members_full_by_rep.tsv"

PUBLIC_TCRS_JSON="${PROCESSED_ROOT}/public_tcrs.json"
PUBLIC_TCR_HLA_NPZ="${PROCESSED_ROOT}/public_tcr_hla_counts.h5"

echo "Base:             ${BASE}"
echo "Processed root:   ${PROCESSED_ROOT}"
echo "mmseqs dir:       ${MMSEQS_DIR}"
echo "mmseqs threads:   ${MMSEQS_THREADS}"
echo "sort threads:     ${SORT_THREADS}"
echo "sort memory:      ${SORT_MEM}"
echo "min cluster size: ${MIN_CLUSTER_SIZE}"
echo "min public donors:${MIN_PUBLIC_DONORS}"
echo "dtype (public HLA): ${PUBLIC_DTYPE}"
echo

mkdir -p "${MMSEQS_DIR}" "${TMP_DIR}"


if [[ ! -s "${INFA}" ]]; then
  echo "[0/5] Exporting bio-identity FASTA and metadata"
  "${PYTHON_BIN}" src/export_bioidentity_fasta.py \
    --base "${BASE}" \
    --processed-root "${PROCESSED_ROOT_NAME}" \
    --out-fasta "mmseqs/$(basename "${INFA}")" \
    --out-meta "mmseqs/$(basename "${META_TSV}")"
else
  echo "[0/5] Found existing FASTA: ${INFA}"
fi


echo "[1/5] mmseqs createdb"
mmseqs createdb "${INFA}" "${OUT_PREFIX}.DB" --dbtype 1

echo "[2/5] mmseqs linclust (100% identity; full coverage on both seqs)"
mmseqs linclust "${OUT_PREFIX}.DB" "${OUT_PREFIX}.CLU" "${TMP_DIR}" \
  --min-seq-id 1.0 \
  -c 1.0 \
  --cov-mode 0 \
  --alignment-mode 3 \
  --cluster-mode 0 \
  --split-memory-limit 40G \
  --threads "${MMSEQS_THREADS}"

echo "[3/5] mmseqs createtsv (rep, member table)"
mmseqs createtsv "${OUT_PREFIX}.DB" "${OUT_PREFIX}.DB" "${OUT_PREFIX}.CLU" "${CLUSTERS_TSV}" \
  --threads "${MMSEQS_THREADS}"

wc -l "${CLUSTERS_TSV}" || true
echo "All clusters TSV (including singletons): ${CLUSTERS_TSV}"


echo "[4/5] Filtering clusters by size >= ${MIN_CLUSTER_SIZE}"
"${PYTHON_BIN}" src/filter_clusters_by_size_stream.py \
  "${MMSEQS_DIR}" \
  "$(basename "${CLUSTERS_TSV}")" \
  "$(basename "${CLUSTERS_GE_TSV}")" \
  --min-size "${MIN_CLUSTER_SIZE}"

echo "[4/5] Building cluster_members_by_seq.sorted.tsv"
awk 'NF >= 2 {print $2 "\t" $1}' "${CLUSTERS_GE_TSV}" \
  | pv \
  | LC_ALL=C sort -t $'\t' -k1,1 \
      -S "${SORT_MEM}" \
      --parallel="${SORT_THREADS}" \
  > "${CLUSTER_MEMBERS_BY_SEQ_SORTED}"

echo "cluster_members_by_seq.sorted.tsv: ${CLUSTER_MEMBERS_BY_SEQ_SORTED}"


echo "[4/5] Sorting all_bioid_metadata.tsv by bioid"
tail -n +2 "${META_TSV}" \
  | pv \
  | LC_ALL=C sort -t $'\t' -k1,1 \
      -S "${SORT_MEM}" \
      --parallel="${SORT_THREADS}" \
  > "${META_SORTED}"

echo "all_bioid_metadata.sorted.tsv: ${META_SORTED}"

echo "[4/5] Joining clusters with metadata"
"${PYTHON_BIN}" src/join_clusters_with_metadata_simple.py \
  "${BASE}" \
  --processed-root "${PROCESSED_ROOT_NAME}" \
  --clusters "mmseqs/$(basename "${CLUSTER_MEMBERS_BY_SEQ_SORTED}")" \
  --metadata "mmseqs/$(basename "${META_SORTED}")" \
  --out-full "mmseqs/$(basename "${CLUSTER_MEMBERS_FULL}")"

echo "[4/5] Sorting cluster_members_full.tsv by representative"
pv "${CLUSTER_MEMBERS_FULL}" \
  | LC_ALL=C sort -t $'\t' -k1,1 \
      -S "${SORT_MEM}" \
      --parallel="${SORT_THREADS}" \
  > "${CLUSTER_MEMBERS_FULL_BY_REP}"

echo "cluster_members_full_by_rep.tsv: ${CLUSTER_MEMBERS_FULL_BY_REP}"


echo "[5/5] Building public TCR × HLA counts (NPZ)"
"${PYTHON_BIN}" src/build_public_tcr_hla_numpy.py \
  --export-root   "${PROCESSED_ROOT}" \
  --public-tcrs-json "${PUBLIC_TCRS_JSON}" \
  --out-npz       "${PUBLIC_TCR_HLA_NPZ}" \
  --min-donors    "${MIN_PUBLIC_DONORS}" \
  --dtype         "${PUBLIC_DTYPE}" \
  --require-v-genes \
  --progress

echo
echo "Done"
echo "  Clusters (all):     ${CLUSTERS_TSV}"
echo "  Clusters (>=${MIN_CLUSTER_SIZE}): ${CLUSTERS_GE_TSV}"
echo "  Cluster members:    ${CLUSTER_MEMBERS_FULL_BY_REP}"
echo "  Public TCRs JSON:   ${PUBLIC_TCRS_JSON}"
echo "  Public TCR × HLA:   ${PUBLIC_TCR_HLA_NPZ}"
