# MLE Pipeline for TCR-HLA Binding Probability Estimation

Estimates per-TCR binding probabilities (`z_probs`) for each HLA allele via maximum likelihood, then analyzes the results.

## Scripts

| Script | Purpose |
|---|---|
| `mle_real_data.py` | Train, merge, and monitor chunk-wise MLE |
| `utils.py` | Model (`SparseTCRModel`), data reader/writer classes |
| `analyze_zprobs.py` | Post-training analysis and plotting |

## Quick Start

```bash
# 1. Train (SLURM array — 5 GPU jobs, 1000 clusters/chunk)
sbatch run_mle.sbatch

# 2. Check progress
python mle_real_data.py --mode status --chunk_size 1000 \
    --h5_data_path data.h5 --donor_matrix_path donor.npz --output_dir out/

# 3. Merge chunks into final H5
python mle_real_data.py --mode merge --chunk_size 1000 \
    --h5_data_path data.h5 --donor_matrix_path donor.npz --output_dir out/

# 4. Analyze
python analyze_zprobs.py --all --h5_path out/dataset_pval.h5 \
    --donor_matrix_path donor.npz --output_dir out/
```

## mle_real_data.py

### Modes

**`--mode train`** — Fits z_probs per chunk. Each chunk saves an independent `.npz` checkpoint.

```bash
# Single chunk (SLURM array)
python mle_real_data.py --mode train --chunk_id $SLURM_ARRAY_TASK_ID ...

# Range of chunks
python mle_real_data.py --mode train --chunk_range 0 100 --resume ...

# Sequential with auto-resume
python mle_real_data.py --mode train --resume ...
```

**`--mode merge`** — Combines all chunk checkpoints into the output H5 (copies original + adds `clusters/z_probs` CSR group).

**`--mode status`** — Prints completion stats, missing chunks, and SLURM command templates.

### SLURM Job Arrays

```bash
#!/bin/bash
#SBATCH --job-name=mle_tcr
#SBATCH --array=0-4
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=01-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=scc-gpu
#SBATCH --output=logs/mle_job_%a.out
#SBATCH --error=logs/mle_job_%a.err

# GPU setup
module purge && source ~/.bashrc
conda activate /path/to/TCRtyper
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export CUDA_HOME=$CONDA_PREFIX
module load gcc/13.2.0-nvptx cuda/12.6.2 cudnn/9.8.0.87-12

# Distribute chunks across jobs
TOTAL_CHUNKS=38742        # = ceil(38741623 / 1000)
N_JOBS=5
CHUNKS_PER_JOB=$(( (TOTAL_CHUNKS + N_JOBS - 1) / N_JOBS ))
START=$(( SLURM_ARRAY_TASK_ID * CHUNKS_PER_JOB ))
END=$(( START + CHUNKS_PER_JOB ))
if [ $END -gt $TOTAL_CHUNKS ]; then END=$TOTAL_CHUNKS; fi

python mle_real_data.py \
    --h5_data_path data.h5 --donor_matrix_path donor.npz \
    --output_dir out/ --chunk_size 1000 \
    --mode train --chunk_range $START $END --resume \
    --device auto --epochs 120 --batch_size 2000 \
    --reduction sum --learning_rate 0.6 --l2_reg 0.0000002
```

Adjust `--array`, `TOTAL_CHUNKS`, and `N_JOBS` to match your setup. Each job processes `CHUNKS_PER_JOB` chunks sequentially with `--resume` skipping any already completed.

After all jobs finish:

```bash
# Verify completion
python mle_real_data.py --mode status --chunk_size 1000 ...

# Merge into final H5
python mle_real_data.py --mode merge --chunk_size 1000 ...
```

### Key Training Arguments
| `--batch_size` | 512 | Mini-batch for GPU memory |
| `--accumulation_steps` | 1 | Gradient accumulation (effective batch = batch_size × N) |
| `--optimizer` | adam | `adam`, `lion`, `lamb`, or `sgd` |
| `--reduction` | sum | `sum` (recommended) or `mean` (legacy) |
| `--learning_rate` | 0.01 | Initial LR (CosineDecay schedule) |
| `--l2_reg` | 1e-5 | L2 regularization strength |
| `--epochs` | 10 | Epochs per chunk |
| `--resume` | off | Skip already-completed chunks |
| `--device` | auto | `auto`, `cpu`, or `gpu` |

### Output Structure

```
output_dir/
├── config.json
├── chunks/
│   ├── chunk_000000.npz    # {cluster_start, cluster_end, binder_sets, z_probs, metadata_json}
│   ├── chunk_000001.npz
│   └── ...
├── dataset_pval.h5         # After merge: original H5 + clusters/z_probs/{indptr,indices,data}
└── mle_training_log.json   # After merge: hyperparams + per-chunk loss/timing
```

### Reading Results

```python
from utils import PublicTcrHlaCsrReaderChunk

with PublicTcrHlaCsrReaderChunk("out/dataset_pval.h5",
        include_counts=True, include_donors=True, include_z_probs=True) as reader:
    for chunk in reader.iter_cluster_chunks(chunk_rows=10000):
        # chunk.cluster_id        — global cluster IDs
        # chunk.n_donors          — donor count per cluster
        # chunk.z_probs_dense     — (n, num_alleles) binding probabilities
        # chunk.counts_dense      — (n, num_alleles) allele occurrence counts
```

## analyze_zprobs.py

Runs on the merged H5. Select analyses with `--all` or individual flags.

### Analyses

**`--donor_explanation`** — For each TCR at each threshold, what fraction of its donors is "explained" (has at least one active allele with z > threshold). Produces AUC curves across 10 explanation levels (≥10%, ≥20%, ..., ≥100%).

**`--hla_diversity`** — Per-HLA enrichment (observed binding / expected by abundance) across thresholds. Checks whether specific alleles dominate predictions.

**`--entropy`** — Shannon entropy, Gini index, max/mean z-prob, and active allele count per TCR. Shows how concentrated or spread out each TCR's binding profile is.

**`--donor_bins`** — Groups TCRs by donor count and compares explanation AUC, entropy, and Gini distributions across groups. Requires `--donor_explanation` and `--entropy` (auto-enabled).

### Key Analysis Arguments

| Arg | Default | Description |
|---|---|---|
| `--threshold_step` | 0.05 | Step between thresholds (0.0 to 1.0) |
| `--explanation_levels` | 10 20 ... 100 | N% levels for explanation curves |
| `--donor_bin_edges` | 1 2 6 11 26 51 101 251 501 | Edges for donor count groups |
| `--chunk_size` | 50000 | Processing chunk size |

### Output Structure

```
output_dir/analysis/
├── analysis.log
├── metrics.h5                            # Per-TCR: cluster_id, n_donors,
│                                         #   explanation_fractions (N×T), explanation_auc,
│                                         #   entropy, gini, max_z_prob, n_active_alleles
├── donor_explanation/
│   ├── explanation_auc_curves.png        # Line plot with AUC per explanation level
│   └── explanation_auc_values.csv
├── hla_diversity/
│   ├── hla_diversity_summary.png         # Enrichment bands + conditional binding
│   ├── hla_top_bottom_enrichment.png     # Bar chart: most over/under-represented HLAs
│   └── hla_diversity_data.h5
├── entropy/
│   └── metric_distributions.png          # 6-panel histogram: entropy, gini, max_z, etc.
├── donor_bins/
│   ├── donor_bin_distribution.png        # Bar chart: TCR counts per donor group
│   ├── donor_bin_explanation_auc.png     # Box plot: AUC by group
│   ├── donor_bin_entropy_gini.png        # Box plot: entropy + gini by group
│   ├── donor_bin_max_zprob.png           # Box plot: max z-prob by group
│   └── donor_bin_stats.csv
└── additional/
    └── summary_statistics.json
```

### metrics.h5 Schema

| Dataset | Shape | Description |
|---|---|---|
| `cluster_id` | (N,) int64 | Global cluster ID (maps to original H5) |
| `n_donors` | (N,) int32 | Number of donors per TCR |
| `thresholds` | (T,) float32 | Threshold values used |
| `explanation_fractions` | (N, T) float16 | Fraction of donors explained per threshold |
| `explanation_auc` | (N,) float32 | AUC of explanation curve per TCR |
| `entropy` | (N,) float32 | Shannon entropy of normalized z_probs |
| `gini` | (N,) float32 | Gini index of z_prob distribution |
| `max_z_prob` | (N,) float32 | Maximum z_prob per TCR |
| `mean_z_prob_nonzero` | (N,) float32 | Mean of nonzero z_probs |
| `n_active_alleles` | (N,) uint16 | Count of alleles with z > 0 |

## Notes

- **`reduction=sum`** is recommended — makes gradients independent of batch size.
- **Gradient accumulation** (`--accumulation_steps N`) simulates large batches without extra GPU memory. Effective batch = batch_size × N.
- **Resume** (`--resume`) is safe for restarts and SLURM preemption — completed chunks are atomic (write to `.tmp.npz`, then rename).
- The analysis script does a **single pass** through the H5, computing all metrics simultaneously.



--------------------
# TCR Binder Set Size Classifier

Predicts how many HLA alleles restrict a given TCR using a classifier trained on synthetic data with uncertainty-aware calibration.

## Requirements

```
tensorflow >= 2.14
numpy, h5py, matplotlib, scikit-learn
```

## Overview

**Train** on synthetic TCR-HLA data (bX_nY folders) → **Infer** on real TCR data (HDF5).

The model outputs a continuous expected binder set size per TCR and flags unreliable predictions using calibrated entropy/margin thresholds.

## Usage

### Training

```bash
python train_and_pred_binder_set.py --mode train \
    --input_train outputs/synthetic_analysis_with_reg \
    --output_dir outputs/synthetic_analysis_with_reg/models \
    --epochs 100 \
    --convergence \
    --k_crossval 10 \
    --val_split 0.1 \
    --num_neurons 128 \
    --num_layers 1 \
    --dropout_rate 0.3 \
    --ordinal_lambda 1.0 \
    --delta 5.0 \
    --max_error_rate 0.05 \
    --batch_size 10000 \
    --lr_schedule cosine \
    --seed 43
```

**Input:** Directory containing `bX_nY/` subfolders (e.g. `b10_n100/`), each with `figures/donor_scores_matrix.npz` and `figures/analysis_arrays.npz`.

**Output:** Model checkpoint, `metadata_train.json` (calibration parameters), confusion matrices, and training curves in `--output_dir`.

### Inference

```bash
python train_and_pred_binder_set.py --mode inference \
    --input_inference outputs/real_data_2/analysis/metrics.h5 \
    --output_dir outputs/synthetic_analysis_with_reg/models \
    --batch_size 10000 \
    --infer_chunk 500000
```

**Input:** HDF5 file with keys `entropy`, `n_donors`, `explanation_fractions`, `n_active_alleles`, `explanation_auc`, `max_z_prob`, `mean_z_prob_nonzero`.

**Output:** Writes predictions back into the same HDF5:

| Key | Description |
|---|---|
| `predicted_binderset_probs_calibrated` | (N, 6) calibrated probabilities over bins [3,5,10,15,25,35] |
| `predicted_binderset_expected` | (N,) continuous expected binder size (main output) |
| `predicted_binderset_best_calibrated` | (N,) argmax discrete binder size |
| `predicted_binderset_entropy` | (N,) predictive entropy |
| `predicted_binderset_margin` | (N,) confidence margin |
| `predicted_binderset_reliable` | (N,) binary: 1 = passes uncertainty thresholds |

Also saves a `.npz` copy and generates diagnostic plots in `figures_inference/`.

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--ordinal_lambda` | 1.0 | Sharpness of distance-weighted soft targets |
| `--delta` | 5.0 | Tolerance band for continuous error: \|b̂ − y\| > δ |
| `--max_error_rate` | 0.05 | Max acceptable error among retained predictions |
| `--k_crossval` | 5 | Folds for cross-validated temperature calibration |
| `--convergence` | True | Early stopping on validation loss (patience=15) |