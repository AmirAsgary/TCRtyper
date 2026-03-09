# TCRtyper: Training, Inference & Analysis

## Overview

TCRtyper predicts TCR–HLA binding probabilities using a transformer model trained on cluster-level TCR data stored in HDF5. Two data pipelines are supported:

| Pipeline | Training data | Inference output | Analysis script |
|----------|--------------|-----------------|-----------------|
| **HDF5** (default) | Streams chunks from `.h5` via Python threads | `predictions.h5` (sparse z_probs appended) | `analyze_zprobs.py` |
| **TFRecord** (faster) | One-time `.h5` → sharded `.tfrecord` conversion, then pure C++ I/O | `predictions.npz` (dense arrays) | `analyze_zprobs_npz.py` |

---

## 1. Required Inputs

All modes need these files:

| File | Description |
|------|-------------|
| `train_ds.h5` | Training HDF5 (clusters, CDR freqs, donors, counts) |
| `valid_ds.h5` | Validation HDF5 (same format, optional) |
| `donor_hla_matrix.npz` | Key `donor_hla_matrix`, shape `(N_donors, A)` |
| `idx_to_hla.json` | Maps `{"0": "HLA-A*01:01", "1": ...}` |
| `hla_embed.npz` | ESM embeddings per allele (optional, improves init) |

---

## 2. Training

### 2a. HDF5 Pipeline (default)

```bash
python train_tcrtyper.py --mode train \
    --train_ds data/train_ds.h5 \
    --valid_ds data/valid_ds.h5 \
    --donor_hla_matrix data/donor_hla_matrix.npz \
    --idx_to_hla data/idx_to_hla.json \
    --hla_embed data/hla_embed.npz \
    --output_dir outputs/exp01 \
    --embed_dim 64 --num_heads 4 --num_layers 2 \
    --batch_size 2000 --epochs 100 \
    --lr 9e-4 --grad_clip 10 \
    --reduction mean \
    --train_hla_head \
    --masking_rate 0.15 \
    --patience 30
```

Data is streamed from HDF5 every epoch via a background prefetch thread. Simple but slower for large datasets due to Python/HDF5 overhead.

### 2b. TFRecord Pipeline (recommended for large data)

```bash
python train_tcrtyper.py --mode train \
    --train_ds data/train_ds.h5 \
    --valid_ds data/valid_ds.h5 \
    --donor_hla_matrix data/donor_hla_matrix.npz \
    --idx_to_hla data/idx_to_hla.json \
    --hla_embed data/hla_embed.npz \
    --output_dir outputs/exp02 \
    --use_tfrecord \
    --tf_record_path outputs/exp02/tfrecords \
    --num_shards 32 \
    --keep_only_upperthan_n_donors 10 \
    --embed_dim 64 --num_heads 4 --num_layers 2 \
    --batch_size 2000 --epochs 240 \
    --lr 9e-4 --grad_clip 10 \
    --reduction mean \
    --train_hla_head \
    --masking_rate 0.15 \
    --patience 30 \
    --resume
```

**What happens:**
1. First run converts `train_ds.h5` → `outputs/exp02/tfrecords/tfrecord_cache_train/` (sharded `.tfrecord` files + `manifest.json`). Same for validation.
2. Subsequent runs (with `--resume`) skip conversion and reuse cached shards.
3. All I/O runs in C++ threads with parallel interleave + prefetch — significantly faster than HDF5 streaming.

**Key TFRecord flags:**

| Flag | Description |
|------|-------------|
| `--use_tfrecord` | Enable TFRecord pipeline |
| `--tf_record_path` | Base directory for cached shards |
| `--num_shards` | Number of shard files (more = better parallelism) |
| `--keep_only_upperthan_n_donors N` | Filter clusters with < N donors during conversion |

---

## 3. Inference

### 3a. HDF5 Inference

```bash
python train_tcrtyper.py --mode inference \
    --train_ds data/train_ds.h5 \
    --inference_ds data/valid_ds.h5 \
    --donor_hla_matrix data/donor_hla_matrix.npz \
    --idx_to_hla data/idx_to_hla.json \
    --hla_embed data/hla_embed.npz \
    --output_dir outputs/exp01 \
    --embed_dim 64 --num_heads 4 --num_layers 2 \
    --batch_size 2000 \
    --reduction mean \
    --train_hla_head \
    --masking_rate 0.0
```

**Output:** `outputs/exp01/predictions.h5` — a copy of the input H5 with `clusters/z_probs` (sparse CSR) appended.

### 3b. TFRecord Inference

```bash
python train_tcrtyper.py --mode inference \
    --train_ds data/train_ds.h5 \
    --donor_hla_matrix data/donor_hla_matrix.npz \
    --idx_to_hla data/idx_to_hla.json \
    --hla_embed data/hla_embed.npz \
    --output_dir outputs/exp02 \
    --embed_dim 64 --num_heads 4 --num_layers 2 \
    --batch_size 2000 \
    --reduction mean \
    --train_hla_head \
    --masking_rate 0.0 \
    --use_tfrecord \
    --tf_record_path outputs/exp02/tfrecords/tfrecord_cache_valid
```

**Note:** `--tf_record_path` points directly to the shard directory (e.g., `tfrecord_cache_valid/`), not the parent.

**Output:** `outputs/exp02/predictions.npz` with keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `z_logits` | `(N, A)` | Raw model output (pre-sigmoid) |
| `z_probs` | `(N, A)` | Binding probabilities `sigmoid(z_logits)` |
| `binder_dense` | `(N, A)` | Binary co-occurrence mask |
| `donor_indices` | `(N, D)` | Padded donor IDs per cluster |
| `n_donors` | `(N,)` | Number of donors per cluster |

### Important: Architecture flags must match training

These flags must be identical between training and inference:
`--embed_dim`, `--num_heads`, `--num_layers`, `--train_hla_head`, `--hla_embed`

---

## 4. Analysis

### 4a. Analyze H5 Predictions

```bash
python analyze_zprobs.py \
    --h5_path outputs/exp01/predictions.h5 \
    --donor_matrix_path data/donor_hla_matrix.npz \
    --output_dir outputs/exp01 \
    --keep_only_upperthan_n_donors 10 \
    --gpu \
    --all
```

### 4b. Analyze NPZ Predictions (TFRecord)

```bash
python analyze_zprobs_npz.py \
    --npz_path outputs/exp02/predictions.npz \
    --donor_matrix_path data/donor_hla_matrix.npz \
    --output_dir outputs/exp02 \
    --keep_only_upperthan_n_donors 10 \
    --gpu \
    --all
```

### Analysis Outputs (both versions)

Both scripts produce identical analysis structure:

```
output_dir/analysis_10/          # (or analysis_npz_10/ for NPZ)
├── metrics.h5                   # Per-TCR metrics (intermediate)
├── donor_explanation/
│   ├── explanation_auc_curves.png
│   └── explanation_auc_values.csv
├── hla_diversity/
│   ├── hla_diversity_summary.png
│   ├── hla_top_bottom_enrichment.png
│   └── hla_diversity_data.h5
├── entropy/
│   └── metric_distributions.png
├── donor_bins/
│   ├── donor_bin_zprobs_summary.png
│   ├── donor_bin_explanation_curves.png
│   ├── donor_bin_entropy_boxplot.png
│   ├── donor_bin_dual_axis_thresholds.png
│   ├── donor_bin_zprob_heatmaps.png
│   └── donor_bin_stats.csv
├── additional/
│   ├── summary_statistics.json
│   └── logit_distributions.png   # NPZ only
└── analysis.log
```

### Available Analyses

| Analysis | Flag | What it shows |
|----------|------|---------------|
| Donor Explanation | `--donor_explanation` | Fraction of donors explained at each z-prob threshold |
| HLA Diversity | `--hla_diversity` | Which alleles dominate predictions, enrichment vs background |
| Entropy | `--entropy` | Distribution of prediction entropy, gini, min/max/mean z-prob |
| Donor Bins | `--donor_bins` | All metrics stratified by donor count groups |
| All | `--all` | Run everything |

### NPZ-Exclusive: Logit Analysis

The NPZ version automatically generates `logit_distributions.png` with:
- Active vs inactive logit histograms
- Per-TCR mean logit comparison
- Logit separation gap (max active − max inactive)

This is not possible with H5 predictions since they only store sigmoid outputs.

---

## 5. Quick Reference

### Full TFRecord Workflow (recommended)

```bash
# 1. Train
python train_tcrtyper.py --mode train \
    --train_ds data/train_ds.h5 --valid_ds data/valid_ds.h5 \
    --donor_hla_matrix data/donor_hla_matrix.npz \
    --idx_to_hla data/idx_to_hla.json \
    --hla_embed data/hla_embed.npz \
    --output_dir outputs/exp01 \
    --use_tfrecord --tf_record_path outputs/exp01/tfrecords --num_shards 32 \
    --embed_dim 64 --num_heads 4 --num_layers 2 \
    --batch_size 2000 --epochs 240 --lr 9e-4 \
    --train_hla_head --masking_rate 0.15 --reduction mean

# 2. Inference on validation set
python train_tcrtyper.py --mode inference \
    --train_ds data/train_ds.h5 \
    --donor_hla_matrix data/donor_hla_matrix.npz \
    --idx_to_hla data/idx_to_hla.json \
    --hla_embed data/hla_embed.npz \
    --output_dir outputs/exp01 \
    --use_tfrecord --tf_record_path outputs/exp01/tfrecords/tfrecord_cache_valid \
    --embed_dim 64 --num_heads 4 --num_layers 2 \
    --batch_size 2000 --train_hla_head --masking_rate 0.0 --reduction mean

# 3. Analyze
python analyze_zprobs_npz.py \
    --npz_path outputs/exp01/predictions.npz \
    --donor_matrix_path data/donor_hla_matrix.npz \
    --output_dir outputs/exp01 \
    --gpu --all
```

### Full HDF5 Workflow

```bash
# 1. Train (same but without --use_tfrecord)
python train_tcrtyper.py --mode train \
    --train_ds data/train_ds.h5 --valid_ds data/valid_ds.h5 \
    --donor_hla_matrix data/donor_hla_matrix.npz \
    --idx_to_hla data/idx_to_hla.json \
    --hla_embed data/hla_embed.npz \
    --output_dir outputs/exp01 \
    --embed_dim 64 --num_heads 4 --num_layers 2 \
    --batch_size 2000 --epochs 100 --lr 9e-4 \
    --train_hla_head --masking_rate 0.15 --reduction mean

# 2. Inference
python train_tcrtyper.py --mode inference \
    --train_ds data/train_ds.h5 --inference_ds data/valid_ds.h5 \
    --donor_hla_matrix data/donor_hla_matrix.npz \
    --idx_to_hla data/idx_to_hla.json \
    --hla_embed data/hla_embed.npz \
    --output_dir outputs/exp01 \
    --embed_dim 64 --num_heads 4 --num_layers 2 \
    --batch_size 2000 --train_hla_head --masking_rate 0.0 --reduction mean

# 3. Analyze
python analyze_zprobs.py \
    --h5_path outputs/exp01/predictions.h5 \
    --donor_matrix_path data/donor_hla_matrix.npz \
    --output_dir outputs/exp01 \
    --gpu --all
```