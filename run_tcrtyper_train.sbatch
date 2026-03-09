#!/bin/bash
#SBATCH --job-name=TCR_Train
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=02-00:00:00
#SBATCH --output=/user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/outputs/exp03_ndonors10_scratch_WithTCR/logs/mle_%x_%A_%a_%j.out
#SBATCH --error=/user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/outputs/exp03_ndonors10_scratch_WithTCR/logs/mle_%x_%A_%a_%j.err
#SBATCH --partition=scc-gpu
#SBATCH -G A100:1

module purge
source ~/.bashrc
mamba activate /user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/envs/TCRtyper

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export CUDA_HOME=$CONDA_PREFIX
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=2"
export TMPDIR=/user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/tmp
mkdir -p $TMPDIR
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"

module load gcc/13.2.0-nvptx
module load cuda/12.6.2
module load cudnn/9.8.0.87-12

DONOR_HLA_MATRIX_PATH="/user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/data/autotcr/donor_hla_matrix.npz"
IDX_TO_HLA="/user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/data/autotcr/id_to_hla.json"
HLA_EMBED="/user/a.hajialiasgarynaj01/u14286/.project/dir.project/amirreza/for_Nikolai/all_mhc_clusters_esm.npz"
OUTPUT_DIR="/user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/outputs/exp03_ndonors10_scratch_WithTCR"

mkdir -p $OUTPUT_DIR/logs

python train_tcrtyper.py --mode train \
    --train_ds data_train/train_ds.h5 \
    --valid_ds data_train/valid_ds.h5 \
    --donor_hla_matrix $DONOR_HLA_MATRIX_PATH \
    --idx_to_hla $IDX_TO_HLA \
    --hla_embed $HLA_EMBED \
    --output_dir $OUTPUT_DIR \
    --tf_record_path outputs/exp02_ndonor10/tfrecords \
    --embed_dim 64 --num_heads 4 --num_layers 2 \
    --batch_size 2000 --epochs 240 \
    --lr 9e-4 \
    --log_step 50 \
    --use_tfrecord --num_shards 32 \
    --grad_clip 10 \
    --reduction "mean" \
    --keep_only_upperthan_n_donors 10 \
    --l2_reg 0.01 \
    --invariant_lambda 0.0 \
    --false_pos_lambda 5.0 \
    --masking_rate 0.15 \
    --patience 30 \
    --train_hla_head \
    --resume