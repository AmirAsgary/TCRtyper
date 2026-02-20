#!/bin/bash
#SBATCH --job-name=tcrtyper
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --time=02-00:00:00
#SBATCH --output=/user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/outputs/exp01/logs/mle_job_%a_2.out
#SBATCH --error=/user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/outputs/exp01/logs/mle_job_%a_2.err
#SBATCH --partition=scc-gpu
#SBATCH -G A100:1


# %a in the filenames above gets replaced by the array task ID automatically
# for cpu
#conda activate /user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/envs/TCRtyper
#export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# for gpu required
module purge
source ~/.bashrc
mamba activate /user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/envs/TCRtyper
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export CUDA_HOME=$CONDA_PREFIX
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=2"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"
module load gcc/13.2.0-nvptx
module load cuda/12.6.2
module load cudnn/9.8.0.87-12
export TMPDIR=/user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/tmp
mkdir -p $TMPDIR
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"


DONOR_HLA_MATRIX_PATH="/user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/data/autotcr/donor_hla_matrix.npz"
IDX_TO_HLA="/user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/data/autotcr/id_to_hla.json"
HLA_EMBED="/user/a.hajialiasgarynaj01/u14286/.project/dir.project/amirreza/for_Nikolai/all_mhc_clusters_esm.npz"
python train_tcrtyper.py --mode train   --train_ds data_train/train_ds.h5 --valid_ds data_train/valid_ds.h5   --donor_hla_matrix $DONOR_HLA_MATRIX_PATH   --idx_to_hla $IDX_TO_HLA   --hla_embed $HLA_EMBED   --output_dir outputs/exp01   --embed_dim 40 --num_heads 4 --num_layers 2   --batch_size 1024 --epochs 50 --lr 1e-3 --log_step 1 --use_tfrecord --num_shards 32