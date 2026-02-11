#!/bin/bash
#SBATCH --job-name=mle_tcr%a
#SBATCH --array=0-387
#SBATCH --cpus-per-task=50
#SBATCH --mem=100G
#SBATCH --time=04-00:00:00
#SBATCH --output=/user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/outputs/real_data_1/logs/chunk_%a.out
#SBATCH --error=/user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/outputs/real_data_1/logs/chunk_%a.err
#SBATCH --partition=soeding


# %a in the filenames above gets replaced by the array task ID automatically

# Activate your conda environment
conda activate /user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/envs/TCRtyper
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Each task trains exactly ONE chunk.
# SLURM sets $SLURM_ARRAY_TASK_ID = 0 for first task, 1 for second, etc.
python src/mle/mle_real_data.py \
    --h5_data_path /user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/data/autotcr/dataset_pval.h5 \
    --donor_matrix_path /user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/data/autotcr/donor_hla_matrix.npz \
    --output_dir /user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/outputs/real_data_1 \
    --chunk_size 100000 \
    --mode train \
    --chunk_id $SLURM_ARRAY_TASK_ID \
    --device cpu \
    --epochs 10 \
    --verbose 2