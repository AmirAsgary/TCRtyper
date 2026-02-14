#!/bin/bash
#SBATCH --job-name=mle_tcr
#SBATCH --array=0-4
#SBATCH --cpus-per-task=10
#SBATCH --mem=63G
#SBATCH --time=02-00:00:00
#SBATCH --output=/user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/outputs/real_data_2/logs/mle_job_%a_2.out
#SBATCH --error=/user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/outputs/real_data_2/logs/mle_job_%a_2.err
#SBATCH --partition=scc-gpu
#SBATCH -G A100:1


# %a in the filenames above gets replaced by the array task ID automatically
# for cpu
#conda activate /user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/envs/TCRtyper
#export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# for gpu required
module purge
source ~/.bashrc
conda activate /user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/envs/TCRtyper
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export CUDA_HOME=$CONDA_PREFIX
module load gcc/13.2.0-nvptx
module load cuda/12.6.2
module load cudnn/9.8.0.87-12
TOTAL_CHUNKS=38742
CHUNKS_PER_JOB=$(( (TOTAL_CHUNKS + 4) / 5 ))  # = 7749
START=$(( SLURM_ARRAY_TASK_ID * CHUNKS_PER_JOB ))
END=$(( START + CHUNKS_PER_JOB ))
export TMPDIR=/user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/tmp
mkdir -p $TMPDIR
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
# Clamp END to total
if [ $END -gt $TOTAL_CHUNKS ]; then
    END=$TOTAL_CHUNKS
fi

echo "Job $SLURM_ARRAY_TASK_ID: chunks $START to $END"
# Each task trains exactly ONE chunk.
# SLURM sets $SLURM_ARRAY_TASK_ID = 0 for first task, 1 for second, etc.
# my setting chunk_size, batch size and etc
python src/mle/mle_real_data.py \
    --h5_data_path /user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/data/autotcr/dataset_pval.h5 \
    --donor_matrix_path /user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/data/autotcr/donor_hla_matrix.npz \
    --output_dir /user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/TCRtyper/outputs/real_data_2 \
    --chunk_size 1000 \
    --mode train \
    --chunk_range $START $END \
    --resume \
    --device auto \
    --verbose 2 \
    --learning_rate 0.4 \
    --batch_size 1000 \
    --l2_reg 0.00002 \
    --reduction sum \
    --converge \
    --converge_patience 10 \
    --converge_tol 6e-5 \
    --converge_max_epochs 1000
