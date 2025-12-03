#!/bin/bash
#SBATCH -p scc-gpu              
#SBATCH -G A100:1               
#SBATCH -c 24                   
#SBATCH --mem=64G               
#SBATCH -J train                
#SBATCH -t 1-00:00:00           
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# Environment setup
source ~/.bashrc
mamba activate TCRtyper
module load gcc/13.2.0
module load cuda/11.8.0 
module load cudnn/8.9.7.29-11
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/sw/rev/25.04/rome_mofed_cuda80_rocky8/linux-rocky8-zen2/gcc-13.2.0/cuda-11.8.0-4oekvl532gamzsqnoiwlgjlpqqhhin6j


# Run the training
srun python train_model.py

