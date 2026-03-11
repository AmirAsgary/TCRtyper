#!/bin/bash
# Submit MLE pipeline jobs for all bX/nY combinations.
# Usage: bash submit_all_mle_jobs.sh
BASE_DATA="data/autotcr/synthetic/binder_set"
DONOR_MATRIX="data/autotcr/donor_hla_matrix.npz"
#OUTPUT_BASE="outputs/synthetic_test_b5_n1_reg0"
OUTPUT_BASE="outputs/cbscratch/synthetic_03_10_2026" # cbscratch is symblink
# Loop over all bX directories (skip non-directory entries like .json files)
for b_dir in "${BASE_DATA}"/b*/; do
    # Extract bX name (e.g. b10)
    b_name=$(basename "$b_dir")
    # Loop over all nY directories inside bX
    for n_dir in "${b_dir}"n*/; do
        n_name=$(basename "$n_dir")
        data_dir="${n_dir}N100000"
        # Skip if N100000 folder does not exist
        [ -d "$data_dir" ] || continue
        job_name="mle_${b_name}_${n_name}"
        out_dir="${OUTPUT_BASE}/${b_name}_${n_name}"
        # Create output dir before submission so SLURM can write logs there
        mkdir -p "${out_dir}"
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=soeding
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=01-04:00:00
#SBATCH --nodes=1
#SBATCH --output=${out_dir}/slurm.%x.%j.out
#SBATCH --error=${out_dir}/slurm.%x.%j.err
module purge
source ~/.bashrc
mamba activate /user/a.hajialiasgarynaj01/u14286/.project/dir.project/Amir/envs/TCRtyper
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export CUDA_HOME=$CONDA_PREFIX
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=2"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"
python src/mle/mle_pipeline_dense.py \
    --data_dir ${data_dir} \
    --donor_matrix ${DONOR_MATRIX} \
    --output_dir ${out_dir} \
    --epochs 150 \
    --batch_size 1000 \
    --analyze_all \
    --max_k 10 \
    --learning_rate 0.1 \
    --B 30 \
    --alpha 2 \
    --alpha_0 1.0 \
    --alpha_1 1.5 \
    --beta 4
EOF
        echo "Submitted: ${job_name} (${data_dir} -> ${out_dir})"
    done
done

#python src/mle/mle_pipeline.py --data_dir data/autotcr/synthetic/binder_set/b5/n10/N100000 --donor_matrix $DONOR_MATRIX --output_dir $OUTPUT_BASE --epochs 100 --batch_size 3000 --l2_reg 0.0 --analyze_all --max_k 10 --learning_rate 0.6