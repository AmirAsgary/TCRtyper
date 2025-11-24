#!/bin/bash
#SBATCH --job-name=rp_val0
#SBATCH --partition=soeding
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12
#SBATCH --time=01-04:00:00
#SBATCH --nodes=1
#SBATCH --output=slurm.%x.%j.out
#SBATCH --error=slurm.%x.%j.err

# -----------------------------
# Parse arguments: --input and --output
# -----------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [[ -z "$INPUT_FILE" ]] || [[ -z "$OUTPUT_FILE" ]]; then
    echo "Usage: sbatch tfrecord_prep.sbatch --input <input.csv> --output <output.tfrecord>"
    exit 1
fi

echo "Input file:  $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"

# -----------------------------
# Environment setup
# -----------------------------
source ~/.bashrc
mamba activate TCRtyper

# -----------------------------
# Run script
# -----------------------------
python tfrecord_prep.py "$INPUT_FILE" "$OUTPUT_FILE"
