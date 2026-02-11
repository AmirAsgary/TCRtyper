# TCR-HLA Binding Model Pipeline

A probabilistic model training pipeline for TCR-HLA binding analysis using maximum likelihood estimation.

## Installation

```bash
pip install -r requirements.txt
```

**Note:** You also need the `dataset_processing` module with `PublicTcrHlaCsrReader` class.

## Files

- `utils.py` - Utility functions, model definition (SparseTCRModel), and analysis functions
- `pipeline.py` - Main CLI pipeline script for training and analysis
- `requirements.txt` - Python dependencies

## Usage

### Single Dataset

```bash
python pipeline.py \
    --data_dir /path/to/data/b10/n100/N100000 \
    --donor_matrix /path/to/donor_hla_matrix.npz \
    --output_dir /path/to/output \
    --analyze_all
```

### Multiple Datasets (via CSV/TSV/JSON config file)

Create a config file (e.g., `config.csv`):
```csv
data_dir,donor_matrix,name,l2_reg
/path/to/b10/n100/N100000,/path/to/donor_hla_matrix.npz,b10_n100,1e-5
/path/to/b25/n200/N100000,/path/to/donor_hla_matrix.npz,b25_n200,1e-5
```

Then run:
```bash
python pipeline.py \
    --df config.csv \
    --output_dir /path/to/output \
    --analyze_all
```

## Command Line Arguments

### Data Input (mutually exclusive)
- `--data_dir` - Path to single dataset directory
- `--df` - Path to CSV/TSV/JSON config file for multiple datasets

### Required
- `--output_dir` - Output directory for results
- `--donor_matrix` - Path to donor HLA matrix (required with `--data_dir`)

### Training Hyperparameters
- `--epochs` - Training epochs (default: 10)
- `--batch_size` - Batch size (default: 512)
- `--learning_rate` - Learning rate (default: 0.01)
- `--beta` - Beta hyperparameter (default: 4.0)
- `--l2_reg` - L2 regularization lambda (default: 1e-5)
- `--pad_token` - Padding token value (default: -1.0)
- `--threshold` - Decision threshold (default: 0.5)

### Analysis Flags
- `--analyze_all` - Run ALL analysis modules
- `--analyze_donors` - Run donor explanation analysis
- `--analyze_predictions` - Run model predictions analysis  
- `--analyze_performance` - Run PR/ROC performance evaluation
- `--analyze_precision_k` - Run Precision@k analysis
- `--max_k` - Max k for Precision@k (default: 20)

### Other
- `--seed` - Random seed (default: 42)
- `--verbose` - Verbosity level (default: 1)

## Config File Format

For batch processing, use CSV, TSV, or JSON with these columns:

| Column | Required | Description |
|--------|----------|-------------|
| data_dir | Yes | Path to dataset directory |
| donor_matrix | Yes | Path to donor HLA matrix |
| name | No | Identifier for this run |
| l2_reg | No | Override L2 regularization |
| epochs | No | Override number of epochs |
| batch_size | No | Override batch size |
| learning_rate | No | Override learning rate |
| beta | No | Override beta parameter |

## Output Structure

```
output_dir/
├── config.json              # Run configuration
├── model.keras              # Trained model
├── history.json             # Training history
├── final_metrics_summary.json
├── precision_at_k.json      # If precision@k enabled
└── figures/
    ├── analysis_plots.png/pdf
    ├── analysis_report.txt
    ├── analysis_arrays.npz
    ├── donor_explanation_plots.png/pdf
    ├── donor_explanation_report.txt
    ├── explanation_curves.csv
    ├── donor_scores_matrix.npz
    ├── performance_curves.png/pdf
    ├── curve_data.npz
    └── threshold_optimization.csv
```

## Examples

### Basic Training Only
```bash
python pipeline.py \
    --data_dir ./data/b10/n100/N100000 \
    --donor_matrix ./data/donor_hla_matrix.npz \
    --output_dir ./output/run1 \
    --epochs 20
```

### Full Analysis with Custom Parameters
```bash
python pipeline.py \
    --data_dir ./data/b25/n200/N100000 \
    --donor_matrix ./data/donor_hla_matrix.npz \
    --output_dir ./output/run2 \
    --epochs 15 \
    --batch_size 1024 \
    --l2_reg 1e-4 \
    --analyze_all \
    --max_k 30
```

### Only Performance Analysis
```bash
python pipeline.py \
    --data_dir ./data/b10/n100/N100000 \
    --donor_matrix ./data/donor_hla_matrix.npz \
    --output_dir ./output/run3 \
    --analyze_performance
```
