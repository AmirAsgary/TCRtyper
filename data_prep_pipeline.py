"""
TCR and MHC Data Processing Pipeline

This script processes T-cell receptor (TCR) and MHC data from input files,
performs sequence mapping, and generates processed output files.
"""

import argparse
import pandas as pd
import numpy as np
import os
import sys
from src import utils


def parse_tuple(value):
    """Parse a tuple from string format '(x,y)' or 'x,y'."""
    value = value.strip('()').strip()
    parts = [int(x.strip()) for x in value.split(',')]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Tuple must have exactly 2 values, got {len(parts)}"
        )
    return tuple(parts)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process TCR and MHC data for machine learning pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('--output_dir', type=str, required=True, help='Output directory for processed data')
    required.add_argument('--patient_df_path', type=str, required=True, help='Path to patient index file (TSV/CSV)')
    required.add_argument('--mhc_maximum_num', type=int, required=True, help='Maximum number of MHC molecules')
    
    # Optional path argument
    parser.add_argument('--relpath_input', type=str, default=None, help='Relative path prefix to prepend to file paths')
    
    # Boolean flags
    parser.add_argument('--add_rel_path', action='store_true', default=False, help='Add relative path prefix to file paths')
    parser.add_argument('--assign_numeric_ids', action='store_true', default=True, help='Assign numeric IDs to samples')
    parser.add_argument('--no_assign_numeric_ids', dest='assign_numeric_ids', action='store_false', help='Do not assign numeric IDs to samples')
    parser.add_argument('--assign_cdr_columns', action='store_true', default=True, help='Assign CDR column names to dataframe')
    parser.add_argument('--no_assign_cdr_columns', dest='assign_cdr_columns', action='store_false', help='Do not assign CDR column names to dataframe')
    
    # Sequence mapping options
    parser.add_argument('--sequence_mapping', action='store_true', default=True, help='Enable sequence mapping to numeric representation')
    parser.add_argument('--no_sequence_mapping', dest='sequence_mapping', action='store_false', help='Disable sequence mapping')
    parser.add_argument('--cdr1_sequence', action='store_true', default=True, help='Map CDR1 sequences (requires --sequence_mapping)')
    parser.add_argument('--cdr2_sequence', action='store_true', default=True, help='Map CDR2 sequences (requires --sequence_mapping)')
    parser.add_argument('--cdr2_5_sequence', action='store_true', default=True, help='Map CDR2.5 sequences (requires --sequence_mapping)')
    
    # ReadandProcess options
    parser.add_argument('--read_and_process', action='store_true', help='Run ReadAndProcess pipeline')
    parser.add_argument('--mhc_num_allele_thr', type=parse_tuple, default=(7, 100), help='MHC allele threshold (min, max). Format: "8,20"')
    parser.add_argument('--tcr_tsv_path_col', type=str, default='relpath_tcr', help='Column name for TCR TSV file paths')
    parser.add_argument('--mhc_arr_path_col', type=str, default='relpath_mask', help='Column name for MHC array file paths')
    parser.add_argument('--cdr1_col', type=str, default='cdr1aa_gapped', help='Column name for CDR1 sequences')
    parser.add_argument('--cdr2_col', type=str, default='cdr2aa_gapped', help='Column name for CDR2 sequences')
    parser.add_argument('--cdr2_5_col', type=str, default='cdr2.5aa_gapped', help='Column name for CDR2.5 sequences')
    parser.add_argument('--cdr3_col', type=str, default='cdr3aa', help='Column name for CDR3 sequences')
    parser.add_argument('--sample_id', type=str, default='sample_id', help='Column name for sample/patient IDs')
    parser.add_argument('--pad_token', type=int, default=-2, help='Padding token value for sequence mapping')
    
    # Train/val split options
    parser.add_argument('--train_val_split', action='store_true', help='Perform train/validation split')
    parser.add_argument('--patients_remained_index', type=str, default=None, help='Path to existing patients_remained_index.csv')
    parser.add_argument('--tcr_donor_ids', type=str, default=None, help='Path to existing tcr_donor_ids.csv')
    parser.add_argument('--datasetwise', action='store_true', default=True, help='Perform dataset-wise train/val split')
    parser.add_argument('--no_datasetwise', dest='datasetwise', action='store_false', help='Skip dataset-wise split')
    parser.add_argument('--patientwise', action='store_true', default=True, help='Perform patient-wise train/val split')
    parser.add_argument('--no_patientwise', dest='patientwise', action='store_false', help='Skip patient-wise split')
    parser.add_argument('--K', type=int, default=5, help='Number of folds for K-fold cross-validation')
    
    args = parser.parse_args()
    
    # Validation
    if args.add_rel_path and not args.relpath_input:
        parser.error("--relpath_input is required when --add_rel_path is enabled")
    
    if args.train_val_split and not (args.datasetwise or args.patientwise):
        parser.error("At least one of --datasetwise or --patientwise must be enabled for train_val_split")
    
    return args


def main():
    """Main processing pipeline."""
    args = parse_args()
    
    print("=" * 70)
    print("TCR AND MHC DATA PROCESSING PIPELINE")
    print("=" * 70)
    
    # Create output directory
    print(f"\n[1/4] Setting up output directory")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"  → Output directory: {args.output_dir}")
    
    # Load patient dataframe
    print(f"\n[2/4] Loading patient index")
    
    if not os.path.exists(args.patient_df_path):
        print(f"✗ Error: Patient index file not found at {args.patient_df_path}")
        sys.exit(1)
    
    try:
        # Detect separator from file extension
        sep = '\t' if args.patient_df_path.endswith('.tsv') else ','
        patient_df = pd.read_csv(args.patient_df_path, sep=sep)
        print(f"  → Loaded {len(patient_df)} patients from: {args.patient_df_path}")
    except Exception as e:
        print(f"✗ Error loading patient data: {e}")
        sys.exit(1)
    
    # Validate required columns
    required_cols = [args.sample_id]
    if args.add_rel_path:
        required_cols.extend([args.tcr_tsv_path_col, args.mhc_arr_path_col])
    
    missing_cols = [col for col in required_cols if col not in patient_df.columns]
    if missing_cols:
        print(f"✗ Error: Missing required columns: {missing_cols}")
        print(f"  Available columns: {list(patient_df.columns)}")
        sys.exit(1)
    
    # Add relative paths if requested
    print(f"\n[3/4] Processing patient data")
    if args.add_rel_path:
        print(f"  → Adding relative path prefix: {args.relpath_input}")
        cols = [args.tcr_tsv_path_col, args.mhc_arr_path_col]
        
        # Ensure paths start with '/' if relpath_input doesn't end with it
        prefix = args.relpath_input if args.relpath_input.endswith('/') else args.relpath_input + '/'
        patient_df[cols] = patient_df[cols].astype(str).apply(lambda x: prefix + x)
    
    # Assign numeric IDs if requested
    if args.assign_numeric_ids:
        print(f"  → Assigning numeric sample IDs")
        patient_df['sample_id_str'] = patient_df[args.sample_id].astype(str)
        patient_df[args.sample_id] = pd.factorize(patient_df['sample_id_str'])[0]
        print(f"    {len(patient_df[args.sample_id].unique())} unique samples")
    
    # Assign CDR column names if requested
    if args.assign_cdr_columns:
        print(f"  → Assigning CDR column names")
        patient_df[args.cdr1_col] = args.cdr1_col 
        patient_df[args.cdr2_col] = args.cdr2_col 
        patient_df[args.cdr2_5_col] = args.cdr2_5_col 
        patient_df[args.cdr3_col] = args.cdr3_col 
    
    # Save processed patient index
    processed_patient_path = os.path.join(args.output_dir, 'patients_index_process.tsv')
    '''
    P = []
    for d in np.unique(patient_df.dataset):
        a = patient_df[patient_df['dataset']==d]
        a = a.iloc[-1:, :]
        print(a)
        P.append(a)

    patient_df = pd.concat(P)
    print(patient_df)
    '''
    try:
        patient_df.to_csv(processed_patient_path, sep='\t', index=False)
        print(f"  → Saved processed patient index to: {processed_patient_path}")
    except Exception as e:
        print(f"✗ Error saving processed patient index: {e}")
        sys.exit(1)
    
    # 1- Initialize ReadAndPreprocess class
    if args.read_and_process:
        print(f"\n[4/4] Running data preprocessing pipeline")
        print(f"  → MHC maximum number: {args.mhc_maximum_num}")
        print(f"  → MHC allele threshold: {args.mhc_num_allele_thr}")
        print(f"  → Sequence mapping: {args.sequence_mapping}")
        
        if args.sequence_mapping:
            print(f"    • CDR1: {args.cdr1_sequence}")
            print(f"    • CDR2: {args.cdr2_sequence}")
            print(f"    • CDR2.5: {args.cdr2_5_sequence}")
            print(f"    • Padding token: {args.pad_token}")
        
        try:
            rap = utils.ReadAndPreprocess(
                df=patient_df,
                csv_output_path=os.path.join(args.output_dir, 'processed'),
                pad_token=args.pad_token,
                tcr_tsv_path=args.tcr_tsv_path_col,
                mhc_arr_path=args.mhc_arr_path_col,
                cdr3_col=args.cdr3_col,
                cdr1_col=args.cdr1_col,
                cdr2_col=args.cdr2_col,
                cdr2_5_col=args.cdr2_5_col,
                id_col=args.sample_id,
                mhc_maximum_num=args.mhc_maximum_num,
                mhc_num_allele_thr=args.mhc_num_allele_thr
            )
            
            rap.call(
                sequence_mapping=args.sequence_mapping,
                cdr1_sequence=args.cdr1_sequence,
                cdr2_sequence=args.cdr2_sequence,
                cdr2_5_sequence=args.cdr2_5_sequence
            )
            print(f"  ✓ Preprocessing completed")
        except Exception as e:
            print(f"✗ Error during preprocessing: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f"\n[4/4] Skipping preprocessing (--read_and_process not set)")
    
    # 2- Train/val splitting
    if args.train_val_split:
        print(f"\n" + "=" * 70)
        print("TRAIN/VALIDATION SPLIT GENERATION")
        print("=" * 70)
        
        # Determine paths
        if args.patients_remained_index:
            patient_index_path = args.patients_remained_index
        else:
            patient_index_path = os.path.join(args.output_dir, 'processed', 'patients_remained_index.csv')
        
        if args.tcr_donor_ids:
            tcr_donor_id_path = args.tcr_donor_ids
        else:
            tcr_donor_id_path = os.path.join(args.output_dir, 'processed', 'tcr_seq.csv')
        
        # Validate paths
        if not os.path.exists(patient_index_path):
            print(f"✗ Error: Patient index not found at {patient_index_path}")
            print(f"  Run with --read_and_process first or provide --patients_remained_index")
            sys.exit(1)
        
        if not os.path.exists(tcr_donor_id_path):
            print(f"✗ Error: TCR donor IDs not found at {tcr_donor_id_path}")
            print(f"  Run with --read_and_process first or provide --tcr_donor_ids")
            sys.exit(1)
        
        print(f"\nConfiguration:")
        print(f"  → Patient index: {patient_index_path}")
        print(f"  → TCR donor IDs: {tcr_donor_id_path}")
        print(f"  → Dataset-wise split: {args.datasetwise}")
        print(f"  → Patient-wise split: {args.patientwise}")
        if args.patientwise:
            print(f"  → K-folds: {args.K}")
        
        output_path = os.path.join(args.output_dir, 'train_val_split')
        
        try:
            crossval = utils.CrossValGen2(
                patient_index_path, 
                tcr_donor_id_path, 
                output_path, 
                datasetwise=args.datasetwise, 
                random_patient_wise=args.patientwise, 
                K=args.K
            )
            crossval.call()
        except Exception as e:
            print(f"\n✗ Error during train/val split: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f"\nSkipping train/val split (--train_val_split not set)")
    
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == '__main__':
    main()