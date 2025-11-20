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
    required.add_argument('--relpath_input', type=str, required=True, help='Relative path to input data directory')
    required.add_argument('--mhc_maximum_num', type=int, required=True, help='Maximum number of MHC molecules')
    # Optional arguments with defaults
    parser.add_argument('--mhc_num_allele_thr', type=parse_tuple, default=(8, 20), help='MHC number of allele threshold as tuple (min, max). Format: "8,20" or "(8,20)"')
    parser.add_argument('--tcr_tsv_path_col', type=str, default='relpath_tcr', help='Column name for TCR TSV file paths')
    parser.add_argument('--mhc_arr_path_col', type=str, default='relpath_mask', help='Column name for MHC array file paths')
    parser.add_argument('--cdr1_col', type=str, default='cdr1aa_gapped', help='Column name for CDR1 sequences')
    parser.add_argument('--cdr2_col', type=str, default='cdr2aa_gapped', help='Column name for CDR2 sequences')
    parser.add_argument('--cdr2_5_col', type=str, default='cdr2.5aa_gapped', help='Column name for CDR2.5 sequences')
    parser.add_argument('--cdr3_col', type=str, default='cdr3aa', help='Column name for CDR3 sequences')
    parser.add_argument('--sample_id', type=str, default='sample_id', help='Column name for sample/patient IDs')
    parser.add_argument('--pad_token', type=int, default=-1, help='Padding token value for sequence mapping')
    # Boolean flags
    parser.add_argument('--add_rel_path', action='store_true', default=True, help='Add relative path prefix to file paths')
    parser.add_argument('--no_add_rel_path', dest='add_rel_path', action='store_false', help='Do not add relative path prefix to file paths')
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
    # train val options
    parser.add_argument('--train_val_split', action='store_true', help='Weather do train val split or not')
    parser.add_argument('--patients_remained_index', type=str, default=None, help='PATH; If ReadandProcess is done once and there is a patients_remained_index.csv file already available')
    parser.add_argument('--tcr_donor_ids', type=str, default=None, help='PATH; If ReadandProcess is done once and there is a tcr_donor_ids.csv file already available')
    parser.add_argument('--no_datasetwise', action='store_false', help='no datasetwise train val split')
    parser.add_argument('--no_patientwise', action='store_false', help='no patientwise train val split')
    parser.add_argument('--K', type=int, default=5, help='K fold cross validation if patientwise is true, default=5')
    args = parser.parse_args()
    # Validation
    if args.add_rel_path and args.relpath_input is None:
        parser.error("--relpath_input is required when --add_rel_path is True")
    
    return args


def main():
    """Main processing pipeline."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load patient dataframe
    patient_df_path = os.path.join(args.relpath_input, 'patients_index.tsv')
    
    if not os.path.exists(patient_df_path):
        print(f"Error: Patient index file not found at {patient_df_path}")
        sys.exit(1)
    
    print(f"Loading patient data from: {patient_df_path}")
    patient_df = pd.read_csv(patient_df_path, sep='\t')
    print(f"Loaded {len(patient_df)} patients")
    
    # Add relative paths if requested
    if args.add_rel_path:
        print(f"Adding relative path prefix: {args.relpath_input}")
        cols = [args.tcr_tsv_path_col, args.mhc_arr_path_col]
        patient_df[cols] = patient_df[cols].astype(str).radd(args.relpath_input)
    
    # Assign numeric IDs if requested
    if args.assign_numeric_ids:
        print("Assigning numeric sample IDs")
        patient_df['sample_id_str'] = patient_df[args.sample_id]
        patient_df[args.sample_id] = pd.factorize(patient_df['sample_id_str'])[0]
    
    # Assign CDR column names if requested
    if args.assign_cdr_columns:
        print("Assigning CDR column names")
        patient_df[args.cdr1_col] = len(patient_df) * [args.cdr1_col]
        patient_df[args.cdr2_col] = len(patient_df) * [args.cdr2_col]
        patient_df[args.cdr2_5_col] = len(patient_df) * [args.cdr2_5_col]
        patient_df[args.cdr3_col] = len(patient_df) * [args.cdr3_col]
    
    # Save processed patient index
    processed_patient_path = os.path.join(args.output_dir, 'patients_index_process.tsv')
    patient_df.to_csv(processed_patient_path, sep='\t', index=False)
    print(f"Saved processed patient index to: {processed_patient_path}")
    
    ##################### 1- Initialize ReadAndPreprocess class
    print("\nInitializing data preprocessing...")
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
    
    # Run processing pipeline
    print("\nStarting TCR and MHC data processing...")
    print(f"  Sequence mapping: {args.sequence_mapping}")
    if args.sequence_mapping:
        print(f"  CDR1 mapping: {args.cdr1_sequence}")
        print(f"  CDR2 mapping: {args.cdr2_sequence}")
        print(f"  CDR2.5 mapping: {args.cdr2_5_sequence}")
    
    rap.call(
        sequence_mapping=args.sequence_mapping,
        cdr1_sequence=args.cdr1_sequence,
        cdr2_sequence=args.cdr2_sequence,
        cdr2_5_sequence=args.cdr2_5_sequence
    )
    ############################ 2- train val splitting
    if args.train_val_split:
        patient_index_path = os.path.join(args.output_dir, 'processed', 'patients_remained_index.csv') if not args.patients_remained_index else args.patients_remained_index
        tcr_donor_id_path = os.path.join(args.output_dir, 'processed', 'tcr_donor_ids.csv') if not args.tcr_donor_ids else args.tcr_donor_ids
        output_path = os.path.join(args.output_dir, 'train_val_split')
        crossval = utils.CrossValGen(patient_index_path, 
                                    tcr_donor_id_path, 
                                    output_path, 
                                    datasetwise=args.no_datasetwise, 
                                    random_patient_wise=args.no_patientwise, 
                                    K=args.K)
    print("\n✓ Pipeline completed successfully!")


if __name__ == '__main__':
    main()