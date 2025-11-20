import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import sys

# Configuration
output_dir = sys.argv[1] if len(sys.argv) > 1 else 'tcr_visualizations'
os.makedirs(output_dir, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load data
print("Loading data... (this may take a while for large datasets)")
tcr_donor_ids = 'data/processed_data/delmonte2023_mitchell2022_musvosvi2022/processed/tcr_donor_ids.csv'
tcr_donor_df = pd.read_csv(tcr_donor_ids, dtype={'sample_id': str, 'cdr1': str, 'cdr2': str, 'cdr2.5': str, 'cdr3': str})
print(f"Data loaded: {len(tcr_donor_df):,} rows")

# IMPORTANT: Each row represents a UNIQUE combination of (CDR1, CDR2, CDR2.5, CDR3)
# This comes from assign_patient_ids_to_tcrs() which groups by all four CDR regions
# The 'sample_id' column contains semicolon-separated patient IDs who share this EXACT TCR
print("\n" + "="*60)
print("PUBLIC TCR DEFINITION")
print("="*60)
print("Public TCRs = TCRs with IDENTICAL CDR1 + CDR2 + CDR2.5 + CDR3")
print("              found in 2 or more patients")
print("Private TCRs = TCRs found in only 1 patient")
print("="*60 + "\n")

# Validate that we have all CDR columns
required_cols = ['cdr1', 'cdr2', 'cdr2.5', 'cdr3', 'sample_id']
assert all(col in tcr_donor_df.columns for col in required_cols), \
    f"Missing required columns! Found: {tcr_donor_df.columns.tolist()}"

# Verify uniqueness of TCR combinations
print("Verifying TCR uniqueness...")
tcr_combination = tcr_donor_df[['cdr1', 'cdr2', 'cdr2.5', 'cdr3']].copy()
duplicate_tcrs = tcr_combination.duplicated().sum()
if duplicate_tcrs > 0:
    print(f"WARNING: Found {duplicate_tcrs} duplicate TCR combinations!")
    print("This suggests the data may not be properly aggregated.")
else:
    print("✓ Confirmed: Each row is a unique (CDR1, CDR2, CDR2.5, CDR3) combination")

# Efficient calculation of number of patients (vectorized)
print("\nCalculating TCR sharing statistics...")
tcr_donor_df['num_patients'] = tcr_donor_df['sample_id'].str.count(';') + 1
tcr_donor_df['tcr_type'] = (tcr_donor_df['num_patients'] >= 2).map({True: 'Public', False: 'Private'})

# Calculate CDR3 length only (vectorized) - for length analysis, not for defining public/private
tcr_donor_df['cdr3_length'] = tcr_donor_df['cdr3'].str.count(';') + 1

# Count total unique patients
print("\nCounting unique patients...")
unique_patients = tcr_donor_df['sample_id'].str.split(';').explode().nunique()
total_patients = unique_patients
print(f"Total unique patients: {unique_patients:,}")

# Calculate fraction of patients for each TCR (for weighting)
tcr_donor_df['patient_fraction'] = tcr_donor_df['num_patients'] / total_patients

print("\n=== Dataset Summary ===")
print(f"Total unique TCR combinations (CDR1+CDR2+CDR2.5+CDR3): {len(tcr_donor_df):,}")
print(f"Private TCRs (1 patient only): {(tcr_donor_df['tcr_type'] == 'Private').sum():,}")
print(f"Public TCRs (≥2 patients, EXACT match on all CDRs): {(tcr_donor_df['tcr_type'] == 'Public').sum():,}")
print(f"Max patients sharing identical TCR: {tcr_donor_df['num_patients'].max()}")

# Pre-compute value counts for efficiency
print("\nComputing frequency distributions...")
sharing_counts = tcr_donor_df['num_patients'].value_counts().sort_index()
type_counts = tcr_donor_df['tcr_type'].value_counts()
cdr3_length_counts = tcr_donor_df['cdr3_length'].value_counts().sort_index()

# CDR germline categories frequencies
print("Computing CDR germline category frequencies...")
cdr1_counts = tcr_donor_df['cdr1'].value_counts().sort_values(ascending=False)
cdr2_counts = tcr_donor_df['cdr2'].value_counts().sort_values(ascending=False)
cdr2_5_counts = tcr_donor_df['cdr2.5'].value_counts().sort_values(ascending=False)

print(f"Unique CDR1 categories: {len(cdr1_counts):,}")
print(f"Unique CDR2 categories: {len(cdr2_counts):,}")
print(f"Unique CDR2.5 categories: {len(cdr2_5_counts):,}")

# Calculate CDR rarity scores (inverse frequency)
total_tcrs = len(tcr_donor_df)
cdr1_rarity = (1 / cdr1_counts).to_dict()
cdr2_rarity = (1 / cdr2_counts).to_dict()
cdr2_5_rarity = (1 / cdr2_5_counts).to_dict()

##############################################
# Per-Patient Statistics
##############################################
print("\n" + "="*60)
print("CALCULATING PER-PATIENT STATISTICS")
print("="*60)
print("Expanding patient IDs (this will take time)...")

# Explode sample_id to create one row per (TCR, patient) pair
tcr_patient_expanded = tcr_donor_df.copy()
tcr_patient_expanded['sample_id_list'] = tcr_patient_expanded['sample_id'].str.split(';')
tcr_patient_expanded = tcr_patient_expanded.explode('sample_id_list')
tcr_patient_expanded.rename(columns={'sample_id_list': 'patient_id'}, inplace=True)
print(f"Expanded to {len(tcr_patient_expanded):,} (TCR, patient) pairs")

# Calculate per-patient statistics
print("Calculating per-patient metrics...")
patient_stats = []

for patient_id, group in tcr_patient_expanded.groupby('patient_id'):
    # 1. Total number of unique TCR combinations (CDR1+CDR2+CDR2.5+CDR3)
    total_tcrs = len(group)
    
    # 2. Total number of public TCRs (shared with ≥1 other patient)
    public_tcrs = (group['tcr_type'] == 'Public').sum()
    
    # 3. Weighted sum of public TCRs (weight = patient_fraction, private TCRs have weight 0)
    weighted_public = group[group['tcr_type'] == 'Public']['patient_fraction'].sum()
    
    # 4. CDR1, CDR2, CDR2.5 statistics
    # Count unique CDRs
    unique_cdr1 = group['cdr1'].nunique()
    unique_cdr2 = group['cdr2'].nunique()
    unique_cdr2_5 = group['cdr2.5'].nunique()
    
    # Fraction of total unique CDRs in the dataset
    cdr1_fraction = unique_cdr1 / len(cdr1_counts)
    cdr2_fraction = unique_cdr2 / len(cdr2_counts)
    cdr2_5_fraction = unique_cdr2_5 / len(cdr2_5_counts)
    
    # Average rarity score (uniqueness) of their CDRs
    cdr1_uniqueness = group['cdr1'].map(cdr1_rarity).mean()
    cdr2_uniqueness = group['cdr2'].map(cdr2_rarity).mean()
    cdr2_5_uniqueness = group['cdr2.5'].map(cdr2_5_rarity).mean()
    
    patient_stats.append({
        'patient_id': patient_id,
        'total_tcrs': total_tcrs,
        'public_tcrs': public_tcrs,
        'private_tcrs': total_tcrs - public_tcrs,
        'weighted_public_tcrs': weighted_public,
        'unique_cdr1': unique_cdr1,
        'unique_cdr2': unique_cdr2,
        'unique_cdr2_5': unique_cdr2_5,
        'cdr1_fraction': cdr1_fraction,
        'cdr2_fraction': cdr2_fraction,
        'cdr2_5_fraction': cdr2_5_fraction,
        'cdr1_uniqueness': cdr1_uniqueness,
        'cdr2_uniqueness': cdr2_uniqueness,
        'cdr2_5_uniqueness': cdr2_5_uniqueness
    })
    
    if len(patient_stats) % 100 == 0:
        print(f"  Processed {len(patient_stats)} patients...")

patient_stats_df = pd.DataFrame(patient_stats)
print(f"\nCompleted! Processed {len(patient_stats_df):,} patients")
patient_stats_df['fraction_of_public_tcrs'] = patient_stats_df['public_tcrs'] / patient_stats_df['total_tcrs']

# Save patient statistics
patient_stats_path = os.path.join(output_dir, 'patient_statistics.csv')
patient_stats_df.to_csv(patient_stats_path, index=False)
print(f"Saved patient statistics to: {patient_stats_path}")

##############################################
# Figure 1: TCR Sharing Overview
##############################################
print("\nGenerating Figure 1: TCR Sharing Overview...")
fig1 = plt.figure(figsize=(18, 6))

ax1 = plt.subplot(1, 3, 1)
ax1.bar(sharing_counts.index, sharing_counts.values, color='steelblue', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Number of Patients Sharing TCR', fontsize=12)
ax1.set_ylabel('Number of TCRs (log scale)', fontsize=12)
ax1.set_title('TCR Sharing Distribution\n(Exact match: CDR1+CDR2+CDR2.5+CDR3)', fontsize=13, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(1, 3, 2)
colors = ['#ff7f0e', '#2ca02c']
wedges, texts, autotexts = ax2.pie(type_counts.values, labels=type_counts.index, 
                                     autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100.*type_counts.sum()):,})',
                                     colors=colors, startangle=90, textprops={'fontsize': 11})
ax2.set_title('Public vs Private TCRs\n(All 4 CDR regions must match)', fontsize=13, fontweight='bold')

ax3 = plt.subplot(1, 3, 3)
cumulative = np.cumsum(sharing_counts.values)
ax3.plot(sharing_counts.index, cumulative, marker='o', linewidth=2, markersize=4, color='darkred')
ax3.fill_between(sharing_counts.index, cumulative, alpha=0.3, color='darkred')
ax3.set_xlabel('Number of Patients', fontsize=12)
ax3.set_ylabel('Cumulative TCR Count', fontsize=12)
ax3.set_title('Cumulative TCR Sharing', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(left=1)

plt.tight_layout()
fig1_path = os.path.join(output_dir, '01_tcr_sharing_overview.png')
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig1_path}")

##############################################
# Figure 2: CDR3 Length Distribution
##############################################
print("Generating Figure 2: CDR3 Length Distribution...")
fig2 = plt.figure(figsize=(14, 6))

ax1 = plt.subplot(1, 2, 1)
ax1.bar(cdr3_length_counts.index, cdr3_length_counts.values, color='coral', edgecolor='black', alpha=0.7, width=0.8)
median_len = tcr_donor_df['cdr3_length'].median()
ax1.axvline(median_len, color='red', linestyle='--', linewidth=2, label=f'Median: {median_len:.1f}')
ax1.set_xlabel('CDR3 Length (amino acids)', fontsize=12)
ax1.set_ylabel('Frequency (log scale)', fontsize=12)
ax1.set_title('CDR3 Length Distribution', fontsize=13, fontweight='bold')
ax1.set_yscale('log')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, axis='y')

ax2 = plt.subplot(1, 2, 2)
bp = ax2.boxplot([tcr_donor_df['cdr3_length']], vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('coral')
bp['boxes'][0].set_alpha(0.7)
ax2.set_ylabel('CDR3 Length (amino acids)', fontsize=12)
ax2.set_title('CDR3 Length Statistics', fontsize=13, fontweight='bold')
ax2.set_xticklabels(['CDR3'])
ax2.grid(True, alpha=0.3, axis='y')
stats_text = f"Mean: {tcr_donor_df['cdr3_length'].mean():.2f}\nMedian: {median_len:.0f}\nStd: {tcr_donor_df['cdr3_length'].std():.2f}"
ax2.text(1.35, tcr_donor_df['cdr3_length'].median(), stats_text, fontsize=11, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
fig2_path = os.path.join(output_dir, '02_cdr3_length_distribution.png')
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig2_path}")

##############################################
# Figure 3: CDR1 Germline Categories
##############################################
print("Generating Figure 3: CDR1 Germline Categories...")
top_n = min(100, len(cdr1_counts))
cdr1_top = cdr1_counts.head(top_n)
fig3 = plt.figure(figsize=(16, max(10, top_n//5)))
ax = fig3.add_subplot(111)
y_pos = np.arange(len(cdr1_top))
bars = ax.barh(y_pos, cdr1_top.values, color='skyblue', edgecolor='black', alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(cdr1_top.index, fontsize=8)
ax.set_xlabel('Frequency (log scale)', fontsize=12)
ax.set_title(f'Top {top_n} CDR1 Germline Sequences (Total: {len(cdr1_counts):,})', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()
plt.tight_layout()
fig3_path = os.path.join(output_dir, '03_cdr1_germline_categories.png')
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig3_path}")
cdr1_counts.to_csv(os.path.join(output_dir, 'cdr1_full_distribution.csv'), header=['count'])

##############################################
# Figure 4: CDR2 Germline Categories
##############################################
print("Generating Figure 4: CDR2 Germline Categories...")
cdr2_top = cdr2_counts.head(top_n)
fig4 = plt.figure(figsize=(16, max(10, top_n//5)))
ax = fig4.add_subplot(111)
y_pos = np.arange(len(cdr2_top))
bars = ax.barh(y_pos, cdr2_top.values, color='lightgreen', edgecolor='black', alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(cdr2_top.index, fontsize=8)
ax.set_xlabel('Frequency (log scale)', fontsize=12)
ax.set_title(f'Top {top_n} CDR2 Germline Sequences (Total: {len(cdr2_counts):,})', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()
plt.tight_layout()
fig4_path = os.path.join(output_dir, '04_cdr2_germline_categories.png')
plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig4_path}")
cdr2_counts.to_csv(os.path.join(output_dir, 'cdr2_full_distribution.csv'), header=['count'])

##############################################
# Figure 5: CDR2.5 Germline Categories
##############################################
print("Generating Figure 5: CDR2.5 Germline Categories...")
cdr2_5_top = cdr2_5_counts.head(top_n)
fig5 = plt.figure(figsize=(16, max(10, top_n//5)))
ax = fig5.add_subplot(111)
y_pos = np.arange(len(cdr2_5_top))
bars = ax.barh(y_pos, cdr2_5_top.values, color='plum', edgecolor='black', alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(cdr2_5_top.index, fontsize=8)
ax.set_xlabel('Frequency (log scale)', fontsize=12)
ax.set_title(f'Top {top_n} CDR2.5 Germline Sequences (Total: {len(cdr2_5_counts):,})', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()
plt.tight_layout()
fig5_path = os.path.join(output_dir, '05_cdr2_5_germline_categories.png')
plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig5_path}")
cdr2_5_counts.to_csv(os.path.join(output_dir, 'cdr2_5_full_distribution.csv'), header=['count'])

##############################################
# Figure 6: Top Shared TCRs
##############################################
print("Generating Figure 6: Top Shared TCRs...")
top_n_shared = min(30, (tcr_donor_df['num_patients'] >= 2).sum())
public_tcrs = tcr_donor_df.nlargest(top_n_shared, 'num_patients')
fig6 = plt.figure(figsize=(12, 10))
ax = fig6.add_subplot(111)
y_pos = np.arange(len(public_tcrs))
bars = ax.barh(y_pos, public_tcrs['num_patients'].values, color='teal', edgecolor='black', alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([f'TCR_{id}' for id in public_tcrs['tcr_id'].values], fontsize=9)
ax.set_xlabel('Number of Patients', fontsize=12)
ax.set_title(f'Top {top_n_shared} Most Shared TCRs\n(Identical CDR1+CDR2+CDR2.5+CDR3)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()
for bar, val in zip(bars, public_tcrs['num_patients'].values):
    ax.text(val, bar.get_y() + bar.get_height()/2, f' {val}', 
            va='center', fontsize=9)
plt.tight_layout()
fig6_path = os.path.join(output_dir, '06_top_shared_tcrs.png')
plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig6_path}")

##############################################
# Figure 7: Per-Patient Statistics Overview
##############################################
print("Generating Figure 7: Per-Patient Statistics...")
fig7 = plt.figure(figsize=(20, 12))

# 7.1 Total TCRs per patient
ax1 = plt.subplot(3, 3, 1)
ax1.hist(patient_stats_df['total_tcrs'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Total TCRs', fontsize=11)
ax1.set_ylabel('Number of Patients', fontsize=11)
ax1.set_title('Total TCRs per Patient', fontsize=12, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# 7.2 Public TCRs per patient
ax2 = plt.subplot(3, 3, 2)
ax2.hist(patient_stats_df['public_tcrs'], bins=50, color='coral', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Public TCRs', fontsize=11)
ax2.set_ylabel('Number of Patients', fontsize=11)
ax2.set_title('Public TCRs per Patient', fontsize=12, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

# 7.3 Weighted public TCRs
ax3 = plt.subplot(3, 3, 3)
ax3.hist(patient_stats_df['weighted_public_tcrs'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
ax3.set_xlabel('Weighted Public TCRs', fontsize=11)
ax3.set_ylabel('Number of Patients', fontsize=11)
ax3.set_title('Weighted Public TCRs per Patient', fontsize=12, fontweight='bold')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

# 7.4 Unique CDR1 per patient
ax4 = plt.subplot(3, 3, 4)
ax4.hist(patient_stats_df['unique_cdr1'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax4.set_xlabel('Unique CDR1 Sequences', fontsize=11)
ax4.set_ylabel('Number of Patients', fontsize=11)
ax4.set_title('Unique CDR1 per Patient', fontsize=12, fontweight='bold')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)

# 7.5 Unique CDR2 per patient
ax5 = plt.subplot(3, 3, 5)
ax5.hist(patient_stats_df['unique_cdr2'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
ax5.set_xlabel('Unique CDR2 Sequences', fontsize=11)
ax5.set_ylabel('Number of Patients', fontsize=11)
ax5.set_title('Unique CDR2 per Patient', fontsize=12, fontweight='bold')
ax5.set_yscale('log')
ax5.grid(True, alpha=0.3)

# 7.6 Unique CDR2.5 per patient
ax6 = plt.subplot(3, 3, 6)
ax6.hist(patient_stats_df['unique_cdr2_5'], bins=50, color='plum', edgecolor='black', alpha=0.7)
ax6.set_xlabel('Unique CDR2.5 Sequences', fontsize=11)
ax6.set_ylabel('Number of Patients', fontsize=11)
ax6.set_title('Unique CDR2.5 per Patient', fontsize=12, fontweight='bold')
ax6.set_yscale('log')
ax6.grid(True, alpha=0.3)

# 7.7 CDR1 uniqueness score
ax7 = plt.subplot(3, 3, 7)
ax7.hist(patient_stats_df['cdr1_uniqueness'], bins=50, color='orange', edgecolor='black', alpha=0.7)
ax7.set_xlabel('CDR1 Uniqueness Score', fontsize=11)
ax7.set_ylabel('Number of Patients', fontsize=11)
ax7.set_title('CDR1 Uniqueness per Patient', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)

# 7.8 CDR2 uniqueness score
ax8 = plt.subplot(3, 3, 8)
ax8.hist(patient_stats_df['cdr2_uniqueness'], bins=50, color='pink', edgecolor='black', alpha=0.7)
ax8.set_xlabel('CDR2 Uniqueness Score', fontsize=11)
ax8.set_ylabel('Number of Patients', fontsize=11)
ax8.set_title('CDR2 Uniqueness per Patient', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)

# 7.9 CDR2.5 uniqueness score
ax9 = plt.subplot(3, 3, 9)
ax9.hist(patient_stats_df['cdr2_5_uniqueness'], bins=50, color='lavender', edgecolor='black', alpha=0.7)
ax9.set_xlabel('CDR2.5 Uniqueness Score', fontsize=11)
ax9.set_ylabel('Number of Patients', fontsize=11)
ax9.set_title('CDR2.5 Uniqueness per Patient', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3)

plt.tight_layout()
fig7_path = os.path.join(output_dir, '07_per_patient_statistics.png')
plt.savefig(fig7_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig7_path}")

##############################################
# Summary Statistics to File
##############################################
print("\nGenerating summary statistics file...")
stats_file = os.path.join(output_dir, 'summary_statistics.txt')
with open(stats_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("TCR DATA SUMMARY STATISTICS\n")
    f.write("="*60 + "\n\n")
    
    f.write("PUBLIC TCR DEFINITION:\n")
    f.write("  Public TCRs = EXACT match on CDR1 + CDR2 + CDR2.5 + CDR3\n")
    f.write("                found in ≥2 patients\n")
    f.write("  Private TCRs = Found in only 1 patient\n\n")
    
    f.write(f"Total unique TCR combinations: {len(tcr_donor_df):,}\n")
    f.write(f"Private TCRs (1 patient): {(tcr_donor_df['tcr_type'] == 'Private').sum():,}\n")
    f.write(f"Public TCRs (≥2 patients): {(tcr_donor_df['tcr_type'] == 'Public').sum():,}\n")
    f.write(f"Total unique patients: {unique_patients:,}\n\n")
    
    f.write("-"*60 + "\n")
    f.write("CDR3 LENGTH STATISTICS\n")
    f.write("-"*60 + "\n")
    f.write(f"Mean: {tcr_donor_df['cdr3_length'].mean():.2f}\n")
    f.write(f"Median: {tcr_donor_df['cdr3_length'].median():.2f}\n")
    f.write(f"Std Dev: {tcr_donor_df['cdr3_length'].std():.2f}\n")
    f.write(f"Min: {tcr_donor_df['cdr3_length'].min()}\n")
    f.write(f"Max: {tcr_donor_df['cdr3_length'].max()}\n\n")
    
    f.write("-"*60 + "\n")
    f.write("TCR SHARING STATISTICS\n")
    f.write("-"*60 + "\n")
    f.write(f"Mean patients per TCR: {tcr_donor_df['num_patients'].mean():.2f}\n")
    f.write(f"Median patients per TCR: {tcr_donor_df['num_patients'].median():.2f}\n")
    f.write(f"Max patients sharing a TCR: {tcr_donor_df['num_patients'].max()}\n\n")
    
    f.write("Sharing Percentiles:\n")
    for p in [50, 75, 90, 95, 99]:
        val = np.percentile(tcr_donor_df['num_patients'], p)
        f.write(f"  {p}th percentile: {val:.1f} patients\n")
    
    f.write("\n" + "-"*60 + "\n")
    f.write("GERMLINE CDR CATEGORIES\n")
    f.write("-"*60 + "\n")
    f.write(f"Unique CDR1 categories: {len(cdr1_counts):,}\n")
    f.write(f"Unique CDR2 categories: {len(cdr2_counts):,}\n")
    f.write(f"Unique CDR2.5 categories: {len(cdr2_5_counts):,}\n")
    
    f.write("\n" + "-"*60 + "\n")
    f.write("PER-PATIENT STATISTICS (AVERAGES)\n")
    f.write("-"*60 + "\n")
    f.write(f"Mean total TCRs per patient: {patient_stats_df['total_tcrs'].mean():.2f}\n")
    f.write(f"Mean public TCRs per patient: {patient_stats_df['public_tcrs'].mean():.2f}\n")
    f.write(f"Mean fraction of public TCRs: {patient_stats_df['fraction_of_public_tcrs'].mean():.4f}\n")
    f.write(f"Mean weighted public TCRs per patient: {patient_stats_df['weighted_public_tcrs'].mean():.4f}\n")
    f.write(f"Mean unique CDR1 per patient: {patient_stats_df['unique_cdr1'].mean():.2f}\n")
    f.write(f"Mean unique CDR2 per patient: {patient_stats_df['unique_cdr2'].mean():.2f}\n")
    f.write(f"Mean unique CDR2.5 per patient: {patient_stats_df['unique_cdr2_5'].mean():.2f}\n")
    f.write(f"Mean CDR1 uniqueness score: {patient_stats_df['cdr1_uniqueness'].mean():.6f}\n")
    f.write(f"Mean CDR2 uniqueness score: {patient_stats_df['cdr2_uniqueness'].mean():.6f}\n")
    f.write(f"Mean CDR2.5 uniqueness score: {patient_stats_df['cdr2_5_uniqueness'].mean():.6f}\n")

print(f"  Saved: {stats_file}")

print("\n" + "="*60)
print("ALL VISUALIZATIONS COMPLETED!")
print("="*60)
print(f"Output directory: {output_dir}")
print(f"Generated files:")
print(f"  - 7 figures (PNG)")
print(f"  - 1 patient statistics CSV")
print(f"  - 3 full CDR distribution CSVs")
print(f"  - 1 summary statistics TXT")
print("\nREMINDER: Public TCRs = Identical CDR1+CDR2+CDR2.5+CDR3 in ≥2 patients")