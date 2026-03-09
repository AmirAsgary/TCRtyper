#!/usr/bin/env python3
"""
Aggregate results from all bX_nY runs and generate comparison heatmaps.
Reads final_metrics_summary.json and precision_at_k.json from each subfolder.
Produces multi-page PDFs when many k-values are requested (4 panels per page).
Usage:
    python aggregate_heatmaps.py \
        --results_dir outputs/synthetic_25_2_25_withreg0001 \
        --k_values 1 3 5 10
"""
import os, json, re, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path


# ═════════════════════════════════════════════════════════════════════
# Constants for multi-page layout
# ═════════════════════════════════════════════════════════════════════
PANELS_PER_PAGE_COLS = 2  # columns per PDF page
PANELS_PER_PAGE_ROWS = 2  # rows per PDF page
PANELS_PER_PAGE = PANELS_PER_PAGE_COLS * PANELS_PER_PAGE_ROWS  # 4 panels/page


def parse_args():
    """Parse command-line arguments for heatmap aggregation."""
    parser = argparse.ArgumentParser(
        description='Aggregate synthetic run results into heatmaps')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Root directory containing all bX_nY result folders')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 3, 5, 10],
                        help='k values for Precision@k heatmaps (default: 1 3 5 10)')
    parser.add_argument('--dpi', type=int, default=200,
                        help='DPI for saved PNG figures (default: 200)')
    parser.add_argument('--cmap', type=str, default='viridis',
                        help='Colormap for heatmaps (default: viridis)')
    return parser.parse_args()


def parse_folder_name(name):
    """Extract numeric b and n values from folder name like 'b10_n250'.
    Returns (b_val, n_val) as integers, or None if parsing fails.
    """
    match = re.match(r'^b(\d+)_n(\d+)$', name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def load_all_results(results_dir):
    """Scan all bX_nY subfolders and load metrics JSONs into a list of dicts.
    Each dict contains: b, n, and all metric values found.
    """
    results_dir = Path(results_dir)
    records = []
    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir():
            continue
        parsed = parse_folder_name(folder.name)
        if parsed is None:
            continue
        b_val, n_val = parsed
        record = {'b': b_val, 'n': n_val, 'folder': folder.name}
        # Load final_metrics_summary.json
        metrics_path = folder / 'final_metrics_summary.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            record.update(metrics)
        # Load precision_at_k.json
        pk_path = folder / 'precision_at_k.json'
        if pk_path.exists():
            with open(pk_path, 'r') as f:
                pk = json.load(f)
            # Flatten precision@k and recall@k into record
            if 'mean_precision_at_k' in pk:
                for k, v in pk['mean_precision_at_k'].items():
                    record[f'precision_at_{k}'] = v
            if 'mean_recall_at_k' in pk:
                for k, v in pk['mean_recall_at_k'].items():
                    record[f'recall_at_{k}'] = v
        records.append(record)
    print(f"Loaded {len(records)} runs from {results_dir}")
    return pd.DataFrame(records)


def build_heatmap_matrix(df, metric_col, b_values, n_values):
    """Pivot a metric column into a 2D numpy array indexed by (n_row, b_col).
    Returns the matrix with np.nan for missing entries.
    """
    matrix = np.full((len(n_values), len(b_values)), np.nan)
    for _, row in df.iterrows():
        b_idx = b_values.index(row['b']) if row['b'] in b_values else None
        n_idx = n_values.index(row['n']) if row['n'] in n_values else None
        if (b_idx is not None and n_idx is not None
                and metric_col in row and pd.notna(row[metric_col])):
            matrix[n_idx, b_idx] = row[metric_col]
    return matrix


def _format_annotation(matrix, decimal_places=2):
    """Build annotation string array: values rounded to `decimal_places`, blanks for NaN.
    Args:
        matrix: 2D numpy array of float values (may contain NaN).
        decimal_places: number of digits after the decimal point.
    Returns:
        2D numpy object array of formatted strings.
    """
    annot = np.empty_like(matrix, dtype=object)
    fmt_str = f'{{:.{decimal_places}f}}'
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i, j]):
                annot[i, j] = ''
            else:
                annot[i, j] = fmt_str.format(matrix[i, j])
    return annot


def plot_single_heatmap(ax, matrix, b_labels, n_labels, title, cmap='viridis',
                        vmin=None, vmax=None, decimal_places=2, annot_size=9):
    """Render one annotated heatmap on the given axes.
    Args:
        ax: matplotlib Axes to draw on.
        matrix: 2D numpy array (n_values x b_values).
        b_labels: list of x-axis tick labels.
        n_labels: list of y-axis tick labels.
        title: panel title string.
        cmap: colormap name.
        vmin/vmax: color scale bounds (None for auto).
        decimal_places: digits after decimal for annotations.
        annot_size: font size for cell annotations.
    """
    annot = _format_annotation(matrix, decimal_places=decimal_places)
    sns.heatmap(matrix, ax=ax, annot=annot, fmt='',
                xticklabels=b_labels, yticklabels=n_labels,
                cmap=cmap, vmin=vmin, vmax=vmax,
                annot_kws={'size': annot_size},
                linewidths=0.5, linecolor='white',
                cbar_kws={'shrink': 0.8})
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Binder Set Size (b)', fontsize=10)
    ax.set_ylabel('Number of Donors (n)', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)


def generate_metrics_heatmaps(df, b_values, n_values, output_dir,
                               cmap='viridis', dpi=200):
    """Generate the multi-panel heatmap figure for core metrics.
    Panels: AUC-ROC, PR-AUC, Best F1, Donor Score, Alleles/TCR, TCR Coverage.
    Saved as both PNG and single-page PDF.
    """
    b_labels = [f'b{b}' for b in b_values]
    n_labels = [f'n{n}' for n in n_values]
    # (column_name, title, vmin, vmax, decimal_places)
    metric_specs = [
        ('auc_roc',               'AUC-ROC',           0.5, 1.0, 2),
        ('average_precision',     'PR-AUC',            0.0, 1.0, 2),
        ('best_f1_score',         'Best F1 Score',     0.0, 1.0, 2),
        ('donor_explanation_mean','Donor Score',        0.0, 1.0, 2),
        ('avg_alleles_per_tcr',   'Alleles/TCR',       None, None, 2),
        ('tcr_coverage_pct',      'TCR Coverage (%)',   None, None, 2),
    ]
    # Keep only metrics present in data
    metric_specs = [s for s in metric_specs if s[0] in df.columns]
    n_metrics = len(metric_specs)
    if n_metrics == 0:
        print("No core metrics found, skipping metrics heatmaps.")
        return
    ncols = 2
    nrows = (n_metrics + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    for idx, (col, title, vmin, vmax, dp) in enumerate(metric_specs):
        matrix = build_heatmap_matrix(df, col, b_values, n_values)
        plot_single_heatmap(axes[idx], matrix, b_labels, n_labels,
                            title=title, cmap=cmap, vmin=vmin, vmax=vmax,
                            decimal_places=dp)
    # Hide unused axes
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle('Synthetic Benchmark — Core Metrics',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    # Save PNG
    png_path = os.path.join(output_dir, 'metrics_heatmaps.png')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    # Save PDF
    pdf_path = os.path.join(output_dir, 'metrics_heatmaps.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def _save_multipage_heatmaps(panels, b_labels, n_labels, suptitle_prefix,
                              output_dir, filename_stem, cmap='viridis',
                              dpi=200):
    """Generate a multi-page PDF of heatmap panels (4 per page, 2x2 grid).
    Also saves a single PNG with all panels for quick preview.
    Args:
        panels: list of (matrix, title, vmin, vmax, decimal_places) tuples.
        b_labels: x-axis tick labels.
        n_labels: y-axis tick labels.
        suptitle_prefix: prefix for page titles (e.g. 'Precision@k').
        output_dir: directory to write output files.
        filename_stem: base name without extension (e.g. 'precision_at_k_heatmaps').
        cmap: colormap name.
        dpi: resolution for PNG output.
    """
    if not panels:
        return
    n_panels = len(panels)
    n_pages = (n_panels + PANELS_PER_PAGE - 1) // PANELS_PER_PAGE
    pdf_path = os.path.join(output_dir, f'{filename_stem}.pdf')
    # ── Multi-page PDF (4 panels per page) ───────────────────────────
    with PdfPages(pdf_path) as pdf:
        for page_idx in range(n_pages):
            start = page_idx * PANELS_PER_PAGE
            end = min(start + PANELS_PER_PAGE, n_panels)
            page_panels = panels[start:end]
            n_on_page = len(page_panels)
            # Always create a 2x2 grid per page
            fig, axes = plt.subplots(
                PANELS_PER_PAGE_ROWS, PANELS_PER_PAGE_COLS,
                figsize=(7 * PANELS_PER_PAGE_COLS, 5 * PANELS_PER_PAGE_ROWS))
            axes = axes.flatten()
            # Plot each panel on this page
            for pidx, (matrix, title, vmin, vmax, dp) in enumerate(page_panels):
                plot_single_heatmap(
                    axes[pidx], matrix, b_labels, n_labels,
                    title=title, cmap=cmap, vmin=vmin, vmax=vmax,
                    decimal_places=dp, annot_size=8)
            # Hide unused axes on the last page
            for pidx in range(n_on_page, PANELS_PER_PAGE):
                axes[pidx].set_visible(False)
            # Page title with page number
            page_label = (f'{suptitle_prefix} — Page {page_idx + 1}/{n_pages}'
                          if n_pages > 1 else suptitle_prefix)
            fig.suptitle(page_label, fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f"Saved: {pdf_path} ({n_pages} page{'s' if n_pages > 1 else ''}, "
          f"{n_panels} panels)")
    # ── Single PNG with all panels (compact overview) ────────────────
    total_cols = min(n_panels, PANELS_PER_PAGE_COLS)
    total_rows = (n_panels + total_cols - 1) // total_cols
    fig, axes = plt.subplots(
        total_rows, total_cols,
        figsize=(7 * total_cols, 5 * total_rows))
    # Ensure axes is always a flat array
    if n_panels == 1:
        axes = [axes]
    else:
        axes = np.asarray(axes).flatten()
    for pidx, (matrix, title, vmin, vmax, dp) in enumerate(panels):
        plot_single_heatmap(
            axes[pidx], matrix, b_labels, n_labels,
            title=title, cmap=cmap, vmin=vmin, vmax=vmax,
            decimal_places=dp, annot_size=7)
    for pidx in range(n_panels, len(axes)):
        axes[pidx].set_visible(False)
    fig.suptitle(suptitle_prefix, fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    png_path = os.path.join(output_dir, f'{filename_stem}.png')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {png_path}")


def generate_precision_at_k_heatmaps(df, b_values, n_values, k_values,
                                      output_dir, cmap='viridis', dpi=200):
    """Generate Precision@k heatmaps as a multi-page PDF (4 per page)."""
    b_labels = [f'b{b}' for b in b_values]
    n_labels = [f'n{n}' for n in n_values]
    # Filter to k values present in data
    available_k = [k for k in k_values if f'precision_at_{k}' in df.columns]
    if not available_k:
        print("No Precision@k data found, skipping.")
        return
    # Build panel list: (matrix, title, vmin, vmax, decimal_places)
    panels = []
    for k in available_k:
        col = f'precision_at_{k}'
        matrix = build_heatmap_matrix(df, col, b_values, n_values)
        panels.append((matrix, f'Precision@{k}', 0.0, 1.0, 2))
    _save_multipage_heatmaps(
        panels, b_labels, n_labels,
        suptitle_prefix='Synthetic Benchmark — Precision@k',
        output_dir=output_dir,
        filename_stem='precision_at_k_heatmaps',
        cmap=cmap, dpi=dpi)


def generate_recall_at_k_heatmaps(df, b_values, n_values, k_values,
                                    output_dir, cmap='viridis', dpi=200):
    """Generate Recall@k heatmaps as a multi-page PDF (4 per page)."""
    b_labels = [f'b{b}' for b in b_values]
    n_labels = [f'n{n}' for n in n_values]
    # Filter to k values present in data
    available_k = [k for k in k_values if f'recall_at_{k}' in df.columns]
    if not available_k:
        print("No Recall@k data found, skipping.")
        return
    # Build panel list: (matrix, title, vmin, vmax, decimal_places)
    panels = []
    for k in available_k:
        col = f'recall_at_{k}'
        matrix = build_heatmap_matrix(df, col, b_values, n_values)
        panels.append((matrix, f'Recall@{k}', 0.0, 1.0, 2))
    _save_multipage_heatmaps(
        panels, b_labels, n_labels,
        suptitle_prefix='Synthetic Benchmark — Recall@k',
        output_dir=output_dir,
        filename_stem='recall_at_k_heatmaps',
        cmap=cmap, dpi=dpi)


def save_summary_csv(df, output_dir):
    """Save the full aggregated results table as CSV for downstream use."""
    out_path = os.path.join(output_dir, 'aggregated_results.csv')
    df_sorted = df.sort_values(['b', 'n']).reset_index(drop=True)
    df_sorted.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main():
    """Load all run results, generate heatmaps, and save summary CSV."""
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = results_dir / 'aggregate_plots'
    os.makedirs(output_dir, exist_ok=True)
    # Load all results into a dataframe
    df = load_all_results(results_dir)
    if df.empty:
        print("No valid bX_nY results found. Exiting.")
        return
    # Extract sorted unique b and n values for axis ordering
    b_values = sorted(df['b'].unique().tolist())
    n_values = sorted(df['n'].unique().tolist())
    print(f"  b values: {b_values}")
    print(f"  n values: {n_values}")
    print(f"  Grid: {len(n_values)} rows x {len(b_values)} cols\n")
    # Generate all heatmap figures
    generate_metrics_heatmaps(df, b_values, n_values, str(output_dir),
                              cmap=args.cmap, dpi=args.dpi)
    generate_precision_at_k_heatmaps(df, b_values, n_values, args.k_values,
                                      str(output_dir), cmap=args.cmap,
                                      dpi=args.dpi)
    generate_recall_at_k_heatmaps(df, b_values, n_values, args.k_values,
                                   str(output_dir), cmap=args.cmap,
                                   dpi=args.dpi)
    # Save combined CSV
    save_summary_csv(df, str(output_dir))
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()