"""
Utility functions and model definitions for TCR-HLA binding analysis.

Requirements:
    - tensorflow>=2.10
    - tensorflow_probability
    - numpy
    - matplotlib
    - seaborn
    - pandas (for precision@k heatmaps)

Author: TCR-HLA Binding Analysis Pipeline
"""
import os, json
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- MODEL DEFINITION ---
class SparseTCRModel(tf.keras.Model):
    """TCR Binding Model using sparse representation with likelihood maximization."""
    def __init__(self, num_tcrs, max_hlas_per_tcr, donor_hla_matrix, binder_sets, 
                 beta=4.0, mode='continuous', pad_token=-1., l2_reg_lambda=1e-5):
        super().__init__()
        self.beta = beta
        self.mode = mode
        self.pad_token = pad_token
        self.l2_reg_lambda = l2_reg_lambda
        # Store donor matrix transposed: (NumAlleles, NumDonors)
        self.X_T = tf.constant(donor_hla_matrix.T, dtype=tf.float32)
        # Store binder_sets with pad replaced by 0 for gathering
        self.binder_sets = tf.constant(np.maximum(binder_sets, 0), dtype=tf.int32)
        self.binder_mask = tf.constant(binder_sets != pad_token, dtype=tf.float32)
        # Embedding layer for z parameters
        self.z_embedding = tf.keras.layers.Embedding(
            input_dim=num_tcrs, output_dim=max_hlas_per_tcr,
            embeddings_initializer=tf.keras.initializers.RandomNormal(mean=-1.25, stddev=0.75),
            name="z_values")
        # Metric trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.final_loss_tracker = tf.keras.metrics.Mean(name="final_loss")
        self.reg_tracker = tf.keras.metrics.Mean(name="reg_term")

    def l2_reg(self, z_logits, mask):
        if self.l2_reg_lambda:
            return self.l2_reg_lambda * tf.reduce_sum(tf.pow(z_logits, 2) * mask)
        return 0.

    def get_z_probabilities(self, z_logits, mask):
        """Toggle between continuous relaxation and binary sampling."""
        if self.mode == 'continuous':
            return tf.sigmoid(z_logits) * mask
        elif self.mode == 'gumbel':
            dist = tfp.distributions.RelaxedBernoulli(temperature=0.5, logits=z_logits)
            z_sampled = dist.sample()
            z_hard = tf.cast(tf.greater(z_sampled, 0.5), tf.float32)
            z_final = tf.stop_gradient(z_hard - z_sampled) + z_sampled
            return z_final * mask
        raise ValueError(f"Unknown mode: {self.mode}")

    def call(self, inputs):
        """Compute negative log-likelihood loss for a batch of TCRs."""
        tcr_idx, pos_donor_indices = inputs
        batch_binder_indices = tf.gather(self.binder_sets, tcr_idx)
        batch_mask = tf.gather(self.binder_mask, tcr_idx)
        z_logits = self.z_embedding(tcr_idx)
        # Compute p_ni (Prob TCR i binds in Donor n)
        relevant_x = tf.gather(self.X_T, batch_binder_indices)
        z_prob = self.get_z_probabilities(z_logits, batch_mask)
        z_prob_expanded = tf.expand_dims(z_prob, axis=-1)
        regularization_term = self.l2_reg(z_logits, batch_mask)
        term_raw = 1.0 - (relevant_x * z_prob_expanded)
        term_safe = tf.maximum(term_raw, 1e-7)
        log_prod = tf.reduce_sum(tf.math.log(term_safe), axis=1)
        p_ni = 1.0 - tf.exp(log_prod)
        p_ni_safe = tf.maximum(p_ni, 1e-7)
        # Positive donors (Reward)
        safe_pos_indices = tf.maximum(pos_donor_indices, 0)
        pos_mask = tf.cast(tf.not_equal(pos_donor_indices, tf.cast(self.pad_token, tf.int32)), tf.float32)
        p_pos = tf.gather(p_ni_safe, safe_pos_indices, batch_dims=1)
        reward = tf.reduce_sum(tf.math.log(p_pos) * pos_mask, axis=1)
        # Negative donors (Penalty via Beta-Binomial)
        n_i = tf.reduce_sum(pos_mask, axis=1)
        sum_p_all = tf.reduce_sum(p_ni_safe, axis=1)
        sum_p_pos = tf.reduce_sum(p_pos * pos_mask, axis=1)
        n_tilde = sum_p_all - sum_p_pos
        penalty = tf.math.lgamma(n_tilde + self.beta) - tf.math.lgamma(n_i + n_tilde + self.beta + 1.0)
        log_likelihood = reward + penalty
        return -tf.reduce_mean(log_likelihood), regularization_term

    @property
    def metrics(self):
        return [self.loss_tracker, self.final_loss_tracker, self.reg_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss, reg_term = self(data, training=True)
            final_loss = loss + reg_term
        gradients = tape.gradient(final_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.final_loss_tracker.update_state(final_loss)
        self.reg_tracker.update_state(reg_term)
        return {m.name: m.result() for m in self.metrics}


def create_dataset(donor_indices, batch_size):
    """Create TF dataset from donor indices."""
    tcr_ids = np.arange(donor_indices.shape[0], dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((tcr_ids, donor_indices))
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def pad_list_to_array(counts_set, max_all, pad_token=-1.):
    """Pad variable-length lists to fixed array."""
    n_samples = len(counts_set)
    result = np.full((n_samples, max_all), pad_token)
    for i, row in enumerate(counts_set):
        result[i, :len(row)] = row
    return result


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def assess_explanation_for_donors(model, donor_indices, donor_hla_matrix, batch_size=1024, 
                                   output_path=None, pad_token=-1.):
    """Check if we can explain TCR presence in donors at various strictness levels."""
    if output_path: os.makedirs(output_path, exist_ok=True)
    report_lines = []
    def log(msg):
        print(msg)
        if output_path: report_lines.append(msg)
    
    log(f"\n{'='*60}\nASSESSING DONOR EXPLANATION\n{'='*60}")
    num_tcrs = donor_indices.shape[0]
    donor_hla_tensor = tf.constant(donor_hla_matrix, dtype=tf.float32)
    all_donor_scores = []
    
    log(f"Processing {num_tcrs} TCRs in batches of {batch_size}...")
    num_batches = int(np.ceil(num_tcrs / batch_size))
    
    for i in range(num_batches):
        start, end = i * batch_size, min((i + 1) * batch_size, num_tcrs)
        tcr_indices = tf.range(start, end, dtype=tf.int32)
        z_logits = model.z_embedding(tcr_indices)
        probs = tf.sigmoid(z_logits) * tf.gather(model.binder_mask, tcr_indices)
        candidates = tf.gather(model.binder_sets, tcr_indices)
        batch_donors = donor_indices[start:end]
        valid_donor_mask = tf.not_equal(batch_donors, tf.cast(pad_token, tf.int32))
        safe_donor_ids = tf.maximum(batch_donors, 0)
        batch_donor_hlas = tf.gather(donor_hla_tensor, safe_donor_ids)
        candidates_tiled = tf.tile(tf.expand_dims(candidates, 1), [1, batch_donor_hlas.shape[1], 1])
        donor_has_candidate = tf.gather(batch_donor_hlas, candidates_tiled, batch_dims=2)
        probs_expanded = tf.expand_dims(probs, 1)
        explanation_scores = probs_expanded * donor_has_candidate
        max_score_per_donor = tf.reduce_max(explanation_scores, axis=2)
        scores_masked = tf.where(valid_donor_mask, max_score_per_donor, pad_token)
        all_donor_scores.append(scores_masked.numpy())
        if i % 10 == 0: print(f"  Batch {i}/{num_batches} done...", end='\r')
    
    donor_scores_matrix = np.concatenate(all_donor_scores)
    log("\n\nAnalysis Complete. Generating Report...")
    
    thresholds = np.linspace(0.01, 0.99, 100)
    curves = {level: [] for level in range(100, 9, -10)}
    curves[1] = []
    total_donors_per_tcr = np.maximum(np.sum(donor_scores_matrix != pad_token, axis=1), 1)
    
    for t in thresholds:
        is_explained = (donor_scores_matrix > t)
        num_explained = np.sum(is_explained, axis=1)
        fraction_explained = num_explained / total_donors_per_tcr
        for level in curves.keys():
            if level == 100: perc = np.mean(fraction_explained == 1.0) * 100
            elif level == 1: perc = np.mean(num_explained >= 1) * 100
            else: perc = np.mean(fraction_explained >= (level / 100.0)) * 100
            curves[level].append(perc)
    
    # Visualization
    fig = plt.figure(figsize=(20, 6))
    plt.subplot(1, 3, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, 11))
    for i, level in enumerate(range(100, 9, -10)):
        label_text = "100% Donors" if level == 100 else f"≥ {level}% Donors"
        plt.plot(thresholds, curves[level], color=colors[i], linewidth=2, label=label_text)
    plt.plot(thresholds, curves[1], color='grey', linewidth=2, linestyle='--', label="≥ 1 Donor")
    plt.title("Explanation Robustness Spectrum")
    plt.xlabel("Binarization Threshold")
    plt.ylabel("% TCRs satisfying condition")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    masked_scores = np.ma.masked_equal(donor_scores_matrix, pad_token)
    min_scores = np.min(masked_scores, axis=1).filled(0.0)
    plt.subplot(1, 3, 2)
    plt.hist(min_scores, bins=50, color='#d65f5f', edgecolor='white', range=(0,1))
    plt.title("Critical Score Distribution")
    plt.xlabel("Probability")
    plt.ylabel("Count of TCRs")
    
    check_t = 0.5
    is_explained_check = (donor_scores_matrix > check_t)
    fracs = np.sum(is_explained_check, axis=1) / total_donors_per_tcr
    plt.subplot(1, 3, 3)
    plt.hist(fracs * 100, bins=20, color='#4c72b0', edgecolor='white', range=(0,100))
    plt.title(f"Fraction of Donors Explained (T={check_t})")
    plt.xlabel("% of Donors Explained")
    plt.ylabel("Count of TCRs")
    plt.tight_layout()
    
    if output_path:
        fig.savefig(os.path.join(output_path, "donor_explanation_plots.png"), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(output_path, "donor_explanation_plots.pdf"), bbox_inches='tight')
        plt.close(fig)
        # Save report and data
        columns = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 1]
        with open(os.path.join(output_path, "donor_explanation_report.txt"), 'w') as f:
            f.write('\n'.join(report_lines))
        curve_data = np.column_stack([thresholds] + [curves[level] for level in columns])
        np.savetxt(os.path.join(output_path, "explanation_curves.csv"), curve_data, delimiter=',',
                   header='threshold,' + ','.join([f'pct_{c}_donors' for c in columns]), comments='')
        np.savez_compressed(os.path.join(output_path, "donor_scores_matrix.npz"),
                           donor_scores=donor_scores_matrix, thresholds=thresholds,
                           total_donors_per_tcr=total_donors_per_tcr)
    else:
        plt.show()
    
    # Summary statistics
    perfect_count = np.sum(fracs == 1.0)
    summary_stats = {
        'num_tcrs': num_tcrs, 'perfect_100pct': int(perfect_count),
        'mean_fraction_explained_t005': float(np.mean(fracs)),
        'median_fraction_explained_t005': float(np.median(fracs)),
    }
    return donor_scores_matrix, summary_stats


def analyze_model_predictions(model, binder_sets, num_total_alleles, threshold=0.5, 
                               output_path=None, pad_token=-1.):
    """Full analysis pipeline with visualizations."""
    if output_path: os.makedirs(output_path, exist_ok=True)
    report_lines = []
    def log(msg):
        print(msg)
        if output_path: report_lines.append(msg)
    
    log(f"\n{'='*50}\nSTARTING MODEL ANALYSIS\n{'='*50}")
    trained_logits = model.z_embedding.get_weights()[0]
    trained_probs = tf.sigmoid(trained_logits).numpy()
    valid_mask = (binder_sets != pad_token)
    viz_probs = trained_probs.copy()
    viz_probs[~valid_mask] = pad_token
    analysis_probs = trained_probs.copy()
    analysis_probs[~valid_mask] = 0.0
    
    # Threshold optimization
    log("\n--- Threshold Optimization Analysis ---")
    threshold_range = np.linspace(0.01, 0.999, 1000)
    coverages, avg_counts = [], []
    idx_99, idx_95, best_tradeoff_idx = -1, -1, -1
    max_tradeoff_score = -float('inf')
    
    for i, t in enumerate(threshold_range):
        matches = np.sum(analysis_probs > t, axis=1)
        cov = np.mean(matches > 0) * 100
        avg = np.mean(matches)
        coverages.append(cov)
        avg_counts.append(avg)
        if cov >= 99.0: idx_99 = i
        if cov >= 95.0: idx_95 = i
        score = cov - (5.0 * avg)
        if score > max_tradeoff_score:
            max_tradeoff_score, best_tradeoff_idx = score, i
    
    def print_stat(name, idx):
        if idx >= 0:
            log(f"Strategy: {name:<25} | Threshold: {threshold_range[idx]:.3f} | Coverage: {coverages[idx]:.2f}%")
    print_stat("'Strict' (99% Coverage)", idx_99)
    print_stat("'Relaxed' (95% Coverage)", idx_95)
    print_stat("'Balanced' (Elbow Point)", best_tradeoff_idx)
    
    # Current threshold stats
    log(f"\n--- Statistics for Threshold ({threshold}) ---")
    final_decisions = (analysis_probs > threshold)
    matches_per_tcr = np.sum(final_decisions, axis=1)
    current_coverage = np.mean(matches_per_tcr > 0) * 100
    current_avg = np.mean(matches_per_tcr)
    current_median = np.median(matches_per_tcr)
    current_max = np.max(matches_per_tcr)
    zero_matches = np.sum(matches_per_tcr == 0)
    log(f"Coverage: {current_coverage:.2f}% | Avg HLAs/TCR: {current_avg:.2f} | Zero matches: {zero_matches}")
    
    # Visualizations
    fig = plt.figure(figsize=(18, 12))
    plt.subplot(2, 2, 1)
    max_probs = np.max(viz_probs, axis=1)
    valid_plot_data = max_probs[max_probs >= 0.0]
    if len(valid_plot_data) > 0:
        plt.hist(valid_plot_data, bins=50, range=(0, 1), color='#4c72b0', edgecolor='white')
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'T={threshold}')
    plt.title("Distribution of Model Confidence")
    plt.legend()
    
    plt.subplot(2, 2, 2)
    ax1 = plt.gca()
    ax1.plot(threshold_range, coverages, 'b-', linewidth=2, label='Coverage %')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Coverage (%)', color='b')
    ax2 = ax1.twinx()
    ax2.plot(threshold_range, avg_counts, 'r--', linewidth=2, label='Avg HLAs/TCR')
    ax2.set_ylabel('Avg Count', color='r')
    plt.title("Optimization Curve")
    
    plt.subplot(2, 2, 3)
    chosen_allele_ids = binder_sets[final_decisions]
    if len(chosen_allele_ids) > 0:
        unique_ids, counts = np.unique(chosen_allele_ids, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        top_n = min(30, len(sorted_indices))
        plt.bar(range(top_n), counts[sorted_indices[:top_n]], color='#55a868')
        plt.title(f"Top {top_n} Predicted Alleles")
    
    plt.subplot(2, 2, 4)
    sample_size = min(20, len(viz_probs))
    random_indices = np.random.choice(len(viz_probs), sample_size, replace=False)
    full_heatmap_matrix = np.full((sample_size, num_total_alleles), 0.0)
    for i, idx in enumerate(random_indices):
        valid_idx = binder_sets[idx] != pad_token
        tcr_allele_ids = binder_sets[idx][valid_idx].astype(int)
        full_heatmap_matrix[i, tcr_allele_ids] = viz_probs[idx][valid_idx]
    plt.imshow(full_heatmap_matrix, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)
    plt.colorbar(label="Probability")
    plt.title(f"Binding Probabilities ({num_total_alleles} Alleles)")
    plt.tight_layout()
    
    if output_path:
        fig.savefig(os.path.join(output_path, "analysis_plots.png"), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(output_path, "analysis_plots.pdf"), bbox_inches='tight')
        plt.close(fig)
        with open(os.path.join(output_path, "analysis_report.txt"), 'w') as f:
            f.write('\n'.join(report_lines))
        np.savetxt(os.path.join(output_path, "threshold_optimization.csv"),
                   np.column_stack([threshold_range, coverages, avg_counts]), delimiter=',',
                   header='threshold,coverage_percent,avg_hlas_per_tcr', comments='')
        np.savez(os.path.join(output_path, "analysis_arrays.npz"), trained_probs=trained_probs,
                 analysis_probs=analysis_probs, final_decisions=final_decisions,
                 matches_per_tcr=matches_per_tcr, threshold_range=threshold_range,
                 coverages=np.array(coverages), avg_counts=np.array(avg_counts))
    else:
        plt.show()
    
    return {
        'coverage': current_coverage, 'avg_hlas_per_tcr': current_avg,
        'median_hlas_per_tcr': current_median, 'max_hlas_per_tcr': current_max,
        'tcrs_with_zero_hlas': zero_matches,
        'threshold_95_coverage': threshold_range[idx_95] if idx_95 >= 0 else None,
        'threshold_99_coverage': threshold_range[idx_99] if idx_99 >= 0 else None,
    }


def evaluate_model_performance(model, binder_sets, true_hla_set, num_total_alleles=440, 
                                output_path=None, pad_token=-1.):
    """Calculate PR Curve, ROC Curve, and statistical metrics (optimized sparse version)."""
    if output_path: os.makedirs(output_path, exist_ok=True)
    print(f"\n--- Performance Evaluation (PR & ROC) ---")
    
    z_logits = model.z_embedding.get_weights()[0]
    candidate_probs = tf.sigmoid(z_logits).numpy()
    num_tcrs = binder_sets.shape[0]
    
    # Build sparse representation
    valid_mask = (binder_sets != pad_token)
    pred_probs_sparse = candidate_probs[valid_mask]
    pred_allele_ids = binder_sets[valid_mask].astype(int)
    pred_tcr_ids = np.repeat(np.arange(num_tcrs), binder_sets.shape[1])[valid_mask.flatten()]
    
    # True allele lookup
    true_allele_sets = []
    for i in range(num_tcrs):
        valid_true = true_hla_set[i] >= 0
        true_allele_sets.append(set(true_hla_set[i][valid_true].astype(int)))
    
    # Create binary labels
    y_true_sparse = np.array([
        1 if pred_allele_ids[j] in true_allele_sets[pred_tcr_ids[j]] else 0
        for j in range(len(pred_probs_sparse))
    ], dtype=np.int32)
    
    total_true_positives = sum(len(s) for s in true_allele_sets)
    true_positives_in_candidates = np.sum(y_true_sparse)
    fn_from_non_candidates = total_true_positives - true_positives_in_candidates
    total_negatives = num_tcrs * num_total_alleles - total_true_positives
    
    # Sort by probability descending
    sorted_indices = np.argsort(-pred_probs_sparse)
    y_true_sorted = y_true_sparse[sorted_indices]
    y_pred_sorted = pred_probs_sparse[sorted_indices]
    
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    
    # Compute curves
    precision = tps / (tps + fps + 1e-7)
    recall = tps / (total_true_positives + 1e-7)
    fpr = fps / (total_negatives + 1e-7)
    tpr = recall
    
    # Metrics
    roc_auc = np.trapz(tpr, fpr)
    average_precision = np.sum(np.diff(np.concatenate([[0], recall])) * precision)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = y_pred_sorted[best_f1_idx] if best_f1_idx < len(y_pred_sorted) else 0.5
    best_f1 = f1_scores[best_f1_idx]
    
    print(f"AUC ROC: {roc_auc:.5f} | AP: {average_precision:.5f} | Best F1: {best_f1:.5f}")
    
    if output_path:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        axes[0].plot(recall, precision, color='#2ca02c', lw=2, label=f'AP = {average_precision:.3f}')
        axes[0].set_title('Precision-Recall Curve')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].legend()
        axes[1].plot(fpr, tpr, color='#1f77b4', lw=2, label=f'AUC = {roc_auc:.3f}')
        axes[1].plot([0, 1], [0, 1], 'gray', linestyle='--')
        axes[1].set_title('ROC Curve')
        axes[1].set_xlabel('FPR')
        axes[1].set_ylabel('TPR')
        axes[1].legend()
        plt.tight_layout()
        fig.savefig(os.path.join(output_path, "performance_curves.png"), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(output_path, "performance_curves.pdf"), bbox_inches='tight')
        plt.close(fig)
        np.savez(os.path.join(output_path, "curve_data.npz"), precision=precision, recall=recall,
                 fpr=fpr, tpr=tpr, y_pred_sorted=y_pred_sorted)
    
    return {'auc': roc_auc, 'ap': average_precision, 'best_f1': best_f1, 'best_threshold': best_threshold}


def compute_precision_at_k(output_dir, data_dir, max_k=20, pad_token=-1.):
    """Compute Precision@k and Recall@k for k=1,2,...,max_k."""
    from dataset_processing.utils import PublicTcrHlaCsrReader
    output_dir, data_dir = Path(output_dir), Path(data_dir)
    
    arrays_path = output_dir / "figures" / "analysis_arrays.npz"
    arrays = np.load(arrays_path)
    trained_probs = arrays['trained_probs']
    
    h5_path = data_dir / 'synthetic_tcr_hla_counts.h5'
    with PublicTcrHlaCsrReader(str(h5_path)) as reader:
        counts_set, max_all = reader.read_sparse_indices()
    
    num_tcrs = len(counts_set)
    binder_sets = np.full((num_tcrs, max_all), pad_token)
    for i, row in enumerate(counts_set):
        binder_sets[i, :len(row)] = row
    
    true_hla_set = np.load(data_dir / "synthetic_binder_sets.npy")
    true_allele_sets = []
    for i in range(num_tcrs):
        valid_mask = true_hla_set[i] >= 0
        true_allele_sets.append(set(true_hla_set[i][valid_mask].astype(int)))
    
    print(f"Computing Precision@k for k=1 to {max_k}...")
    precision_at_k = {k: [] for k in range(1, max_k + 1)}
    recall_at_k = {k: [] for k in range(1, max_k + 1)}
    
    for i in range(num_tcrs):
        valid_mask = binder_sets[i] != pad_token
        candidate_ids = binder_sets[i][valid_mask].astype(int)
        candidate_probs = trained_probs[i][valid_mask]
        sorted_indices = np.argsort(-candidate_probs)
        sorted_candidate_ids = candidate_ids[sorted_indices]
        true_set = true_allele_sets[i]
        num_true = len(true_set)
        
        hits_so_far = 0
        for k in range(1, min(max_k + 1, len(sorted_candidate_ids) + 1)):
            if sorted_candidate_ids[k-1] in true_set:
                hits_so_far += 1
            precision_at_k[k].append(hits_so_far / k)
            recall_at_k[k].append(hits_so_far / num_true if num_true > 0 else 0.0)
        for k in range(len(sorted_candidate_ids) + 1, max_k + 1):
            precision_at_k[k].append(hits_so_far / k)
            recall_at_k[k].append(hits_so_far / num_true if num_true > 0 else 0.0)
    
    results = {
        'mean_precision_at_k': {k: np.mean(v) for k, v in precision_at_k.items()},
        'std_precision_at_k': {k: np.std(v) for k, v in precision_at_k.items()},
        'mean_recall_at_k': {k: np.mean(v) for k, v in recall_at_k.items()},
        'num_tcrs': num_tcrs,
    }
    return results


def plot_precision_at_k_heatmap(all_results, k_values=[1, 2, 3, 5, 10], output_path=None):
    """Create heatmaps of Precision@k across configurations."""
    import pandas as pd
    b_vals = sorted(set(int(k.split('_')[0].replace('b', '')) for k in all_results.keys()))
    n_vals = sorted(set(int(k.split('_')[1].replace('n', '')) for k in all_results.keys()))
    columns = [f'b{b}' for b in b_vals]
    indexes = [f'n{n}' for n in n_vals]
    
    fig, axes = plt.subplots(1, len(k_values), figsize=(5*len(k_values), 4))
    if len(k_values) == 1: axes = [axes]
    
    for ax, k in zip(axes, k_values):
        df = pd.DataFrame(columns=columns, index=indexes, dtype=float)
        for config_name, results in all_results.items():
            n, b = config_name.split('_')[1], config_name.split('_')[0]
            df.loc[n, b] = results['mean_precision_at_k'][k]
        df = df.astype(float)
        sns.heatmap(df, annot=True, fmt='.3f', cmap='viridis', ax=ax, vmin=0, vmax=1)
        ax.set_title(f'Precision@{k}')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_precision_at_k_curves(all_results, configs_to_plot=None, max_k=20, output_path=None):
    """Plot Precision@k curves for configurations."""
    if configs_to_plot is None:
        configs_to_plot = list(all_results.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(configs_to_plot)))
    
    for config_name, color in zip(sorted(configs_to_plot), colors):
        if config_name not in all_results: continue
        results = all_results[config_name]
        k_vals = list(range(1, max_k + 1))
        precisions = [results['mean_precision_at_k'][k] for k in k_vals]
        ax.plot(k_vals, precisions, marker='o', markersize=3, label=config_name, color=color)
    
    ax.set_xlabel('k')
    ax.set_ylabel('Precision@k')
    ax.set_title('Precision@k Curves')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def save_metrics_json(output_path, perf_metrics, analysis_results, donor_stats, threshold=0.5):
    """Save final metrics summary to JSON."""
    simple_dict = {
        "auc_roc": float(perf_metrics['auc']),
        "average_precision": float(perf_metrics['ap']),
        "best_f1_score": float(perf_metrics['best_f1']),
        "tcr_coverage_pct": float(analysis_results['coverage']),
        "avg_alleles_per_tcr": float(analysis_results['avg_hlas_per_tcr']),
        "donor_explanation_mean": float(donor_stats['mean_fraction_explained_t005']),
    }
    json_path = os.path.join(output_path, "final_metrics_summary.json")
    with open(json_path, 'w') as f:
        json.dump(simple_dict, f, indent=4, cls=NumpyEncoder)
    print(f"Metrics saved to: {json_path}")
    return simple_dict
