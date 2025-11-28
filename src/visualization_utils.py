"""
Visualization utilities for TCRtyper model debugging and analysis.

This module provides the TestModeVisualizer class which generates comprehensive
visualizations of model internals, intermediate tensors, and likelihood components
during TEST_MODE training runs.

Usage:
    from src.visualization_utils import TestModeVisualizer
    
    visualizer = TestModeVisualizer(output_dir='output/test_mode')
    visualizer.plot_gamma(gamma, sample_indices=[0, 1, 2])
    visualizer.save_summary_statistics(...)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from typing import List, Dict, Optional, Union, Tuple


class TestModeVisualizer:
    """
    A comprehensive visualization class for TCRtyper TEST_MODE analysis.
    
    This class provides methods to visualize all intermediate tensors and 
    likelihood components produced during a forward pass of the TCRtyper model.
    Each visualization method is self-contained and can be called independently.
    
    Attributes:
        output_dir (str): Directory where all visualization outputs are saved.
        dpi (int): Resolution for saved figures (dots per inch).
        default_figsize (tuple): Default figure size for plots.
    
    Example:
        >>> visualizer = TestModeVisualizer(output_dir='output/debug')
        >>> visualizer.plot_input_sequences(tcr_seqs, sample_indices=[0, 1, 2])
        >>> visualizer.plot_attention_weights(att_score, sample_indices=[0, 1, 2])
    """
    
    def __init__(self, output_dir: str, dpi: int = 150, default_figsize: Tuple[int, int] = (15, 4)):
        """
        Initialize the TestModeVisualizer.
        
        Args:
            output_dir: Directory path where all visualizations will be saved.
                       Created automatically if it doesn't exist.
            dpi: Resolution for saved figures. Higher values produce sharper images
                 but larger file sizes. Default is 150.
            default_figsize: Default (width, height) tuple for figure dimensions.
        """
        self.output_dir = output_dir
        self.dpi = dpi
        self.default_figsize = default_figsize
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"TestModeVisualizer initialized. Output directory: {self.output_dir}")
    
    def _to_numpy(self, tensor: Union[tf.Tensor, np.ndarray]) -> np.ndarray:
        """
        Safely convert a tensor to numpy array.
        
        This helper handles both TensorFlow tensors and numpy arrays,
        ensuring consistent behavior regardless of input type.
        """
        if hasattr(tensor, 'numpy'):
            return tensor.numpy()
        return np.array(tensor)
    
    def _save_figure(self, filename: str) -> None:
        """Save the current figure and close it to free memory."""
        filepath = os.path.join(self.output_dir, f'{filename}.png')
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}.png")
    
    def _create_heatmap_subplots(
        self,
        data: Union[tf.Tensor, np.ndarray],
        title: str,
        filename: str,
        sample_indices: List[int],
        xlabel: str = '',
        ylabel: str = '',
        cmap: str = 'viridis',
        figsize: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        Internal method to create heatmap subplots for multiple samples.
        
        This is the core visualization method used by many public methods.
        It handles tensors of various dimensionalities by applying appropriate
        transformations (e.g., averaging over attention heads).
        
        Args:
            data: Input tensor of shape (B, ...) where B is batch size.
            title: Main title displayed above all subplots.
            filename: Output filename (without .png extension).
            sample_indices: List of batch indices to visualize.
            xlabel: Label for x-axis of each subplot.
            ylabel: Label for y-axis of each subplot.
            cmap: Matplotlib colormap name for the heatmap.
            figsize: Optional custom figure size; uses default if None.
        """
        data_np = self._to_numpy(data)
        n_samples = len(sample_indices)
        figsize = figsize or self.default_figsize
        
        fig, axes = plt.subplots(1, n_samples, figsize=figsize)
        # Ensure axes is always iterable (handles single subplot case)
        if n_samples == 1:
            axes = [axes]
        
        for ax, sample_idx in zip(axes, sample_indices):
            sample_data = data_np[sample_idx]
            
            # Handle different tensor dimensionalities appropriately
            if len(sample_data.shape) == 1:
                # 1D data: reshape to row vector for heatmap display
                sample_data = sample_data.reshape(1, -1)
            elif len(sample_data.shape) == 3:
                # 3D data (e.g., attention with heads): average over first dim
                sample_data = np.mean(sample_data, axis=0)
            
            im = ax.imshow(sample_data, aspect='auto', cmap=cmap)
            ax.set_title(f'Sample {sample_idx}')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        self._save_figure(filename)
    
    def _create_bar_subplots(
        self,
        data: Union[tf.Tensor, np.ndarray],
        title: str,
        filename: str,
        sample_indices: List[int],
        xlabel: str = '',
        ylabel: str = '',
        figsize: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        Internal method to create bar chart subplots for 1D data.
        
        Useful for visualizing per-sample scalar values or short vectors.
        """
        data_np = self._to_numpy(data)
        n_samples = len(sample_indices)
        figsize = figsize or self.default_figsize
        
        fig, axes = plt.subplots(1, n_samples, figsize=figsize)
        if n_samples == 1:
            axes = [axes]
        
        for ax, sample_idx in zip(axes, sample_indices):
            sample_data = data_np[sample_idx]
            
            if len(sample_data.shape) == 0:
                # Scalar value: display as single bar
                ax.bar([0], [float(sample_data)])
                ax.set_xticks([0])
                ax.set_xticklabels(['Value'])
            else:
                ax.bar(range(len(sample_data)), sample_data)
            
            ax.set_title(f'Sample {sample_idx}')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        self._save_figure(filename)
    
    # =========================================================================
    # PUBLIC VISUALIZATION METHODS (1-20)
    # Each method corresponds to one of the 20 visualization steps
    # =========================================================================
    
    def plot_input_sequences(
        self,
        tcr_seqs: Union[tf.Tensor, np.ndarray],
        sample_indices: List[int]
    ) -> None:
        """
        Step 1: Visualize input TCR sequences as heatmaps of token IDs.
        
        Each position in the sequence is colored according to its amino acid
        token ID, allowing you to see the sequence structure and padding.
        
        Args:
            tcr_seqs: Input sequences tensor of shape (B, seq_len).
            sample_indices: Which samples from the batch to visualize.
        """
        self._create_heatmap_subplots(
            data=tcr_seqs,
            title='Input TCR Sequences (Token IDs)',
            filename='01_tcr_seqs',
            sample_indices=sample_indices,
            xlabel='Position',
            ylabel='',
            cmap='viridis'
        )
    
    def plot_donor_ids(
        self,
        tcr_donor_ids: Union[tf.Tensor, np.ndarray],
        sample_indices: List[int]
    ) -> None:
        """
        Step 2: Visualize donor IDs associated with each TCR.
        
        Shows which donors contributed each TCR sequence. Padded positions
        (where no donor exists) will have the pad token value.
        
        Args:
            tcr_donor_ids: Tensor of shape (B, max_donors) with donor indices.
            sample_indices: Which samples from the batch to visualize.
        """
        self._create_heatmap_subplots(
            data=tcr_donor_ids,
            title='TCR Donor IDs (Padded)',
            filename='02_tcr_donor_ids',
            sample_indices=sample_indices,
            xlabel='Donor Index',
            ylabel='',
            cmap='tab20'
        )
    
    def plot_masked_embedding(
        self,
        me: Union[tf.Tensor, np.ndarray],
        sample_indices: List[int]
    ) -> None:
        """
        Step 3: Visualize the masked embedding layer output.
        
        Shows how each amino acid token is embedded into the model's
        latent space. Masked/padded positions should show zeros.
        
        Args:
            me: Embedding tensor of shape (B, seq_len, embed_dim).
            sample_indices: Which samples from the batch to visualize.
        """
        self._create_heatmap_subplots(
            data=me,
            title='Masked Embedding Output (B, S, D)',
            filename='03_masked_embedding',
            sample_indices=sample_indices,
            xlabel='Embedding Dim',
            ylabel='Position',
            cmap='coolwarm'
        )
    
    def plot_positional_encoding(
        self,
        pe: Union[tf.Tensor, np.ndarray],
        sample_indices: List[int]
    ) -> None:
        """
        Step 4: Visualize output after positional encoding is added.
        
        The sinusoidal positional encoding pattern should be visible,
        especially in the embedding dimensions. Compare with step 3
        to see how position information is injected.
        
        Args:
            pe: Tensor after positional encoding, shape (B, seq_len, embed_dim).
            sample_indices: Which samples from the batch to visualize.
        """
        self._create_heatmap_subplots(
            data=pe,
            title='After Positional Encoding (B, S, D)',
            filename='04_positional_encoding',
            sample_indices=sample_indices,
            xlabel='Embedding Dim',
            ylabel='Position',
            cmap='coolwarm'
        )
    
    def plot_attention_heads(
        self,
        att_score: Union[tf.Tensor, np.ndarray],
        sample_indices: List[int],
        figsize: Tuple[int, int] = (12, 10)
    ) -> None:
        """
        Step 5: Visualize attention weights for all heads of each sample.
        
        Creates one figure per sample, with subplots for each attention head.
        This reveals what positions each head is attending to, helping
        diagnose whether attention patterns are sensible.
        
        Args:
            att_score: Attention weights of shape (B, heads, seq, seq).
            sample_indices: Which samples to create figures for.
            figsize: Figure size for each sample's attention visualization.
        """
        att_np = self._to_numpy(att_score)
        
        for sample_idx in sample_indices:
            sample_att = att_np[sample_idx]  # Shape: (heads, seq, seq)
            n_heads = sample_att.shape[0]
            
            # Create grid layout for attention heads
            n_cols = min(4, n_heads)
            n_rows = (n_heads + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = np.atleast_2d(axes).flatten()
            
            for head_idx in range(n_heads):
                ax = axes[head_idx]
                im = ax.imshow(sample_att[head_idx], aspect='auto', cmap='Blues')
                ax.set_title(f'Head {head_idx}')
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Hide any unused subplot axes
            for idx in range(n_heads, len(axes)):
                axes[idx].axis('off')
            
            fig.suptitle(f'Self-Attention Weights (Sample {sample_idx})', fontsize=14)
            plt.tight_layout()
            self._save_figure(f'05_attention_heads_sample{sample_idx}')
    
    def plot_attention_averaged(
        self,
        att_score: Union[tf.Tensor, np.ndarray],
        sample_indices: List[int]
    ) -> None:
        """
        Step 6: Visualize attention weights averaged over all heads.
        
        Provides a simpler view of the overall attention pattern by
        averaging across heads. Useful for getting a quick overview.
        
        Args:
            att_score: Attention weights of shape (B, heads, seq, seq).
            sample_indices: Which samples from the batch to visualize.
        """
        # Average over the heads dimension (axis=1)
        att_avg = tf.reduce_mean(att_score, axis=1)
        
        self._create_heatmap_subplots(
            data=att_avg,
            title='Self-Attention (Averaged over Heads)',
            filename='06_attention_avg',
            sample_indices=sample_indices,
            xlabel='Key Position',
            ylabel='Query Position',
            cmap='Blues'
        )
    
    def plot_mlp_output(
        self,
        mlp1: Union[tf.Tensor, np.ndarray],
        sample_indices: List[int]
    ) -> None:
        """
        Step 7: Visualize the MLP layer output after attention.
        
        Shows the transformed representations after the feed-forward
        network processes the attention output.
        
        Args:
            mlp1: MLP output tensor of shape (B, seq_len, hidden_dim).
            sample_indices: Which samples from the batch to visualize.
        """
        self._create_heatmap_subplots(
            data=mlp1,
            title='MLP1 Output (B, S, 128)',
            filename='07_mlp1_output',
            sample_indices=sample_indices,
            xlabel='Hidden Dim',
            ylabel='Position',
            cmap='coolwarm'
        )
    
    def plot_pooled_representation(
        self,
        pooled: Union[tf.Tensor, np.ndarray],
        sample_indices: List[int]
    ) -> None:
        """
        Step 8: Visualize the pooled sequence representation.
        
        After global average pooling, each sequence is reduced to a
        single vector. This is the final representation before the
        output heads (gamma and q).
        
        Args:
            pooled: Pooled tensor of shape (B, hidden_dim).
            sample_indices: Which samples from the batch to visualize.
        """
        self._create_heatmap_subplots(
            data=pooled,
            title='Pooled Representation (B, 128)',
            filename='08_pooled',
            sample_indices=sample_indices,
            xlabel='Hidden Dim',
            ylabel='',
            cmap='coolwarm'
        )
    
    def plot_gamma(
        self,
        gamma: Union[tf.Tensor, np.ndarray],
        sample_indices: List[int]
    ) -> None:
        """
        Step 9: Visualize gamma predictions (HLA binding probabilities).
        
        Gamma represents the predicted probability that each TCR binds
        to each HLA allotype. Values close to 1 indicate predicted binding.
        
        Args:
            gamma: Prediction tensor of shape (B, num_alleles).
            sample_indices: Which samples from the batch to visualize.
        """
        self._create_heatmap_subplots(
            data=gamma,
            title='Gamma Predictions (HLA Binding Probabilities)',
            filename='09_gamma',
            sample_indices=sample_indices,
            xlabel='HLA Allele Index',
            ylabel='',
            cmap='Reds'
        )
    
    def plot_q_probability(
        self,
        q: Union[tf.Tensor, np.ndarray],
        sample_indices: List[int]
    ) -> None:
        """
        Step 10: Visualize q predictions (sampling probability).
        
        The q parameter represents the probability that a TCR appears
        in a sample given that the donor carries the matching HLA allele.
        
        Args:
            q: Probability tensor of shape (B, 1) or (B,).
            sample_indices: Which samples from the batch to visualize.
        """
        # Squeeze to ensure shape is (B,) for bar plot
        q_squeezed = tf.squeeze(q, axis=-1) if len(q.shape) > 1 else q
        
        self._create_bar_subplots(
            data=q_squeezed,
            title='Q Predictions (Sampling Probability)',
            filename='10_q_prob',
            sample_indices=sample_indices,
            xlabel='',
            ylabel='q value'
        )
    
    def plot_ni_size(
        self,
        Ni_size: Union[tf.Tensor, np.ndarray],
        sample_indices: List[int]
    ) -> None:
        """
        Step 11: Visualize Ni_size (number of donors per TCR).
        
        Shows how many donors contributed each TCR sequence in the batch.
        This is |N_i| in the likelihood equation.
        
        Args:
            Ni_size: Count tensor of shape (B,).
            sample_indices: Which samples from the batch to visualize.
        """
        self._create_bar_subplots(
            data=Ni_size,
            title='Ni_size (Number of Donors per TCR)',
            filename='11_Ni_size',
            sample_indices=sample_indices,
            xlabel='',
            ylabel='Count'
        )
    
    def plot_ni_matrix(
        self,
        Ni: Union[tf.Tensor, np.ndarray],
        gamma_donor_id_mask: Union[tf.Tensor, np.ndarray],
        sample_indices: List[int]
    ) -> None:
        """
        Step 12: Visualize Ni matrix (donor MHC profiles for each TCR).
        
        For each TCR, shows the HLA alleles carried by each of its donors.
        This is the x_{na} matrix from the likelihood, gathered per TCR.
        
        Args:
            Ni: Tensor of shape (B, max_donors, num_alleles) with binary HLA indicators.
            gamma_donor_id_mask: Mask tensor of shape (B, max_donors) indicating valid donors.
            sample_indices: Which samples to create figures for.
        """
        Ni_np = self._to_numpy(Ni)
        mask_np = self._to_numpy(gamma_donor_id_mask)
        
        for sample_idx in sample_indices:
            ni_sample = Ni_np[sample_idx]  # Shape: (max_donors, num_alleles)
            mask_sample = mask_np[sample_idx]  # Shape: (max_donors,)
            valid_donors = int(np.sum(mask_sample))
            
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Only show valid (non-padded) donors
            if valid_donors > 0:
                im = ax.imshow(ni_sample[:valid_donors, :], aspect='auto', cmap='Blues')
                ax.set_title(f'Ni Matrix (Donor MHC Profiles) - Sample {sample_idx} ({valid_donors} donors)')
            else:
                # Handle edge case of no valid donors
                im = ax.imshow(ni_sample[:1, :], aspect='auto', cmap='Blues')
                ax.set_title(f'Ni Matrix - Sample {sample_idx} (No valid donors)')
            
            ax.set_xlabel('HLA Allele Index')
            ax.set_ylabel('Donor Index')
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            self._save_figure(f'12_Ni_matrix_sample{sample_idx}')
    
    def plot_donor_id_mask(
        self,
        gamma_donor_id_mask: Union[tf.Tensor, np.ndarray],
        sample_indices: List[int]
    ) -> None:
        """
        Step 13: Visualize the donor ID mask.
        
        Shows which donor positions are valid (1) versus padded (0).
        This mask is applied to prevent padded positions from affecting
        the likelihood calculation.
        
        Args:
            gamma_donor_id_mask: Binary mask of shape (B, max_donors).
            sample_indices: Which samples from the batch to visualize.
        """
        self._create_heatmap_subplots(
            data=gamma_donor_id_mask,
            title='Gamma Donor ID Mask (Valid=1, Padded=0)',
            filename='13_donor_id_mask',
            sample_indices=sample_indices,
            xlabel='Donor Index',
            ylabel='',
            cmap='binary'
        )
    
    def plot_pn(
        self,
        pn: Union[tf.Tensor, np.ndarray],
        sample_indices: List[int]
    ) -> None:
        """
        Step 14: Visualize p_ni values (TCR-donor binding probability).
        
        This is the probability that TCR i can bind to an HLA allotype
        present in donor n, computed as: p_ni = 1 - prod_a(1 - gamma_ia)^x_na
        
        Args:
            pn: Probability tensor of shape (B, max_donors).
            sample_indices: Which samples from the batch to visualize.
        """
        self._create_heatmap_subplots(
            data=pn,
            title='p_ni Values (TCR-Donor Binding Probability)',
            filename='14_pn',
            sample_indices=sample_indices,
            xlabel='Donor Index',
            ylabel='',
            cmap='Greens'
        )
    
    def plot_numerator(
        self,
        numerator: Union[tf.Tensor, np.ndarray],
        sample_indices: List[int]
    ) -> None:
        """
        Step 15: Visualize the likelihood numerator (q * p_ni).
        
        This is the probability of observing TCR i in sample n,
        which appears in the log-odds ratio of the first term.
        
        Args:
            numerator: Tensor of shape (B, max_donors).
            sample_indices: Which samples from the batch to visualize.
        """
        self._create_heatmap_subplots(
            data=numerator,
            title='Numerator (q × p_ni)',
            filename='15_numerator',
            sample_indices=sample_indices,
            xlabel='Donor Index',
            ylabel='',
            cmap='Oranges'
        )
    
    def plot_denominator(
        self,
        denominator: Union[tf.Tensor, np.ndarray],
        sample_indices: List[int]
    ) -> None:
        """
        Step 16: Visualize the likelihood denominator (1 - q * p_ni).
        
        This is the probability of NOT observing TCR i in sample n,
        the complement of the numerator in the log-odds ratio.
        
        Args:
            denominator: Tensor of shape (B, max_donors).
            sample_indices: Which samples from the batch to visualize.
        """
        self._create_heatmap_subplots(
            data=denominator,
            title='Denominator (1 - q × p_ni)',
            filename='16_denominator',
            sample_indices=sample_indices,
            xlabel='Donor Index',
            ylabel='',
            cmap='Purples'
        )
    
    def plot_likelihood_terms(
        self,
        first_term: Union[tf.Tensor, np.ndarray],
        second_term: Union[tf.Tensor, np.ndarray],
        sample_indices: List[int],
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Step 17: Compare first and second terms of the likelihood.
        
        The likelihood is: LL_i = first_term - second_term
        where first_term = sum_n log(q*p_ni / (1 - q*p_ni))
        and second_term = |N_i| * q * sum_a(N_a * gamma_ia)
        
        Args:
            first_term: Tensor of shape (B,) with first term values.
            second_term: Tensor of shape (B,) with second term values.
            sample_indices: Which samples to compare.
            figsize: Figure size for the comparison plot.
        """
        first_np = self._to_numpy(first_term)
        second_np = self._to_numpy(second_term)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(sample_indices))
        width = 0.35
        
        first_vals = [first_np[idx] for idx in sample_indices]
        second_vals = [second_np[idx] for idx in sample_indices]
        
        ax.bar(x - width/2, first_vals, width, label='First Term', color='steelblue')
        ax.bar(x + width/2, second_vals, width, label='Second Term', color='coral')
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Value')
        ax.set_title('Likelihood Components: First vs Second Term')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Sample {i}' for i in sample_indices])
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        self._save_figure('17_likelihood_terms')
    
    def plot_all_scalars(
        self,
        first_term: Union[tf.Tensor, np.ndarray],
        second_term: Union[tf.Tensor, np.ndarray],
        Ni_size: Union[tf.Tensor, np.ndarray],
        q: Union[tf.Tensor, np.ndarray],
        sample_indices: List[int],
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Step 18: Compare all scalar metrics across samples.
        
        Provides a comprehensive view of all per-sample scalar values
        including likelihood terms, donor counts, and q probability.
        
        Args:
            first_term: First likelihood term, shape (B,).
            second_term: Second likelihood term, shape (B,).
            Ni_size: Number of donors per TCR, shape (B,).
            q: Sampling probability, shape (B, 1) or (B,).
            sample_indices: Which samples to compare.
            figsize: Figure size for the comparison plot.
        """
        # Convert all to numpy
        first_np = self._to_numpy(first_term)
        second_np = self._to_numpy(second_term)
        ni_np = self._to_numpy(Ni_size)
        q_np = self._to_numpy(tf.squeeze(q, axis=-1) if len(q.shape) > 1 else q)
        
        # Compute LL = first - second
        ll_np = first_np - second_np
        
        # Prepare data dictionary
        scalars = {
            'First Term': first_np,
            'Second Term': second_np,
            'LL (First - Second)': ll_np,
            'Ni_size': ni_np,
            'q': q_np
        }
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(sample_indices))
        n_metrics = len(scalars)
        width = 0.8 / n_metrics
        
        for i, (name, data) in enumerate(scalars.items()):
            values = [data[idx] for idx in sample_indices]
            offset = (i - n_metrics / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=name)
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Value')
        ax.set_title('All Scalar Metrics per Sample')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Sample {i}' for i in sample_indices])
        ax.legend(loc='upper right', fontsize=8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        self._save_figure('18_all_scalars')
    
    def plot_gamma_histogram(
        self,
        gamma: Union[tf.Tensor, np.ndarray],
        sample_indices: List[int],
        figsize: Tuple[int, int] = (15, 4)
    ) -> None:
        """
        Step 19: Plot histogram of gamma predictions.
        
        Shows the distribution of predicted HLA binding probabilities.
        Ideally, most values should be near 0 (no binding) with a few
        near 1 (strong binding prediction) for sparse predictions.
        
        Args:
            gamma: Prediction tensor of shape (B, num_alleles).
            sample_indices: Which samples to create histograms for.
            figsize: Figure size for the histogram plot.
        """
        gamma_np = self._to_numpy(gamma)
        n_samples = len(sample_indices)
        
        fig, axes = plt.subplots(1, n_samples, figsize=figsize)
        if n_samples == 1:
            axes = [axes]
        
        for ax, sample_idx in zip(axes, sample_indices):
            gamma_vals = gamma_np[sample_idx]
            
            ax.hist(gamma_vals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_title(f'Sample {sample_idx}')
            ax.set_xlabel('Gamma Value')
            ax.set_ylabel('Count')
            ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold=0.5')
            ax.legend()
            
            # Add statistics annotation
            mean_val = np.mean(gamma_vals)
            max_val = np.max(gamma_vals)
            above_threshold = np.sum(gamma_vals > 0.5)
            ax.text(0.95, 0.95, f'Mean: {mean_val:.4f}\nMax: {max_val:.4f}\n>0.5: {above_threshold}',
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Distribution of Gamma Predictions (HLA Binding Probabilities)', fontsize=14)
        plt.tight_layout()
        self._save_figure('19_gamma_histogram')
    
    def save_summary_statistics(
        self,
        tcr_seqs: Union[tf.Tensor, np.ndarray],
        gamma: Union[tf.Tensor, np.ndarray],
        q: Union[tf.Tensor, np.ndarray],
        att_score: Union[tf.Tensor, np.ndarray],
        Ni: Union[tf.Tensor, np.ndarray],
        pn: Union[tf.Tensor, np.ndarray],
        Ni_size: Union[tf.Tensor, np.ndarray],
        first_term: Union[tf.Tensor, np.ndarray],
        second_term: Union[tf.Tensor, np.ndarray],
        sample_indices: List[int]
    ) -> None:
        """
        Step 20: Save comprehensive summary statistics to a text file.
        
        Generates a detailed report with tensor shapes and per-sample
        statistics for debugging and analysis purposes.
        
        Args:
            tcr_seqs: Input sequences tensor.
            gamma: HLA binding predictions.
            q: Sampling probability.
            att_score: Attention weights.
            Ni: Donor MHC profiles.
            pn: TCR-donor binding probabilities.
            Ni_size: Number of donors per TCR.
            first_term: First likelihood term.
            second_term: Second likelihood term.
            sample_indices: Which samples to include in statistics.
        """
        # Convert tensors to numpy for statistics
        gamma_np = self._to_numpy(gamma)
        q_np = self._to_numpy(q)
        pn_np = self._to_numpy(pn)
        Ni_size_np = self._to_numpy(Ni_size)
        first_np = self._to_numpy(first_term)
        second_np = self._to_numpy(second_term)
        
        # Compute LL batch
        ll_batch = first_np - second_np
        
        summary_path = os.path.join(self.output_dir, 'summary_statistics.txt')
        
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("TCRtyper TEST MODE - SUMMARY STATISTICS\n")
            f.write("=" * 70 + "\n\n")
            
            # Tensor shapes section
            f.write("--- Tensor Shapes ---\n")
            f.write(f"tcr_seqs:           {tcr_seqs.shape}\n")
            f.write(f"gamma:              {gamma.shape}\n")
            f.write(f"q:                  {q.shape}\n")
            f.write(f"att_score:          {att_score.shape}\n")
            f.write(f"Ni:                 {Ni.shape}\n")
            f.write(f"pn:                 {pn.shape}\n")
            f.write(f"Ni_size:            {Ni_size.shape}\n")
            f.write(f"first_term:         {first_term.shape}\n")
            f.write(f"second_term:        {second_term.shape}\n\n")
            
            # Per-sample statistics
            f.write("--- Per-Sample Statistics ---\n")
            for idx in sample_indices:
                f.write(f"\nSample {idx}:\n")
                f.write("-" * 40 + "\n")
                
                # Donor count
                ni_count = Ni_size_np[idx]
                f.write(f"  Ni_size (num donors):     {ni_count:.0f}\n")
                
                # Q value
                q_val = q_np[idx][0] if len(q_np[idx].shape) > 0 else q_np[idx]
                f.write(f"  q (sampling prob):        {q_val:.6f}\n")
                
                # Gamma statistics
                gamma_sample = gamma_np[idx]
                f.write(f"  gamma mean:               {np.mean(gamma_sample):.6f}\n")
                f.write(f"  gamma std:                {np.std(gamma_sample):.6f}\n")
                f.write(f"  gamma min:                {np.min(gamma_sample):.6f}\n")
                f.write(f"  gamma max:                {np.max(gamma_sample):.6f}\n")
                f.write(f"  gamma > 0.5 count:        {np.sum(gamma_sample > 0.5)}\n")
                f.write(f"  gamma > 0.9 count:        {np.sum(gamma_sample > 0.9)}\n")
                
                # p_n statistics (only for valid donors)
                valid_count = int(ni_count)
                if valid_count > 0:
                    pn_valid = pn_np[idx][:valid_count]
                    f.write(f"  pn mean (valid donors):   {np.mean(pn_valid):.6f}\n")
                    f.write(f"  pn max (valid donors):    {np.max(pn_valid):.6f}\n")
                else:
                    f.write(f"  pn mean (valid donors):   N/A (no valid donors)\n")
                
                # Likelihood terms
                f.write(f"  First term:               {first_np[idx]:.6f}\n")
                f.write(f"  Second term:              {second_np[idx]:.6f}\n")
                f.write(f"  LL_i (first - second):    {ll_batch[idx]:.6f}\n")
            
            # Global statistics
            f.write("\n" + "=" * 70 + "\n")
            f.write("--- Global Statistics (All Samples) ---\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Batch size:              {gamma.shape[0]}\n")
            f.write(f"Number of HLA alleles:   {gamma.shape[1]}\n\n")
            
            f.write(f"Mean LL_batch:           {np.mean(ll_batch):.6f}\n")
            f.write(f"Std LL_batch:            {np.std(ll_batch):.6f}\n")
            f.write(f"Min LL_batch:            {np.min(ll_batch):.6f}\n")
            f.write(f"Max LL_batch:            {np.max(ll_batch):.6f}\n\n")
            
            f.write(f"Mean gamma (global):     {np.mean(gamma_np):.6f}\n")
            f.write(f"Mean q (global):         {np.mean(q_np):.6f}\n")
            f.write(f"Mean Ni_size:            {np.mean(Ni_size_np):.2f}\n")
            f.write(f"Max Ni_size:             {np.max(Ni_size_np):.0f}\n")
        
        print(f"  Saved: summary_statistics.txt")
    
    def print_tensor_shapes(
        self,
        tensors_dict: Dict[str, Union[tf.Tensor, np.ndarray]]
    ) -> None:
        """
        Utility method to print shapes of all provided tensors.
        
        Useful for quick debugging without saving to file.
        
        Args:
            tensors_dict: Dictionary mapping tensor names to tensor values.
        """
        print("\n--- Tensor Shapes ---")
        for name, tensor in tensors_dict.items():
            shape = tensor.shape if hasattr(tensor, 'shape') else 'N/A'
            print(f"  {name:25s} {shape}")
        print()