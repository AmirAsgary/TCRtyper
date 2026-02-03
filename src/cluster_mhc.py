#!/usr/bin/env python3
"""
Protein Embedding Clustering via Transformer Autoencoder

This script clusters protein sequence embeddings using a Transformer Autoencoder
to learn a compressed latent representation, followed by HDBSCAN clustering.

Usage:
    python cluster_embeddings.py \
        --embeddings embeddings.npy \
        --annotations annotations.csv \
        --output_dir ./results \
        --latent_dim 64 \
        --monitor_converge \
        --min_cluster_size 50
"""

import argparse
import os
import sys
from pathlib import Path

# Add these with your other imports
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Important for headless/cluster environments (SLURM)

try:
    import umap
except ImportError:
    print("Warning: umap-learn not installed. Visualization will be skipped.")
    umap = None

# Set TensorFlow memory growth before importing TF
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING
import numpy as np
import pandas as pd
import h5py
import hdbscan
import tensorflow as tf

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")

from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.saving import register_keras_serializable



@register_keras_serializable()
class PositionalEncoding(layers.Layer):
    """
    Sinusoidal positional encoding layer.
    Adds position information to the input embeddings.
    """
    
    def __init__(self, max_seq_len=5000, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
    
    def build(self, input_shape):
        _, seq_len, embed_dim = input_shape
        
        # Create positional encoding matrix
        position = np.arange(self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        
        pe = np.zeros((self.max_seq_len, embed_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[:embed_dim // 2] if embed_dim % 2 else div_term)
        
        self.pe = tf.constant(pe, dtype=tf.float32)
        super().build(input_shape)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({"max_seq_len": self.max_seq_len})
        return config




@register_keras_serializable()
class TransformerEncoderLayer(layers.Layer):
    """
    Single Transformer Encoder layer with multi-head attention and feedforward network.
    """
    
    def __init__(self, embed_dim, num_heads=8, ff_dim=None, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim or embed_dim * 4
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.dropout_rate
        )
        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation='gelu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.embed_dim),
            layers.Dropout(self.dropout_rate)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(self.dropout_rate)
        super().build(input_shape)
    
    def call(self, x, training=False, mask=None):
        # Multi-head self-attention with residual connection
        attn_output = self.attention(x, x, attention_mask=mask, training=training)
        attn_output = self.dropout(attn_output, training=training)
        x = self.layernorm1(x + attn_output)
        
        # Feedforward network with residual connection
        ffn_output = self.ffn(x, training=training)
        x = self.layernorm2(x + ffn_output)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate
        })
        return config




@register_keras_serializable()
class TransformerDecoderLayer(layers.Layer):
    """
    Single Transformer Decoder layer for reconstruction.
    """
    
    def __init__(self, embed_dim, num_heads=8, ff_dim=None, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim or embed_dim * 4
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.dropout_rate
        )
        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation='gelu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.embed_dim),
            layers.Dropout(self.dropout_rate)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(self.dropout_rate)
        super().build(input_shape)
    
    def call(self, x, training=False):
        # Self-attention with residual connection
        attn_output = self.attention(x, x, training=training)
        attn_output = self.dropout(attn_output, training=training)
        x = self.layernorm1(x + attn_output)
        
        # Feedforward network with residual connection
        ffn_output = self.ffn(x, training=training)
        x = self.layernorm2(x + ffn_output)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate
        })
        return config



@register_keras_serializable()
class LatentBottleneck(layers.Layer):
    """
    Compresses sequence representation to a fixed-size latent vector.
    Uses attention pooling to aggregate sequence information.
    """
    
    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
    
    def build(self, input_shape):
        embed_dim = input_shape[-1]
        
        # Learnable query for attention pooling
        self.query = self.add_weight(
            shape=(1, 1, embed_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_query'
        )
        
        self.attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=embed_dim // 4
        )
        
        self.to_latent = layers.Dense(self.latent_dim, activation='linear', name='to_latent')
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        
        super().build(input_shape)
    
    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        
        # Expand query to batch size
        query = tf.tile(self.query, [batch_size, 1, 1])
        
        # Attention pooling: query attends to all sequence positions
        pooled = self.attention(query, x, training=training)  # (batch, 1, embed_dim)
        pooled = self.layernorm(pooled)
        pooled = tf.squeeze(pooled, axis=1)  # (batch, embed_dim)
        
        # Project to latent dimension
        latent = self.to_latent(pooled)  # (batch, latent_dim)
        
        return latent
    
    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim})
        return config



@register_keras_serializable()
class LatentExpansion(layers.Layer):
    """
    Expands latent vector back to sequence representation.
    """
    
    def __init__(self, seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.embed_dim = embed_dim
    
    def build(self, input_shape):
        self.expand = layers.Dense(self.seq_len * self.embed_dim, activation='linear')
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        super().build(input_shape)
    
    def call(self, x, training=False):
        # Expand latent to sequence
        expanded = self.expand(x)  # (batch, seq_len * embed_dim)
        expanded = tf.reshape(expanded, (-1, self.seq_len, self.embed_dim))
        expanded = self.layernorm(expanded)
        return expanded
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_len": self.seq_len,
            "embed_dim": self.embed_dim
        })
        return config




class TransformerAutoencoder:
    """
    Transformer-based Autoencoder for learning latent representations
    of protein sequence embeddings.
    
    Architecture:
        Input (seq_len, input_dim) 
        -> Input Projection (seq_len, embed_dim)
        -> Positional Encoding
        -> Transformer Encoder Layer
        -> Latent Bottleneck (latent_dim)
        -> Latent Expansion (seq_len, embed_dim)
        -> Positional Encoding
        -> Transformer Decoder Layer
        -> Output Projection (seq_len, input_dim)
    """
    
    def __init__(
        self,
        seq_len,
        input_dim,
        embed_dim=256,
        latent_dim=64,
        num_heads=8,
        ff_dim=None,
        dropout_rate=0.1
    ):
        self.seq_len = seq_len
        self.input_dim = input_dim  # Original embedding dimension (e.g., 1280 for ESM)
        self.embed_dim = embed_dim  # Reduced working dimension for transformer
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim or embed_dim * 4
        self.dropout_rate = dropout_rate
        
        self.model = self._build_model()
        self.encoder = self._build_encoder()
    
    def _build_model(self):
        """Build the full autoencoder model."""
        inputs = layers.Input(shape=(self.seq_len, self.input_dim), name='input')
        
        # Input projection: reduce from input_dim to embed_dim
        x = layers.Dense(self.embed_dim, name='input_projection')(inputs)
        x = layers.LayerNormalization(epsilon=1e-6, name='input_norm')(x)
        
        # Add positional encoding
        x = PositionalEncoding(max_seq_len=self.seq_len + 100, name='pos_encoding')(x)
        
        # Encoder: Single transformer encoder layer
        x = TransformerEncoderLayer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout_rate=self.dropout_rate,
            name='encoder_layer'
        )(x)
        
        # Bottleneck: Compress to latent space
        latent = LatentBottleneck(
            latent_dim=self.latent_dim,
            name='latent_bottleneck'
        )(x)
        
        # Expansion: Expand latent back to sequence
        x = LatentExpansion(
            seq_len=self.seq_len,
            embed_dim=self.embed_dim,
            name='latent_expansion'
        )(latent)
        
        # Add positional encoding for decoder
        x = PositionalEncoding(max_seq_len=self.seq_len + 100, name='decoder_pos_encoding')(x)
        
        # Decoder: Single transformer decoder layer
        x = TransformerDecoderLayer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout_rate=self.dropout_rate,
            name='decoder_layer'
        )(x)
        
        # Output projection: expand from embed_dim back to input_dim
        outputs = layers.Dense(self.input_dim, name='output_projection')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='transformer_autoencoder')
        return model
    
    def _build_encoder(self):
        """Build encoder-only model for extracting latent representations."""
        inputs = layers.Input(shape=(self.seq_len, self.input_dim), name='encoder_input')
        
        # Reuse layers from full model
        x = self.model.get_layer('input_projection')(inputs)
        x = self.model.get_layer('input_norm')(x)
        x = self.model.get_layer('pos_encoding')(x)
        x = self.model.get_layer('encoder_layer')(x)
        latent = self.model.get_layer('latent_bottleneck')(x)
        
        encoder = Model(inputs=inputs, outputs=latent, name='encoder')
        return encoder
    
    def compile(self, learning_rate=1e-4):
        """Compile the model with optimizer and loss."""
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-5
        )
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
    
    def summary(self):
        """Print model summary."""
        self.model.summary()



class ConvergenceMonitor(callbacks.Callback):
    """
    Custom callback to monitor training convergence and stop when converged.
    
    Convergence is detected when:
    1. Loss improvement is below a threshold for a certain number of epochs
    2. Loss is stable (low variance) over recent epochs
    """
    
    def __init__(
        self,
        patience=15,
        min_delta=1e-5,
        variance_window=10,
        variance_threshold=1e-6,
        verbose=1
    ):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.variance_window = variance_window
        self.variance_threshold = variance_threshold
        self.verbose = verbose
        
        self.best_loss = np.inf
        self.wait = 0
        self.loss_history = []
        self.stopped_epoch = 0
        self.converged = False
    
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        self.loss_history.append(current_loss)
        
        # Check for improvement
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
        
        # Check variance-based convergence
        variance_converged = False
        if len(self.loss_history) >= self.variance_window:
            recent_losses = self.loss_history[-self.variance_window:]
            loss_variance = np.var(recent_losses)
            variance_converged = loss_variance < self.variance_threshold
            
            if self.verbose:
                print(f"  [Convergence Monitor] Loss variance (last {self.variance_window} epochs): {loss_variance:.2e}")
        
        # Stop if converged
        if self.wait >= self.patience or variance_converged:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            self.converged = True
            
            reason = "low variance" if variance_converged else "no improvement"
            if self.verbose:
                print(f"\n>>> Training converged at epoch {epoch + 1} ({reason})")
                print(f">>> Best loss: {self.best_loss:.6f}")
    
    def on_train_end(self, logs=None):
        if self.converged and self.verbose:
            print(f"\nConvergence achieved. Final training loss: {self.loss_history[-1]:.6f}")



def load_embeddings(embeddings_path, memmap=False):
    """Load embeddings from numpy file.
    
    Parameters:
    -----------
    embeddings_path : str
        Path to .npy file
    memmap : bool
        If True, load as memory-mapped file to reduce RAM usage
    """
    print(f"Loading embeddings from: {embeddings_path}")
    
    if memmap:
        print("  Using memory-mapped loading (reduced RAM usage)")
        embeddings = np.load(embeddings_path, mmap_mode='r')
        # Copy to writable array since we need to shuffle for training
        # But this is still more memory efficient as numpy handles it smartly
    else:
        embeddings = np.load(embeddings_path)
    
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    print(f"  Size: {embeddings.nbytes / 1e9:.2f} GB")
    
    return embeddings


def load_annotations(annotations_path):
    """Load allele annotations from CSV file."""
    print(f"Loading annotations from: {annotations_path}")
    df = pd.read_csv(annotations_path)
    
    if 'allele' not in df.columns:
        raise ValueError("CSV file must contain an 'allele' column")
    
    alleles = df['allele'].values
    print(f"  Number of alleles: {len(alleles)}")
    print(f"  Unique alleles: {len(np.unique(alleles))}")
    
    return alleles


def validate_data(embeddings, alleles):
    """Validate that embeddings and alleles have matching dimensions."""
    if len(embeddings) != len(alleles):
        raise ValueError(
            f"Mismatch between embeddings ({len(embeddings)}) and alleles ({len(alleles)})"
        )
    print(f"Data validation passed: {len(embeddings)} samples")




def cluster_latent_space(
    latent_embeddings,
    min_cluster_size=50,
    min_samples=None,
    cluster_selection_epsilon=0.0,
    metric='euclidean',
    cluster_selection_method='eom'
):
    """
    Perform HDBSCAN clustering on latent embeddings.
    
    Parameters:
    -----------
    latent_embeddings : np.ndarray
        Latent space representations (n_samples, latent_dim)
    min_cluster_size : int
        Minimum cluster size for HDBSCAN
    min_samples : int or None
        Minimum samples for core points (defaults to min_cluster_size)
    cluster_selection_epsilon : float
        Distance threshold for cluster selection
    metric : str
        Distance metric to use
    cluster_selection_method : str
        Method for selecting clusters ('eom' or 'leaf')
    
    Returns:
    --------
    labels : np.ndarray
        Cluster labels for each sample (-1 indicates noise)
    clusterer : hdbscan.HDBSCAN
        Fitted HDBSCAN object
    """
    print("\nPerforming HDBSCAN clustering...")
    print(f"  min_cluster_size: {min_cluster_size}")
    print(f"  min_samples: {min_samples or min_cluster_size}")
    print(f"  metric: {metric}")
    print(f"  cluster_selection_method: {cluster_selection_method}")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=-1
    )
    
    labels = clusterer.fit_predict(latent_embeddings)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    
    print(f"\nClustering results:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Noise points: {n_noise} ({100 * n_noise / len(labels):.1f}%)")
    
    # Print cluster size distribution
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(counts) > 0:
        print(f"  Cluster size range: {counts.min()} - {counts.max()}")
        print(f"  Mean cluster size: {counts.mean():.1f}")
    
    return labels, clusterer



def save_results(output_path, latent_embeddings, cluster_labels, alleles):
    """
    Save results to HDF5 file.
    
    Structure:
    - /latent: Latent space embeddings (n_samples, latent_dim)
    - /clusters: Cluster labels (n_samples,)
    - /alleles: Allele names (n_samples,) as variable-length strings
    """
    print(f"\nSaving results to: {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Save latent embeddings
        f.create_dataset(
            'latent',
            data=latent_embeddings,
            compression='gzip',
            compression_opts=4
        )
        
        # Save cluster labels
        f.create_dataset(
            'clusters',
            data=cluster_labels,
            compression='gzip',
            compression_opts=4
        )
        
        # Save alleles as variable-length strings
        dt = h5py.special_dtype(vlen=str)
        alleles_ds = f.create_dataset('alleles', shape=(len(alleles),), dtype=dt)
        alleles_ds[:] = alleles
        
        # Add metadata
        f.attrs['n_samples'] = len(latent_embeddings)
        f.attrs['latent_dim'] = latent_embeddings.shape[1]
        f.attrs['n_clusters'] = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        f.attrs['n_noise'] = int((cluster_labels == -1).sum())
        f.attrs['n_unique_alleles'] = len(np.unique(alleles))
    
    print(f"  Saved {len(latent_embeddings)} samples")
    print(f"  Latent dimension: {latent_embeddings.shape[1]}")
    


class EmbeddingDataGenerator:
    """
    Generator class for memory-efficient training.
    Uses index-based access to avoid loading entire dataset into GPU.
    """
    
    def __init__(self, embeddings, indices, shuffle=False):
        self.embeddings = embeddings
        self.indices = indices.copy()
        self.shuffle = shuffle
        self.seq_len = embeddings.shape[1]
        self.embed_dim = embeddings.shape[2]
    
    def __call__(self):
        indices = self.indices.copy()
        if self.shuffle:
            np.random.shuffle(indices)
        
        for idx in indices:
            x = np.asarray(self.embeddings[idx], dtype=np.float32)
            yield x, x


def create_tf_dataset_generator(embeddings, batch_size, validation_split=0.1):
    """
    Create tf.data.Dataset using a generator for very large datasets.
    
    This is memory efficient as it only loads one batch at a time to GPU.
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Input embeddings (n_samples, seq_len, embed_dim)
    batch_size : int
        Training batch size
    validation_split : float
        Fraction of data to use for validation
    
    Returns:
    --------
    train_dataset : tf.data.Dataset
        Training dataset
    val_dataset : tf.data.Dataset
        Validation dataset
    n_train : int
        Number of training samples
    n_val : int
        Number of validation samples
    """
    n_samples = len(embeddings)
    n_val = int(n_samples * validation_split)
    n_train = n_samples - n_val
    
    # Shuffle indices once for train/val split
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    seq_len, embed_dim = embeddings.shape[1], embeddings.shape[2]
    
    output_signature = (
        tf.TensorSpec(shape=(seq_len, embed_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(seq_len, embed_dim), dtype=tf.float32)
    )
    
    # Create generator instances
    train_gen = EmbeddingDataGenerator(embeddings, train_indices, shuffle=True)
    val_gen = EmbeddingDataGenerator(embeddings, val_indices, shuffle=False)
    
    train_dataset = tf.data.Dataset.from_generator(
        train_gen,
        output_signature=output_signature
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_generator(
        val_gen,
        output_signature=output_signature
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, n_train, n_val



def train_autoencoder(
    embeddings,
    latent_dim,
    embed_dim=256,
    ff_dim=None,
    epochs=None,
    monitor_converge=False,
    batch_size=32,
    learning_rate=1e-4,
    validation_split=0.1,
    patience=15,
    min_delta=1e-5,
    num_heads=8,
    dropout_rate=0.1
):
    """
    Train the Transformer Autoencoder.
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Input embeddings (n_samples, seq_len, input_dim)
    latent_dim : int
        Dimension of latent space
    embed_dim : int
        Working dimension for transformer layers (projects input_dim -> embed_dim)
    ff_dim : int or None
        Feedforward dimension (defaults to embed_dim * 4)
    epochs : int or None
        Number of training epochs (if not using convergence monitoring)
    monitor_converge : bool
        Whether to use convergence monitoring
    batch_size : int
        Training batch size
    learning_rate : float
        Learning rate for optimizer
    validation_split : float
        Fraction of data to use for validation
    patience : int
        Patience for convergence monitoring
    min_delta : float
        Minimum improvement for convergence
    num_heads : int
        Number of attention heads
    dropout_rate : float
        Dropout rate
    
    Returns:
    --------
    autoencoder : TransformerAutoencoder
        Trained autoencoder
    history : keras.callbacks.History
        Training history
    """
    _, seq_len, input_dim = embeddings.shape
    ff_dim = ff_dim or embed_dim * 4
    
    print("\n" + "=" * 60)
    print("Building Transformer Autoencoder")
    print("=" * 60)
    print(f"  Sequence length: {seq_len}")
    print(f"  Input dimension: {input_dim}")
    print(f"  Transformer embed dimension: {embed_dim}")
    print(f"  Feedforward dimension: {ff_dim}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Number of attention heads: {num_heads}")
    print(f"  Dropout rate: {dropout_rate}")
    
    # Build model
    autoencoder = TransformerAutoencoder(
        seq_len=seq_len,
        input_dim=input_dim,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout_rate=dropout_rate
    )
    
    autoencoder.compile(learning_rate=learning_rate)
    autoencoder.summary()
    
    # Setup callbacks
    callback_list = []
    
    if monitor_converge:
        print("\n>>> Using convergence monitoring (will stop when converged)")
        convergence_callback = ConvergenceMonitor(
            patience=patience,
            min_delta=min_delta,
            verbose=1
        )
        callback_list.append(convergence_callback)
        max_epochs = 1000  # Upper limit when monitoring convergence
    else:
        if epochs is None:
            epochs = 100
        max_epochs = epochs
        print(f"\n>>> Training for {epochs} epochs")
    
    # Add learning rate scheduler
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    callback_list.append(lr_scheduler)
    
    # Create memory-efficient datasets
    print("\nCreating memory-efficient data pipeline...")
    train_dataset, val_dataset, n_train, n_val = create_tf_dataset_generator(
        embeddings=embeddings,
        batch_size=batch_size,
        validation_split=validation_split
    )
    print(f"  Training samples: {n_train}")
    print(f"  Validation samples: {n_val}")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per epoch: {n_train // batch_size}")
    
    # Training
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    history = autoencoder.model.fit(
        train_dataset,
        epochs=max_epochs,
        validation_data=val_dataset,
        callbacks=callback_list,
        verbose=1
    )
    
    return autoencoder, history

def visualize_clusters(latent_embeddings, cluster_labels, alleles, output_dir):
    """
    Generate UMAP visualizations:
    1. Colored by HDBSCAN Cluster ID
    2. Colored by Simplified HLA Allele (e.g., HLA-A*01)
    """
    if umap is None:
        print("Skipping visualization: umap-learn module not found.")
        return

    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    # 1. Process Alleles (Simplify to 2-digit resolution)
    # Extract everything before the first colon (e.g., "HLA-A*01:290" -> "HLA-A*01")
    simple_alleles = np.array([a.split(':')[0] for a in alleles])
    unique_alleles = np.unique(simple_alleles)
    print(f"Unique simplified alleles found: {len(unique_alleles)}")

    # 2. Run UMAP (Reduce to 2D)
    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(
        n_components=2, 
        n_neighbors=3,
        min_dist=0.1,
        metric='euclidean'
    )
    embedding_2d = reducer.fit_transform(latent_embeddings)
    
    # --- PLOT 1: HDBSCAN CLUSTERS ---
    print("Generating cluster plot...")
    plt.figure(figsize=(12, 10))
    
    noise_mask = (cluster_labels == -1)
    cluster_mask = ~noise_mask
    
    # Plot Noise (Grey)
    if noise_mask.any():
        plt.scatter(
            embedding_2d[noise_mask, 0],
            embedding_2d[noise_mask, 1],
            c='lightgrey', s=10, alpha=0.3, label='Noise'
        )
        
    # Plot Clusters
    if cluster_mask.any():
        scatter = plt.scatter(
            embedding_2d[cluster_mask, 0],
            embedding_2d[cluster_mask, 1],
            c=cluster_labels[cluster_mask],
            cmap='tab20', s=20, alpha=0.7
        )
        plt.colorbar(scatter, label='Cluster ID')

    plt.title(f'Latent Space by HDBSCAN Cluster\n(n={len(latent_embeddings)})')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    out_path_cluster = Path(output_dir) / 'umap_clusters.png'
    plt.savefig(out_path_cluster, dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- PLOT 2: HLA ALLELES ---
    print("Generating allele plot...")
    plt.figure(figsize=(14, 10))  # Slightly wider for legend
    
    # Map strings to integers for coloring
    # We use a distinct colormap (tab20) and cycle through it if >20 alleles
    label_to_id = {label: i for i, label in enumerate(unique_alleles)}
    allele_ids = np.array([label_to_id[a] for a in simple_alleles])
    
    # Use a colormap with enough contrast
    cmap = plt.get_cmap('tab20')
    
    # Plot each allele group separately to build a proper legend
    # (Doing this loop ensures the legend handles the labels correctly)
    for i, label in enumerate(unique_alleles):
        mask = (simple_alleles == label)
        plt.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            color=cmap(i % 20), # Cycle colors if > 20 alleles
            s=20,
            alpha=0.7,
            label=label
        )
        
    plt.title(f'Latent Space by HLA Allele (2-digit)\n(Resolution: {unique_alleles[0]}...)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    # Place legend outside the plot to avoid covering data
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, ncol=1, fontsize='small')
    
    out_path_allele = Path(output_dir) / 'umap_alleles.png'
    plt.savefig(out_path_allele, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved cluster plot to: {out_path_cluster}")
    print(f"  Saved allele plot to: {out_path_allele}")






def main():
    parser = argparse.ArgumentParser(
        description='Cluster protein embeddings using Transformer Autoencoder + HDBSCAN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--embeddings',
        type=str,
        required=True,
        help='Path to embeddings numpy file (.npy)'
    )
    parser.add_argument(
        '--annotations',
        type=str,
        required=True,
        help='Path to CSV file with allele annotations'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save output files'
    )
    
    # Model architecture
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=64,
        help='Dimension of latent space'
    )
    parser.add_argument(
        '--embed_dim',
        type=int,
        default=256,
        help='Working dimension for transformer layers (input is projected to this dimension)'
    )
    parser.add_argument(
        '--ff_dim',
        type=int,
        default=None,
        help='Feedforward dimension in transformer layers (defaults to embed_dim * 4)'
    )
    parser.add_argument(
        '--num_heads',
        type=int,
        default=8,
        help='Number of attention heads in transformer'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (ignored if --monitor_converge is set)'
    )
    parser.add_argument(
        '--monitor_converge',
        action='store_true',
        help='Enable convergence monitoring (stops training when converged)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=15,
        help='Patience for convergence monitoring'
    )
    parser.add_argument(
        '--min_delta',
        type=float,
        default=1e-5,
        help='Minimum improvement threshold for convergence'
    )
    parser.add_argument(
        '--mixed_precision',
        action='store_true',
        help='Enable mixed precision training (float16) to reduce memory usage'
    )
    parser.add_argument(
        '--memmap',
        action='store_true',
        help='Load embeddings as memory-mapped file to reduce RAM usage'
    )
    
    # HDBSCAN parameters
    parser.add_argument(
        '--min_cluster_size',
        type=int,
        default=50,
        help='Minimum cluster size for HDBSCAN'
    )
    parser.add_argument(
        '--min_samples',
        type=int,
        default=None,
        help='Minimum samples for HDBSCAN core points (defaults to min_cluster_size)'
    )
    parser.add_argument(
        '--cluster_epsilon',
        type=float,
        default=0.0,
        help='Cluster selection epsilon for HDBSCAN'
    )
    parser.add_argument(
        '--cluster_metric',
        type=str,
        default='euclidean',
        choices=['euclidean', 'cosine', 'manhattan'],
        help='Distance metric for HDBSCAN'
    )
    parser.add_argument(
        '--cluster_method',
        type=str,
        default='eom',
        choices=['eom', 'leaf'],
        help='Cluster selection method for HDBSCAN'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.monitor_converge and args.epochs is None:
        args.epochs = 100
        print(f"No --epochs specified and --monitor_converge not set. Using default: {args.epochs} epochs")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Protein Embedding Clustering Pipeline")
    print("=" * 60)
    
    # Enable mixed precision if requested
    if args.mixed_precision:
        print("Enabling mixed precision training (float16)...")
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("  Mixed precision enabled")
    
    # Load data
    embeddings = load_embeddings(args.embeddings, memmap=args.memmap)
    alleles = load_annotations(args.annotations)
    validate_data(embeddings, alleles)
    
    # Convert memmap to regular array for training (needed for shuffling)
    if args.memmap:
        print("\nConverting memory-mapped embeddings to regular array...")
        embeddings = np.array(embeddings, dtype=np.float32)
        print(f"  Conversion complete")
    
    # Train autoencoder
    autoencoder, history = train_autoencoder(
        embeddings=embeddings,
        latent_dim=args.latent_dim,
        embed_dim=args.embed_dim,
        ff_dim=args.ff_dim,
        epochs=args.epochs,
        monitor_converge=args.monitor_converge,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        min_delta=args.min_delta,
        num_heads=args.num_heads,
        dropout_rate=args.dropout
    )
    
    # Extract latent representations
    print("\n" + "=" * 60)
    print("Extracting Latent Representations")
    print("=" * 60)
    
    # Extract latent embeddings in batches to avoid OOM
    print("Extracting latent embeddings in batches...")
    latent_embeddings = []
    n_samples = len(embeddings)
    
    for i in range(0, n_samples, args.batch_size):
        batch = embeddings[i:min(i + args.batch_size, n_samples)]
        batch_latent = autoencoder.encoder.predict(batch, verbose=0)
        latent_embeddings.append(batch_latent)
        
        if (i // args.batch_size) % 100 == 0:
            print(f"  Processed {min(i + args.batch_size, n_samples)}/{n_samples} samples...")
    
    latent_embeddings = np.concatenate(latent_embeddings, axis=0)
    print(f"Latent embeddings shape: {latent_embeddings.shape}")
    
    # Cluster latent space
    cluster_labels, clusterer = cluster_latent_space(
        latent_embeddings=latent_embeddings,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_epsilon,
        metric=args.cluster_metric,
        cluster_selection_method=args.cluster_method
    )
    
    # Save results
    output_file = output_dir / 'mhc_clusters.h5'
    save_results(output_file, latent_embeddings, cluster_labels, alleles)
    
    visualize_clusters(
            latent_embeddings=latent_embeddings,
            cluster_labels=cluster_labels,
            alleles=alleles,  # <--- Pass the alleles array here
            output_dir=output_dir
        )

    # Save training history
    history_file = output_dir / 'training_history.npy'
    np.save(history_file, history.history)
    print(f"Saved training history to: {history_file}")
    
    

    # Print summary
    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"Output file: {output_file}")
    print(f"  - /latent: Latent embeddings ({latent_embeddings.shape})")
    print(f"  - /clusters: Cluster labels ({cluster_labels.shape})")
    print(f"  - /alleles: Allele names ({len(alleles)})")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


