import keras
from keras import layers
import tensorflow as tf
import src
from src.constants import PHYSICHE_PROPERTIES_IDX



@tf.keras.utils.register_keras_serializable(package='custom_layers', name='PositionalEncoding')
class PositionalEncoding(keras.layers.Layer):
    """
    Sinusoidal Positional Encoding layer that applies encodings
    only to non-masked tokens.
    
    Args:
        embed_dim (int): Dimension of embeddings (must match input last dim).
        pos_range (int): Maximum sequence length expected (used to precompute encodings).
        mask_token (float): Value for masked tokens.
        pad_token (float): Value for padded tokens.
    """
    def __init__(self, embed_dim, pos_range=100, mask_token=-1., pad_token=-2., name='positional_encoding', **kwargs):
        super().__init__(name=name, **kwargs)
        self.embed_dim = embed_dim
        self.pos_range = pos_range
        self.mask_token = mask_token
        self.pad_token = pad_token

    def build(self, input_shape):
        # Create (1, pos_range, embed_dim) encoding matrix
        pos = tf.range(self.pos_range, dtype=tf.float32)[:, tf.newaxis]  # (pos_range, 1)
        # Use integer range for dimension indices, do integer division, then cast
        i = tf.range(self.embed_dim, dtype=tf.int32)[tf.newaxis, :]  # (1, embed_dim) as int32
        i_div_2 = i // 2  # Integer division on int32 tensor
        # Now cast to float for computation
        i_div_2_float = tf.cast(i_div_2, tf.float32)
        embed_dim_float = tf.cast(self.embed_dim, tf.float32)
        power = -(2.0 * i_div_2_float)
        power = power / embed_dim_float
        # Compute angle rates
        angle_rates = tf.exp(tf.math.log(300.0) * power)
        angle_rads = pos * angle_rates  # (pos_range, embed_dim)
        # Apply sin to even indices, cos to odd indices
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)  # (pos_range, embed_dim)
        pos_encoding = pos_encoding[tf.newaxis, ...]  # (1, pos_range, embed_dim)
        # Store in compute dtype to reduce casts
        self.pos_encoding = tf.cast(pos_encoding, dtype=self.compute_dtype)

    def call(self, x, mask):
        """
        Args:
            x: Input tensor of shape (B, N, D)
            mask: Tensor of shape (B, N)
        
        Returns:
            Tensor with positional encodings added for masked and non padded tokens.
        """
        seq_len = tf.shape(x)[1]
        pe = self.pos_encoding[:, :seq_len, :]  # (1, N, D)
        mask = tf.cast(mask[:, :, tf.newaxis], x.dtype)  # (B, N, 1)
        mask = tf.where(mask == self.pad_token, tf.cast(0.0, x.dtype), tf.cast(1.0, x.dtype))
        pe = tf.cast(pe, x.dtype) * mask  # zero out positions where mask is 0
        return x + pe
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'pos_range': self.pos_range,
            'mask_token': self.mask_token,
            'pad_token': self.pad_token,
        })
        return config




@keras.utils.register_keras_serializable(package='custom_layers', name='MaskedEmbedding')
class MaskedEmbedding(keras.layers.Layer):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        mask_token=-1,
        pad_token=-2,
        name='masked_embedding',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mask_token = mask_token
        self.pad_token = pad_token

    def build(self, input_shape):
        # Learnable embedding matrix
        self.embedding = self.add_weight(
            name="embedding_matrix",
            shape=(self.vocab_size+1, self.embed_dim),
            initializer='random_normal',
            trainable=True,
        )

    @tf.function(reduce_retracing=True)
    def call(self, x, mask):
        """
        Args:
            x: integer input tokens, shape (B, N)
            mask: mask values, shape (B, N),
                  contains original IDs so we check pad/mask tokens
        Returns:
            Embeddings with padded/masked positions set to zero.
        """
        # Standard embedding lookup
        x = tf.cast(x, dtype=tf.int32) #(B,S)
        x = tf.where(x == tf.cast(self.pad_token, tf.int32), self.vocab_size, x)
        embedded = tf.nn.embedding_lookup(self.embedding, x)
        mask = tf.cast(mask, embedded.dtype)

        # Identify padded or masked tokens
        bad = (mask == self.pad_token) | (mask == self.mask_token)

        # Convert to (0 for masked/padded, 1 for valid)
        keep_mask = tf.where(bad, 0.0, 1.0)
        keep_mask = tf.cast(keep_mask, embedded.dtype)

        # Zero-out embeddings at masked/padded positions
        embedded = embedded * keep_mask[:, :, tf.newaxis]

        return embedded


    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'mask_token': self.mask_token,
            'pad_token': self.pad_token,
        })
        return config



@keras.utils.register_keras_serializable(package='custom_layers', name='MaskedEmbeddingOHE')
class MaskedEmbeddingOHE(keras.layers.Layer):
    def __init__(
        self,
        vocab_size,
        mask_token=-1,
        pad_token=-2,
        name='masked_embedding_ohe',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.mask_token = mask_token
        self.pad_token = pad_token
        # For one-hot encoding, embed_dim is vocab_size+1 (including pad token)
        self.embed_dim = vocab_size + 1

    def build(self, input_shape):
        # No learnable weights needed for one-hot encoding
        super().build(input_shape)

    @tf.function(reduce_retracing=True)
    def call(self, x, mask):
        """
        Args:
            x: integer input tokens, shape (B, N)
            mask: mask values, shape (B, N),
                  contains original IDs so we check pad/mask tokens
        Returns:
            One-hot encoded embeddings with padded/masked positions set to zero.
        """
        # Cast to int32
        x = tf.cast(x, dtype=tf.int32)  # (B, S)
        
        # Map pad_token to vocab_size
        x = tf.where(x == tf.cast(self.pad_token, tf.int32), self.vocab_size, x)
        
        # One-hot encode: shape (B, S, vocab_size+1)
        embedded = tf.one_hot(x, depth=self.vocab_size + 1, dtype=tf.float32)
        
        # Cast mask to same dtype
        mask = tf.cast(mask, embedded.dtype)

        # Identify padded or masked tokens
        bad = (mask == self.pad_token) | (mask == self.mask_token)

        # Convert to (0 for masked/padded, 1 for valid)
        keep_mask = tf.where(bad, 0.0, 1.0)
        keep_mask = tf.cast(keep_mask, embedded.dtype)

        # Zero-out embeddings at masked/padded positions
        embedded = embedded * keep_mask[:, :, tf.newaxis]

        return embedded

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'mask_token': self.mask_token,
            'pad_token': self.pad_token,
        })
        return config





def _neg_inf(dtype):
    """Return a large negative value for the given dtype."""
    return tf.constant(-1e9, dtype=dtype)

def compute_rope_rotations(dim, max_seq_len, base=10000.0, dtype=tf.float32):
    """
    Compute rotation matrices for RoPE.
    
    Args:
        dim: Feature dimension (must be even).
        max_seq_len: Maximum sequence length.
        base: Base for computing frequencies (default 10000).
        dtype: Data type for computations.
    
    Returns:
        Tuple of (cos, sin) matrices of shape (max_seq_len, dim).
    """
    # Compute theta values: theta_j = base^(-2j/d)
    inv_freq = 1.0 / (base ** (tf.range(0, dim, 2, dtype=dtype) / dim))
    
    # Compute positions * frequencies: shape (max_seq_len, dim//2)
    positions = tf.range(max_seq_len, dtype=dtype)
    freqs = tf.einsum('i,j->ij', positions, inv_freq)
    
    # Duplicate each frequency for the pair rotation: shape (max_seq_len, dim)
    freqs = tf.repeat(freqs, 2, axis=-1)
    
    # Compute cos and sin
    cos = tf.cos(freqs)
    sin = tf.sin(freqs)
    
    return cos, sin

def apply_rope(x, cos, sin, start_index=0):
    """
    Apply rotary position embedding to input tensor.
    
    Args:
        x: Input tensor of shape (batch, seq_len, heads, dim) or (batch, seq_len, dim).
        cos: Cosine values of shape (max_seq_len, dim).
        sin: Sine values of shape (max_seq_len, dim).
        start_index: Starting position index for the sequence.
    
    Returns:
        Tensor with RoPE applied.
    """
    seq_len = tf.shape(x)[1]
    
    # Select relevant positions
    cos = cos[start_index:start_index + seq_len]
    sin = sin[start_index:start_index + seq_len]
    
    # Reshape for broadcasting
    if len(x.shape) == 4:  # (batch, seq, heads, dim)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    else:  # (batch, seq, dim)
        cos = cos[None, :, :]
        sin = sin[None, :, :]
    
    # Split into pairs and rotate
    # x = [x0, x1, x2, x3, ...] -> rotate pairs (x0,x1), (x2,x3), ...
    x1 = x[..., 0::2]  # Even indices
    x2 = x[..., 1::2]  # Odd indices
    
    # Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
    cos_half = cos[..., 0::2]
    sin_half = sin[..., 0::2]
    
    rotated_x1 = x1 * cos_half - x2 * sin_half
    rotated_x2 = x1 * sin_half + x2 * cos_half
    
    # Interleave back
    rotated = tf.stack([rotated_x1, rotated_x2], axis=-1)
    rotated = tf.reshape(rotated, tf.shape(x))
    
    return rotated

@tf.keras.utils.register_keras_serializable(package='custom_layers', name='AttentionLayer')
class AttentionLayer(keras.layers.Layer):
    """
    Custom multi-head attention layer supporting self- and cross-attention.

    Args:
        query_dim (int): Input feature dimension for query.
        context_dim (int): Input feature dimension for context (key and value).
        output_dim (int): Output feature dimension.
        type (str): 'self' or 'cross'.
        heads (int): Number of attention heads.
        resnet (bool): Whether to use residual connection.
        return_att_weights (bool): Whether to return attention weights.
        name (str): Layer name.
        epsilon (float): Epsilon for layer normalization.
        gate (bool): Whether to use gating mechanism.
        mask_token (float): Value for masked tokens.
        pad_token (float): Value for padded tokens.
        use_rope (bool): Whether to use Rotary Position Embedding.
        rope_max_seq_len (int): Maximum sequence length for RoPE.
        rope_base (float): Base for RoPE frequency computation.
    """

    def __init__(self, query_dim, context_dim, output_dim, type, heads=4,
                 resnet=True, return_att_weights=False, name='attention',
                 epsilon=1e-6, gate=True, mask_token=-1., pad_token=-2.,
                 use_rope=False, rope_max_seq_len=2048, rope_base=10000.0,
                 kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=0.05, seed=42),
                 bias_initializer=keras.initializers.Zeros(),
                 gate_initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42),
                 **kwargs):
        super().__init__(name=name, **kwargs)
        assert isinstance(query_dim, int) and isinstance(context_dim, int) and isinstance(output_dim, int)
        assert type in ['self', 'cross']
        if resnet:
            assert query_dim == output_dim
        
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.type = type
        self.heads = heads
        self.resnet = resnet
        self.return_att_weights = return_att_weights
        self.epsilon = epsilon
        self.gate = gate
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.gate_initializer = gate_initializer
        self.att_dim = output_dim // heads  # Attention dimension per head
        
        # RoPE parameters
        self.use_rope = use_rope
        self.rope_max_seq_len = rope_max_seq_len
        self.rope_base = rope_base
        
        # Validate att_dim for RoPE (must be even)
        if self.use_rope and self.att_dim % 2 != 0:
            raise ValueError(f"When use_rope=True, att_dim must be even. Got {self.att_dim}")

    def build(self, input_shape):
        # Projection weights
        self.q_proj = self.add_weight(
            shape=(self.heads, self.query_dim, self.att_dim),
            initializer=self.kernel_initializer,
            trainable=True,
            name=f'q_proj_{self.name}'
        )
        self.k_proj = self.add_weight(
            shape=(self.heads, self.context_dim, self.att_dim),
            initializer=self.kernel_initializer,
            trainable=True,
            name=f'k_proj_{self.name}'
        )
        self.v_proj = self.add_weight(
            shape=(self.heads, self.context_dim, self.att_dim),
            initializer=self.kernel_initializer,
            trainable=True,
            name=f'v_proj_{self.name}'
        )
        
        if self.gate:
            self.g = self.add_weight(
                shape=(self.heads, self.query_dim, self.att_dim),
                initializer=self.gate_initializer,
                trainable=True,
                name=f'gate_{self.name}'
            )
        
        # Layer normalization parameters - norm_in
        self.gamma_in = self.add_weight(
            shape=(self.query_dim,),
            initializer='ones',
            trainable=True,
            name=f'gamma_in_{self.name}'
        )
        self.beta_in = self.add_weight(
            shape=(self.query_dim,),
            initializer='zeros',
            trainable=True,
            name=f'beta_in_{self.name}'
        )
        
        # Layer normalization parameters - norm_context (for cross-attention)
        if self.type == 'cross':
            self.gamma_context = self.add_weight(
                shape=(self.context_dim,),
                initializer='ones',
                trainable=True,
                name=f'gamma_context_{self.name}'
            )
            self.beta_context = self.add_weight(
                shape=(self.context_dim,),
                initializer='zeros',
                trainable=True,
                name=f'beta_context_{self.name}'
            )
        
        # Layer normalization parameters - norm_out
        self.gamma_out = self.add_weight(
            shape=(self.output_dim,),
            initializer='ones',
            trainable=True,
            name=f'gamma_out_{self.name}'
        )
        self.beta_out = self.add_weight(
            shape=(self.output_dim,),
            initializer='zeros',
            trainable=True,
            name=f'beta_out_{self.name}'
        )
        
        # Output projection
        self.out_w = self.add_weight(
            shape=(self.heads * self.att_dim, self.output_dim),
            initializer=self.kernel_initializer,
            trainable=True,
            name=f'outw_{self.name}'
        )
        self.out_b = self.add_weight(
            shape=(self.output_dim,),
            initializer=self.bias_initializer,
            trainable=True,
            name=f'outb_{self.name}'
        )
        
        # Compute scale in compute dtype
        self.scale = 1.0 / tf.math.sqrt(
            tf.cast(self.att_dim, keras.mixed_precision.global_policy().compute_dtype)
        )
        
        # Precompute RoPE rotations if enabled
        if self.use_rope:
            cos, sin = compute_rope_rotations(
                self.att_dim,
                self.rope_max_seq_len,
                base=self.rope_base,
                dtype=self.compute_dtype
            )
            # Store as non-trainable weights
            self.rope_cos = self.add_weight(
                shape=cos.shape,
                initializer=keras.initializers.Constant(cos.numpy()),
                trainable=False,
                name=f'rope_cos_{self.name}'
            )
            self.rope_sin = self.add_weight(
                shape=sin.shape,
                initializer=keras.initializers.Constant(sin.numpy()),
                trainable=False,
                name=f'rope_sin_{self.name}'
            )

    def layer_norm(self, x, gamma, beta):
        """
        Custom layer normalization implementation.
        
        Args:
            x: Input tensor of shape (B, N, D)
            gamma: Scale parameter of shape (D,)
            beta: Shift parameter of shape (D,)
            
        Returns:
            Normalized tensor of same shape as x
        """
        # Compute mean and variance across the feature dimension (axis=-1)
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / tf.sqrt(variance + self.epsilon)
        
        # Scale and shift
        return gamma * x_norm + beta

    @tf.function
    def call(self, x, mask, context=None, context_mask=None, start_index=0):
        """
        Args:
            x: Tensor of shape (B, N, query_dim) for query.
            mask: Tensor of shape (B, N).
            context: Tensor of shape (B, M, context_dim) for key/value in cross-attention.
            context_mask: Tensor of shape (B, M) for context.
            start_index: Starting position for RoPE (useful for cached decoding).
        """
        mask = tf.cast(mask, self.compute_dtype)
        
        # Prepare inputs
        if self.type == 'self':
            context = x
            context_mask = mask
            q_input = k_input = v_input = self.layer_norm(x, self.gamma_in, self.beta_in)
            mask_q = mask_k = tf.where(mask == self.pad_token, 0., 1.)
        else:
            assert context is not None and context_mask is not None
            q_input = self.layer_norm(x, self.gamma_in, self.beta_in)
            k_input = v_input = self.layer_norm(context, self.gamma_context, self.beta_context)
            mask_q = tf.where(mask == self.pad_token, 0., 1.)
            mask_k = tf.where(tf.cast(context_mask, self.compute_dtype) == self.pad_token, 0., 1.)

        # Project query, key, value
        q = tf.einsum('bnd,hde->bhne', q_input, self.q_proj)
        k = tf.einsum('bmd,hde->bhme', k_input, self.k_proj)
        v = tf.einsum('bmd,hde->bhme', v_input, self.v_proj)

        # Apply RoPE to queries and keys (not values)
        if self.use_rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin, start_index=start_index)
            # For cross-attention, only apply RoPE to query (since context may have different positions)
            if self.type == 'self':
                k = apply_rope(k, self.rope_cos, self.rope_sin, start_index=start_index)

        # Compute attention scores
        att = tf.einsum('bhne,bhme->bhnm', q, k) * tf.cast(self.scale, self.compute_dtype)
        
        # Apply mask
        mask_q_exp = tf.expand_dims(mask_q, axis=1)
        mask_k_exp = tf.expand_dims(mask_k, axis=1)
        attention_mask = tf.einsum('bqn,bkm->bqnm', mask_q_exp, mask_k_exp)
        attention_mask = tf.broadcast_to(attention_mask, tf.shape(att))
        att += (1.0 - attention_mask) * _neg_inf(att.dtype)
        att = tf.nn.softmax(att, axis=-1) * attention_mask

        # Compute output
        out = tf.einsum('bhnm,bhme->bhne', att, v)
        
        if self.gate:
            g = tf.einsum('bnd,hde->bhne', tf.cast(q_input, self.compute_dtype), 
                         tf.cast(self.g, self.compute_dtype))
            g = tf.nn.sigmoid(g)
            out *= g

        # Reshape and project output
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [tf.shape(x)[0], tf.shape(x)[1], self.heads * self.att_dim])
        out = tf.matmul(out, tf.cast(self.out_w, self.compute_dtype)) + tf.cast(self.out_b, self.compute_dtype)

        # Residual connection
        if self.resnet:
            out += tf.cast(x, self.compute_dtype)

        # Final normalization and masking
        out = self.layer_norm(out, self.gamma_out, self.beta_out)
        mask_exp = tf.expand_dims(mask_q, axis=-1)
        out *= mask_exp

        return (out, att) if self.return_att_weights else out

    def get_config(self):
        config = super().get_config()
        config.update({
            'query_dim': self.query_dim,
            'context_dim': self.context_dim,
            'output_dim': self.output_dim,
            'type': self.type,
            'heads': self.heads,
            'resnet': self.resnet,
            'return_att_weights': self.return_att_weights,
            'epsilon': self.epsilon,
            'gate': self.gate,
            'mask_token': self.mask_token,
            'pad_token': self.pad_token,
            'use_rope': self.use_rope,
            'rope_max_seq_len': self.rope_max_seq_len,
            'rope_base': self.rope_base,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'gate_initializer': self.gate_initializer
        })
        return config



class Likelihood(keras.losses.Loss):
    def __init__(self, donor_mhc: tf.Tensor, 
                 pad_token: int | float = -1, 
                 test_mode: bool = False, 
                 N : int = 702,
                use_softmax_loss: bool = True,
                softmax_loss_weight: float = 1.0,
                softmax_temperature: float = 1.0,
                **kwargs):
        super().__init__()
        self.donor_mhc = tf.constant(donor_mhc) #(N, A)
        self.pad_token = tf.cast(pad_token, tf.int32)
        self.test_mode = test_mode
        self.N = N # johannes said not to use this, bcz it is wrong. My idea was to use it to divide self.Na on it to decrease the magnitude of it.
        assert len(tf.shape(self.donor_mhc)) == 2, f'the shape of donor_mhc should be (donor, mhc_allele), found {tf.shape(donor_mhc)}'
        self.Na = tf.cast(tf.reduce_sum(self.donor_mhc, axis=0), tf.float32) #(A,)
        self.use_softmax_loss = use_softmax_loss
        self.softmax_loss_weight = softmax_loss_weight
        self.softmax_temperature = softmax_temperature
        #self.Na = tf.cast(tf.reduce_sum(self.donor_mhc, axis=0), tf.float32) / tf.cast(N, tf.float32)  # (A,)

    def call(self, gamma, q, gamma_donor_id):
        '''
        gamma: output of model. dimension (batch, mhc_allele) or (B,A)
        q: output of model. dimension (batch,)
        gamma_donor_id: padded integers of donor ids per each tcr. dimension (batch, padded_donor_id) or (B, D_i). map tcr to donors.
        '''
        #### Calculate The second Term ####
        if len(tf.shape(q)) == 2: q = tf.squeeze(q, axis=-1) #(B,1) --> (B,)
        #TODO clipping might be dangerous bcz the porbs can be so small. we should return in log space instead of probs
        #gamma = tf.clip_by_value(gamma, 1e-7, 1.0 - 1e-7)
        #q = tf.clip_by_value(q, 1e-7, 1.0 - 1e-7)
        # |Ni_size| * q * Sum^A( Na * gamma_ia )
        Ni_size, Ni, gamma_donor_id_mask = self.calculate_Ni_Nisize(gamma_donor_id) #(B,) and (B, N_i, A) and (B, N_i)
        #second_term = q * Ni_size * tf.reduce_sum(self.Na[tf.newaxis, :] * gamma, axis=-1) #(B,) * (B,) * Sum^A ( (B, A) * (B, A)) --> (B,)
        second_term = q * tf.reduce_sum(self.Na[tf.newaxis, :] * gamma, axis=-1) #
        #### Calculate The first Term ####
        # Sum^Ni (ln( qi*pni / 1 - qi*pni))
        ## calculate pni: 1 - Prod( 1 - gamma_ia) ** x_na
        pn = self.calculate_pni(Ni, gamma, gamma_donor_id_mask) # (B, N_i)
        # TODO do in log space in the model
        numerator = tf.multiply(q[:, tf.newaxis], pn) # (B,1) * (B, N_i)
        denominator = 1. - numerator
        first_term = tf.math.log(numerator) - tf.math.log(denominator + 1e-15)
        # apply mask, because padded ones are now log(0) == 1 
        first_term = first_term * gamma_donor_id_mask
        first_term = tf.reduce_sum(first_term, axis=-1) #(B, N_i) --> (B,)
        LL_batch = first_term - second_term
        # ===== SOFTMAX LOSS (optional) =====
        if self.use_softmax_loss:
            cce = self.softmax_loss(Ni, gamma, gamma_donor_id_mask)  # (B,)
        else:
            cce = tf.zeros_like(LL_batch)
        if not self.test_mode:
            return -tf.reduce_mean(LL_batch), self.softmax_loss_weight * tf.reduce_mean(cce)
        else: #(B,) and (B, N_i, A) and (B, N_i), (B,)
            return (Ni_size, #(B,) --> for each B, stores how many donors they have that sequence
                    Ni, #(B,N_i,A) --> for each B, gathers all ohe of mhcs for each of their donors N_i. since it is padded, should be masked out by gamma_donor_id_mask
                    gamma_donor_id_mask, #(B,N_i) --> for each B, lists all donor id masks, 0 for padded ones and 1 for not padded ones
                    pn, # (B, N_i) for each B, and for each of their donors has calculate pn
                    numerator, # (B, N_i) --> q * pn
                    denominator, # (B, N_i) --> 1 - q * pn
                    first_term, #(B,)
                    second_term, #(B,)
                    cce,) # (B,)
         
    def softmax_loss(self, Ni, gamma, gamma_donor_id_mask):
        """
        Compute cross-entropy between gamma predictions and empirical HLA distribution.
        The target distribution is derived from the HLA profiles of donors containing
        each TCR. Alleles appearing in more donors get higher target probability.
        Args:
            Ni: HLA profiles of donors, shape (B, N_i, A), binary values
            gamma: Model predictions, shape (B, A), values in (0, 1)
            gamma_donor_id_mask: Mask for valid donors, shape (B, N_i)
        Returns:
            Cross-entropy loss per sample, shape (B,)
        """
        # Sum HLA counts across donors: (B, N_i, A) -> (B, A)
        # This gives the count of how many donors have each allele
        hla_counts = tf.reduce_sum(Ni, axis=1)  # (B, A)
        # Convert counts to soft target distribution using temperature-scaled softmax
        # Lower temperature = sharper distribution (more confident about top alleles)
        # Higher temperature = softer distribution (more uniform)
        y_true = tf.nn.softmax(
            tf.cast(hla_counts, tf.float32) / self.softmax_temperature, 
            axis=-1
        )  # (B, A)
        # The model's gamma predictions serve as the predicted distribution
        # Normalize gamma to be a proper distribution for cross-entropy
        y_pred = gamma / (tf.reduce_sum(gamma, axis=-1, keepdims=True) + 1e-10)  # (B, A)
        # Compute cross-entropy: -sum(y_true * log(y_pred))
        cce = -tf.reduce_sum(y_true * tf.math.log(y_pred + 1e-10), axis=-1)  # (B,)
        return cce
        
    def calculate_Ni_Nisize(self, gamma_donor_id): #(B, max)
        ## calculate count of donors per each tcr
        gamma_donor_id_count = tf.where(gamma_donor_id == self.pad_token, 0., 1.) #(B, pad_donor_id)
        Ni_size = tf.reduce_sum(gamma_donor_id_count, axis=-1) #(B,) each has the number of Ni 
        ## extract (Ni, A) from (N, A)
        # First, we need to create a masking for padded tokens
        gamma_donor_id_converted = tf.where(gamma_donor_id == self.pad_token, 0, gamma_donor_id) # just to make sure tf.gather does not raise error
        gamma_donor_id_converted = tf.cast(gamma_donor_id_converted, dtype=tf.int32)
        gamma_donor_id_mask = tf.where(gamma_donor_id == self.pad_token, 0., 1) #(B,D_i) or (B,N_i) pads == 0, normal == 1
        Ni = tf.gather(self.donor_mhc, gamma_donor_id_converted, axis=0) # (N, A), (B,D_i) --> (B, N_i, A) N_i are simply gathered D_is , D_i is index and is padded. len(N_i) == len(D_i)
        Ni = tf.multiply(Ni, tf.cast(gamma_donor_id_mask[:, :, tf.newaxis], tf.int32)) #(B,N_i,A) masked out
        return tf.cast(Ni_size, tf.float32), tf.cast(Ni, tf.float32), gamma_donor_id_mask #(B,) (B,N_i,A) masked out and (B,N_i) pads == 0, normal == 1

    def calculate_pni(self, Ni, gamma, gamma_donor_id_mask): #(B,N_i,A) and (B,A)
        # pni = 1 - Prod (1 - gamma_ia) ^ xna
        # Ni has only 0 and 1 now. Also, the N_i dim is padded to maximum number of donors for a tcr.
        # gamma should be expanded from (B,A) to (B, N_i, A)
        gamma_expanded = tf.expand_dims(gamma, axis=1) #(B, 1, A)
        gamma_expanded = tf.broadcast_to(gamma_expanded, shape=tf.shape(Ni)) #(B, N_i, A)
        # output = (1 - gamma)^ xna
        output = tf.pow(1. - gamma_expanded, Ni) #(B, N_i, A)
        # 1. - Prod(output)
        #pni = 1. - tf.reduce_prod(output, axis=-1) #(B, N_i)
        log_prod = tf.reduce_sum(tf.math.log(output + 1e-10), axis=-1)
        pni = 1. - tf.exp(log_prod)
        # apply mask
        pni = pni * gamma_donor_id_mask #(B, N_i)
        return pni


@tf.keras.utils.register_keras_serializable(package="custom_layers", name='MasedDense')
class MaskedDense(keras.layers.Layer):
    def __init__(self, units, pad_token=-2., use_bias=True, name='maskeddense', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.pad_token = pad_token
        self.use_bias = use_bias
        self.name

    def build(self, input_shapes):
        """input_shapes = (x_shape, mask_shape)"""
        x_shape, _ = input_shapes
        last_dim = x_shape[-1]

        # Weight matrix
        self.W = self.add_weight(
            name="kernel",
            shape=(last_dim, self.units),
            initializer="glorot_uniform",
            trainable=True
        )

        # Optional bias
        if self.use_bias:
            self.b = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer="zeros",
                trainable=True
            )
        else:
            self.b = None

    def call(self, inputs):
        """
        inputs: tuple (x, mask)
        x:    (B, Seq, D)
        mask: (B, Seq)
        """
        x, mask = inputs

        # Dense projection
        y = tf.matmul(x, self.W)  # (B, Seq, units)
        if self.use_bias:
            y = y + self.b
        y = tf.nn.relu(y)

        # Make mask binary: 1 = keep, 0 = pad
        mask_binary = tf.cast(mask != self.pad_token, y.dtype)  # (B, Seq)
        mask_binary = mask_binary[:, :, tf.newaxis]             # (B, Seq, 1)

        # Zero-out padded tokens
        return y * mask_binary

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "pad_token": self.pad_token,
            "use_bias": self.use_bias,
            "name": self.name
        })
        return config


@tf.keras.utils.register_keras_serializable(package="custom_layers", name='QDense')
class QDense(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros",
        name='QDense', bound=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.name = name
        self.bound = bound
        if self.bound:
            self.bound = tf.constant(self.bound, dtype=tf.float32)
    def build(self, input_shape):
        # input_shape = (batch, features)
        input_dim = int(input_shape[-1])

        # Weight matrix W with Glorot (Xavier) initialization
        self.W = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            trainable=True,
            name="kernel"
        )
        # Bias vector b
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                trainable=True,
                name="bias"
            )
        else:
            self.b = None

    def call(self, inputs):
        # Compute XW
        outputs = tf.matmul(inputs, self.W)

        # Add bias if enabled
        if self.b is not None:
            outputs = outputs + self.b
        if self.bound:
            outputs = tf.multiply(outputs, self.bound)
        outputs = tf.nn.sigmoid(outputs)
        return outputs


@tf.keras.utils.register_keras_serializable(package="custom_layers", name='QDense')
class SpatialTemporalDropout1D(keras.layers.Layer):
    """Drops entire feature channels AND entire timesteps."""
    def __init__(self, feature_rate=0.1, timestep_rate=0.1, name='spatial_dropout', **kwargs):
        super().__init__(**kwargs)
        self.feature_rate = feature_rate
        self.timestep_rate = timestep_rate
        self.name = name
    @tf.function(reduce_retracing=True)
    def call(self, inputs, training=None):
        if not training:
            return inputs
        shape = tf.shape(inputs)  # (B, T, F)
        # Drop entire features: mask shape (B, 1, F)
        feature_mask = tf.random.uniform((shape[0], 1, shape[2])) > self.feature_rate
        # Drop entire timesteps: mask shape (B, T, 1)
        timestep_mask = tf.random.uniform((shape[0], shape[1], 1)) > self.timestep_rate
        # Combine masks
        combined_mask = tf.cast(feature_mask & timestep_mask, inputs.dtype)
        # Scale to maintain expected value
        keep_prob = (1 - self.feature_rate) * (1 - self.timestep_rate)
        return inputs * combined_mask / keep_prob
    def get_config(self):
        config = super().get_config()
        config.update({
            "feature_rate": self.feature_rate,
            "timestep_rate": self.timestep_rate,
            'name': self.name
        })
        return config



def log1mexp(x):
    """
    Compute log(1 - exp(x)) in a numerically stable way.
    
    This function handles two regimes:
    - When x is very negative: exp(x) ≈ 0, so log(1 - exp(x)) ≈ log(1) = 0
      Here we use log1p(-exp(x)) which is stable for small exp(x)
    - When x is close to 0: exp(x) ≈ 1, so 1 - exp(x) ≈ 0
      Here we use log(-expm1(x)) where expm1(x) = exp(x) - 1 is stable near 0
    
    The crossover point is at x = log(0.5) ≈ -0.693
    
    Args:
        x: Input tensor, should be negative (since we're computing log(1 - p) where p < 1)
    
    Returns:
        log(1 - exp(x))
    """
    threshold = tf.constant(-0.693, dtype=x.dtype)  # log(0.5)
    
    # Clip x to avoid edge cases at extreme values
    x = tf.clip_by_value(x, -100.0, -1e-7)
    
    return tf.where(
        x < threshold,
        tf.math.log1p(-tf.exp(x)),     # Stable when exp(x) is small (x very negative)
        tf.math.log(-tf.math.expm1(x))  # Stable when exp(x) is close to 1 (x close to 0)
    )


class LogSpaceLikelihood(keras.losses.Loss):
    """
    Numerically stable likelihood computation using log-space arithmetic.
    
    Instead of working with probabilities directly, this implementation
    works with log-probabilities throughout, converting products to sums
    and avoiding underflow/overflow issues.
    """
    
    def __init__(
        self, 
        donor_mhc: tf.Tensor, #(N, A) binary that is 1 if a donor has one mhc and 0 if not
        pad_token: int | float = -2,
        test_mode: bool = False,
        use_softmax_loss: bool = True,
        softmax_loss_weight: float = 1.0,
        softmax_temperature: float = 1.0,
        fix_q = True, #If True, Q will be fixed to the calculations in the document $q_i = |\mathcal{N}_i| A  / \big(|\mathcal{N}_i| A + 24 N\big)$
        num_mhc=358, # total number of mhcs in the dataset. Based on the input data, some mhcs might not be available in the data, and they will be masked out.
        **kwargs
    ):
        super().__init__()
        self.donor_mhc = tf.cast(donor_mhc, dtype=tf.float32)
        self.pad_token = tf.cast(pad_token, tf.int32)
        self.test_mode = test_mode
        self.use_softmax_loss = use_softmax_loss
        self.softmax_loss_weight = softmax_loss_weight
        self.softmax_temperature = softmax_temperature
        self.N = 702. # total number of donors
        self.fix_q = fix_q
        
        # Precompute Na and log(Na) for the second term
        # Na is the count of donors carrying each allele, shape (A,)
        self.Na = tf.reduce_sum(self.donor_mhc, axis=0)  # (A,)
        self.log_Na = tf.math.log(self.Na + 1e-10)  # (A,)
        # Mask for valid alleles (present in cohort)
        self.valid_allele_mask = tf.cast(self.Na > 0, tf.float32)  # (A,)
        self.BCE = keras.losses.BinaryCrossentropy(from_logits=True, reduction=None)
        self.A = tf.constant(num_mhc, dtype=tf.float32)
        self.log_A = tf.math.log(self.A)
    
    def call(self, gamma_logits, q_logits, gamma_donor_id, delta_logits):
        # Ensure q_logits has correct shape
        if len(tf.shape(q_logits)) == 2:
            q_logits = tf.squeeze(q_logits, axis=-1)  # (B,)
        
        # Compute log probabilities in numerically stable way from logits:
        # The aim is not to directly convert them to probabilities, because they can become too small
        # Assume z is the logits
        # log(γ) = log(sigmoid(z)) = log_sigmoid(z)
        # log(1 - γ) = log(1 - sigmoid(z)) = log(sigmoid(-z))
        log_gamma = tf.math.log_sigmoid(gamma_logits)  # (B, A)
        log_one_minus_gamma = tf.math.log_sigmoid(-gamma_logits)  # (B, A)
        log_q = tf.math.log_sigmoid(q_logits)  # (B,)
            
        # Get donor information
        Ni_size, Ni, gamma_donor_id_mask = self.calculate_Ni_Nisize(gamma_donor_id) #(B,), (B,N_i,A) masked out, (B,N_i) pads == 0, normal == 1
        if self.fix_q: 
            log_q = tf.exp(log_q) * 0. # zero outing the q (B,)
            # calculate log(q_i) = log(Ni_size * A) - log(Ni_size * A + 24N)
            log_numerator = tf.math.log(Ni_size) + self.log_A #(B,) + scalar
            log_N_times_24 = tf.math.log(24.0) + tf.math.log(self.N) #(1,) + (1,)
            log_N_times_24 = tf.broadcast_to(log_N_times_24, tf.shape(log_numerator)) #(B,)
            log_denominator = tf.math.reduce_logsumexp(tf.stack([log_numerator, log_N_times_24], axis=-1), axis=-1) #(B,)
            term = log_numerator - log_denominator
            log_q = log_q + term #(B,)
        
        # ===== FIRST TERM: Sum over donors containing each TCR =====
        # Compute log(p_ni) = log(1 - prod_a(1-γ)^x_na)
        log_p_ni = self.calculate_log_pni(Ni, log_one_minus_gamma, gamma_donor_id_mask)
        
        # Compute log(q * p_ni) = log(q) + log(p_ni)
        log_qp = log_q[:, tf.newaxis] + log_p_ni  # (B, N_i)
        
        # Compute log(1 - q * p_ni) using log1mexp
        log_one_minus_qp = log1mexp(log_qp)  # (B, N_i)
        
        # Log-odds: log(qp / (1-qp)) = log(qp) - log(1-qp)
        log_odds = log_qp - log_one_minus_qp  # (B, N_i)
        
        # Apply mask and sum over donors
        first_term = log_odds * gamma_donor_id_mask  # (B, N_i)
        first_term = tf.reduce_sum(first_term, axis=-1)  # (B,)

        # ===== SECOND TERM: Penalty term =====
        # q * sum_a(Na * γ)
        # It's a sum (not a product) and is less numerically sensitive
        # We compute: exp(log_q + logsumexp(log_Na + log_gamma))
        #           = exp(log_q) * sum_a(exp(log_Na + log_gamma))
        #           = q * sum_a(Na * γ)
        # Note: self.log_Na has shape (A,), log_gamma has shape (B, A)
        # Broadcasting: (A,) + (B, A) -> (B, A)
        log_Na_gamma = self.log_Na[tf.newaxis, :] + log_gamma  # (B, A)
        log_sum_Na_gamma = tf.reduce_logsumexp(log_Na_gamma, axis=-1)  # (B,)
        log_second_term = log_q + log_sum_Na_gamma  # (B,)
        second_term = tf.math.exp(log_second_term)  # (B,)

        # ===== LIKELIHOOD =====
        LL_batch = first_term - second_term  # (B,)

        # ===== SOFTMAX LOSS (optional) =====
        if self.use_softmax_loss:
            true_probs = self.delta_loss(Ni) #(B,A)
            bce = self.BCE(true_probs, delta_logits)

        else:
            bce = tf.zeros_like(LL_batch)
        
        # regularization term
        reg_term = self.regularization_term_alleles_notavail(gamma_logits) #(B,)
        # ===== TOTAL LOSS =====
        if not self.test_mode:
            return -LL_batch + reg_term, self.softmax_loss_weight * bce #(B,) , (B)

        else:
            return (Ni_size, Ni, gamma_donor_id_mask, log_p_ni,
                    log_qp, log_one_minus_qp, first_term, second_term, bce,
                    log_gamma, log_one_minus_gamma, true_probs, tf.nn.sigmoid(delta_logits))

    def calculate_log_pni(self, Ni, log_one_minus_gamma, gamma_donor_id_mask):
        """
        Compute log(p_ni) = log(1 - prod_a(1-γ)^x_na) in a numerically stable way.

        Args:
            Ni: HLA profiles of donors, shape (B, N_i, A), binary values
            log_one_minus_gamma: log(1-γ), shape (B, A)
            gamma_donor_id_mask: Mask for valid donors, shape (B, N_i) --> pads are zero, rest are 1
        
        Returns:
            log(p_ni), shape (B, N_i)
        """
        # p_ni = 1 - prod_a(1-γ_ia)^x_na
        # log(p_ni) = log(1 - exp(sum_a(x_na * log(1-γ_ia))))

        # Expand log(1-γ) from (B, A) to (B, N_i, A) for element-wise multiplication with Ni
        log_omg_expanded = tf.expand_dims(log_one_minus_gamma, axis=1)  # (B, 1, A)
        log_omg_expanded = tf.broadcast_to(log_omg_expanded, tf.shape(Ni))  # (B, N_i, A)
        
        # log(prod_a(1-γ)^x_na) = sum_a(x_na * log(1-γ))
        # Ni contains x_na values (0 or 1), so this selects only alleles the donor has
        log_prod = tf.reduce_sum(Ni * log_omg_expanded, axis=-1)  # (B, N_i)
        
        # log(p_ni) = log(1 - exp(log_prod)) = log1mexp(log_prod)
        log_p_ni = log1mexp(log_prod)  # (B, N_i)
        
        # Mask out log_p_ni for donors that do not exist, by setting them to -100.0
        # This represents effectively log(0) and will be masked out in the sum
        # gamma_donor_id_mask stores pads as 0. and valid donors as 1.
        log_p_ni = tf.where(
            gamma_donor_id_mask > 0.5, 
            log_p_ni, 
            tf.constant(-100.0, dtype=log_p_ni.dtype)
        )
        
        return log_p_ni  # masked out with dim (B, N_i)

    def calculate_Ni_Nisize(self, gamma_donor_id):
        """Calculate donor counts and gather HLA profiles."""
        gamma_donor_id_count = tf.where(gamma_donor_id == self.pad_token, 0., 1.)
        Ni_size = tf.reduce_sum(gamma_donor_id_count, axis=-1)
        
        gamma_donor_id_safe = tf.where(gamma_donor_id == self.pad_token, 0, gamma_donor_id)
        gamma_donor_id_safe = tf.cast(gamma_donor_id_safe, tf.int32)
        gamma_donor_id_mask = tf.where(gamma_donor_id == self.pad_token, 0., 1.)
        
        Ni = tf.gather(self.donor_mhc, gamma_donor_id_safe, axis=0)
        Ni = Ni * tf.cast(gamma_donor_id_mask[:, :, tf.newaxis], Ni.dtype)
        
        return tf.cast(Ni_size, tf.float32), tf.cast(Ni, tf.float32), gamma_donor_id_mask #(B,) (B,N_i,A) masked out and (B,N_i) pads == 0, normal == 1

    def delta_loss(self, Ni):
        """
        Args:
            Ni: HLA profiles of donors, shape (B, N_i, A)
        
        Returns:
            CCE loss per sample, shape (B,)
        """
        #### --- Defining Parameters --- ####
        N = self.N #scalar --> total number of donors
        ##
        log_Na = self.log_Na #(A,) --> for each hla, how many donors we have
        ##
        log_Nia = tf.math.log(tf.reduce_sum(Ni, axis=1) + 1e-7)#(B, A) --> for each tcr, how many hlas we have 
        ## Ni valid (B,1) --> count how many donors we have for a given tcr
        mask_valid = tf.reduce_any(tf.not_equal(Ni, 0), axis=-1)
        mask_valid = tf.cast(mask_valid, tf.float32)  # (B, N_i)
        Ni_valid = tf.reduce_sum(mask_valid, axis=1, keepdims=True)  # (B,1)
        #### --- Calculations --- ####
        # 1- Cohort level background frequencies:
        # f_a = Na / N --> log(f_a) = log(Na) - log(N)
        log_fa = log_Na - tf.math.log(N) # (A,)
        log_fa = tf.clip_by_value(log_fa, clip_value_min=-100., clip_value_max=0.0) # to keep always below 1.

        # 2- TCR i and HLA a co-occurance frequency
        # p_ia = N_ia / N_i --> log(p_ia) = log(N_ia) - log(Ni_valid)
        log_pia = log_Nia - tf.math.log(Ni_valid + 1e-7) #(B, A)
        log_pia = tf.clip_by_value(log_pia, clip_value_min=-100., clip_value_max=0.0) # to keep always below 1.

        # 3- Compute enrichment scores for each allele
        # E_ia =  p_ia / f_a --> log(E_ia) = log(p_ia) - log(f_a)

        log_Eia = log_pia - tf.expand_dims(log_fa, axis=0) #(B, A)
        log_Eia = tf.where(
        self.valid_allele_mask[tf.newaxis, :] > 0.5, #(1,A)
        log_Eia,
        tf.constant(-100., dtype=log_Eia.dtype)  # zero probability
        )
    
        # 4- Compute conditional/binding probability
        # p_a_binds_i = E_ia / SUM_b(E_ib) --> log(p_a_binds_i) = log(E_ia) - log(SUM_b(E_ib))
        #log_p_a_binds_i = log_Eia - tf.reduce_logsumexp(log_Eia, axis=-1, keepdims=True) # (B,A)
        # Calculate loss CCE
        #pred_probs = tf.nn.softmax(delta_logits / self.softmax_temperature, axis=-1) #(B,A)
        #true_probs = tf.exp(log_p_a_binds_i) #(B,A)
        # Compute scaled log odds
        # log E_ia - log max(E_ia)
        
        log_Eia_max = tf.reduce_max(log_Eia, axis=-1, keepdims=True)  #(B,1)
        log_scaled = log_Eia - log_Eia_max  #(B,A)
        true_probs = tf.exp(log_scaled)  #(B,A)

        return true_probs

    def regularization_term_alleles_notavail(self, gamma_logits):
                    # In your training loss
        invalid_mask = 1.0 - self.valid_allele_mask  # 1 for Na=0, 0 otherwise
        gamma_probs = tf.nn.sigmoid(gamma_logits)

        # Penalize non-zero gamma for invalid alleles
        invalid_gamma_penalty = tf.reduce_mean(
            gamma_probs * invalid_mask[tf.newaxis, :],  # (B, A)
            axis=-1  # (B,)
            )
        return invalid_gamma_penalty
    






class LogSpaceExactLikelihood(keras.losses.Loss):
    """
    ===================================================================================
    EXACT LIKELIHOOD COMPUTATION (EQUATION 4)
    ===================================================================================
    
    From the document, Equation 4:
    
        LL_i = Σ_{n ∈ N_i} ln(q_i * p_ni / (1 - q_i * p_ni)) + Σ_{n=1}^{N} ln(1 - q_i * p_ni)
               \___________ First Term ___________________/   \____ Second Term ____/
    
    Where:
        - N_i = set of donors whose sample contains TCR sequence i
        - p_ni = probability that TCR i can bind to an HLA allotype in donor n
        - q_i = probability of observing TCR i given donor has the right HLA
        - N = total number of donors (702 in our case)
    
    And p_ni is defined by Equation 1:
    
        p_ni = 1 - Π_{a=1}^{A} (1 - γ_ia)^{x_na}
    
    Where:
        - γ_ia = probability that TCR i binds to HLA allele a
        - x_na = 1 if donor n has allele a, 0 otherwise
    
    ===================================================================================
    DIFFERENCE FROM EQUATION 8 (SIMPLIFIED LIKELIHOOD)
    ===================================================================================
    
    Equation 8 approximates the second term:
    
        Σ_{n=1}^{N} ln(1 - q_i * p_ni) ≈ -q_i * Σ_{a=1}^{A} N_a * γ_ia
    
    This approximation assumes:
        1. q_i * p_ni << 1 for all donors (so ln(1-x) ≈ -x)
        2. Each TCR binds to only ONE HLA allele (for product-to-sum conversion)
    
    YOUR CONCERN: These assumptions may not hold!
        - For public TCRs with |N_i|=10, q_i ≈ 0.26 (not small!)
        - Cross-reactive TCRs bind multiple alleles
    
    THIS CLASS: Computes the EXACT second term by summing over ALL N donors.
    
    ===================================================================================
    COMPLEXITY TRADE-OFF
    ===================================================================================
    
    Simplified (Eq 8): O(B * |N_i| * A) for first term + O(B * A) for second term
    Exact (Eq 4):      O(B * |N_i| * A) for first term + O(B * N * A) for second term
    
    For B=5000, N=702, A=358: exact version adds ~1.3 billion operations per batch
    """
    
    def __init__(
        self, 
        donor_mhc: tf.Tensor,
        pad_token: int | float = -2,
        test_mode: bool = False,
        use_softmax_loss: bool = True,
        softmax_loss_weight: float = 1.0,
        softmax_temperature: float = 1.0,
        fix_q: bool = True,
        num_mhc: int = 358,
        N: float = 702.,
        **kwargs
    ):
        """
        Initialize the exact likelihood loss.
        
        Args:
            donor_mhc: Binary tensor of shape (N_expanded, A) where:
                - N_expanded = max_patient_id + 1 (may include zero rows for missing IDs)
                - A = number of HLA alleles
                - donor_mhc[n, a] = 1 if donor n has allele a, 0 otherwise
                
                IMPORTANT: In your code, donor_mhc is expanded to handle non-contiguous
                patient IDs. Many rows may be all zeros (invalid donors).
                
            pad_token: Value used to pad variable-length donor ID lists (default: -2)
            
            test_mode: If True, return intermediate tensors for debugging
            
            use_softmax_loss: If True, compute auxiliary BCE loss on delta_logits
            
            softmax_loss_weight: Weight for the auxiliary loss
            
            softmax_temperature: Temperature for softmax in auxiliary loss
            
            fix_q: If True, use fixed formula for q_i instead of learning it:
                   q_i = |N_i| * A / (|N_i| * A + 24N)
                   
            num_mhc: Total number of HLA alleles (A = 358)
            
            N: ACTUAL number of valid donors (702), NOT the expanded array size!
               CHANGE FROM ORIGINAL: Made this an explicit parameter instead of inferring
               from donor_mhc.shape[0], which would give wrong value for expanded arrays.
        """
        super().__init__()
        
        # Store donor HLA profiles
        # Shape: (N_expanded, A) where N_expanded >= N (may have zero rows)
        self.donor_mhc = tf.cast(donor_mhc, dtype=tf.float32)
        
        self.pad_token = tf.cast(pad_token, tf.int32)
        self.test_mode = test_mode
        self.use_softmax_loss = use_softmax_loss
        self.softmax_loss_weight = softmax_loss_weight
        self.softmax_temperature = softmax_temperature
        self.fix_q = fix_q
        
        # CHANGE: Use explicit N parameter, not inferred from array
        # Original code had: self.N = 702. (hardcoded)
        # We keep this behavior but make it a parameter
        self.N = N
        
        # Precompute N_a = number of donors carrying each allele
        # Used in: auxiliary loss (delta_loss) and regularization
        # Shape: (A,)
        self.Na = tf.reduce_sum(self.donor_mhc, axis=0)
        self.log_Na = tf.math.log(self.Na + 1e-10)
        
        # Mask for valid alleles (those present in at least one donor)
        # Shape: (A,) - 1.0 if Na > 0, else 0.0
        self.valid_allele_mask = tf.cast(self.Na > 0, tf.float32)
        
        # =========================================================================
        # NEW: Precompute valid donor mask
        # =========================================================================
        # Problem: donor_mhc may have rows that are all zeros (invalid donors)
        # These occur because patient IDs are non-contiguous and we expanded the array
        # 
        # For invalid donors:
        #   - All x_na = 0
        #   - p_ni = 1 - Π(1-γ)^0 = 1 - 1 = 0
        #   - ln(1 - q*0) = ln(1) = 0 (contributes nothing to second term)
        #
        # But numerically, we need to handle this carefully to avoid log1mexp(0) issues
        #
        # Shape: (N_expanded,) - 1.0 if donor has at least one allele, else 0.0
        donor_allele_count = tf.reduce_sum(self.donor_mhc, axis=-1)  # (N_expanded,)
        self.valid_donor_mask = tf.cast(donor_allele_count > 0, tf.float32)
        self.num_valid_donors = tf.reduce_sum(self.valid_donor_mask)
        # =========================================================================
        
        self.BCE = keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
        self.A = tf.constant(num_mhc, dtype=tf.float32)
        self.log_A = tf.math.log(self.A)
    
    def call(self, gamma_logits, q_logits, gamma_donor_id, delta_logits):
        """
        Compute the exact negative log-likelihood loss.
        
        =======================================================================
        INPUT SHAPES
        =======================================================================
        
        gamma_logits: (B, A)
            - B = batch size (number of TCR sequences in this batch)
            - A = number of HLA alleles (358)
            - These are LOGITS, not probabilities!
            - γ_ia = sigmoid(gamma_logits[i, a])
            
        q_logits: (B,) or (B, 1)
            - LOGITS for q_i probability
            - q_i = sigmoid(q_logits[i])
            - If fix_q=True, these are ignored and q is computed from formula
            
        gamma_donor_id: (B, max_Ni)
            - For each TCR i, lists the donor IDs that contain this TCR
            - max_Ni = maximum number of donors for any TCR in this batch
            - Padded with pad_token (-2) for TCRs with fewer donors
            - Example: TCR 0 appears in donors [5, 12, 47], TCR 1 in [3, 8]
              gamma_donor_id = [[5, 12, 47], [3, 8, -2]]
              
        delta_logits: (B, A)
            - Auxiliary output for BCE loss (helps training)
            - Not used in main likelihood computation
        
        =======================================================================
        OUTPUT
        =======================================================================
        
        If test_mode=False:
            Returns: (negative_LL + regularization, auxiliary_BCE_loss)
            Both have shape (B,)
            
        If test_mode=True:
            Returns tuple of intermediate tensors for debugging
        """
        
        # =====================================================================
        # STEP 1: Prepare q_logits shape
        # =====================================================================
        # Model may output (B, 1), we need (B,)
        if len(tf.shape(q_logits)) == 2:
            q_logits = tf.squeeze(q_logits, axis=-1)
        # q_logits shape: (B,)
        
        # =====================================================================
        # STEP 2: Convert logits to log-probabilities (numerically stable)
        # =====================================================================
        # We work in LOG SPACE to avoid underflow/overflow
        #
        # For logits z:
        #   γ = sigmoid(z) = 1 / (1 + exp(-z))
        #   log(γ) = log(sigmoid(z)) = -log(1 + exp(-z)) = log_sigmoid(z)
        #   log(1-γ) = log(1 - sigmoid(z)) = log(sigmoid(-z)) = log_sigmoid(-z)
        #
        # TensorFlow's log_sigmoid is numerically stable for all z values
        
        log_gamma = tf.math.log_sigmoid(gamma_logits)
        # Shape: (B, A)
        # Values: negative (since γ < 1, log(γ) < 0)
        # Example: if γ = 0.01, log(γ) ≈ -4.6
        
        log_one_minus_gamma = tf.math.log_sigmoid(-gamma_logits)
        # Shape: (B, A)
        # Values: negative (since 1-γ < 1)
        # Example: if γ = 0.01, log(1-γ) ≈ -0.01
        
        log_q = tf.math.log_sigmoid(q_logits)
        # Shape: (B,)
        
        # =====================================================================
        # STEP 3: Get donor information for N_i (donors containing each TCR)
        # =====================================================================
        Ni_size, Ni, gamma_donor_id_mask = self.calculate_Ni_Nisize(gamma_donor_id)
        #
        # Ni_size: (B,)
        #   - Number of donors for each TCR (|N_i|)
        #   - Example: [3, 2] if TCR 0 has 3 donors, TCR 1 has 2
        #
        # Ni: (B, max_Ni, A)
        #   - HLA profiles of donors in N_i
        #   - Ni[i, j, a] = 1 if j-th donor of TCR i has allele a
        #   - Padded rows are all zeros
        #
        # gamma_donor_id_mask: (B, max_Ni)
        #   - 1.0 for valid donors, 0.0 for padded positions
        #   - Example: [[1, 1, 1], [1, 1, 0]] for above example
        
        # =====================================================================
        # STEP 4: Compute q_i (optionally fixed to formula)
        # =====================================================================
        if self.fix_q:
            # Use formula from document:
            # q_i = |N_i| * A / (|N_i| * A + 24N)
            #
            # In log space:
            # log(q_i) = log(|N_i| * A) - log(|N_i| * A + 24N)
            #          = log(|N_i|) + log(A) - log(|N_i| * A + 24N)
            #
            # For the denominator, use logsumexp for stability:
            # log(a + b) = logsumexp([log(a), log(b)])
            
            log_q = tf.zeros_like(log_q)  # Zero out learned q
            
            log_numerator = tf.math.log(Ni_size + 1e-10) + self.log_A
            # Shape: (B,)
            # = log(|N_i|) + log(A)
            
            log_N_times_24 = tf.math.log(24.0) + tf.math.log(self.num_valid_donors)
            # Scalar: log(24 * 702) ≈ 9.73
            
            log_N_times_24 = tf.broadcast_to(log_N_times_24, tf.shape(log_numerator))
            # Shape: (B,)
            
            # log(|N_i|*A + 24N) = logsumexp([log(|N_i|*A), log(24N)])
            log_denominator = tf.math.reduce_logsumexp(
                tf.stack([log_numerator, log_N_times_24], axis=-1), 
                axis=-1
            )
            # Shape: (B,)
            
            log_q = log_numerator - log_denominator
            # Shape: (B,)
            # Example: |N_i|=2, A=358, N=702
            #   numerator = 2 * 358 = 716
            #   denominator = 716 + 24*702 = 716 + 16848 = 17564
            #   q = 716/17564 ≈ 0.041
            #   log(q) ≈ -3.2
        
        # =====================================================================
        # STEP 5: Compute FIRST TERM (sum over donors in N_i)
        # =====================================================================
        # First term: Σ_{n ∈ N_i} ln(q_i * p_ni / (1 - q_i * p_ni))
        #           = Σ_{n ∈ N_i} [ln(q_i * p_ni) - ln(1 - q_i * p_ni)]
        #           = Σ_{n ∈ N_i} [log_q + log_p_ni - ln(1 - exp(log_q + log_p_ni))]
        
        # Step 5a: Compute log(p_ni) for donors in N_i
        log_p_ni_Ni = self.calculate_log_pni(Ni, log_one_minus_gamma, gamma_donor_id_mask)
        # Shape: (B, max_Ni)
        # Values: negative (since p_ni < 1)
        # Padded positions: -100 (represents log(0))
        
        # Step 5b: Compute log(q * p_ni) = log(q) + log(p_ni)
        log_qp_Ni = log_q[:, tf.newaxis] + log_p_ni_Ni
        # Shape: (B, max_Ni)
        # Broadcast: (B, 1) + (B, max_Ni) -> (B, max_Ni)
        
        # Step 5c: Compute log(1 - q * p_ni) using log1mexp
        # Since log_qp_Ni = log(q*p), we have q*p = exp(log_qp_Ni)
        # So log(1 - q*p) = log(1 - exp(log_qp_Ni)) = log1mexp(log_qp_Ni)
        log_one_minus_qp_Ni = log1mexp(log_qp_Ni)
        # Shape: (B, max_Ni)
        
        # Step 5d: Compute log-odds
        log_odds_Ni = log_qp_Ni - log_one_minus_qp_Ni
        # Shape: (B, max_Ni)
        # = ln(q*p / (1 - q*p))
        
        # Step 5e: Apply mask and sum
        first_term = tf.reduce_sum(log_odds_Ni * gamma_donor_id_mask, axis=-1)
        # Shape: (B,)
        # Mask zeros out padded positions before summing
        
        # =====================================================================
        # STEP 6: Compute SECOND TERM (sum over ALL donors) - THIS IS THE KEY CHANGE!
        # =====================================================================
        # Second term: Σ_{n=1}^{N} ln(1 - q_i * p_ni)
        #
        # ORIGINAL (Eq 8): Approximated as -q_i * Σ_a N_a * γ_ia
        # NEW (Eq 4): Compute EXACTLY by summing over all N donors
        #
        # Step 6a: Compute log(p_ni) for ALL donors
        log_p_ni_all = self.calculate_log_pni_all_donors(log_one_minus_gamma)
        # Shape: (B, N_expanded)
        # For each TCR in batch, compute p_ni for every donor in the cohort
        
        # Step 6b: Compute log(q * p_ni) for all donors
        log_qp_all = log_q[:, tf.newaxis] + log_p_ni_all
        # Shape: (B, N_expanded)
        
        # Step 6c: Compute log(1 - q * p_ni) for all donors
        log_one_minus_qp_all = log1mexp(log_qp_all)
        # Shape: (B, N_expanded)
        
        # Step 6d: Apply valid donor mask
        # For invalid donors (zero rows in donor_mhc):
        #   - p_ni = 0 (no alleles to bind)
        #   - ln(1 - q*0) = ln(1) = 0 (no contribution)
        # We explicitly mask to avoid numerical issues from log1mexp
        log_one_minus_qp_all = log_one_minus_qp_all * self.valid_donor_mask[tf.newaxis, :]
        # Shape: (B, N_expanded)
        # Invalid donors now contribute 0 to the sum
        
        # Step 6e: Sum over all donors
        second_term = tf.reduce_sum(log_one_minus_qp_all, axis=-1)
        # Shape: (B,)
        # This is NEGATIVE (sum of log values < 0)
        
        # =====================================================================
        # STEP 7: Compute total log-likelihood
        # =====================================================================
        # LL = first_term + second_term
        #
        # SIGN CONVENTION:
        # - first_term: can be positive or negative (log-odds)
        # - second_term: NEGATIVE (sum of ln(1-x) where x > 0)
        #
        # In Equation 8, second term was SUBTRACTED as positive penalty: LL = first - penalty
        # In Equation 4, second term is ADDED as negative value: LL = first + negative
        # Mathematically equivalent, but be careful with signs!
        
        LL_batch = first_term + second_term
        # Shape: (B,)
        
        # =====================================================================
        # STEP 8: Auxiliary losses (same as original)
        # =====================================================================
        if self.use_softmax_loss:
            true_probs = self.delta_loss(Ni)  # (B, A)
            bce = self.BCE(true_probs, delta_logits)  # (B,)
        else:
            bce = tf.zeros_like(LL_batch)
        
        gamma_probs = tf.nn.sigmoid(gamma_logits)
        reg_term = self.regularization_term_alleles_notavail(gamma_probs)  # (B,)
        reg_term2 = self.regularization_term_penalize_same_gamma_for_one_allele(gamma_probs, Ni_size, gamma_donor_id, lambda_reg=0.1)
        
        # =====================================================================
        # STEP 9: Return loss
        # =====================================================================
        if not self.test_mode:
            # Return NEGATIVE log-likelihood (for minimization)
            return -LL_batch + reg_term , self.softmax_loss_weight * bce, -reg_term2
        else:
            return (Ni_size, Ni, gamma_donor_id_mask, log_p_ni_Ni, log_p_ni_all,
                    log_qp_Ni, log_one_minus_qp_Ni, first_term, second_term, bce,
                    log_gamma, log_one_minus_gamma, true_probs, tf.nn.sigmoid(delta_logits))
    
    def calculate_log_pni_all_donors(self, log_one_minus_gamma):
        """
        =======================================================================
        NEW METHOD: Compute log(p_ni) for ALL donors (for exact second term)
        =======================================================================
        
        This is the KEY ADDITION for exact likelihood.
        
        Recall:
            p_ni = 1 - Π_{a=1}^{A} (1 - γ_ia)^{x_na}
        
        In log space:
            log(Π (1-γ)^x) = Σ x * log(1-γ)   [product -> sum]
            
        Let log_prod = Σ_a x_na * log(1-γ_ia)
        Then p_ni = 1 - exp(log_prod)
        And log(p_ni) = log(1 - exp(log_prod)) = log1mexp(log_prod)
        
        Args:
            log_one_minus_gamma: log(1-γ), shape (B, A)
        
        Returns:
            log(p_ni) for all donors, shape (B, N_expanded)
        
        Tensor operations:
            donor_mhc:           (N_expanded, A)  - all donor HLA profiles
            log_one_minus_gamma: (B, A)           - model predictions
            
            Broadcast multiply:  (1, N_expanded, A) * (B, 1, A) -> (B, N_expanded, A)
            Sum over alleles:    (B, N_expanded, A) -> (B, N_expanded)
        """
        # Expand dimensions for broadcasting
        log_omg_expanded = tf.expand_dims(log_one_minus_gamma, axis=1)
        # Shape: (B, 1, A)
        
        donor_mhc_expanded = tf.expand_dims(self.donor_mhc, axis=0)
        # Shape: (1, N_expanded, A)
        
        # Compute log(Π (1-γ)^x) = Σ x * log(1-γ)
        # Only positions where x_na = 1 contribute (binary multiplication)
        log_prod = tf.reduce_sum(donor_mhc_expanded * log_omg_expanded, axis=-1)
        # Shape: (B, N_expanded)
        #
        # For valid donors: log_prod < 0 (sum of negative terms)
        # For INVALID donors (all x_na = 0): log_prod = 0 (empty sum)
        
        # Handle invalid donors to avoid log1mexp(0) issues
        # When log_prod = 0: exp(0) = 1, so p_ni = 1-1 = 0, log(0) = -inf
        # We set log_prod to -100 for invalid donors, then mask the result
        log_prod = tf.where(
            self.valid_donor_mask[tf.newaxis, :] > 0.5,  # (1, N_expanded)
            log_prod,
            tf.constant(-100.0, dtype=log_prod.dtype)
        )
        # Shape: (B, N_expanded)
        
        # Compute log(p_ni) = log(1 - exp(log_prod))
        log_p_ni = log1mexp(log_prod)
        # Shape: (B, N_expanded)
        # Important: Now for the invalid donors, pni is 1, and log(pni) = 0. This is not interpretable!
        # Why a donor without any HLA should have prob to have all TCRs?
        # we should set them to zero porb, which means log prob of (log_pni) -100
        log_p_ni = tf.where(self.valid_donor_mask[tf.newaxis, :] > 0.5, log_p_ni, tf.constant(-100.0, dtype=log_prod.dtype))
        return log_p_ni
    
    def calculate_log_pni(self, Ni, log_one_minus_gamma, gamma_donor_id_mask):
        """
        Compute log(p_ni) for donors in N_i only (for first term).
        
        SAME AS ORIGINAL - no changes needed.
        
        Args:
            Ni: HLA profiles of donors in N_i
                Shape: (B, max_Ni, A)
                Values: 0 or 1 (binary)
                Padded rows: all zeros
                
            log_one_minus_gamma: log(1-γ)
                Shape: (B, A)
                
            gamma_donor_id_mask: Mask for valid donors
                Shape: (B, max_Ni)
                Values: 1.0 for valid, 0.0 for padded
        
        Returns:
            log(p_ni)
            Shape: (B, max_Ni)
            Padded positions: -100 (log(0))
        """
        # Expand log(1-γ) for broadcasting
        log_omg_expanded = tf.expand_dims(log_one_minus_gamma, axis=1)
        # Shape: (B, 1, A)
        
        log_omg_expanded = tf.broadcast_to(log_omg_expanded, tf.shape(Ni))
        # Shape: (B, max_Ni, A)
        
        # Compute log(Π (1-γ)^x) = Σ x * log(1-γ)
        log_prod = tf.reduce_sum(Ni * log_omg_expanded, axis=-1)
        # Shape: (B, max_Ni)
        
        # Compute log(p_ni) = log(1 - exp(log_prod))
        log_p_ni = log1mexp(log_prod)
        # Shape: (B, max_Ni)
        
        # Mask padded positions
        log_p_ni = tf.where(
            gamma_donor_id_mask > 0.5,
            log_p_ni,
            tf.constant(-100.0, dtype=log_p_ni.dtype)  # log(0)
        )
        
        return log_p_ni
    
    def calculate_Ni_Nisize(self, gamma_donor_id):
        """
        Calculate donor counts and gather HLA profiles for N_i.
        
        SAME AS ORIGINAL - no changes needed.
        
        Args:
            gamma_donor_id: Donor IDs for each TCR
                Shape: (B, max_Ni)
                Values: donor indices, or pad_token (-2) for padding
        
        Returns:
            Ni_size: (B,) - number of valid donors per TCR
            Ni: (B, max_Ni, A) - HLA profiles, masked
            gamma_donor_id_mask: (B, max_Ni) - mask (1=valid, 0=pad)
        """
        # Count valid donors per TCR
        gamma_donor_id_count = tf.where(gamma_donor_id == self.pad_token, 0., 1.)
        Ni_size = tf.reduce_sum(gamma_donor_id_count, axis=-1)
        # Shape: (B,)
        
        # Create safe indices for gather (replace pad_token with 0)
        gamma_donor_id_safe = tf.where(gamma_donor_id == self.pad_token, 0, gamma_donor_id)
        gamma_donor_id_safe = tf.cast(gamma_donor_id_safe, tf.int32)
        
        # Create mask
        gamma_donor_id_mask = tf.where(gamma_donor_id == self.pad_token, 0., 1.)
        # Shape: (B, max_Ni)
        
        # Gather HLA profiles
        Ni = tf.gather(self.donor_mhc, gamma_donor_id_safe, axis=0)
        # Shape: (B, max_Ni, A)
        
        # Apply mask
        Ni = Ni * tf.cast(gamma_donor_id_mask[:, :, tf.newaxis], Ni.dtype)
        
        return tf.cast(Ni_size, tf.float32), tf.cast(Ni, tf.float32), gamma_donor_id_mask
    
    def delta_loss(self, Ni):
        """
        Compute auxiliary target probabilities based on HLA enrichment.
        
        SAME AS ORIGINAL - no changes needed.
        """
        N = self.num_valid_donors
        log_Na = self.log_Na
        
        log_Nia = tf.math.log(tf.reduce_sum(Ni, axis=1) + 1e-7)
        
        mask_valid = tf.reduce_any(tf.not_equal(Ni, 0), axis=-1)
        mask_valid = tf.cast(mask_valid, tf.float32)
        Ni_valid = tf.reduce_sum(mask_valid, axis=1, keepdims=True)
        
        log_fa = log_Na - tf.math.log(N)
        log_fa = tf.clip_by_value(log_fa, -100., 0.0)
        
        log_pia = log_Nia - tf.math.log(Ni_valid + 1e-7)
        log_pia = tf.clip_by_value(log_pia, -100., 0.0)
        
        log_Eia = log_pia - tf.expand_dims(log_fa, axis=0)
        
        log_Eia = tf.where(
            self.valid_allele_mask[tf.newaxis, :] > 0.5,
            log_Eia,
            tf.constant(-100., dtype=log_Eia.dtype)
        )
        
        log_Eia_max = tf.reduce_max(log_Eia, axis=-1, keepdims=True)
        log_scaled = log_Eia - log_Eia_max
        true_probs = tf.exp(log_scaled)
        
        return true_probs
    
    def regularization_term_alleles_notavail(self, gamma_probs):
        """
        Penalize non-zero gamma for alleles not present in cohort.
        
        SAME AS ORIGINAL - no changes needed.
        """
        invalid_mask = 1.0 - self.valid_allele_mask
        
        invalid_gamma_penalty = tf.reduce_mean(
            gamma_probs * invalid_mask[tf.newaxis, :],
            axis=-1
        )
        return invalid_gamma_penalty
    
    def regularization_term_penalize_same_gamma_for_one_allele(self, gamma_probs, Ni_size, gamma_donor_id, lambda_reg=1.):
        '''
        Ri = −λ ∑_a(γia · max (0, 1 − e_ia)) --> enrichment penalyt 0 if enrichment is below expected enrichment, penalize
        E_Mia = na*​∣Ni​∣/N --> expected number of cooccurences
        mia --> when tcr i occures with allele a
        e_ia = M_ia / E_Mia
        inputs:
            gamma_probs: (B, A)
            Ni_size: (B,)
            gamma_donor_id: (B, max_Ni), padded by PAD_TOKEN

        '''
        Na = self.Na #(A,)
        donor_mhc = self.donor_mhc # (N_expanded, A)
        N = self.num_valid_donors # scalar
        mhc_val_mask = tf.expand_dims(self.valid_allele_mask, axis=0) # (1,A)
        # calculate M_ia
        # first multi hot encode (mhe) gamma_donor_id (B, max_Ni) --> (B, N_expanded), invalid indices out of [0,depth−1] are set to zero.
        # make sure pad tokens are invalid (-2.) to become all zero
        gamma_donor_id = tf.where(gamma_donor_id == self.pad_token, -2, gamma_donor_id)
        gamma_donor_id = tf.cast(gamma_donor_id, tf.int32)
        gamma_donor_id_expanded = tf.keras.ops.multi_hot(gamma_donor_id, num_classes=tf.cast(tf.shape(donor_mhc)[0], tf.int32)) #(B, N_expanded), pads are all zero
        # Now map to MHC mhe
        M_ia = tf.matmul(gamma_donor_id_expanded, tf.cast(donor_mhc, tf.float32)) * mhc_val_mask # (B, N_expanded) . (N_expanded, A) --> (B,A) * (B,A), 1> if a is present in donors of i
        # calculate regularization:
        # 1- E_Mia = na*​∣Ni​∣/N
        E_Mia = (Na[tf.newaxis, :] * Ni_size[:, tf.newaxis])  / N # (1,A) * (B,1) --> (B,A) - N is never zero bcz it is our number of patients
        # 2- M_ia / E_Mia
        e_ia = M_ia / (E_Mia + 1e-10) # (B,A), E_Mia can be zero for some alleles, as some alleles have no occurance
        e_ia_masked_out = e_ia * mhc_val_mask # (B,A) * (1,A) = (B,A) zeros out those allaes that are zero
        # 3- γia · max (0, 1 − e_ia)
        R_ia = gamma_probs * tf.math.maximum(0, 1.-e_ia_masked_out) #(B,A)
        # 4- R_i = −λ ∑_a(R_ia)
        R_i = -lambda_reg * tf.reduce_sum(R_ia, axis=-1) #(B,)
        return R_i


        






    