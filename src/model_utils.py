import keras
from keras import layers
import tensorflow as tf



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
            initializer="glorot_uniform",
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
            initializer='glorot_uniform',
            trainable=True,
            name=f'q_proj_{self.name}'
        )
        self.k_proj = self.add_weight(
            shape=(self.heads, self.context_dim, self.att_dim),
            initializer='glorot_uniform',
            trainable=True,
            name=f'k_proj_{self.name}'
        )
        self.v_proj = self.add_weight(
            shape=(self.heads, self.context_dim, self.att_dim),
            initializer='glorot_uniform',
            trainable=True,
            name=f'v_proj_{self.name}'
        )
        
        if self.gate:
            self.g = self.add_weight(
                shape=(self.heads, self.query_dim, self.att_dim),
                initializer='random_uniform',
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
            initializer='glorot_uniform',
            trainable=True,
            name=f'outw_{self.name}'
        )
        self.out_b = self.add_weight(
            shape=(self.output_dim,),
            initializer='zeros',
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
        })
        return config



class Likelihood(keras.losses.Loss):
    def __init__(self, donor_mhc: tf.Tensor, 
                 pad_token = -1, **kwargs):
        super().__init__()
        self.donor_mhc = tf.constant(donor_mhc) #(N, A)
        self.pad_token = tf.cast(pad_token, tf.int32)
        assert len(tf.shape(self.donor_mhc)) == 2, f'the shape of donor_mhc should be (donor, mhc_allele), found {tf.shape(donor_mhc)}'
        self.Na = tf.cast(tf.reduce_sum(self.donor_mhc, axis=0), tf.float32) #(A,)

    def call(self, gamma, q, gamma_donor_id):
        '''
        gamma: output of model. dimension (batch, mhc_allele) or (B,A)
        q: output of model. dimension (batch,)
        gamma_donor_id: padded integers of donor ids per each tcr. dimension (batch, padded_donor_id) or (B, D_i). map tcr to donors.
        '''
        #### Calculate The second Term ####
        if len(tf.shape(q)) == 2: q = tf.squeeze(q, axis=-1) #(B,1) --> (B,)
        gamma = tf.clip_by_value(gamma, 1e-7, 1.0 - 1e-7)
        q = tf.clip_by_value(q, 1e-7, 1.0 - 1e-7)
        # |Ni_size| * Sum^A( Na * gamma_ia )
        Ni_size, Ni, gamma_donor_id_mask = self.calculate_Ni_Nisize(gamma_donor_id) #(B,) and (B, N_i, A) and (B, N_i)
        second_term = q * Ni_size * tf.reduce_sum(self.Na[tf.newaxis, :] * gamma, axis=-1) #(B,) * (B,) * Sum^A ( (B, A) * (B, A)) --> (B,)

        #### Calculate The first Term ####
        # Sum^Ni (ln( qi*pni / 1 - qi*pni))
        ## calculate pni: 1 - Prod( 1 - gamma_ia) ** x_na
        pn = self.calculate_pni(Ni, gamma, gamma_donor_id_mask) # (B, N_i)
        numerator = tf.multiply(q[:, tf.newaxis], pn) # (B,1) * (B, N_i)
        denominator = 1. - numerator
        first_term = tf.math.log(numerator + 1e-10) - tf.math.log(denominator + 1e-10)
        # apply mask, because padded ones are now log(0) == 1 
        first_term = first_term * gamma_donor_id_mask
        first_term = tf.reduce_sum(first_term, axis=-1) #(B, N_i) --> (B,)
        
        LL_batch = first_term - second_term
        return -tf.reduce_mean(LL_batch) #-tf.reduce_sum(LL_batch) 

    def calculate_Ni_Nisize(self, gamma_donor_id): #(B, max)
        # calculate count of donors per each tcr
        gamma_donor_id_count = tf.where(gamma_donor_id == self.pad_token, 0., 1.) #(B, pad_donor_id)
        Ni_size = tf.reduce_sum(gamma_donor_id_count, axis=-1) #(B,) each has the number of Ni 
        ## extract (Ni, A) from (N, A)
        # First, we need to create a masking for padded tokens
        gamma_donor_id_converted = tf.where(gamma_donor_id == self.pad_token, 0, gamma_donor_id) # just to make sure tf.gather does not raise error
        gamma_donor_id_converted = tf.cast(gamma_donor_id_converted, dtype=tf.int32)
        gamma_donor_id_mask = tf.where(gamma_donor_id == self.pad_token, 0., 1) #(B,D_i) or (B,N_i)
        Ni = tf.gather(self.donor_mhc, gamma_donor_id_converted, axis=0) # (N, A), (B,D_i) --> (B, N_i, A) N_i are simply gathered D_is , D_i is index and is padded. len(N_i) == len(D_i)
        Ni = tf.multiply(Ni, tf.cast(gamma_donor_id_mask[:, :, tf.newaxis], tf.int32)) #(B,N_i,A) masked out
        return tf.cast(Ni_size, tf.float32), tf.cast(Ni, tf.float32), gamma_donor_id_mask

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


@tf.keras.utils.register_keras_serializable(package="custom_layers")
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