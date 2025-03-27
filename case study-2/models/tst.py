import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package="TimeSeriesTransformer")
class PositionalEncoding(layers.Layer):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Create positional encoding matrix in __init__ instead of build
        position = np.arange(max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((max_seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        # Create as a tf Variable instead of a constant
        self.pos_encoding = self.add_weight(
            name="positional_encoding",
            shape=(max_seq_len, d_model),
            initializer=tf.constant_initializer(pos_encoding),
            trainable=False
        )
        
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        # Make sure we don't exceed the maximum sequence length
        seq_len = tf.minimum(seq_len, self.max_seq_len)
        return inputs + tf.gather(self.pos_encoding, tf.range(seq_len))
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model
        })
        return config

# Multi-Head Attention with scaling
@register_keras_serializable(package="TimeSeriesTransformer")
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Masking if needed
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        output = self.dense(output)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads
        })
        return config

# Transformer Block
@register_keras_serializable(package="TimeSeriesTransformer")
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn1 = layers.Dense(dff, activation='relu')
        self.ffn2 = layers.Dense(d_model)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        # Multi-head attention with residual connection
        attn_output = self.mha(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed forward with residual connection
        ffn_output = self.ffn1(out1)
        ffn_output = self.ffn2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate
        })
        return config

# Time Series Transformer for Regression
@register_keras_serializable(package="Custom")
class TimeSeriesTransformer(Model):
    def __init__(self, 
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 max_seq_len,
                 num_features,
                 output_dim=1,
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.num_features = num_features
        self.output_dim = output_dim
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        # Input projection
        self.input_projection = layers.Dense(d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(max_seq_len, d_model)
        
        # Transformer blocks
        self.transformer_blocks = []
        for _ in range(num_layers):
            self.transformer_blocks.append(
                TransformerBlock(d_model, num_heads, dff, dropout_rate)
            )
        
        # Output layers
        self.dropout = layers.Dropout(dropout_rate)
        self.global_pooling = layers.GlobalAveragePooling1D()
        
        # For regression: no activation function
        self.final_layer = layers.Dense(output_dim)
        
    def call(self, inputs, training=False):
        # Input projection
        x = self.input_projection(inputs)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        
        # Global pooling
        x = self.global_pooling(x)
        x = self.dropout(x, training=training)
        
        # Final output (regression)
        return self.final_layer(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "max_seq_len": self.max_seq_len,
            "num_features": self.num_features,
            "output_dim": self.output_dim,
            "dropout_rate": self.dropout_rate
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        kwargs = {k: v for k, v in config.items() 
                 if k not in ['name', 'trainable', 'dtype']}
        return cls(**kwargs)

def create_transformer_model(sequence_length, num_features, num_layers=4, d_model=128, 
                           num_heads=8, dff=256, dropout_rate=0.1, learning_rate=0.001):
    model = TimeSeriesTransformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        max_seq_len=sequence_length,
        num_features=num_features,
        output_dim=1,
        dropout_rate=dropout_rate
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mae')
    
    return model