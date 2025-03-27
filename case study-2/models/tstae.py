import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
import os
from tensorflow.keras.saving import register_keras_serializable

@tf.keras.saving.register_keras_serializable()
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention = tf.matmul(q, k, transpose_b=True)
        scaled_attention = scaled_attention / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        
        attention_weights = tf.nn.softmax(scaled_attention, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        return self.dense(output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads
        })
        return config

@tf.keras.saving.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout
        
        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
    
    def call(self, inputs, training=False):
        attn_output = self.mha(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout": self.dropout_rate
        })
        return config

@tf.keras.saving.register_keras_serializable()
class PositionalEncoding(layers.Layer):
    def __init__(self, max_steps, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_steps = max_steps
        self.d_model = d_model
        
        position = tf.range(max_steps, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))
        
        pe = tf.zeros((max_steps, d_model))
        sine_indices = tf.stack([
            tf.repeat(tf.range(max_steps), tf.shape(div_term)),
            tf.tile(tf.range(0, d_model, 2), [max_steps])
        ], axis=1)
        sine_updates = tf.reshape(tf.sin(position * div_term), [-1])
        pe = tf.tensor_scatter_nd_update(pe, sine_indices, sine_updates)
        
        if d_model > 1:
            cosine_indices = tf.stack([
                tf.repeat(tf.range(max_steps), tf.shape(div_term)),
                tf.tile(tf.range(1, d_model, 2), [max_steps])
            ], axis=1)
            cosine_updates = tf.reshape(tf.cos(position * div_term), [-1])
            pe = tf.tensor_scatter_nd_update(pe, cosine_indices, cosine_updates)
        
        self.pe = pe[tf.newaxis, :, :]
        
    def call(self, inputs):
        return inputs + self.pe[:, :tf.shape(inputs)[1], :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "max_steps": self.max_steps,
            "d_model": self.d_model
        })
        return config


@tf.keras.saving.register_keras_serializable()
class TimeSeriesTransformerAutoencoderRegression(Model):
    def __init__(self, 
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 max_seq_len,
                 num_features,
                 output_dim=1,  
                 dropout_rate=0.1,
                 bottleneck_dim=None,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.num_features = num_features
        self.dff = dff
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.bottleneck_dim = bottleneck_dim or (d_model // 2)
        
        # Input projection
        self.input_projection = layers.Dense(d_model)
        self.pos_encoding = PositionalEncoding(max_seq_len, d_model)
        
        # Encoder transformer blocks
        self.encoder_blocks = [
            TransformerBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        
        # Improved bottleneck
        self.bottleneck_down = layers.Dense(self.bottleneck_dim, activation='relu')
        self.bottleneck_norm = layers.LayerNormalization(epsilon=1e-6)
        self.bottleneck_up = layers.Dense(d_model)
        
        # Decoder transformer blocks
        self.decoder_blocks = [
            TransformerBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        
        # Output layers
        self.reconstruction_layer = layers.Dense(num_features)
        self.global_pooling = layers.GlobalAveragePooling1D()
        self.regression_dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.regression_dense2 = layers.Dense(64, activation='relu')
        self.regression_layer = layers.Dense(output_dim)  
    
    def call(self, inputs, training=False, regression_only=False):
        # Encoder
        x = self.input_projection(inputs)
        x = self.pos_encoding(x)
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, training=training)
        
        encoded = x
        
        # Regression branch
        reg_features = self.global_pooling(encoded)
        reg_features = self.regression_dense1(reg_features)
        reg_features = self.dropout(reg_features, training=training)
        reg_features = self.regression_dense2(reg_features)
        regression_output = self.regression_layer(reg_features)
        
        if regression_only:
            return regression_output
        
        # Decoder branch with improved bottleneck
        decoder_features = self.bottleneck_down(encoded)
        decoder_features = self.bottleneck_norm(decoder_features)
        decoder_features = self.bottleneck_up(decoder_features)
        
        for decoder_block in self.decoder_blocks:
            decoder_features = decoder_block(decoder_features, training=training)
        
        reconstructed = self.reconstruction_layer(decoder_features)
        
        return {
            'reconstruction_output': reconstructed,
            'regression_output': regression_output
        }
    
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
            "dropout_rate": self.dropout_rate,
            "bottleneck_dim": self.bottleneck_dim
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        # Remove Keras-specific arguments if present
        kwargs = {k: v for k, v in config.items() 
                 if k not in ['name', 'trainable', 'dtype']}
        return cls(**kwargs)

def create_transformer_autoencoder_model(sequence_length, num_features, num_layers=4, 
                                       d_model=128, num_heads=8, dff=256, 
                                       dropout_rate=0.1, bottleneck_dim=None,
                                       learning_rate=0.001, reconstruction_weight=0.3,
                                       regression_weight=0.7):
    model = TimeSeriesTransformerAutoencoderRegression(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        max_seq_len=sequence_length,
        num_features=num_features,
        dropout_rate=dropout_rate,
        bottleneck_dim=bottleneck_dim
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss={
            'reconstruction_output': 'mse',
            'regression_output': 'huber'
        },
        loss_weights={
            'reconstruction_output': reconstruction_weight,
            'regression_output': regression_weight
        }
    )
    
    return model