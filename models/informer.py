def build_informer(seq_length, input_features, num_classes=5):

    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras import layers
    from tensorflow.keras.saving import register_keras_serializable
    

    @register_keras_serializable()
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
            
            self.pe = pe[tf.newaxis, :, :]  # Add batch dimension
            
        def call(self, inputs):
            return inputs + self.pe[:, :tf.shape(inputs)[1], :]
        
        def get_config(self):
            config = super().get_config()
            config.update({
                "max_steps": self.max_steps,
                "d_model": self.d_model
            })
            return config

    @register_keras_serializable()
    class ProbSparseAttention(layers.Layer):
        def __init__(self, d_model, num_heads, factor=5, **kwargs):
            super().__init__(**kwargs)
            self.d_model = d_model
            self.num_heads = num_heads
            self.factor = factor
            self.depth = d_model // num_heads
            
            self.wq = layers.Dense(d_model)
            self.wk = layers.Dense(d_model)
            self.wv = layers.Dense(d_model)
            self.dense = layers.Dense(d_model)
        
        def _prob_QK(self, Q, K, sample_k):
            B, H, L_Q, D = tf.shape(Q)[0], tf.shape(Q)[1], tf.shape(Q)[2], tf.shape(Q)[3]
            L_K = tf.shape(K)[2]
            
            Q_K = tf.matmul(Q, K, transpose_b=True)
            Q_K = Q_K / tf.math.sqrt(tf.cast(self.depth, tf.float32))
            
            M = tf.math.reduce_max(Q_K, axis=-1, keepdims=True)
            Q_K = Q_K - M
            Q_K = tf.exp(Q_K)
            
            sample_size = tf.minimum(L_K, sample_k)
            mean_attention = tf.reduce_mean(Q_K, axis=2)
            _, indices = tf.nn.top_k(mean_attention, k=sample_size)
            
            return indices
        
        def call(self, inputs, training=None):
            batch_size = tf.shape(inputs)[0]
            seq_len = tf.shape(inputs)[1]
            
            Q = self.wq(inputs)
            K = self.wk(inputs)
            V = self.wv(inputs)
            
            Q = tf.reshape(Q, (batch_size, -1, self.num_heads, self.depth))
            Q = tf.transpose(Q, perm=[0, 2, 1, 3])
            K = tf.reshape(K, (batch_size, -1, self.num_heads, self.depth))
            K = tf.transpose(K, perm=[0, 2, 1, 3])
            V = tf.reshape(V, (batch_size, -1, self.num_heads, self.depth))
            V = tf.transpose(V, perm=[0, 2, 1, 3])
            
            L_K = tf.shape(K)[2]
            sample_k = tf.cast(tf.math.log(tf.cast(L_K, tf.float32)) * self.factor, tf.int32)
            sample_k = tf.minimum(sample_k, L_K)
            
            indices = self._prob_QK(Q, K, sample_k)
            
            batch_indices = tf.range(batch_size)[:, tf.newaxis, tf.newaxis]
            batch_indices = tf.tile(batch_indices, [1, self.num_heads, sample_k])
            head_indices = tf.range(self.num_heads)[tf.newaxis, :, tf.newaxis]
            head_indices = tf.tile(head_indices, [batch_size, 1, sample_k])
            
            gather_indices = tf.stack([batch_indices, head_indices, indices], axis=-1)
            
            K_sampled = tf.gather_nd(K, gather_indices)
            V_sampled = tf.gather_nd(V, gather_indices)
            
            attention_scores = tf.matmul(Q, K_sampled, transpose_b=True)
            attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.depth, tf.float32))
            
            attention_weights = tf.nn.softmax(attention_scores, axis=-1)
            output = tf.matmul(attention_weights, V_sampled)
            
            output = tf.transpose(output, perm=[0, 2, 1, 3])
            output = tf.reshape(output, (batch_size, -1, self.d_model))
            
            return self.dense(output)
        
        def get_config(self):
            config = super().get_config()
            config.update({
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "factor": self.factor
            })
            return config

    @register_keras_serializable()
    class InformerBlock(layers.Layer):
        def __init__(self, d_model, num_heads, dff, dropout=0.1, factor=5, **kwargs):
            super().__init__(**kwargs)
            self.d_model = d_model
            self.num_heads = num_heads
            self.dff = dff
            self.dropout_rate = dropout
            self.factor = factor
            
            self.prob_attention = ProbSparseAttention(d_model, num_heads, factor)
            self.ffn = tf.keras.Sequential([
                layers.Dense(dff, activation='relu'),
                layers.Dense(d_model)
            ])
            
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
            
            self.dropout1 = layers.Dropout(dropout)
            self.dropout2 = layers.Dropout(dropout)
        
        def call(self, inputs, training=None):
            attn_output = self.prob_attention(inputs, training=training)
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
                "dropout": self.dropout_rate,
                "factor": self.factor
            })
            return config

    @register_keras_serializable()
    class TimeSeriesInformer(Model):
        def __init__(self, 
                    num_layers,
                    d_model,
                    num_heads,
                    dff,
                    max_seq_len,
                    num_features,
                    num_classes,
                    dropout_rate=0.1,
                    factor=5,
                    **kwargs):
            super().__init__(**kwargs)
            
            self.num_layers = num_layers
            self.d_model = d_model
            self.num_heads = num_heads
            self.max_seq_len = max_seq_len
            self.num_features = num_features
            self.num_classes = num_classes
            self.dropout_rate = dropout_rate
            self.factor = factor
            self.dff = dff
            
            self.input_projection = layers.Dense(d_model)
            self.pos_encoding = PositionalEncoding(max_seq_len, d_model)
            
            self.informer_blocks = [
                InformerBlock(d_model, num_heads, dff, dropout_rate, factor)
                for _ in range(num_layers)
            ]
            
            self.dropout = layers.Dropout(dropout_rate)
            self.global_pooling = layers.GlobalAveragePooling1D()
            self.final_layer = layers.Dense(num_classes, activation='softmax')
            
        def call(self, inputs, training=None):
            x = self.input_projection(inputs)
            x = self.pos_encoding(x)
            
            for informer_block in self.informer_blocks:
                x = informer_block(x, training=training)
            
            x = self.global_pooling(x)
            x = self.dropout(x, training=training)
            
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
                "num_classes": self.num_classes,
                "dropout_rate": self.dropout_rate,
                "factor": self.factor
            })
            return config
            
        @classmethod
        def from_config(cls, config):
            return cls(**config)

    # Model parameters
    num_layers = 4
    d_model = 128
    num_heads = 8
    dff = 256
    max_seq_len = seq_length
    num_features = input_features
    dropout_rate = 0.1
    factor = 5

    # Create and compile the Informer model
    ts_informer = TimeSeriesInformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        max_seq_len=max_seq_len,
        num_features=num_features,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        factor=factor
    )

    # Compile the model
    ts_informer.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return ts_informer