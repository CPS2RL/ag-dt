import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.saving import register_keras_serializable

# Load your data (assuming it’s the data you provided)
data = pd.read_csv("data/quincy_faulty-test.csv")  # Replace with your file path or use the provided text
data = data.ffill()
features = data.drop(columns=["timestamp_utc", "datetime", "Class"])
labels = data["Class"].map({"clean": 0, "random": 1, "drift": 2, "malfunction": 3, "bias": 4}).values

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Create sequences
def create_sequences(data, labels, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(labels[i + seq_length - 1])
    return np.array(X), np.array(y)

seq_length = 48
num_classes=5
X_test, y_test = create_sequences(features_scaled, labels, seq_length)

# Convert labels to one-hot encoding
y_test = to_categorical(y_test, num_classes=5)


@register_keras_serializable()
class PositionalEncoding(layers.Layer):
    def __init__(self, max_steps, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_steps = max_steps
        self.d_model = d_model
        
        # Create positional encoding matrix once during initialization
        position = tf.range(max_steps, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))
        
        pe = tf.zeros((max_steps, d_model))
        # Use scatter_nd to update sine values
        sine_indices = tf.stack([
            tf.repeat(tf.range(max_steps), tf.shape(div_term)),
            tf.tile(tf.range(0, d_model, 2), [max_steps])
        ], axis=1)
        sine_updates = tf.reshape(tf.sin(position * div_term), [-1])
        pe = tf.tensor_scatter_nd_update(pe, sine_indices, sine_updates)
        
        # Use scatter_nd to update cosine values
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
num_features = X_test.shape[2]
num_classes = 5
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

loaded_model = tf.keras.models.load_model('model/resnet.keras')


import time
import numpy as np

def measure_inference_time(model, X_test, y_test):
    """
    Measure inference time
    """
    # Measure baseline overhead
    c = 0
    t1 = time.time()
    for i in range(10):
        x = 2
    t1 = (time.time() - t1) / 10
    
    # Store timing results
    arr = []
    
    # Process each test sample
    for i in range(len(X_test[10])):
        # Get single test sample
        test_sample = X_test[i:i+1]  # Keep the batch dimension
        
        # Measure inference time with 50 iterations
        t2 = time.time()
        for _ in range(10):
            _ = model.predict(test_sample, verbose=0)
        t2 = (time.time() - t2) / 10
        
        # Calculate net inference time
        inference_time = t2 - t1
        arr.append(inference_time)
    
    # Calculate statistics
    avg_time = np.mean(arr)
    std_time = np.std(arr)
    
    #print("\nInference Time Statistics:")
    print(f"Average inference time per sample: {avg_time:.4f} seconds")
    #print(f"Standard deviation: {std_time:.4f} seconds")
    #print(f"Min time: {min(arr):.4f} seconds")
    #print(f"Max time: {max(arr):.4f} seconds")
    
    return arr

import numpy as np
import psutil
from memory_profiler import memory_usage

def measure_subset_memory_usage(model, X_test, start_idx=0, num_samples=100, num_runs=10):
    """
    Measure memory usage for processing a subset of test data
    
    Parameters:
    -----------
    model : ML model object
        The model to evaluate
    X_test : numpy.ndarray
        Test data
    start_idx : int
        Starting index for the subset
    num_samples : int
        Number of samples to include in the subset
    num_runs : int
        Number of times to repeat the measurement for reliability
    """
    memory_results = []
    
    # Get test subset using array indexing
    X_subset = X_test[start_idx:start_idx + num_samples]
    
    # Get baseline memory
    baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    print(f"Baseline memory: {baseline_memory:.2f} MB")
    
    print(f"\nMeasuring memory usage for {len(X_subset)} samples ({num_runs} runs)...")
    
    # Function to measure
    def predict_subset():
        return model.predict(X_subset)
    
    # Repeat measurement multiple times
    for i in range(num_runs):
        # Memory profiling for the subset
        mem_usage = memory_usage(
            (predict_subset, (), {}),
            interval=0.005,  # Adjusted to 5ms sampling interval
            max_iterations=1,
            include_children=True
        )
        
        # Calculate peak memory usage for this run
        peak_memory = max(mem_usage) - baseline_memory
        memory_results.append(peak_memory)
        print(f"Run {i+1}/{num_runs}: Peak memory usage: {peak_memory:.2f} MB")
    
    # Calculate statistics
    memory_stats = {
        'mean': np.mean(memory_results),
        'std': np.std(memory_results),
        'min': np.min(memory_results),
        'max': np.max(memory_results),
        'per_sample_mean': np.mean(memory_results) / len(X_subset)
    }
    
    #print("\nMemory Usage Statistics (for subset):")
    #print(f"Subset size: {len(X_subset)} samples")
    #print(f"Average peak memory for subset: {memory_stats['mean']:.2f} MB")
    #print(f"Standard deviation: {memory_stats['std']:.2f} MB")
    #print(f"Min peak memory: {memory_stats['min']:.2f} MB")
    #print(f"Max peak memory: {memory_stats['max']:.2f} MB")
    print(f"Average memory per sample: {memory_stats['per_sample_mean']:.4f} MB")
    
    return {
        'memory_results': memory_results,
        'memory_stats': memory_stats,
        'baseline_memory': baseline_memory,
        'subset_size': len(X_subset)
    }
    

print("Starting inference time measurement...")
inference_times = measure_inference_time(model=loaded_model, X_test=X_test, y_test=y_test)

# Usage example:
#print("Starting memory profiling for subset...")
#memory_metrics = measure_subset_memory_usage(model=loaded_model,X_test=X_test,start_idx=0,num_samples=5000, num_runs= 10)

