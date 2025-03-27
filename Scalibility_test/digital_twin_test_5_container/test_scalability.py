import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import numpy as np
import pandas as pd
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
import os
import socket
import csv
import time
import fcntl
import psutil
from memory_profiler import memory_usage

# Load your data
data = pd.read_csv("data/quincy_faulty-test.csv")
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

loaded_model = tf.keras.models.load_model('model/lstm.keras')

def measure_inference_time(model, X_test, y_test):
    """
    Measure inference time
    """
    # Measure baseline overhead
    c = 0
    t1 = time.time()
    for i in range(100):
        x = 2
    t1 = (time.time() - t1) / 100
    
    arr = []
    
    # Process each test sample
    for i in range(min(len(X_test), 10))
        # Get single test sample
        test_sample = X_test[i:i+1]  # Keep the batch dimension
        
        # Measure inference time with 100 iterations
        t2 = time.time()
        for _ in range(100):
            _ = model.predict(test_sample, verbose=0)
        t2 = (time.time() - t2) / 100
        
        # Calculate net inference time
        inference_time = t2 - t1
        arr.append(inference_time)
    
    # Calculate statistics
    avg_time = np.mean(arr)
    std_time = np.std(arr)
    
    print(f"Average inference time per sample: {avg_time:.4f} seconds")
    
    return arr, avg_time

def measure_subset_memory_usage(model, X_test, start_idx=0, num_samples=5000, num_runs=10):
    memory_results = []
    
    # Get test subset using array indexing
    X_subset = X_test[start_idx:start_idx + num_samples]
    
    # Get baseline memory
    baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

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
    memory_stats = {
        'mean': np.mean(memory_results),
        'std': np.std(memory_results),
        'min': np.min(memory_results),
        'max': np.max(memory_results),
        'per_sample_mean': np.mean(memory_results) / len(X_subset)
    }

    print(f"Average memory per sample: {memory_stats['per_sample_mean']:.4f} MB")
    
    return {
        'memory_results': memory_results,
        'memory_stats': memory_stats,
        'baseline_memory': baseline_memory,
        'subset_size': len(X_subset)
    }

def save_to_csv(avg_inference_time, memory_data):
    # Get container hostname
    hostname = socket.gethostname()
    
    # Extract only the memory per sample metric
    avg_memory_per_sample = memory_data['memory_stats']['per_sample_mean']
    
    # Ensure results directory exists
    os.makedirs('/app/results', exist_ok=True)
    
    # Define the CSV file path
    csv_file = '/app/results/all_5_containers_metrics_lstm.csv'
    
    # Prepare row data (only the requested metrics)
    row = [
        hostname,
        avg_inference_time,
        avg_memory_per_sample
    ]
    
    # Check if file exists
    file_exists = os.path.exists(csv_file)
    
    # Use file locking for safe concurrent access
    with open(csv_file, 'a', newline='') as f:
        # Acquire an exclusive lock
        fcntl.flock(f, fcntl.LOCK_EX)
        
        try:
            writer = csv.writer(f)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow([
                    'container', 
                    'avg_inference_time_seconds',
                    'avg_memory_per_sample_mb'
                ])
            
            # Write the data row
            writer.writerow(row)
            
        finally:
            # Always release the lock
            fcntl.flock(f, fcntl.LOCK_UN)
    
    print(f"Results for container {hostname} saved to CSV file")

# Main execution
print("Starting inference time measurement...")
inference_times, avg_inference_time = measure_inference_time(model=loaded_model, X_test=X_test, y_test=y_test)

print("\nStarting memory usage measurement...")
memory_data = measure_subset_memory_usage(
    model=loaded_model, 
    X_test=X_test, 
    start_idx=0,
    num_samples=5000,
    num_runs=10     
)
save_to_csv(avg_inference_time, memory_data)

print(f"Container {socket.gethostname()} completed testing.")
