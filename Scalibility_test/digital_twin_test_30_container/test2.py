import numpy as np
import psutil
from memory_profiler import memory_usage


def measure_subset_memory_usage(model, X_test, start_idx=0, num_samples=100, num_runs=10):
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
    
    print("\nMemory Usage Statistics (for subset):")
    print(f"Subset size: {len(X_subset)} samples")
    print(f"Average peak memory for subset: {memory_stats['mean']:.2f} MB")
    print(f"Standard deviation: {memory_stats['std']:.2f} MB")
    print(f"Min peak memory: {memory_stats['min']:.2f} MB")
    print(f"Max peak memory: {memory_stats['max']:.2f} MB")
    print(f"Average memory per sample: {memory_stats['per_sample_mean']:.4f} MB")
    
    return {
        'memory_results': memory_results,
        'memory_stats': memory_stats,
        'baseline_memory': baseline_memory,
        'subset_size': len(X_subset)
    }
    
    
