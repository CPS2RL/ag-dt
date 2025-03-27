import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def evaluate(model, X_test, y_test, scaler_y, plot_results=True, num_samples=5):

    y_pred = model.predict(X_test)
    
    y_test_orig = scaler_y.inverse_transform(y_test)
    y_pred_orig = scaler_y.inverse_transform(y_pred)
    
    metrics = {
        'MAE': mean_absolute_error(y_test_orig, y_pred_orig),
        'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
        'R2': r2_score(y_test_orig, y_pred_orig)
    }
    
    print("\nMetrics on Test Data:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\nFirst {num_samples} Actual vs Predicted Values:")
    for i in range(min(num_samples, len(y_test_orig))):
        print(f"Sample {i+1}: Actual = {y_test_orig[i][0]:.4f}, Predicted = {y_pred_orig[i][0]:.4f}, " 
              f"Difference = {y_test_orig[i][0] - y_pred_orig[i][0]:.4f}")
    
    
    return metrics, y_test_orig, y_pred_orig