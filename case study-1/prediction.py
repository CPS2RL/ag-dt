import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import save_weather_gan, load_weather_gan

def evaluate_predictions(loaded_gan, X_array, y_array, target_datetimes, feature_columns, scaled_features):

    loaded_gan, loaded_feature_columns, loaded_scaled_features = load_weather_gan('weather_gan_model')
    y_pred = loaded_gan.generator.predict(X_array)

    results_df = pd.DataFrame()
    results_df['datetime'] = target_datetimes
    results_df['Date'] = pd.to_datetime(results_df['datetime']).dt.strftime('%m/%d/%Y')
    results_df['Time'] = pd.to_datetime(results_df['datetime']).dt.strftime('%H:%M')

    for i, feature in enumerate(feature_columns):

        feature_scaler = scaled_features[feature]['scaler']
        pred_values = y_pred[:, i].reshape(-1, 1)
        actual_values = y_array[:, i].reshape(-1, 1)
        
        pred_original = feature_scaler.inverse_transform(pred_values)
        actual_original = feature_scaler.inverse_transform(actual_values)
        

        results_df[f'{feature}_Predicted'] = pred_original.flatten()
        results_df[f'{feature}_Actual'] = actual_original.flatten()

 
    column_order = ['Date', 'Time', 'datetime']
    for feature in feature_columns:
        column_order.extend([f'{feature}_Predicted', f'{feature}_Actual'])

    column_order = [col for col in column_order if col in results_df.columns]
    results_df = results_df[column_order]

    metrics = {}
    for feature in feature_columns:
        mae = mean_absolute_error(results_df[f'{feature}_Actual'], results_df[f'{feature}_Predicted'])
        rmse = np.sqrt(mean_squared_error(results_df[f'{feature}_Actual'], results_df[f'{feature}_Predicted']))
        r2 = r2_score(results_df[f'{feature}_Actual'], results_df[f'{feature}_Predicted'])
        
        metrics[feature] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        print(f"\nMetrics for {feature}:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")

    # Print the first 10 rows to verify
    print("\nFirst 10 rows of predictions:")
    print(results_df.head(10))

    return results_df, metrics

# Example usage:
"""
# Assuming you have these variables defined:
loaded_gan = ...  # Your loaded WeatherGAN model
X_array = ...     # Your test input data
y_array = ...     # Your test target data
target_datetimes = ...  # Your datetime values
feature_columns = ['Air Temperature', 'Dew Point', 'Rel. Hum', 'Atm. Press', 
                  'Solar Radiation', 'Wind Speed', 'Wind Gust']
scaled_features = ...  # Your dictionary of scalers

# Call the function
results_df, metrics = evaluate_predictions(
    loaded_gan=loaded_gan,
    X_array=X_array,
    y_array=y_array,
    target_datetimes=target_datetimes,
    feature_columns=feature_columns,
    scaled_features=scaled_features,
    filter_date='09/05/2024'  # Optional
)
"""