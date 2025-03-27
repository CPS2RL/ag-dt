import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df = df.sort_values('datetime').ffill()


    feature_columns = [
        'Air Temperature', 'Dew Point', 'Rel. Hum', 'Atm. Press',
        'Solar Radiation', 'Wind Speed', 'Wind Gust'
    ]

    df['minute_sin'] = np.sin(2 * np.pi * (df['datetime'].dt.hour * 60 + df['datetime'].dt.minute) / (24 * 60))
    df['minute_cos'] = np.cos(2 * np.pi * (df['datetime'].dt.hour * 60 + df['datetime'].dt.minute) / (24 * 60))
    df['month_sin'] = np.sin(2 * np.pi * df['datetime'].dt.month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12.0)
    df['day_sin'] = np.sin(2 * np.pi * df['datetime'].dt.day / 31.0)
    df['day_cos'] = np.cos(2 * np.pi * df['datetime'].dt.day / 31.0)
    df['hour_of_day'] = df['datetime'].dt.hour
    df['Solar_Radiation_Original'] = df['Solar Radiation'].copy()

    scaler = MinMaxScaler()
    scaled_features = {}
    for feature in feature_columns:
        scaled_values = scaler.fit_transform(df[[feature]]).flatten()
        scaled_features[feature] = {
            'values': scaled_values,
            'scaler': scaler.fit(df[[feature]])
        }

    weather_scaled = np.column_stack([scaled_features[feature]['values'] for feature in feature_columns])
    time_features = ['minute_sin', 'minute_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
    time_data = df[time_features].values
    scaled_data = np.hstack([weather_scaled, time_data])

    sequence_length = 24
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(weather_scaled[i + sequence_length])

    return np.array(X), np.array(y), feature_columns, scaled_features, df