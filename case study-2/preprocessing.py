import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import tensorflow as tf

def preprocess_and_split_data(data, target_column='FST_EB', sequence_length=48, stride=1, train_split=0.8):

    df = data.copy()
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    
 
    df['Hour'] = df['datetime'].dt.hour
    df['DayOfWeek'] = df['datetime'].dt.dayofweek
    df['Month'] = df['datetime'].dt.month
    

    features = ['Air Temperature', 'Dew Point', 'Solar Radiation', 'Wind Speed', 
               'Hour', 'DayOfWeek', 'Month']
    

    df[features] = df[features].ffill().bfill()
    
  
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    X = scaler_X.fit_transform(df[features])
    y = scaler_y.fit_transform(df[[target_column]])
    

    X_sequences = []
    y_sequences = []
    
    for i in range(0, len(df) - sequence_length, stride):
        X_sequences.append(X[i:(i + sequence_length)])
        y_sequences.append(y[i + sequence_length])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    

    split_idx = int(len(X_sequences) * train_split)
    X_train = X_sequences[:split_idx]
    y_train = y_sequences[:split_idx]
    X_test = X_sequences[split_idx:]
    y_test = y_sequences[split_idx:]
    
    return X_train, y_train, X_test, y_test, scaler_X, scaler_y