import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

label_map = {"clean": 0, "random": 1, "drift": 2, "malfunction": 3, "bias": 4}

def create_sequences(data, labels, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(labels[i + seq_length - 1])
    return np.array(X), np.array(y)


def load_and_preprocess_data(filepath, seq_length=48):
    data = pd.read_csv(filepath).ffill()
    features = data.drop(columns=["timestamp_utc", "datetime", "Class"])
    labels = data["Class"].map(label_map).values

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X, y = create_sequences(features_scaled, labels, seq_length)
    return X, y, scaler


