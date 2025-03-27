import requests
import pandas 
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
from preprocessing import load_and_preprocess_data


# Data Collection From Zentra Cloud Server
def get_with_credentials(tok, uri, **kwargs):
    token = tok if tok.lower().startswith("token") else f"Token {tok}"
    headers = {"Authorization": token}
    return requests.get(uri, headers=headers, **kwargs)


def get_readings_response(sn, start_date, end_date, **extra_kwargs_for_endpoint):
    server = extra_kwargs_for_endpoint.get("server", "https://zentracloud.com")
    url = f"{server}/api/v4/get_readings/"

    default_args = {
        'output_format': "df",
        'per_page': 100000,
        'page_num': 1,
        'sort_by': 'asc',
        'start_date': start_date,
        'end_date': end_date
    }
    data = {**default_args, **extra_kwargs_for_endpoint, "device_sn": sn}
    tok = extra_kwargs_for_endpoint.pop("token", "")
    return get_with_credentials(tok, url, params=data)

def get_readings_dataframe(sn,start_date,end_date,**extra_kwargs_for_endpoint):
    res = get_readings_response(sn,start_date,end_date,**extra_kwargs_for_endpoint)
    if(res.ok):
        data = res.json()
        return pandas.DataFrame(**json.loads(data["data"]))
    return res


tok = " "
sn=" "
start_date=" "
end_date=" "

server="https://zentracloud.com"
df = get_readings_dataframe(sn, start_date, end_date, token=tok, server=server)

# data preparation
pivoted_df = df.pivot_table(
    index=['timestamp_utc', 'datetime'],
    columns='measurement',
    values='value',
    aggfunc='first' 
).reset_index()

pivoted_df.columns.name = None
pivoted_df.columns = [col if col is not None else '' for col in pivoted_df.columns] 
df = pivoted_df.drop(['X-axis Level', 'Y-axis Level'], axis=1)

# Processing for model inference
seq_length = 48
X_test, y_test, _ = load_and_preprocess_data(df, seq_length)

# Model Loading
model = tf.keras.models.load_model('Saved-Models/resnet.keras')

# Model Inference
y_pred = model.predict(X_test)

predicted_classes = np.argmax(y_pred, axis=1)
confidence_scores = np.max(y_pred, axis=1)

# Create reverse mapping for class interpretation
class_mapping = {"clean": 0, "random": 1, "drift": 2, "malfunction": 3, "bias": 4}

confidence_thresholds = { 'clean': 0.99, 'random': 0.9, 'malfunction': 0.9, 'drift': 0.9, 'bias': 0.9}

# Convert numeric predictions to class labels with confidence thresholds
predicted_labels = []
for pred, conf in zip(predicted_classes, confidence_scores):
    predicted_class = class_mapping[pred]
    if conf >= confidence_thresholds[predicted_class]:
        predicted_labels.append(predicted_class)

# Print predictions and their confidence scores
for label, conf in zip(predicted_labels[:100000], confidence_scores[:100000]):
    print(f"Class: {label}, Confidence: {conf:.4f}")



# Predict Fruit Surface Temperature

df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df = df.sort_values('DateTime')


df['Hour'] = df['DateTime'].dt.hour
df['DayOfWeek'] = df['DateTime'].dt.dayofweek
df['Month'] = df['DateTime'].dt.month
features = ['Air Temperature', 'Dew Point', 'Solar Radiation', 'Wind Speed', 
           'Hour', 'DayOfWeek', 'Month']
target = 'FST_EB'

df[features] = df[features].ffill().bfill()

scaler_X = RobustScaler()
scaler_y = RobustScaler()

X = scaler_X.fit_transform(df[features])
y = scaler_y.fit_transform(df[[target]])

sequence_length = 24*2 
stride = 1  
X_sequences = []
y_sequences = []

for i in range(0, len(df) - sequence_length, stride):
    X_sequences.append(X[i:(i + sequence_length)])
    y_sequences.append(y[i + sequence_length])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

model = tf.keras.models.load_model('fst_model.keras')
predictions = model.predict(X_sequences)
predictions = scaler_y.inverse_transform(predictions)

# Print some sample predictions
print(predictions[:5])