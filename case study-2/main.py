import pandas as pd
import tensorflow as tf
from preprocessing import preprocess_and_split_data
from utils import train
from evaluate import evaluate
from models.tcn import create_tcn_model

data = pd.read_csv('data/FST-quincy.csv')

X_train, y_train, X_test, y_test, scaler_X, scaler_y = preprocess_and_split_data(data)

input_shape = (X_train.shape[1], X_train.shape[2]) 
model = create_tcn_model(input_shape)

#history, model = train(model, X_train, y_train, epochs=5)

#model.save('Saved-weights/tcn.keras')

model = tf.keras.models.load_model('Saved-weights/tcn.keras')

metrics, y_test_orig, y_pred_orig = evaluate(model, X_test, y_test, scaler_y)


