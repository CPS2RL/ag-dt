import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Input, Conv1D, BatchNormalization, 
                                   Activation, Add, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.saving import register_keras_serializable



@register_keras_serializable()
class LSTMAutoencoderRegression(Model):
    def __init__(self,
                 sequence_length,
                 num_features,
                 output_dim=1, 
                 lstm_units=128,
                 latent_dim=64,
                 dropout_rate=0.1):
        super().__init__()
        # Store parameters for serialization
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.output_dim = output_dim 
        self.lstm_units = lstm_units
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        
        # Encoder
        self.encoder_lstm1 = layers.LSTM(lstm_units, return_sequences=True)
        self.encoder_lstm2 = layers.LSTM(lstm_units // 2, return_sequences=True)
        self.encoder_lstm3 = layers.LSTM(latent_dim, return_sequences=True)
        
        # Regression branch
        self.global_pooling = layers.GlobalAveragePooling1D()
        self.regression_dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(dropout_rate)
        self.regression_dense2 = layers.Dense(64, activation='relu')
        self.regression_output = layers.Dense(output_dim)
        
        # Decoder
        self.decoder_lstm1 = layers.LSTM(latent_dim, return_sequences=True)
        self.decoder_lstm2 = layers.LSTM(lstm_units // 2, return_sequences=True)
        self.decoder_lstm3 = layers.LSTM(lstm_units, return_sequences=True)
        self.decoder_output = layers.Dense(num_features)
        
    def call(self, inputs, training=False, regression_only=False):
        # Encoder
        x = self.encoder_lstm1(inputs)
        x = self.encoder_lstm2(x)
        encoded = self.encoder_lstm3(x)
        
        # Regression branch
        reg_features = self.global_pooling(encoded)
        reg_features = self.regression_dense1(reg_features)
        reg_features = self.dropout1(reg_features, training=training)
        reg_features = self.regression_dense2(reg_features)
        regression = self.regression_output(reg_features)
        
        if regression_only:
            return regression
        
        # Decoder branch
        decoded = self.decoder_lstm1(encoded)
        decoded = self.decoder_lstm2(decoded)
        decoded = self.decoder_lstm3(decoded)
        reconstructed = self.decoder_output(decoded)
        
        return {
            'reconstruction_output': reconstructed,
            'regression_output': regression
        }
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "num_features": self.num_features,
            "output_dim": self.output_dim,
            "lstm_units": self.lstm_units,
            "latent_dim": self.latent_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        init_params = {
            "sequence_length": config["sequence_length"],
            "num_features": config["num_features"],
            "output_dim": config["output_dim"],
            "lstm_units": config["lstm_units"],
            "latent_dim": config["latent_dim"],
            "dropout_rate": config["dropout_rate"]
        }
        return cls(**init_params)

def create_lstm_autoencoder_model(sequence_length, num_features, output_dim=1,
                                lstm_units=128, latent_dim=64, dropout_rate=0.1,
                                learning_rate=0.001, reconstruction_weight=0.3,
                                regression_weight=0.7):
    model = LSTMAutoencoderRegression(
        sequence_length=sequence_length,
        num_features=num_features,
        output_dim=output_dim,
        lstm_units=lstm_units,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate
    )


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'reconstruction_output': 'mse', 
            'regression_output': 'mse'     
        },
        metrics={
            'reconstruction_output': ['mae'], 
            'regression_output': ['mae']       
        },
        # Loss weights
        loss_weights={
            'reconstruction_output': reconstruction_weight,
            'regression_output': regression_weight
        }
    )
    
    return model