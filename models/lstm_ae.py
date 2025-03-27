import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.saving import register_keras_serializable

def build_lstm_autoencoder(seq_length, num_features, num_classes, lstm_units=128, latent_dim=64, dropout_rate=0.1):
    # Create model class
    @register_keras_serializable()
    class LSTMAutoencoder(Model):
        def __init__(self,
                    seq_length,
                    num_features,
                    num_classes,
                    lstm_units=128,
                    latent_dim=64,
                    dropout_rate=0.1):
            super().__init__()
            # Store parameters for serialization
            self.seq_length = seq_length
            self.num_features = num_features
            self.num_classes = num_classes
            self.lstm_units = lstm_units
            self.latent_dim = latent_dim
            self.dropout_rate = dropout_rate
            
            # Encoder
            self.encoder_lstm1 = layers.LSTM(lstm_units, return_sequences=True)
            self.encoder_lstm2 = layers.LSTM(lstm_units // 2, return_sequences=True)
            self.encoder_lstm3 = layers.LSTM(latent_dim, return_sequences=True)
            
            # Classifier branch
            self.global_pooling = layers.GlobalAveragePooling1D()
            self.classifier_dense1 = layers.Dense(128, activation='relu')
            self.dropout1 = layers.Dropout(dropout_rate)
            self.classifier_dense2 = layers.Dense(64, activation='relu')
            self.classifier_output = layers.Dense(num_classes, activation='softmax')
            
            # Decoder
            self.decoder_lstm1 = layers.LSTM(latent_dim, return_sequences=True)
            self.decoder_lstm2 = layers.LSTM(lstm_units // 2, return_sequences=True)
            self.decoder_lstm3 = layers.LSTM(lstm_units, return_sequences=True)
            self.decoder_output = layers.Dense(num_features)
            
        def call(self, inputs, training=False):
            # Encoder
            x = self.encoder_lstm1(inputs)
            x = self.encoder_lstm2(x)
            encoded = self.encoder_lstm3(x)
            
            # Classification branch
            class_features = self.global_pooling(encoded)
            class_features = self.classifier_dense1(class_features)
            class_features = self.dropout1(class_features, training=training)
            class_features = self.classifier_dense2(class_features)
            classified = self.classifier_output(class_features)
            
            # Decoder branch
            decoded = self.decoder_lstm1(encoded)
            decoded = self.decoder_lstm2(decoded)
            decoded = self.decoder_lstm3(decoded)
            reconstructed = self.decoder_output(decoded)
            
            return {
                'reconstruction_output': reconstructed,
                'classification_output': classified
            }
        
        def get_config(self):
            config = super().get_config()
            config.update({
                "seq_length": self.seq_length,
                "num_features": self.num_features,
                "num_classes": self.num_classes,
                "lstm_units": self.lstm_units,
                "latent_dim": self.latent_dim,
                "dropout_rate": self.dropout_rate,
            })
            return config

        @classmethod
        def from_config(cls, config):
            # Extract only the parameters we need for initialization
            init_params = {
                "seq_length": config["seq_length"],
                "num_features": config["num_features"],
                "num_classes": config["num_classes"],
                "lstm_units": config["lstm_units"],
                "latent_dim": config["latent_dim"],
                "dropout_rate": config["dropout_rate"]
            }
            return cls(**init_params)
    
    # Create model instance
    model = LSTMAutoencoder(
        seq_length=seq_length,
        num_features=num_features,
        num_classes=num_classes,
        lstm_units=lstm_units,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate
    )
    
    # Build model with sample input
    sample_input = tf.zeros((1, seq_length, num_features))
    _ = model(sample_input)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'reconstruction_output': 'mse',
            'classification_output': 'categorical_crossentropy'
        },
        loss_weights={
            'reconstruction_output': 0.3,
            'classification_output': 0.7
        },
        metrics={
            'classification_output': ['accuracy']
        }
    )
    
    return model


