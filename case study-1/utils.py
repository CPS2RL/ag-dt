import os
import pickle
import tensorflow as tf
from gan import custom_loss, WeatherGAN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model

def save_weather_gan(gan, feature_columns, scaled_features, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    gan.generator.save(os.path.join(save_dir, 'generator.keras'))
    gan.discriminator.save(os.path.join(save_dir, 'discriminator.keras'))
    
    with open(os.path.join(save_dir, 'feature_columns.pkl'), 'wb') as f:
        pickle.dump(feature_columns, f)
    
    scalers_path = os.path.join(save_dir, 'scalers')
    if not os.path.exists(scalers_path):
        os.makedirs(scalers_path)
        
    for feature in feature_columns:
        with open(os.path.join(scalers_path, f'{feature.replace(" ", "_")}_scaler.pkl'), 'wb') as f:
            pickle.dump(scaled_features[feature]['scaler'], f)

def load_weather_gan(load_dir):
    # Load generator
    generator_path = os.path.join(load_dir, 'generator.keras')
    generator = tf.keras.models.load_model(
        generator_path, 
        custom_objects={
            'custom_loss': custom_loss
        }
    )
    
    # Load discriminator
    discriminator_path = os.path.join(load_dir, 'discriminator.keras')
    discriminator = tf.keras.models.load_model(discriminator_path)
    
    # Load feature columns
    with open(os.path.join(load_dir, 'feature_columns.pkl'), 'rb') as f:
        feature_columns = pickle.load(f)
    
    # Load scalers for each feature
    scaled_features = {}
    scalers_path = os.path.join(load_dir, 'scalers')
    
    for feature in feature_columns:
        feature_file_name = f'{feature.replace(" ", "_")}_scaler.pkl'
        with open(os.path.join(scalers_path, feature_file_name), 'rb') as f:
            scaler = pickle.load(f)
            scaled_features[feature] = {'scaler': scaler}
    
    # Reconstruct GAN
    gan = WeatherGAN(generator, discriminator)
    gan.compile(
        g_optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        d_optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        loss_fn=tf.keras.losses.BinaryCrossentropy()
    )
    
    return gan, feature_columns, scaled_features