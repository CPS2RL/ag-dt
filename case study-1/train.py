from gan import WeatherGAN, build_generator, build_discriminator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

def train_gan(X_train, y_train, sequence_length, n_features, output_dim):
  
    generator = build_generator(sequence_length, n_features, output_dim)
    discriminator = build_discriminator(sequence_length=1, n_features=output_dim)
    
    gan = WeatherGAN(generator, discriminator)
    gan.compile(
        g_optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        d_optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        loss_fn=tf.keras.losses.BinaryCrossentropy()
    )
    
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=20, min_lr=1e-6)
    ]
    

    history = gan.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        callbacks=callbacks
    )
    
    return gan, history