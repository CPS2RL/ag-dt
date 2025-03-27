import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Dropout, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam

def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def build_generator(sequence_length, n_features, output_dim):
    input_layer = Input(shape=(sequence_length, n_features))
    
    x = LSTM(128, return_sequences=True)(input_layer)
    x = Dropout(0.2)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(128)(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(output_dim, activation='linear')(x)
    
    return Model(inputs=input_layer, outputs=output_layer, name='generator')

def build_discriminator(sequence_length, n_features):
    input_layer = Input(shape=(sequence_length, n_features))
    
    x = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
    x = Dropout(0.2)(x)
    x = LSTM(64)(x)
    x = Dropout(0.2)(x)
    x = Dense(32)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=input_layer, outputs=output_layer, name='discriminator')

class WeatherGAN(Model):
    def __init__(self, generator, discriminator):
        super(WeatherGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        
    def call(self, inputs, training=None):
        generated_data = self.generator(inputs, training=training)
        return generated_data
        
    def compile(self, g_optimizer, d_optimizer, loss_fn, **kwargs):
        super(WeatherGAN, self).compile(loss=custom_loss, **kwargs)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
        
    def train_step(self, data):
        X_batch, y_batch = data
        
        y_batch_reshaped = tf.expand_dims(y_batch, axis=1)
        
        with tf.GradientTape() as d_tape:
            generated_data = self.generator(X_batch, training=True)
            generated_data_reshaped = tf.expand_dims(generated_data, axis=1)
            
            real_output = self.discriminator(y_batch_reshaped, training=True)
            fake_output = self.discriminator(generated_data_reshaped, training=True)
            
            d_loss_real = self.loss_fn(tf.ones_like(real_output), real_output)
            d_loss_fake = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            d_loss = d_loss_real + d_loss_fake
            
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        with tf.GradientTape() as g_tape:
            generated_data = self.generator(X_batch, training=True)
            generated_data_reshaped = tf.expand_dims(generated_data, axis=1)
            
            fake_output = self.discriminator(generated_data_reshaped, training=True)
            
            g_loss_adv = self.loss_fn(tf.ones_like(fake_output), fake_output)
            g_loss_pred = custom_loss(y_batch, generated_data)
            g_loss = g_loss_adv + 10.0 * g_loss_pred
            
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "g_loss_adv": g_loss_adv,
            "g_loss_pred": g_loss_pred
        }