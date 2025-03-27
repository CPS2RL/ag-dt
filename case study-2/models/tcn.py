import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv1D, BatchNormalization, Activation, Add, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

def create_tcn_model(input_shape, nb_filters_list=[64, 32], kernel_size=3, dropout_rate=0.2, 
                    dilations=[1, 2, 4, 8], learning_rate=0.001):

    # TCN Block Definition
    def TCN_block(input_layer, dilation_rate, nb_filters, kernel_size, padding, dropout_rate=0.0):
        conv1 = tf.keras.layers.Conv1D(
            filters=nb_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding=padding,
            kernel_initializer='he_normal'
        )(input_layer)
        batch1 = BatchNormalization()(conv1)
        act1 = Activation('relu')(batch1)
        drop1 = tf.keras.layers.Dropout(dropout_rate)(act1)

        conv2 = tf.keras.layers.Conv1D(
            filters=nb_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding=padding,
            kernel_initializer='he_normal'
        )(drop1)
        batch2 = BatchNormalization()(conv2)
        act2 = Activation('relu')(batch2)
        drop2 = tf.keras.layers.Dropout(dropout_rate)(act2)

        if input_layer.shape[-1] != nb_filters:
            downsample = tf.keras.layers.Conv1D(
                filters=nb_filters,
                kernel_size=1,
                padding='same',
                kernel_initializer='he_normal'
            )(input_layer)
            residual = downsample
        else:
            residual = input_layer

        return tf.keras.layers.add([drop2, residual])
    
    # Build the model
    input_layer = Input(shape=input_shape)
    x = input_layer
    
    # Create stacks of TCN blocks with different filter sizes
    for nb_filters in nb_filters_list:
        for dilation_rate in dilations:
            x = TCN_block(
                x,
                dilation_rate=dilation_rate,
                nb_filters=nb_filters,
                kernel_size=kernel_size,
                padding='causal',
                dropout_rate=dropout_rate
            )
    
    # Global pooling and dense layers
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output_layer = Dense(1)(x)
    
    # Create and compile the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mae')
    
    return model