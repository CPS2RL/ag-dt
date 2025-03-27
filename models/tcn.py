import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation
from tensorflow.keras.models import Model

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

def create_tcn_model(input_shape, num_classes, filters1=64, filters2=32, kernel_size=3, dropout_rate=0.2):
    # Input layer
    input_layer = Input(shape=input_shape)
    x = input_layer
    
    # First stack of TCN blocks
    nb_filters = filters1
    for dilation_rate in [1, 2, 4, 8]:
        x = TCN_block(
            x,
            dilation_rate=dilation_rate,
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            padding='causal',
            dropout_rate=dropout_rate
        )
    
    # Second stack with fewer filters
    nb_filters = filters2
    for dilation_rate in [1, 2, 4, 8]:
        x = TCN_block(
            x,
            dilation_rate=dilation_rate,
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            padding='causal',
            dropout_rate=dropout_rate
        )
    
    # Global pooling and dense layers
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    tcn_model = Model(inputs=input_layer, outputs=output_layer)
    tcn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return tcn_model
