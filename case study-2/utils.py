import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv1D, BatchNormalization, Activation, Add, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np

def train(model, X_train, y_train, batch_size=32, epochs=100, 
                    validation_split=0.2, patience_stopping=10, patience_lr=5):

    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=patience_stopping, 
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=patience_lr, 
        min_lr=1e-5,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return history, model