import tensorflow as tf
from preprocessing import load_and_preprocess_data
from models.lstm import build_lstm_model
from tensorflow.keras.utils import to_categorical

def train_model():
    seq_length = 48
    num_classes = 5

    X_train, y_train, _ = load_and_preprocess_data("data/quincy_faulty-train.csv", seq_length)
    y_train = to_categorical(y_train, num_classes)
    
    model = build_lstm_model(input_shape=X_train.shape[1:], num_classes=num_classes)
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',factor=0.5,patience=2,min_lr=1e-6,verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True,verbose=1)
    
    history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2, callbacks=[early_stopping,lr_scheduler],verbose=1)

    model.save("Saved-Weights/lstm.keras")
    print("Model trained and saved to Saved-Weights/lstm.keras")
    
if __name__ == "__main__":
    train_model()

