import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from preprocessing import load_and_preprocess_data
from models.tst import build_tst
from models.informer import build_informer
from models.lstm_ae import build_lstm_autoencoder
from models.tst_ae import build_transformer_autoencoder

def evaluate_model():
    seq_length = 48
    num_classes = 5
    X_test, y_test, _ = load_and_preprocess_data("data/quincy_faulty-test.csv", seq_length)
    y_test = to_categorical(y_test, num_classes)
    

    #For TST,Informer, TST-AE, LSTM-AE just load the model for loading their parameters
    #model = build_tst(seq_length=num_classes,input_features=X_test.shape[2],num_classes=5)
    #model = build_informer(seq_length=seq_length,input_features=X_test.shape[2],num_classes=5)
    #model = build_transformer_autoencoder(max_seq_len=48,num_features=X_test.shape[2], num_classes=5, num_layers=4,d_model=128,num_heads=8,dff=256,dropout_rate=0.1,bottleneck_dim=64)
    
    

# The model is already compiled and ready for training


    model = load_model("Saved-Weights/lstm.keras")
    y_pred = model.predict(X_test)
    
    #This portion is for TST-AE & LSTM-AE
    #reconstructed_sequences = y_pred['reconstruction_output']
    #y_pred = y_pred['classification_output']

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_classes, y_pred_classes, average='weighted')

    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    class_names = ["clean", "random", "drift", "malfunction", "bias"]
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes, target_names=class_names, digits=4))

if __name__ == "__main__":
    evaluate_model()

