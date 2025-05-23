import tensorflow as tf
from preprocessing import load_and_preprocess_data
from train import train_gan
from utils import save_weather_gan, load_weather_gan
from visualization import plot_predictions
from prediction import evaluate_predictions

def main():

    #file_path = 'data/quincy-train.csv'
    #X_train, y_train, feature_columns, scaled_features, df = load_and_preprocess_data(file_path)
    
    # Train GAN
    #sequence_length = 24
    #n_features = X_train.shape[2]
    #utput_dim = y_train.shape[1]
    # gan, history = train_gan(X_train, y_train, sequence_length, n_features, output_dim)
    
    # Save the model
    # save_dir = 'weather_gan_model'
    # save_weather_gan(gan, feature_columns, scaled_features, save_dir)

    # test GAN

    file_path_test = 'data/quincy-test.csv'
    X_test, y_test, feature_columns, scaled_features, df_test = load_and_preprocess_data(file_path_test)
    sequence_length = 48
    target_datetimes = df_test['datetime'].iloc[sequence_length:].reset_index(drop=True)
    
    # Load the model
    loaded_gan, loaded_feature_columns, loaded_scaled_features = load_weather_gan('weather_gan_model')
    results_df, metrics = evaluate_predictions(loaded_gan=loaded_gan, X_array=X_test, y_array=y_test, target_datetimes=target_datetimes, feature_columns=feature_columns, scaled_features=scaled_features)


if __name__ == "__main__":
    main()
