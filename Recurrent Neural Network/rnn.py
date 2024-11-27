import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam
from sklearn.model_selection import KFold

DATASET_PATH = "../Datasets/hourly/"

def normalize_data(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

def import_data(file_path):
    df = pd.read_csv(file_path, sep=";", header=None)
    return df.values.astype("float32")

def build_lstm_model(input_shape, optimizer="adam"):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dense(1, activation="relu")
    ])
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    return model

def train_and_evaluate(X_train, Y_train, X_test, Y_test, model):
    model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=1)
    mse = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test MSE: {mse}")
    return mse

def main():
    train_data = import_data(os.path.join(DATASET_PATH, "weather_train.csv"))
    X_train, Y_train = train_data[:, :-1], train_data[:, -1]
    X_train = np.expand_dims(X_train, axis=1)  # Add time-step dimension

    test_data = import_data(os.path.join(DATASET_PATH, "weather_test.csv"))
    X_test, Y_test = test_data[:, :-1], test_data[:, -1]
    X_test = np.expand_dims(X_test, axis=1)
    
    model = build_lstm_model(X_train.shape[1:])
    train_and_evaluate(X_train, Y_train, X_test, Y_test, model)

if __name__ == "__main__":
    main()
