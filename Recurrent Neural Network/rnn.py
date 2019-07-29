import numpy as np
import math
import matplotlib as mpl
from matplotlib.image import imread
from random import randint

import theano
import keras
import pandas

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras import optimizers
import keras.utils
import keras.layers
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import copy
import csv


mpl.use('Agg')
import matplotlib.pyplot as plt

DATASET_PATH = "../Datasets/hourly/"

#Set y values of data to lie between 0 and 1
def normalize_data(dataset, data_min, data_max):
    data_std = (dataset - data_min) / (data_max - data_min)
    test_scaled = data_std * (np.amax(data_std) - np.amin(data_std)) + np.amin(data_std)
    return test_scaled

#Import and pre-process data for future applications
def import_data(train_dataframe, dev_dataframe, test_dataframe):
    dataset = train_dataframe.values
    dataset = dataset.astype('float32')

    #Include all 12 initial factors (Year ; Month ; Hour ; Day ; Cloud Coverage ; Visibility ; Temperature ; Dew Point ;
    #Relative Humidity ; Wind Speed ; Station Pressure ; Altimeter
    max_test = np.max(dataset[:,12])
    min_test = np.min(dataset[:,12])
    scale_factor = max_test - min_test
    max = np.empty(13)
    min = np.empty(13)

    #Create training dataset
    for i in range(0,13):
        min[i] = np.amin(dataset[:,i],axis = 0)
        max[i] = np.amax(dataset[:,i],axis = 0)
        dataset[:,i] = normalize_data(dataset[:, i], min[i], max[i])

    train_data = dataset[:,0:12]
    train_labels = dataset[:,12]

    # Create dev dataset
    dataset = dev_dataframe.values
    dataset = dataset.astype('float32')

    for i in range(0, 13):
        dataset[:, i] = normalize_data(dataset[:, i], min[i], max[i])

    dev_data = dataset[:,0:12]
    dev_labels = dataset[:,12]

    # Create test dataset
    dataset = test_dataframe.values
    dataset = dataset.astype('float32')

    for i in range(0, 13):
        dataset[:, i] = normalize_data(dataset[:, i], min[i], max[i])

    test_data = dataset[:, 0:12]
    test_labels = dataset[:, 12]

    return train_data, train_labels, dev_data, dev_labels, test_data, test_labels, scale_factor

#Construt and return Keras RNN model
def build_model(init_type = 'glorot_uniform', optimizer = 'adam', num_features = 12):
    model = Sequential()
    layers = [num_features, 64, 64, 1, 1]
    model.add(keras.layers.LSTM(
        layers[0],
        input_shape = (None, num_features),
        return_sequences=True))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.LSTM(
        layers[1],
        kernel_initializer = init_type,
        return_sequences=True
        #bias_initializer = 'zeros'
    ))
    model.add(keras.layers.Dropout(0.2))

    model.add(Dense(
        layers[2], activation='tanh',
        kernel_initializer=init_type,
        input_shape = (None, 1)
        ))
    model.add(Dense(
        layers[3]))

    model.add(Activation("relu"))

    #Alternative parameters:
    #momentum = 0.8
    #learning_rate = 0.1
    #epochs = 100
    #decay_rate = learning_rate / 100
    #sgd = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    #model.compile(loss="binary_crossentropy", optimizer=sgd)
    rms = keras.optimizers.RMSprop(lr=0.002, rho=0.9, epsilon=1e-08, decay=0.01)
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    return model

#Save output predictions for graphing and inspection
def write_to_csv(prediction, filename):
    print("Writing to CSV...")
    with open(filename, 'w') as file:
        for i in range(prediction.shape[0]):
            file.write("%.5f" % prediction[i][0][0])
            file.write('\n')
    print("...finished!")

#Return MSE error values of all three data sets based on a single model
def evaluate(model, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, scale_factor):
    scores = model.evaluate(X_train, Y_train, verbose = 0) * scale_factor * scale_factor
    print("train: ", model.metrics_names, ": ", scores)
    scores = model.evaluate(X_dev, Y_dev, verbose = 0) * scale_factor * scale_factor
    print("dev: ", model.metrics_names, ": ", scores)
    scores = model.evaluate(X_test, Y_test, verbose = 0) * scale_factor * scale_factor
    print("test: ", model.metrics_names, ": ", scores)

#Calculate MSE between two arrays of values
def mse(predicted, observed):
    return np.sum(np.multiply((predicted - observed),(predicted - observed)))/predicted.shape[0]

def main():
    plt.switch_backend('tkAgg')

    #Import test data (6027, 13)
    train_dataframe = pandas.read_csv(DATASET_PATH + 'weather_train.csv', sep=";", engine='python', header = None)
    dev_dataframe = pandas.read_csv(DATASET_PATH + 'weather_dev.csv', sep=";", engine='python', header = None)
    test_dataframe = pandas.read_csv(DATASET_PATH + 'weather_test.csv', sep=";", engine='python', header = None)
    train_data, train_labels, dev_data, dev_labels, test_data, test_labels, scale_factor = import_data(train_dataframe, dev_dataframe, test_dataframe)


    time_steps = 1
    assert(train_data.shape[0] % time_steps == 0)

    X_train = np.reshape(train_data, (train_data.shape[0] // time_steps, time_steps, train_data.shape[1]))
    X_dev = np.reshape(dev_data, (dev_data.shape[0] // time_steps, time_steps, dev_data.shape[1]))
    X_test = np.reshape(test_data, (test_data.shape[0] // time_steps, time_steps, test_data.shape[1]))
    Y_train = np.reshape(train_labels, (train_labels.shape[0] // time_steps, time_steps, 1))
    Y_dev = np.reshape(dev_labels, (dev_labels.shape[0] // time_steps, time_steps, 1))
    Y_test = np.reshape(test_labels, (test_labels.shape[0] // time_steps, time_steps, 1))

    model = build_model('glorot_uniform', 'adam')

    #Standard vanilla LSTM model

    model_fit_epochs = 100
    print("X_train shape: ",X_train.shape, " Y_train shape: ",Y_train.shape)

    model.fit(
        X_train, Y_train,
        batch_size = 16, epochs = model_fit_epochs)
    trainset_predicted = model.predict(X_train)
    devset_predicted = model.predict(X_dev)
    testset_predicted = model.predict(X_test)

    print("Train MSE: ", mse(trainset_predicted, Y_train) * scale_factor * scale_factor)
    print("Dev MSE: ", mse(devset_predicted, Y_dev) * scale_factor * scale_factor)
    print("Test MSE: ", mse(testset_predicted, Y_test) * scale_factor * scale_factor)

    #Adaboost model (ensemble learning)

    #cv_model = KerasClassifier(build_fn=build_model, epochs=100, batch_size=16, verbose=0)
    adaboost = AdaBoostClassifier(base_estimator=build_model, learning_rate=0.01, algorithm='SAMME')
    adaboost.fit(train_data, train_labels)

    trainset_predicted = adaboost.predict(X_train)
    devset_predicted = adaboost.predict(X_dev)
    testset_predicted = adaboost.predict(X_test)

    print("Train MSE: ", mse(trainset_predicted, Y_train) * scale_factor * scale_factor)
    print("Dev MSE: ", mse(devset_predicted, Y_dev) * scale_factor * scale_factor)
    print("Test MSE: ", mse(testset_predicted, Y_test) * scale_factor * scale_factor)

    # K-fold cross validation (K = 10):
    kf = KFold(n_splits=10, shuffle=True)
    # Loop through the indices the split() method returns
    for index, (train_indices, val_indices) in enumerate(kf.split(X_train, Y_train)):
        print("Training on fold " + str(index + 1) + "/10...")
        # Generate batches from indices
        xtrain, xval = X_train[train_indices], X_train[val_indices]
        ytrain, yval = Y_train[train_indices], Y_train[val_indices]
        # Clear model, and create it
        model = None
        model = build_model()

        model.fit(
            xtrain, ytrain,
            batch_size=16, epochs=model_fit_epochs)
        testset_predicted = model.predict(xval)
        print("Test MSE: ", mse(testset_predicted, yval))

    #Grid search to optimize model params

    init = ['glorot_uniform', 'normal', 'uniform']
    epochs = [50, 100, 150]
    batches = [8, 16, 32]
    optimizers = ['rmsprop', 'adam', 'adaboost']
    optimal_params = np.empty(4)
    minimum_error = 2e63

    for init_type in init:
        for epoch in epochs:
            for batch in batches:
                for optimizer in optimizers:
                    model = None
                    model = build_model(init_type, optimizer)

                    model.fit(
                        X_train, Y_train,
                        batch_size=batch, epochs=epoch)
                    predicted = model.predict(X_test)
                    error = mse(predicted, Y_test)
                    if error < minimum_error:
                        error = minimum_error
                        optimal_params = [init_type, epoch, batch, optimizer]

    print("optimal params: ", optimal_params)
    print("minimized error: ", minimum_error)

    write_to_csv(trainset_predicted,'nn_trainset_prediction.csv')
    write_to_csv(devset_predicted,'nn_devset_prediction.csv')
    write_to_csv(testset_predicted, 'nn_testset_prediction.csv')

    return

if __name__ == '__main__':
    main()
