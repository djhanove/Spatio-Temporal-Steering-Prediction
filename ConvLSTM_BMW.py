from time import time
import os, math, timeit, json, csv
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import device_lib
import keras

from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import ConvLSTM2D
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.models import Model
from keras import optimizers
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.models import load_model



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import re
import cv2 as cv
import scipy
from multiprocessing.dummy import Pool as ThreadPool

FIT_NN = True
VALIDATE_NN = True
CV_LOAD_IMAGE_GRAYSCALE = 0 # read in images as greyscale
FULL_PATH_TO_COMPILED_CSV_FILE = "C:\\Users\\dhanover\\Documents\\AI-Repo-master\\Autopilot-TensorFlow-master\\Autopilot-TensorFlow-master\\data.csv"
BATCH_SIZE = 8
TRAIN_LENGTH = 400 #pick a # divisible by 10
VAL_LENGTH = 4000
SEQUENCE_LENGTH = 10
TRAIN_BATCHES = int(TRAIN_LENGTH/SEQUENCE_LENGTH)
VAL_BATCHES = int(VAL_LENGTH/SEQUENCE_LENGTH)

TBCALLBACK = keras.callbacks.TensorBoard(log_dir='./Keras_Logs', histogram_freq=0,  
          write_graph=True, write_images=True)

MODEL_FILEPATH = "C:\\Users\\dhanover\\Documents\\AI-Repo-master\\Autopilot-TensorFlow-master\\Autopilot-TensorFlow-master\\Checkpoint\\weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
CHECKPOINT = ModelCheckpoint(MODEL_FILEPATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
EARLY_STOP = keras.callbacks.EarlyStopping(monitor='val_loss', patience=45)


def main():
    """    
    Info:
    -----------------
    Estimate Steering Angle from Forward Facing Camera Greyscale image of 66x200 and vertical+horizontal optical flow tensors
    data set is from https://github.com/SullyChen/Autopilot-TensorFlow
    Network design based on Nvidia Pilot Net + BMW paper from ICRA2019
    Recurrent convolutional LSTM blocks forward outputs and inner states to future predictions

    Steps:
    -----------------
    - Load the Driving Data Set and Optical Flow images into memory.
    - Build the Conv LSTM Network.
    - Normalize Training and Validation Data
    - Feed the I/O data to the network and train it.
    - Validate the training.
    """
    # /*******1 Setup inputs*********/
    images, steering_angle = import_data(FULL_PATH_TO_COMPILED_CSV_FILE, False)
    print("/*******1 Setup inputs Done*********/")

    # /*******2 Run NN********/
    model = build_model() 
    evaluation, image_mean, image_scale, horz_mean, horz_scale, vert_mean, vert_scale, Output_mean, Output_scale = fit_NN(images, steering_angle, model)
    #plot_history(evaluation)
    print("/*******2 Run NN Done*********/")

    # /******3 Validate NN*******/
    if VALIDATE_NN:
        xval, yval = import_data(FULL_PATH_TO_COMPILED_CSV_FILE, True) 
        validate_fit_TF(xval, yval, model, image_mean, image_scale, horz_mean, horz_scale, vert_mean, vert_scale, Output_mean, Output_scale)
        print("/*******3 Validate NN Done*********/")

def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def build_model():
    """
    Info:
    -------------------
    Create Convolutional LSTM based off Nvidia Pilot Net and BMW ICRA2019 work
    """    
    inputs = Input(shape=(SEQUENCE_LENGTH, 66, 200, 3))
    x = ConvLSTM2D(24, 5, strides = (2,2), activation = "relu", data_format="channels_last", return_sequences=True)(inputs)
    x = ConvLSTM2D(36, 5,  strides = (2,2), activation = "relu", return_sequences=True)(x)
    x = ConvLSTM2D(48, 5, strides = (2,2), activation = "relu", return_sequences=True)(x)
    x = ConvLSTM2D(64,3, activation = "relu", return_sequences=True)(x)
    x = ConvLSTM2D(64,3, activation = "relu", return_sequences=True)(x)
    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(1152, activation = "relu"))(x)
    x = Dropout(0.41)(x)
    x = TimeDistributed(Dense(512, activation = "relu"))(x)
    x = Dropout(0.41)(x)
    x = LSTM(units=128, return_sequences=True)(x)
    predictions = Dense(1)(x)
    
    optimizer = optimizers.Adam()
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss='mean_squared_error',
                    optimizer=optimizer,
                    metrics=['mean_absolute_error', 'mean_squared_error'])
    print(model.summary())
    return model

def fit_NN(images, steering_angle,model):
    """
    Info:
    -----------------
    Fit the neural network to the data
    """ 
    _start = timeit.timeit()
    x_train, y_train, imageMean, imageStd, horzMean, horzStd, vertMean, vertStd, yMean, yStd = preprocess_data(images, steering_angle)
    nanXcheck = np.argwhere(np.isnan(x_train))
    nanYcheck = np.argwhere(np.isnan(y_train))
    
    lrate = LearningRateScheduler(step_decay)

    CALLBACKS_LIST = [EARLY_STOP, TBCALLBACK, CHECKPOINT, lrate]
    if FIT_NN:
        history = model.fit(x_train, y_train, epochs=50, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=CALLBACKS_LIST, shuffle=True)
    else:
        history = load_model("C:\\Users\\dhanover\\Documents\\AI-Repo-master\\Autopilot-TensorFlow-master\\Autopilot-TensorFlow-master\\Checkpoint\\weights-improvement-07-1.04.hdf5")
    return history, imageMean, imageStd, horzMean, horzStd, vertMean, vertStd, yMean, yStd

def plot_history(history):
    """
    Info:
    -------------------
    Plot network performance over time
    """
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error ')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
            label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error ')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
            label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()
    plt.show()

def preprocess_data(image_list, steering_angle):
    """
    Info:
    -----------------
    Try and load images into memory, calc optical flow, then scale the data sets for both images and optical flow seperately
    NOTE: You can load everything into RAM if you have 16Gb. The Python thread will use up about ~10Gb

    Steps:
    -----------------
    - Normalize the data set (3@66x200) (depthxheightxwidth)
    """
    images = [] #init palceholder for all images to be loaded into mem
    horizontal = []
    vertical = []
    for i in image_list:
        images.append(cv.imread(f'C:\\Users\\dhanover\\Documents\\AI-Repo-master\\Autopilot-TensorFlow-master\\Autopilot-TensorFlow-master\\preprocessed_data\\data\\{i}', CV_LOAD_IMAGE_GRAYSCALE))
        horizontal.append(cv.imread(f'C:\\Users\\dhanover\\Documents\\AI-Repo-master\\Autopilot-TensorFlow-master\\Autopilot-TensorFlow-master\\preprocessed_data\\horizontal_flow\\{i}', CV_LOAD_IMAGE_GRAYSCALE))
        vertical.append(cv.imread(f'C:\\Users\\dhanover\\Documents\\AI-Repo-master\\Autopilot-TensorFlow-master\\Autopilot-TensorFlow-master\\preprocessed_data\\vertical_flow\\{i}', CV_LOAD_IMAGE_GRAYSCALE))

    # Fit only to the training data (image, horizontal, vertical optical flow)
    imageMean = np.mean(images)
    imageStd = np.std(images)
    normalizedImages = (images-imageMean)/imageStd

    horzMean = np.mean(horizontal)
    horzStd = np.std(horizontal)
    normalizedHorz = np.array((horizontal-horzMean)/horzStd)

    vertMean = np.mean(vertical)
    vertStd = np.std(vertical)
    normalizedVert = np.array((vertical-vertMean)/vertStd)
    
    yMean = np.mean(steering_angle)
    yStd = np.std(steering_angle)
    normalizedY = np.array((steering_angle-yMean)/yStd)
    Y = normalizedY.reshape(TRAIN_BATCHES,SEQUENCE_LENGTH,1) 
    X = []
    for k in range(len(normalizedImages)):
        X.append(np.dstack((normalizedImages[k], normalizedHorz[k], normalizedVert[k]))) #stack all the frames for a 3D tensor

    X = np.array(X)
    X = X.reshape(TRAIN_BATCHES,SEQUENCE_LENGTH,66,200,3)

    return X, Y, imageMean, imageStd, horzMean, horzStd, vertMean, vertStd, yMean, yStd

def import_data(fullpathtocompiledcsvfile, flag):
    """
    Info:
    -----------------
    Import images and steering signals
    """
    df = pd.read_csv(fullpathtocompiledcsvfile)

    if flag == False:
        image_list = df.loc[0:TRAIN_LENGTH-1, 'Images']
        ys = df.loc[0:TRAIN_LENGTH-1,'Steer_deg']
    else:
        image_list = df.loc[TRAIN_LENGTH:TRAIN_LENGTH+VAL_LENGTH-1,'Images']
        ys = df.loc[TRAIN_LENGTH:TRAIN_LENGTH+VAL_LENGTH-1,'Steer_deg']
    return image_list, ys

def validate_fit_TF(xval,yval,model, imageMEAN, imageSTD, horzMEAN, horzSTD, vertMEAN, vertSTD, outputMEAN, outputSTD):
    """
    Info:
    -----------------
    Load a validation set, run the network with the normalization scalars in place, then plot the results relative to the GT
    """

    images = []
    horizontal = []
    vertical = []
    for i in xval:
        images.append(cv.imread(f'C:\\Users\\dhanover\\Documents\\AI-Repo-master\\Autopilot-TensorFlow-master\\Autopilot-TensorFlow-master\\preprocessed_data\\data\\{i}', CV_LOAD_IMAGE_GRAYSCALE))
        horizontal.append(cv.imread(f'C:\\Users\\dhanover\\Documents\\AI-Repo-master\\Autopilot-TensorFlow-master\\Autopilot-TensorFlow-master\\preprocessed_data\\horizontal_flow\\{i}', CV_LOAD_IMAGE_GRAYSCALE))
        vertical.append(cv.imread(f'C:\\Users\\dhanover\\Documents\\AI-Repo-master\\Autopilot-TensorFlow-master\\Autopilot-TensorFlow-master\\preprocessed_data\\vertical_flow\\{i}', CV_LOAD_IMAGE_GRAYSCALE))
            
    
    imageAdj = (images - imageMEAN) / imageSTD
    horzAdj = (horizontal - horzMEAN) / horzSTD
    vertAdj = (vertical - vertMEAN) / vertSTD
    yAdj = (yval - outputMEAN) / outputSTD
    print(outputMEAN)
    yAdj = np.array(yAdj)
    X_val = []
    for k in range(len(imageAdj)):
        X_val.append(np.dstack((imageAdj[k], horzAdj[k], vertAdj[k]))) #stack all the frames for a 3D tensor

    X_val = np.array(X_val)
    X_val = X_val.reshape(VAL_BATCHES,SEQUENCE_LENGTH,66,200,3)
    
    test_predictions = model.predict(X_val)
    outputNN = np.multiply(test_predictions, outputSTD) + outputMEAN
    outputNN = np.array(outputNN)
    outputNN = outputNN.reshape(VAL_LENGTH)
    plt.figure()
    plt.plot(yAdj) 
    plt.plot(outputNN)
    plt.legend((yAdj, outputNN), ('Measured Steering Angle','Predicted Steering Angle'))
    plt.xlabel('Steps')
    plt.ylabel('Steering Angle (deg)')
    plt.show()

if __name__ == "__main__":
    main()