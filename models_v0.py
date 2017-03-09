import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization

import theano.tensor as T
from keras import backend as K


def model_v01():

    # Define model
    autoencoder = models.Sequential()
    # Encoder
    autoencoder.add(Layer(input_shape=(3, 64, 64)))
    autoencoder.add(
        Convolution2D(32, 3, 3, activation='relu', border_mode='same', input_shape=(3, 64, 64), dim_ordering='th'))
    autoencoder.add(MaxPooling2D((2, 2), border_mode='same', dim_ordering='th'))
    autoencoder.add(
        Convolution2D(16, 3, 3, activation='relu', border_mode='same', input_shape=(32, 32, 32), dim_ordering='th'))
    autoencoder.add(MaxPooling2D((2, 2), border_mode='same', dim_ordering='th'))
    autoencoder.add(
        Convolution2D(16, 3, 3, activation='relu', border_mode='same', input_shape=(16, 16, 16), dim_ordering='th'))
    autoencoder.add(MaxPooling2D((2, 2), border_mode='same', dim_ordering='th'))
    # Output : (16, 8, 8)

    # Decoder
    autoencoder.add(
        Convolution2D(16, 3, 3, activation='relu', border_mode='same', input_shape=(16, 8, 8), dim_ordering='th'))
    autoencoder.add(UpSampling2D((2, 2), dim_ordering='th'))
    autoencoder.add(
        Convolution2D(16, 3, 3, activation='relu', border_mode='same', input_shape=(16, 16, 16), dim_ordering='th'))
    autoencoder.add(UpSampling2D((2, 2), dim_ordering='th'))
    autoencoder.add(
        Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same', input_shape=(16, 32, 32), dim_ordering='th'))
    # Output : (3, 32, 32)

    autoencoder.compile(optimizer='adadelta', loss='mse')

    return autoencoder



