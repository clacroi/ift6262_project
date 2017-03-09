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

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return autoencoder


def generate_samples_v01(fn_list, samples_per_epoch, batch_size):
    while 1:
        for i in range(0, samples_per_epoch, batch_size):
            batch_images = []
            for fn in fn_list[i:i + batch_size]:
                im = mpimg.imread(fn)
                batch_images.append(im.transpose(2, 0, 1))

            batch_X = np.array(batch_images) / 255.0
            yield (batch_X, batch_X[:,:,16:48,16:48])



