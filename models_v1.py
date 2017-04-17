import numpy as np
import matplotlib.image as mpimg

import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed

import theano.tensor as T
from keras import backend as K

def model_v10():

    # Define model
    autoencoder = models.Sequential()
    # Encoder
    autoencoder.add(Layer(input_shape=(3, 64, 64)))
    autoencoder.add(
        Convolution2D(32, 5, 5, border_mode='same', input_shape=(3, 64, 64), dim_ordering='th'))
    autoencoder.add(BatchNormalization(mode=0, axis=1))
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPooling2D((2, 2), border_mode='same', dim_ordering='th'))
    autoencoder.add(
        Convolution2D(32, 4, 4, border_mode='same', input_shape=(64, 32, 32), dim_ordering='th'))
    autoencoder.add(BatchNormalization(mode=0, axis=1))
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPooling2D((2, 2), border_mode='same', dim_ordering='th'))
    autoencoder.add(
        Convolution2D(64, 3, 3, border_mode='same', input_shape=(32, 16, 16), dim_ordering='th'))
    autoencoder.add(BatchNormalization(mode=0, axis=1))
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPooling2D((2, 2), border_mode='same', dim_ordering='th'))
    # Output : (16, 8, 8)

    # Intermediate layer
    autoencoder.add(Flatten())
    autoencoder.add(Dense(4096))
    autoencoder.add(Reshape((64, 8, 8)))

    # Decoder
    autoencoder.add(
        Convolution2D(64, 3, 3, activation='relu', border_mode='same', input_shape=(16, 8, 8), dim_ordering='th'))
    autoencoder.add(UpSampling2D((2, 2), dim_ordering='th'))
    autoencoder.add(
        Convolution2D(32, 4, 4, activation='relu', border_mode='same', input_shape=(32, 16, 16), dim_ordering='th'))
    autoencoder.add(UpSampling2D((2, 2), dim_ordering='th'))
    autoencoder.add(
        Convolution2D(3, 5, 5, activation='sigmoid', border_mode='same', input_shape=(3, 32, 32), dim_ordering='th'))
    # Output : (3, 32, 32)

    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

def generate_samples_v10(samples_per_epoch, batch_size, path, fn_list):

    while 1:
        for i in range(0, samples_per_epoch, batch_size):
            batch_images = []
            for fn in fn_list[i:i + batch_size]:
                im = mpimg.imread(path + fn)
                batch_images.append(im.transpose(2, 0, 1))

            batch_X = np.array(batch_images) / 255.0
            batch_Y = batch_X[:,:,16:48,16:48].copy()
            batch_X[:,:,16:48,16:48] = 0.0

            yield (batch_X, batch_Y)

def model_v11():

    # Define model
    autoencoder = models.Sequential()
    # Encoder
    autoencoder.add(Layer(input_shape=(3, 64, 64)))
    autoencoder.add(
        Convolution2D(32, 5, 5, border_mode='same', input_shape=(3, 64, 64), dim_ordering='th'))
    autoencoder.add(BatchNormalization(mode=0, axis=1))
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPooling2D((2, 2), border_mode='same', dim_ordering='th'))
    autoencoder.add(
        Convolution2D(32, 4, 4, border_mode='same', input_shape=(64, 32, 32), dim_ordering='th'))
    autoencoder.add(BatchNormalization(mode=0, axis=1))
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPooling2D((2, 2), border_mode='same', dim_ordering='th'))
    autoencoder.add(
        Convolution2D(64, 3, 3, border_mode='same', input_shape=(32, 16, 16), dim_ordering='th'))
    autoencoder.add(BatchNormalization(mode=0, axis=1))
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPooling2D((2, 2), border_mode='same', dim_ordering='th'))
    # Output : (64, 8, 8)

    # Intermediate layer
    autoencoder.add(Flatten())
    autoencoder.add(Dense(512))
    autoencoder.add(Dense(4096))
    autoencoder.add(Reshape((64, 8, 8)))

    # Decoder
    autoencoder.add(
        Convolution2D(64, 3, 3, activation='relu', border_mode='same', input_shape=(16, 8, 8), dim_ordering='th'))
    autoencoder.add(UpSampling2D((2, 2), dim_ordering='th'))
    autoencoder.add(
        Convolution2D(32, 4, 4, activation='relu', border_mode='same', input_shape=(32, 16, 16), dim_ordering='th'))
    autoencoder.add(UpSampling2D((2, 2), dim_ordering='th'))
    autoencoder.add(
        Convolution2D(3, 5, 5, activation='sigmoid', border_mode='same', input_shape=(3, 32, 32), dim_ordering='th'))
    # Output : (3, 32, 32)

    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


def model_v12():

    encoder = models.Sequential()

    encoder.add(
        Convolution2D(32, 5, 5, border_mode='valid', input_shape=(3, 64, 64), dim_ordering='th'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))

    encoder.add(
        Convolution2D(64, 5, 5, border_mode='valid', subsample=(2,2), input_shape=(32, 60, 60), dim_ordering='th'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))

    encoder.add(
        Convolution2D(64, 5, 5, border_mode='valid', input_shape=(64, 32, 32), dim_ordering='th'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))

    encoder.add(
        Convolution2D(128, 5, 5, border_mode='valid', subsample=(2,2), input_shape=(64, 24, 24), dim_ordering='th'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))

    encoder.add(
        Convolution2D(256, 3, 3, border_mode='valid', input_shape=(128, 10, 10), dim_ordering='th'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))

    encoder.add(
        Convolution2D(256, 3, 3, border_mode='valid', input_shape=(256, 6, 6), dim_ordering='th'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))

    encoder.add(Flatten())
    encoder.add(Dense(1024))
    encoder.add(Dense(4096))
    encoder.add(Reshape((256, 4, 4)))

    # Decoder
    decoder = models.Sequential()
    decoder.add(encoder)

    decoder.add(UpSampling2D((2, 2), dim_ordering='th'))
    decoder.add(
        Convolution2D(128, 3, 3, activation='relu', border_mode='same', input_shape=(256, 4, 4), dim_ordering='th'))

    decoder.add(UpSampling2D((2, 2), dim_ordering='th'))
    decoder.add(
        Convolution2D(64, 3, 3, activation='relu', border_mode='same', input_shape=(128, 8, 8), dim_ordering='th'))

    decoder.add(UpSampling2D((2, 2), dim_ordering='th'))
    decoder.add(
        Convolution2D(32, 3, 3, activation='relu', border_mode='same', input_shape=(64, 16, 16), dim_ordering='th'))

    decoder.add(
        Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same', input_shape=(32, 32, 32), dim_ordering='th'))

    decoder.compile(optimizer='adam', loss='mse')

    return decoder

def model_v13():

    encoder = models.Sequential()

    encoder.add(
        Convolution2D(32, 5, 5, border_mode='valid', input_shape=(3, 64, 64), dim_ordering='th'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))
    encoder.add(Dropout(0.15))

    encoder.add(
        Convolution2D(64, 5, 5, border_mode='valid', subsample=(2,2), input_shape=(32, 60, 60), dim_ordering='th'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))
    encoder.add(Dropout(0.15))

    encoder.add(
        Convolution2D(64, 5, 5, border_mode='valid', input_shape=(64, 32, 32), dim_ordering='th'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))

    encoder.add(
        Convolution2D(128, 5, 5, border_mode='valid', subsample=(2,2), input_shape=(64, 24, 24), dim_ordering='th'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))
    encoder.add(Dropout(0.15))

    encoder.add(
        Convolution2D(256, 3, 3, border_mode='valid', input_shape=(128, 10, 10), dim_ordering='th'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))
    encoder.add(Dropout(0.15))

    encoder.add(
        Convolution2D(256, 3, 3, border_mode='valid', input_shape=(256, 6, 6), dim_ordering='th'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))
    encoder.add(Dropout(0.15))

    encoder.add(Flatten())
    encoder.add(Dense(1024))
    encoder.add(Dense(4096))
    encoder.add(Reshape((256, 4, 4)))

    # Decoder
    decoder = models.Sequential()
    decoder.add(encoder)

    decoder.add(UpSampling2D((2, 2), dim_ordering='th'))
    decoder.add(
        Convolution2D(128, 3, 3, activation='relu', border_mode='same', input_shape=(256, 4, 4), dim_ordering='th'))

    decoder.add(UpSampling2D((2, 2), dim_ordering='th'))
    decoder.add(
        Convolution2D(64, 3, 3, activation='relu', border_mode='same', input_shape=(128, 8, 8), dim_ordering='th'))

    decoder.add(UpSampling2D((2, 2), dim_ordering='th'))
    decoder.add(
        Convolution2D(32, 3, 3, activation='relu', border_mode='same', input_shape=(64, 16, 16), dim_ordering='th'))

    decoder.add(
        Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same', input_shape=(32, 32, 32), dim_ordering='th'))

    decoder.compile(optimizer='adam', loss='mse')

    return decoder