import numpy as np
import matplotlib.image as mpimg

import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization


def model_v10():

    # Define model
    autoencoder = models.Sequential()
    # Encoder
    autoencoder.add(Layer(input_shape=(3, 64, 64)))

    autoencoder.add(
        Convolution2D(32, 5, 5, padding='same', input_shape=(3, 64, 64), data_format='channels_first'))
    autoencoder.add(BatchNormalization(mode=0, axis=1))
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))

    autoencoder.add(
        Convolution2D(32, 4, 4, padding='same', input_shape=(32, 32, 32), data_format='channels_first'))
    autoencoder.add(BatchNormalization(mode=0, axis=1))
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))

    autoencoder.add(
        Convolution2D(64, 3, 3, padding='same', input_shape=(32, 16, 16), data_format='channels_first'))
    autoencoder.add(BatchNormalization(mode=0, axis=1))
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))
    # Output : (64, 8, 8)

    # Intermediate layer
    autoencoder.add(Flatten())
    autoencoder.add(Dense(4096))
    autoencoder.add(Reshape((64, 8, 8)))

    # Decoder
    autoencoder.add(
        Convolution2D(64, 3, 3, activation='relu', padding='same', input_shape=(64, 8, 8), data_format='channels_first'))
    autoencoder.add(UpSampling2D((2, 2), data_format='channels_first'))

    autoencoder.add(
        Convolution2D(32, 4, 4, activation='relu', padding='same', input_shape=(64, 16, 16), data_format='channels_first'))
    autoencoder.add(UpSampling2D((2, 2), data_format='channels_first'))

    autoencoder.add(
        Convolution2D(3, 5, 5, activation='sigmoid', padding='same', input_shape=(32, 32, 32), data_format='channels_first'))
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
        Convolution2D(32, 5, 5, padding='same', input_shape=(3, 64, 64), data_format='channels_first'))
    autoencoder.add(BatchNormalization(mode=0, axis=1))
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))

    autoencoder.add(
        Convolution2D(32, 4, 4, padding='same', input_shape=(32, 32, 32), data_format='channels_first'))
    autoencoder.add(BatchNormalization(mode=0, axis=1))
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))

    autoencoder.add(
        Convolution2D(64, 3, 3, padding='same', input_shape=(32, 16, 16), data_format='channels_first'))
    autoencoder.add(BatchNormalization(mode=0, axis=1))
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))
    # Output : (64, 8, 8)

    # Intermediate layer
    autoencoder.add(Flatten())
    autoencoder.add(Dense(512))
    autoencoder.add(Dense(4096))
    autoencoder.add(Reshape((64, 8, 8)))

    # Decoder
    autoencoder.add(
        Convolution2D(64, 3, 3, activation='relu', padding='same', input_shape=(64, 8, 8), data_format='channels_first'))
    autoencoder.add(UpSampling2D((2, 2), data_format='channels_first'))

    autoencoder.add(
        Convolution2D(32, 4, 4, activation='relu', padding='same', input_shape=(32, 16, 16), data_format='channels_first'))
    autoencoder.add(UpSampling2D((2, 2), data_format='channels_first'))

    autoencoder.add(
        Convolution2D(3, 5, 5, activation='sigmoid', padding='same', input_shape=(3, 32, 32), data_format='channels_first'))
    # Output : (3, 32, 32)

    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


def model_v12():

    encoder = models.Sequential()

    encoder.add(
        Convolution2D(32, 5, 5, padding='valid', input_shape=(3, 64, 64), data_format='channels_first'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))

    encoder.add(
        Convolution2D(64, 5, 5, padding='valid', subsample=(2,2), input_shape=(32, 60, 60), data_format='channels_first'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))

    encoder.add(
        Convolution2D(64, 5, 5, padding='valid', input_shape=(64, 28, 28), data_format='channels_first'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))

    encoder.add(
        Convolution2D(128, 5, 5, padding='valid', subsample=(2,2), input_shape=(64, 24, 24), data_format='channels_first'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))

    encoder.add(
        Convolution2D(256, 3, 3, padding='valid', input_shape=(128, 10, 10), data_format='channels_first'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))

    encoder.add(
        Convolution2D(256, 3, 3, padding='valid', input_shape=(256, 6, 6), data_format='channels_first'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))

    encoder.add(Flatten())
    encoder.add(Dense(1024))
    encoder.add(Dense(4096))
    encoder.add(Reshape((256, 4, 4)))

    # Decoder
    decoder = models.Sequential()
    decoder.add(encoder)

    decoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    decoder.add(
        Convolution2D(128, 3, 3, activation='relu', padding='same', input_shape=(256, 8, 8), data_format='channels_first'))

    decoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    decoder.add(
        Convolution2D(64, 3, 3, activation='relu', padding='same', input_shape=(128, 16, 16), data_format='channels_first'))

    decoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    decoder.add(
        Convolution2D(32, 3, 3, activation='relu', padding='same', input_shape=(64, 32, 32), data_format='channels_first'))

    decoder.add(
        Convolution2D(3, 3, 3, activation='sigmoid', padding='same', input_shape=(32, 32, 32), data_format='channels_first'))

    decoder.compile(optimizer='adam', loss='mse')

    return decoder

def model_v13():

    encoder = models.Sequential()

    encoder.add(
        Convolution2D(32, 5, 5, padding='valid', input_shape=(3, 64, 64), data_format='channels_first'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))
    encoder.add(Dropout(0.15))

    encoder.add(
        Convolution2D(64, 5, 5, padding='valid', subsample=(2,2), input_shape=(32, 60, 60), data_format='channels_first'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))
    encoder.add(Dropout(0.15))

    encoder.add(
        Convolution2D(64, 5, 5, padding='valid', input_shape=(64, 28, 28), data_format='channels_first'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))

    encoder.add(
        Convolution2D(128, 5, 5, padding='valid', subsample=(2,2), input_shape=(64, 24, 24), data_format='channels_first'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))
    encoder.add(Dropout(0.15))

    encoder.add(
        Convolution2D(256, 3, 3, padding='valid', input_shape=(128, 10, 10), data_format='channels_first'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('relu'))
    encoder.add(Dropout(0.15))

    encoder.add(
        Convolution2D(256, 3, 3, padding='valid', input_shape=(256, 6, 6), data_format='channels_first'))
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

    decoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    decoder.add(
        Convolution2D(128, 3, 3, activation='relu', padding='same', input_shape=(256, 8, 8), data_format='channels_first'))

    decoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    decoder.add(
        Convolution2D(64, 3, 3, activation='relu', padding='same', input_shape=(128, 16, 16), data_format='channels_first'))

    decoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    decoder.add(
        Convolution2D(32, 3, 3, activation='relu', padding='same', input_shape=(64, 32, 32), data_format='channels_first'))

    decoder.add(
        Convolution2D(3, 3, 3, activation='sigmoid', padding='same', input_shape=(32, 32, 32), data_format='channels_first'))

    decoder.compile(optimizer='adam', loss='mse')

    return decoder

def model_v14():

    encoder = models.Sequential()

    encoder.add(
        Convolution2D(32, 5, 5, padding='valid', input_shape=(3, 64, 64), data_format='channels_first'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('tanh'))
    encoder.add(Dropout(0.15))

    encoder.add(
        Convolution2D(64, 5, 5, padding='valid', subsample=(2,2), input_shape=(32, 60, 60), data_format='channels_first'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('tanh'))
    encoder.add(Dropout(0.15))

    encoder.add(
        Convolution2D(64, 5, 5, padding='valid', input_shape=(64, 28, 28), data_format='channels_first'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('tanh'))

    encoder.add(
        Convolution2D(128, 5, 5, padding='valid', subsample=(2,2), input_shape=(64, 24, 24), data_format='channels_first'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('tanh'))
    encoder.add(Dropout(0.15))

    encoder.add(
        Convolution2D(256, 3, 3, padding='valid', input_shape=(128, 10, 10), data_format='channels_first'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('tanh'))
    encoder.add(Dropout(0.15))

    encoder.add(
        Convolution2D(256, 3, 3, padding='valid', input_shape=(256, 6, 6), data_format='channels_first'))
    encoder.add(BatchNormalization(mode=0, axis=1))
    encoder.add(Activation('tanh'))
    encoder.add(Dropout(0.15))

    encoder.add(Flatten())
    encoder.add(Dense(1024))
    encoder.add(Dense(4096))
    encoder.add(Reshape((256, 4, 4)))

    # Decoder
    decoder = models.Sequential()
    decoder.add(encoder)

    decoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    decoder.add(
        Convolution2D(128, 3, 3, activation='tanh', padding='same', input_shape=(256, 8, 8), data_format='channels_first'))

    decoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    decoder.add(
        Convolution2D(64, 3, 3, activation='tanh', padding='same', input_shape=(128, 16, 16), data_format='channels_first'))

    decoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    decoder.add(
        Convolution2D(32, 3, 3, activation='tanh', padding='same', input_shape=(64, 32, 32), data_format='channels_first'))

    decoder.add(
        Convolution2D(3, 3, 3, activation='sigmoid', padding='same', input_shape=(32, 32, 32), data_format='channels_first'))

    decoder.compile(optimizer='adam', loss='mse')

    return decoder