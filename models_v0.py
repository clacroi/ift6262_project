import numpy as np
import matplotlib.image as mpimg

import keras.models as models
from keras.models import Model
from keras.layers.core import Layer, Dense, Flatten, Reshape
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed

import theano.tensor as T
from keras import backend as K


def model_v01():

    # Define model
    autoencoder = models.Sequential()
    # Encoder
    autoencoder.add(Layer(input_shape=( 64, 64)))
    autoencoder.add(
        Conv2D(32, 3, activation='relu', padding='same', input_shape=(3, 64, 64), data_format='channels_first'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))
    autoencoder.add(
        Conv2D(16,  3, activation='relu', padding='same', input_shape=(32, 32, 32), data_format='channels_first'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))
    autoencoder.add(
        Conv2D(16,  3, activation='relu', padding='same', input_shape=(16, 16, 16), data_format='channels_first'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))
    # Output : (16, 8, 8)

    # Decoder
    autoencoder.add(
        Conv2D(16,  3, activation='relu', padding='same', input_shape=(16, 8, 8), data_format='channels_first'))
    autoencoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    autoencoder.add(
        Conv2D(16,  3, activation='relu', padding='same', input_shape=(16, 16, 16), data_format='channels_first'))
    autoencoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    autoencoder.add(
        Conv2D(3,  3, activation='sigmoid', padding='same', input_shape=(16, 32, 32), data_format='channels_first'))
    # Output : (3, 32, 32)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return autoencoder


def generate_samples_v01(samples_per_epoch, batch_size, fn_list):
    while 1:
        for i in range(0, samples_per_epoch, batch_size):
            batch_images = []
            for fn in fn_list[i:i + batch_size]:
                im = mpimg.imread(fn)
                batch_images.append(im.transpose(2, 0, 1))

            batch_X = np.array(batch_images) / 255.0
            yield (batch_X, batch_X[:,:,16:48,16:48])


def model_v02():

    # Define model
    autoencoder = models.Sequential()
    # Encoder
    autoencoder.add(Layer(input_shape=(3, 64, 64)))
    autoencoder.add(
        Conv2D(32,  3, activation='relu', padding='same', input_shape=(3, 64, 64), data_format='channels_first'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))
    autoencoder.add(
        Conv2D(16,  3, activation='relu', padding='same', input_shape=(32, 32, 32), data_format='channels_first'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))
    autoencoder.add(
        Conv2D(16,  3, activation='relu', padding='same', input_shape=(16, 16, 16), data_format='channels_first'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))
    # Output : (16, 8, 8)

    # Decoder
    autoencoder.add(
        Conv2D(16,  3, activation='relu', padding='same', input_shape=(16, 8, 8), data_format='channels_first'))
    autoencoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    autoencoder.add(
        Conv2D(16,  3, activation='relu', padding='same', input_shape=(16, 16, 16), data_format='channels_first'))
    autoencoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    autoencoder.add(
        Conv2D(3,  3, activation='sigmoid', padding='same', input_shape=(16, 32, 32), data_format='channels_first'))
    # Output : (3, 32, 32)

    autoencoder.compile(optimizer='adadelta', loss='mse')

    return autoencoder

def generate_samples_v02(samples_per_epoch, batch_size, path, fn_list, meanStd_dict):
    while 1:
        for i in range(0, samples_per_epoch, batch_size):
            batch_images = []
            for fn in fn_list[i:i + batch_size]:
                im = mpimg.imread(path + fn)
                mean, std = meanStd_dict[fn]

                norm_im = np.zeros((3, 64, 64))
                for k in range(3):
                    norm_im[k, :, :] = (im[:, :, k] - mean[2-k]) / std[2-k]

                batch_images.append(norm_im)

            batch_X = np.array(batch_images)
            batch_Y = batch_X[:,:,16:48,16:48].copy()
            batch_X[:,:,16:48,16:48] = 0.0

            yield (batch_X, batch_Y)

def model_v03():

    # Define model
    autoencoder = models.Sequential()
    # Encoder
    autoencoder.add(Layer(input_shape=(3, 64, 64)))
    autoencoder.add(
        Conv2D(64, 5, activation='relu', padding='same', input_shape=(3, 64, 64), data_format='channels_first'))
    autoencoder.add(BatchNormalization(mode=0, axis=1))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))
    autoencoder.add(
        Conv2D(32, 4, activation='relu', padding='same', input_shape=(64, 32, 32), data_format='channels_first'))
    autoencoder.add(BatchNormalization(mode=0, axis=1))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))
    autoencoder.add(
        Conv2D(16,  3, activation='relu', padding='same', input_shape=(32, 16, 16), data_format='channels_first'))
    autoencoder.add(BatchNormalization(mode=0, axis=1))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))
    # Output : (64, 8, 8)

    # Intermediate layer
    autoencoder.add(Reshape((16, 64)))
    autoencoder.add(TimeDistributed(Dense(64), input_shape=(16, 64)))
    autoencoder.add(Reshape((16, 8, 8)))

    # Decoder
    autoencoder.add(
        Conv2D(16,  3, activation='relu', padding='same', input_shape=(16, 8, 8), data_format='channels_first'))
    autoencoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    autoencoder.add(
        Conv2D(32, 4, activation='relu', padding='same', input_shape=(32, 16, 16), data_format='channels_first'))
    autoencoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    autoencoder.add(
        Conv2D(3, 5, 5, activation='sigmoid', padding='same', input_shape=(3, 32, 32), data_format='channels_first'))
    # Output : (3, 32, 32)

    autoencoder.compile(optimizer='adadelta', loss='mse')

    return autoencoder

def model_v04():

    # Define model
    autoencoder = models.Sequential()
    # Encoder
    autoencoder.add(Layer(input_shape=(3, 64, 64)))
    autoencoder.add(
        Conv2D(32, 5, activation='relu', padding='same', input_shape=(3, 64, 64), data_format='channels_first'))
    autoencoder.add(BatchNormalization(axis=1))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))
    autoencoder.add(
        Conv2D(32, 4, activation='relu', padding='same', input_shape=(64, 32, 32), data_format='channels_first'))
    autoencoder.add(BatchNormalization(axis=1))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))
    autoencoder.add(
        Conv2D(64,  3, activation='relu', padding='same', input_shape=(32, 16, 16), data_format='channels_first'))
    autoencoder.add(BatchNormalization(axis=1))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))
    # Output : (16, 8, 8)

    # Intermediate layer
    autoencoder.add(Reshape((64, 64)))
    autoencoder.add(TimeDistributed(Dense(64), input_shape=(64, 64)))
    autoencoder.add(Reshape((64, 8, 8)))

    # Decoder
    autoencoder.add(
        Conv2D(64,  3, activation='relu', padding='same', input_shape=(64, 8, 8), data_format='channels_first'))
    autoencoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    autoencoder.add(
        Conv2D(32, 4, activation='relu', padding='same', input_shape=(32, 16, 16), data_format='channels_first'))
    autoencoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    autoencoder.add(
        Conv2D(3, 5, activation='sigmoid', padding='same', input_shape=(3, 32, 32), data_format='channels_first'))
    # Output : (3, 32, 32)

    autoencoder.compile(optimizer='adadelta', loss='mse')

    return autoencoder

def model_v04():

    # Define model
    autoencoder = models.Sequential()
    # Encoder
    autoencoder.add(Layer(input_shape=(3, 64, 64)))
    autoencoder.add(
        Conv2D(32, 5, activation='relu', padding='same', input_shape=(3, 64, 64), data_format='channels_first'))
    autoencoder.add(BatchNormalization(axis=1))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))
    autoencoder.add(
        Conv2D(32, 4, activation='relu', padding='same', input_shape=(64, 32, 32), data_format='channels_first'))
    autoencoder.add(BatchNormalization(axis=1))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))
    autoencoder.add(
        Conv2D(64,  3, activation='relu', padding='same', input_shape=(32, 16, 16), data_format='channels_first'))
    autoencoder.add(BatchNormalization(axis=1))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))
    # Output : (64, 8, 8)

    # Intermediate layer
    autoencoder.add(Reshape((64, 64)))
    autoencoder.add(TimeDistributed(Dense(64), input_shape=(64, 64)))
    autoencoder.add(Reshape((64, 8, 8)))

    # Decoder
    autoencoder.add(
        Conv2D(64,  3, activation='relu', padding='same', input_shape=(64, 8, 8), data_format='channels_first'))
    autoencoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    autoencoder.add(
        Conv2D(32, 4, activation='relu', padding='same', input_shape=(64, 16, 16), data_format='channels_first'))
    autoencoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    autoencoder.add(
        Conv2D(3, 5, padding='same', input_shape=(3, 32, 32), data_format='channels_first'))
    # Output : (3, 32, 32)

    autoencoder.compile(optimizer='adadelta', loss='mse')

    return autoencoder

def model_v041():

    # Encoder
    inputs = Input(shape=(3,64,64))
    encoder = Conv2D(32, 5, activation='relu', padding='same', input_shape=(3, 64, 64), data_format='channels_first')(inputs)
    encoder = BatchNormalization(axis=1)(encoder)
    encoder = MaxPooling2D((2, 2), padding='same', data_format='channels_first')(encoder)
    encoder = Conv2D(32, 4, activation='relu', padding='same', input_shape=(32, 32, 32), data_format='channels_first')(encoder)
    encoder = BatchNormalization(axis=1)(encoder)
    encoder = MaxPooling2D((2, 2), padding='same', data_format='channels_first')(encoder)
    encoder = Conv2D(64,  3, activation='relu', padding='same', input_shape=(32, 16, 16), data_format='channels_first')(encoder)
    encoder = BatchNormalization(axis=1)(encoder)
    encoder = MaxPooling2D((2, 2), padding='same', data_format='channels_first')(encoder)
    # Output : (64, 8, 8)

    # Intermediate layer
    encoder = Flatten()(encoder)
    encoder = Dense(4096)(encoder)
    decoder = Reshape((-1, 64, 8, 8))(encoder)

    # Decoder
    decoder = Conv2D(64,  3, activation='relu', padding='same', input_shape=(64, 8, 8), data_format='channels_first')(decoder)
    decoder = UpSampling2D((2, 2), data_format='channels_first')(decoder)
    decoder = Conv2D(32, 4, activation='relu', padding='same', input_shape=(64, 16, 16), data_format='channels_first')(decoder)
    decoder = UpSampling2D((2, 2), data_format='channels_first')(decoder)
    decoder = Conv2D(3, 5, padding='same', input_shape=(32, 32, 32), data_format='channels_first')(decoder)
    # Output : (3, 32, 32)

    model = Model(inputs=inputs, outputs=decoder)
    model.compile(optimizer='adam', loss='mse')

    return model