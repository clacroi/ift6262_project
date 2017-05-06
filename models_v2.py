import numpy as np
import matplotlib.image as mpimg

import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute, RepeatVector
from keras.layers import merge
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization

def model_v20():

    # Construct layer for text processing
    text_branch = models.Sequential()
    text_branch.add(Dense(60, input_dim=2048))
    text_branch.add(RepeatVector(4096))
    text_branch.add(Permute((2, 1), input_shape=(4096, 60)))
    text_branch.add(Reshape((60, 64, 64), input_shape=(60, 4096)))
    text_branch.add(BatchNormalization())
    text_branch.add(Activation('relu'))
    text_branch.add(Permute((2, 3, 1), input_shape=(60, 64, 64))) # permute tensors for concatenating

    # Construct input layer for images
    image_branch = models.Sequential()
    image_branch.add(Permute((2, 3, 1), input_shape=(3, 64, 64))) # permute tensors for concatenating

    # Construct encoder
    encoder = models.Sequential()
    encoder.add(merged)

    encoder.add(
        Conv2D(64, 5, padding='same', input_shape=(3, 64, 64), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
    encoder.add(Activation('relu'))
    encoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))

    encoder.add(
            Conv2D(64, 4, padding='same', input_shape=(64, 32, 32), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
    encoder.add(Activation('relu'))
    encoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))

    encoder.add(
        Conv2D(128, 3, padding='same', input_shape=(32, 16, 16), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
    encoder.add(Activation('relu'))
    encoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))
        # Output : (16, 8, 8)

    encoder.add(Flatten())
    encoder.add(Dense(512))

    # Construct decoder
    decoder = models.Sequential()
    decoder.add(encoder)
    decoder.add(Dense(4096))
    decoder.add(Reshape((64, 8, 8)))

    decoder.add(
        Conv2D(128, 3, activation='relu', padding='same', input_shape=(64, 8, 8), data_format='channels_first'))
    decoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    decoder.add(
        Conv2D(64, 4, activation='relu', padding='same', input_shape=(32, 16, 16), data_format='channels_first'))
    decoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    decoder.add(
        Conv2D(3, 5,activation='sigmoid', padding='same', input_shape=(3, 32, 32), data_format='channels_first'))
    
    decoder.compile(optimizer='adam', loss='mse')
    
    return decoder

