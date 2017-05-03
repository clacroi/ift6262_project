import numpy as np
import matplotlib.image as mpimg

import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute, RepeatVector
from keras.layers import merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Deconvolution2D
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
        Convolution2D(64, 5, 5, padding='same', input_shape=(3, 64, 64), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
    encoder.add(Activation('relu'))
    encoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))

    encoder.add(
            Convolution2D(64, 4, 4, padding='same', input_shape=(64, 32, 32), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
    encoder.add(Activation('relu'))
    encoder.add(MaxPooling2D((2, 2), padding='same', data_format='channels_first'))

    encoder.add(
        Convolution2D(128, 3, 3, padding='same', input_shape=(32, 16, 16), data_format='channels_first'))
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
        Convolution2D(128, 3, 3, activation='relu', padding='same', input_shape=(64, 8, 8), data_format='channels_first'))
    decoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    decoder.add(
        Convolution2D(64, 4, 4, activation='relu', padding='same', input_shape=(32, 16, 16), data_format='channels_first'))
    decoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    decoder.add(
        Convolution2D(3, 5, 5, activation='sigmoid', padding='same', input_shape=(3, 32, 32), data_format='channels_first'))
    
    decoder.compile(optimizer='adam', loss='mse')
    
    return decoder


def generate_samples_v20(samples_per_epoch, batch_size, path, fn_list,
                         captions_dict, vectorizer, svd):

    while 1:
        for i in range(0, samples_per_epoch, batch_size):
            batch_embeddings = []
            batch_images = []
            for fn in fn_list[i:i + batch_size]:

                # Construct image captions embedding
                im_captions = captions_dict[fn.split(".")[0]]
                batch_embeddings.append(" ".join(im_captions))

                # Construct image
                im = mpimg.imread(path + fn)
                batch_images.append(im.transpose(2, 0, 1))

            batch_XE = np.array(svd.transform(vectorizer.transform(batch_embeddings)))
            batch_XI = np.array(batch_images) / 255.0
            batch_Y = batch_XI[:,:,16:48,16:48].copy()
            batch_XI[:,:,16:48,16:48] = 0.0

            yield ([batch_XI, batch_XE], batch_Y)
