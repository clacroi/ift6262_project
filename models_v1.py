import numpy as np
import matplotlib.image as mpimg

import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Lambda
from keras.layers.convolutional import Conv2D, UpSampling2D, Cropping2D, ZeroPadding2D
from keras.layers.pooling import MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Multiply, Add, Concatenate
from keras import layers
from keras.layers import Input
from keras.models import Model

import theano.tensor as T
from keras import backend as K

def model_v10():

    # Define model
    autoencoder = models.Sequential()
    # Encoder
    autoencoder.add(Layer(input_shape=(3, 64, 64)))

    autoencoder.add(
        Conv2D(32, 5, padding='same', input_shape=(3, 64, 64), data_format='channels_first'))
    autoencoder.add(BatchNormalization(axis=1))
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPool2D((2, 2), padding='same', data_format='channels_first'))

    autoencoder.add(
        Conv2D(32, 4, padding='same', input_shape=(32, 32, 32), data_format='channels_first'))
    autoencoder.add(BatchNormalization(axis=1))
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPool2D((2, 2), padding='same', data_format='channels_first'))

    autoencoder.add(
        Conv2D(64, 3, padding='same', input_shape=(32, 16, 16), data_format='channels_first'))
    autoencoder.add(BatchNormalization(axis=1))
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPool2D((2, 2), padding='same', data_format='channels_first'))
    # Output : (64, 8, 8)

    # Intermediate layer
    autoencoder.add(Flatten())
    autoencoder.add(Dense(4096))
    autoencoder.add(Reshape((64, 8, 8)))

    # Decoder
    autoencoder.add(
        Conv2D(64, 3, activation='relu', padding='same', input_shape=(64, 8, 8), data_format='channels_first'))
    autoencoder.add(UpSampling2D((2, 2), data_format='channels_first'))

    autoencoder.add(
        Conv2D(32, 4, activation='relu', padding='same', input_shape=(64, 16, 16), data_format='channels_first'))
    autoencoder.add(UpSampling2D((2, 2), data_format='channels_first'))

    autoencoder.add(
        Conv2D(3, 5, activation='sigmoid', padding='same', input_shape=(32, 32, 32), data_format='channels_first'))
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
        Conv2D(32, 5, padding='same', input_shape=(3, 64, 64), data_format='channels_first'))
    autoencoder.add(BatchNormalization(axis=1))
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPool2D((2, 2), padding='same', data_format='channels_first'))

    autoencoder.add(
        Conv2D(32, 4, padding='same', input_shape=(32, 32, 32), data_format='channels_first'))
    autoencoder.add(BatchNormalization(axis=1))
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPool2D((2, 2), padding='same', data_format='channels_first'))

    autoencoder.add(
        Conv2D(64, 3, padding='same', input_shape=(32, 16, 16), data_format='channels_first'))
    autoencoder.add(BatchNormalization(axis=1))
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPool2D((2, 2), padding='same', data_format='channels_first'))
    # Output : (64, 8, 8)

    # Intermediate layer
    autoencoder.add(Flatten())
    autoencoder.add(Dense(512))
    autoencoder.add(Dense(4096))
    autoencoder.add(Reshape((64, 8, 8)))

    # Decoder
    autoencoder.add(
        Conv2D(64, 3, activation='relu', padding='same', input_shape=(64, 8, 8), data_format='channels_first'))
    autoencoder.add(UpSampling2D((2, 2), data_format='channels_first'))

    autoencoder.add(
        Conv2D(32, 4, activation='relu', padding='same', input_shape=(32, 16, 16), data_format='channels_first'))
    autoencoder.add(UpSampling2D((2, 2), data_format='channels_first'))

    autoencoder.add(
        Conv2D(3, 5, activation='sigmoid', padding='same', input_shape=(3, 32, 32), data_format='channels_first'))
    # Output : (3, 32, 32)

    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


def model_v12():

    encoder = models.Sequential()

    encoder.add(
        Conv2D(32, 5, padding='valid', input_shape=(3, 64, 64), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
    encoder.add(Activation('relu'))

    encoder.add(
        Conv2D(64, 5, padding='valid', subsample=(2,2), input_shape=(32, 60, 60), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
    encoder.add(Activation('relu'))

    encoder.add(
        Conv2D(64, 5, padding='valid', input_shape=(64, 28, 28), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
    encoder.add(Activation('relu'))

    encoder.add(
        Conv2D(128, 5, padding='valid', subsample=(2,2), input_shape=(64, 24, 24), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
    encoder.add(Activation('relu'))

    encoder.add(
        Conv2D(256, 3, padding='valid', input_shape=(128, 10, 10), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
    encoder.add(Activation('relu'))

    encoder.add(
        Conv2D(256, 3, padding='valid', input_shape=(256, 6, 6), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
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
        Conv2D(128, 3, activation='relu', padding='same', input_shape=(256, 8, 8), data_format='channels_first'))

    decoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    decoder.add(
        Conv2D(64, 3, activation='relu', padding='same', input_shape=(128, 16, 16), data_format='channels_first'))

    decoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    decoder.add(
        Conv2D(32, 3, activation='relu', padding='same', input_shape=(64, 32, 32), data_format='channels_first'))

    decoder.add(
        Conv2D(3, 3, activation='sigmoid', padding='same', input_shape=(32, 32, 32), data_format='channels_first'))

    decoder.compile(optimizer='adam', loss='mse')

    return decoder

def model_v13():

    encoder = models.Sequential()

    encoder.add(
        Conv2D(32, 5, padding='valid', input_shape=(3, 64, 64), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
    encoder.add(Activation('relu'))
    encoder.add(Dropout(0.15))

    encoder.add(
        Conv2D(64, 5, padding='valid', subsample=(2,2), input_shape=(32, 60, 60), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
    encoder.add(Activation('relu'))
    encoder.add(Dropout(0.15))

    encoder.add(
        Conv2D(64, 5, padding='valid', input_shape=(64, 28, 28), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
    encoder.add(Activation('relu'))

    encoder.add(
        Conv2D(128, 5, padding='valid', subsample=(2,2), input_shape=(64, 24, 24), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
    encoder.add(Activation('relu'))
    encoder.add(Dropout(0.15))

    encoder.add(
        Conv2D(256, 3, padding='valid', input_shape=(128, 10, 10), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
    encoder.add(Activation('relu'))
    encoder.add(Dropout(0.15))

    encoder.add(
        Conv2D(256, 3, padding='valid', input_shape=(256, 6, 6), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
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
        Conv2D(128, 3, activation='relu', padding='same', input_shape=(256, 8, 8), data_format='channels_first'))

    decoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    decoder.add(
        Conv2D(64, 3, activation='relu', padding='same', input_shape=(128, 16, 16), data_format='channels_first'))

    decoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    decoder.add(
        Conv2D(32, 3, activation='relu', padding='same', input_shape=(64, 32, 32), data_format='channels_first'))

    decoder.add(
        Conv2D(3, 3, activation='sigmoid', padding='same', input_shape=(32, 32, 32), data_format='channels_first'))

    decoder.compile(optimizer='adam', loss='mse')

    return decoder

def model_v14():

    encoder = models.Sequential()

    encoder.add(
        Conv2D(32, 5, padding='valid', input_shape=(3, 64, 64), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
    encoder.add(Activation('tanh'))
    encoder.add(Dropout(0.15))

    encoder.add(
        Conv2D(64, 5, padding='valid', subsample=(2,2), input_shape=(32, 60, 60), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
    encoder.add(Activation('tanh'))
    encoder.add(Dropout(0.15))

    encoder.add(
        Conv2D(64, 5, padding='valid', input_shape=(64, 28, 28), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
    encoder.add(Activation('tanh'))

    encoder.add(
        Conv2D(128, 5, padding='valid', subsample=(2,2), input_shape=(64, 24, 24), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
    encoder.add(Activation('tanh'))
    encoder.add(Dropout(0.15))

    encoder.add(
        Conv2D(256, 3, padding='valid', input_shape=(128, 10, 10), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
    encoder.add(Activation('tanh'))
    encoder.add(Dropout(0.15))

    encoder.add(
        Conv2D(256, 3, padding='valid', input_shape=(256, 6, 6), data_format='channels_first'))
    encoder.add(BatchNormalization(axis=1))
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
        Conv2D(128, 3, activation='tanh', padding='same', input_shape=(256, 8, 8), data_format='channels_first'))

    decoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    decoder.add(
        Conv2D(64, 3, activation='tanh', padding='same', input_shape=(128, 16, 16), data_format='channels_first'))

    decoder.add(UpSampling2D((2, 2), data_format='channels_first'))
    decoder.add(
        Conv2D(32, 3, activation='tanh', padding='same', input_shape=(64, 32, 32), data_format='channels_first'))

    decoder.add(
        Conv2D(3, 3, activation='sigmoid', padding='same', input_shape=(32, 32, 32), data_format='channels_first'))

    decoder.compile(optimizer='adam', loss='mse')

    return decoder

def model_v15():

    # Encoder
    im = Input(shape=(3, 64, 64), name='full image')
    cond = Lambda(Zero64CenterPadding, output_shape=(3,64,64))(im)

    encoder = Conv2D(32, 3, activation='relu', padding='same', input_shape=(3, 64, 64), data_format='channels_first')(im)
    encoder = BatchNormalization(axis=1)(encoder)
    encoder = MaxPool2D((2, 2), padding='same', data_format='channels_first')(encoder)

    encoder = Conv2D(64, 3, activation='relu', padding='same', input_shape=(32, 64, 64), data_format='channels_first')(encoder)
    encoder = BatchNormalization(axis=1)(encoder)
    encoder = MaxPool2D((2, 2), padding='same', data_format='channels_first')(encoder)

    encoder = Conv2D(128, 3, activation='relu', padding='same', input_shape=(64, 32, 32), data_format='channels_first')(encoder)
    encoder = BatchNormalization(axis=1)(encoder)
    encoder = MaxPool2D((2, 2), padding='same', data_format='channels_first')(encoder)

    encoder = Conv2D(256, 3, activation='relu', padding='same', input_shape=(128, 16, 16), data_format='channels_first')(encoder)
    encoder = BatchNormalization(axis=1)(encoder)
    encoder = MaxPool2D((2, 2), padding='same', data_format='channels_first')(encoder)

    encoder = Conv2D(512, 3, activation='relu', padding='same', input_shape=(256, 8, 8), data_format='channels_first')(encoder)
    encoder = BatchNormalization(axis=1)(encoder)
    #encoder = Cropping2D(cropping=((1,3), (1, 3)), data_format='channels_first')
    encoder = Lambda(center4_slice, output_shape=(512,2,2))(encoder)
    # Output : (64, 8, 8)

    # Intermediate layer
    encoder = Flatten()(encoder)
    encoder = Dense(1024)(encoder)

    # Decoder
    # z branch
    dec1_1 = Dense(2048)(encoder)
    dec1_1 = Reshape((512, 2, 2))(dec1_1)
    UpSampling2D((2, 2), data_format='channels_first')(dec1_1)
    dec1_1 = Conv2D(256, 3, activation='relu', padding='same', input_shape=(512, 4, 4), data_format='channels_first')(dec1_1)
    # 256*4*4


    dec1_2 = UpSampling2D((2, 2), data_format='channels_first')(dec1_1)
    dec1_2 = Conv2D(128, 3, activation='relu', padding='same', input_shape=(256, 8, 8), data_format='channels_first')(dec1_2)
    # 128*8*8

    dec1_3 = UpSampling2D((2, 2), data_format='channels_first')(dec1_2)
    dec1_3 = Conv2D(64, 3, activation='relu', padding='same', input_shape=(128, 16, 16), data_format='channels_first')(dec1_3)
    dec1_3 = UpSampling2D((2, 2), data_format='channels_first')(dec1_3)
    # 64*16*16

    # Conditional border branch
    dec2_1 = Conv2D(64, 3, activation='relu', padding='same', input_shape=(3, 64, 64), data_format='channels_first')(cond)
    dec2_1 = MaxPool2D((2, 2), padding='same', data_format='channels_first')(dec2_1)
    # 64*32*32

    dec2_2 = Conv2D(128, 3, activation='relu', padding='same', input_shape=(64, 32, 32), data_format='channels_first')(dec2_1)
    dec2_2 = MaxPool2D((2, 2), padding='same', data_format='channels_first')(dec2_2)
    # 128*16*16

    dec2_3 = Conv2D(256, 3, activation='relu', padding='same', input_shape=(128, 16, 16), data_format='channels_first')(dec2_2)
    dec2_3 = MaxPool2D((2, 2), padding='same', data_format='channels_first')(dec2_3)
    # 256*8*8

    
    # Merged layers
    dec1 = ZeroPadding2D(padding=3, data_format='channels_first')(dec1_1)
    dec1 = layers.add([dec1, Lambda(Zero8CenterPadding, output_shape=(256,8,8))(dec2_3)])
    dec1 = Conv2D(256, 3, activation='relu', padding='same', input_shape=(256, 8, 8), data_format='channels_first')(dec1)
    dec1 = UpSampling2D((2, 2), data_format='channels_first')(dec1)
    dec1 = Conv2D(128, 3, activation='relu', padding='same', input_shape=(256, 16, 16), data_format='channels_first')(dec1)
    # 128*16*16

    dec2 = ZeroPadding2D(padding=4, data_format='channels_first')(dec1_2)
    dec2 = layers.add([dec2, Lambda(Zero16CenterPadding, output_shape=(128,16,16))(dec2_2)])
    dec2 = layers.concatenate([dec1, dec2])
    dec2 = Conv2D(128, 3, activation='relu', padding='same', input_shape=(256, 16, 16), data_format='channels_first')(dec2)
    dec2 = UpSampling2D((2, 2), data_format='channels_first')(dec2)
    dec2 = Conv2D(64, 3, activation='relu', padding='same', input_shape=(128, 32, 32), data_format='channels_first')(dec2)
    # 64*32*32

    dec3 = ZeroPadding2D(padding=8, data_format='channels_first')(dec1_3)
    dec3 = layers.add([dec3, Lambda(Zero32CenterPadding, output_shape=(64,32,32))(dec2_1)])
    dec3 = layers.concatenate([dec2, dec3])
    dec3 = Conv2D(64, 3, activation='relu', padding='same', input_shape=(128,32,32), data_format='channels_first')(dec3)
    dec3 = UpSampling2D((2, 2), data_format='channels_first')(dec3)
    dec3 = Conv2D(32, 3, activation='relu', padding='same', input_shape=(64,64,64), data_format='channels_first')(dec3)
    # 32*64*64

    dec4 = Conv2D(16, 3, activation='sigmoid', padding='same', input_shape=(32,64,64), data_format='channels_first')(dec3)
    # 3*64*64 --> 3 channels image

    decoder_outputs = Lambda(center64_slice, output_shape=(3,32,32))(dec4)
    # 3*32*32

    model = Model(inputs=im, outputs=decoder_outputs)
    model.compile(optimizer='adam', loss='mse')

    return model

def generate_samples_v15(samples_per_epoch, batch_size, path, fn_list):

    while 1:
        for i in range(0, samples_per_epoch, batch_size):
            batch_images = []
            for fn in fn_list[i:i + batch_size]:
                im = mpimg.imread(path + fn)
                batch_images.append(im.transpose(2, 0, 1))

            batch_X = np.array(batch_images) / 255.0
            batch_Y = batch_X[:,:,16:48,16:48].copy()
            batch_X_cond = batch_X.copy()
            batch_X_cond[:,:,16:48,16:48] = 0.0

            yield ([batch_X, batch_X_cond], batch_Y)

def center4_slice(x):
    return x[:,1:3,1:3]

def center64_slice(x):
    return x[:,16:48,16:48]

# def floatX(x):
#     return np.asarray(x,dtype=K.config.floatX)
#
# def init_weights(shape):
#     i = shape[0]/4
#     j = 3 * (shape[0]/4) + 1
#     mask = np.zeros(shape)
#     mask[i:j, i:j] = np.ones((shape[0]/2, shape[0]/2))
#     return floatX(mask)

def Zero4CenterPadding(x):
    mask = np.ones((4,4))
    mask[1:3,1:3] = np.zeros((2,2))
    return x * mask

def Zero8CenterPadding(x):
    mask = np.ones((8,8))
    mask[2:6,2:6] = np.zeros((4,4))
    return x * mask

def Zero16CenterPadding(x):
    mask = np.ones((16,16))
    mask[4:12,4:12] = np.zeros((8,8))
    return x * mask

def Zero32CenterPadding(x):
    mask = np.ones((32,32))
    mask[8:24,8:24] = np.zeros((16,16))
    return x * mask

def Zero32CenterPadding(x):
    mask = np.ones((32,32))
    mask[8:24,8:24] = np.zeros((16,16))
    return x * mask

def Zero64CenterPadding(x):
    mask = np.ones((64,64))
    mask[16:48,16:48] = np.zeros((32,32))
    return x * mask