import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Lambda
from keras.layers.convolutional import Conv2D, UpSampling2D, Cropping2D, ZeroPadding2D
from keras.layers.pooling import MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Multiply, Add, Concatenate
from keras import layers
from keras.layers import Input
from keras.models import Model
from keras.losses import mean_squared_error

from custom_layers import *

from keras import backend as K

BATCH_SIZE = 40
N_Z = 512

def model_vae_10(batch_size, original_dim):
    # Encoder
    im = Input(shape=(3, 64, 64), name='full image')
    cond = Lambda(Zero64CenterPadding, output_shape=(3, 64, 64))(im)

    encoder = Conv2D(32, 3, activation='relu', padding='same', input_shape=(3, 64, 64), data_format='channels_first')(
        im)
    encoder = BatchNormalization(axis=1)(encoder)

    encoder = Conv2D(64, 3, activation='relu', padding='same', input_shape=(32, 64, 64), data_format='channels_first')(
        encoder)
    encoder = BatchNormalization(axis=1)(encoder)
    encoder = MaxPool2D((2, 2), padding='same', data_format='channels_first')(encoder)

    encoder = Conv2D(128, 3, activation='relu', padding='same', input_shape=(64, 32, 32), data_format='channels_first')(
        encoder)
    encoder = BatchNormalization(axis=1)(encoder)
    encoder = MaxPool2D((2, 2), padding='same', data_format='channels_first')(encoder)

    encoder = Conv2D(256, 3, activation='relu', padding='same', input_shape=(128, 16, 16),
                     data_format='channels_first')(encoder)
    encoder = BatchNormalization(axis=1)(encoder)
    encoder = MaxPool2D((2, 2), padding='same', data_format='channels_first')(encoder)

    encoder = Conv2D(512, 3, activation='relu', padding='same', input_shape=(256, 8, 8), data_format='channels_first')(
        encoder)
    encoder = BatchNormalization(axis=1)(encoder)
    encoder = MaxPool2D((2, 2), padding='same', data_format='channels_first')(encoder)
    encoder = Cropping2D(cropping=1, data_format='channels_first')(encoder)
    # Output : (512, 2, 2)
    encoder = Flatten()(encoder)

    # Sample z ~ Q(z|X,y)
    enc_m = Dense(512, activation='linear')(encoder)
    enc_s = Dense(512, activation='linear')(encoder)
    z = Lambda(sample_z, output_shape=(512,))([enc_m, enc_s])


    # Decoder : P(c|z,b)

    # z branch
    dec1_1 = Dense(2048)(z)
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
    dec1 = layers.add([dec1, Lambda(Zero8CenterPadding, output_shape=(256, 8, 8))(dec2_3)])
    dec1 = Conv2D(256, 3, activation='relu', padding='same', input_shape=(256, 8, 8), data_format='channels_first')(dec1)
    dec1 = UpSampling2D((2, 2), data_format='channels_first')(dec1)
    dec1 = Conv2D(128, 3, activation='relu', padding='same', input_shape=(256, 16, 16), data_format='channels_first')(dec1)
    # 128*16*16

    dec2 = ZeroPadding2D(padding=6, data_format='channels_first')(dec1_2)
    dec2 = layers.add([dec2, Lambda(Zero16CenterPadding, output_shape=(128, 16, 16))(dec2_2)])
    dec2 = layers.concatenate([dec1, dec2], axis=1)
    dec2 = Conv2D(128, 3, activation='relu', padding='same', input_shape=(256, 16, 16), data_format='channels_first')(dec2)
    dec2 = UpSampling2D((2, 2), data_format='channels_first')(dec2)
    dec2 = Conv2D(64, 3, activation='relu', padding='same', input_shape=(128, 32, 32), data_format='channels_first')(dec2)
    # 64*32*32

    dec3 = ZeroPadding2D(padding=8, data_format='channels_first')(dec1_3)
    dec3 = layers.add([dec3, Lambda(Zero32CenterPadding, output_shape=(64, 32, 32))(dec2_1)])
    dec3 = layers.concatenate([dec2, dec3], axis=1)
    dec3 = Conv2D(64, 3, activation='relu', padding='same', input_shape=(128, 32, 32), data_format='channels_first')(
        dec3)
    dec3 = UpSampling2D((2, 2), data_format='channels_first')(dec3)
    dec3 = Conv2D(32, 3, activation='relu', padding='same', input_shape=(64, 64, 64), data_format='channels_first')(
        dec3)
    # 32*64*64

    dec4 = Conv2D(3, 3, activation='sigmoid', padding='same', input_shape=(32, 64, 64), data_format='channels_first')(
        dec3)
    # decoder_outputs = Lambda(center64_slice, output_shape=(3,32,32))(dec4)
    decoder_outputs = Cropping2D(cropping=16, data_format='channels_first')(dec4)
    # 3*32*32 --> 3 channels image

    def vae_loss(y_true, y_pred):
        """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """

        # E[log P(X|z,y)]
        recon =  mean_squared_error(y_true, y_pred)

        # D_KL(Q(z|X,y) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kl = 0.5 * K.sum(K.exp(enc_s) + enc_m**2 - 1. - enc_s, axis=-1)

        return recon + kl

    model = Model(inputs=im, outputs=decoder_outputs)
    model.compile(optimizer='adam', loss=vae_loss)

    return model

def sample_z(args):
    mu, log_sigma = args
    #eps = np.random.randn(200, 512).astype('float32')
    eps = K.random_normal(shape=(BATCH_SIZE, N_Z), mean=0.,stddev=1.0)
    return mu + K.exp(log_sigma / 2) * eps

