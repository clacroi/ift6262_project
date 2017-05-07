import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Lambda
from keras.layers.convolutional import Conv2D, UpSampling2D, Cropping2D, ZeroPadding2D
from keras.layers.pooling import MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras import layers
from keras.layers import Input
from keras.models import Model
from keras.losses import mean_squared_error

from custom_layers import *

from keras import backend as K

BATCH_SIZE = 100
N_Z = 512

def model_vae_10(batch_size, original_dim):

    # Encoder layers
    im = Input(shape=(3, 64, 64), name='full image')
    cond = Lambda(Zero64CenterPadding, output_shape=(3, 64, 64))(im)

    encoder = Conv2D(32, 3, activation='tanh', padding='same', input_shape=(3, 64, 64), data_format='channels_first')(im)
    encoder = BatchNormalization(axis=1)(encoder)

    encoder = Conv2D(64, 3, activation='tanh', padding='same', input_shape=(32, 64, 64), data_format='channels_first')(encoder)
    encoder = BatchNormalization(axis=1)(encoder)
    encoder = MaxPool2D((2, 2), padding='same', data_format='channels_first')(encoder)

    encoder = Conv2D(128, 3, activation='tanh', padding='same', input_shape=(64, 32, 32), data_format='channels_first')(encoder)
    encoder = BatchNormalization(axis=1)(encoder)
    encoder = MaxPool2D((2, 2), padding='same', data_format='channels_first')(encoder)

    encoder = Conv2D(256, 3, activation='tanh', padding='same', input_shape=(128, 16, 16), data_format='channels_first')(encoder)
    encoder = BatchNormalization(axis=1)(encoder)
    encoder = MaxPool2D((2, 2), padding='same', data_format='channels_first')(encoder)

    encoder = Conv2D(512, 3, activation='tanh', padding='same', input_shape=(256, 8, 8), data_format='channels_first')(encoder)
    encoder = BatchNormalization(axis=1)(encoder)
    encoder = MaxPool2D((2, 2), padding='same', data_format='channels_first')(encoder)
    encoder = Cropping2D(cropping=1, data_format='channels_first')(encoder)
    # 512*2*2

    encoder = Flatten()(encoder)
    enc_m = Dense(512, activation='linear')(encoder)
    enc_s = Dense(512, activation='linear')(encoder)

    # Sample z ~ Q(z|X,y)
    z = Lambda(sample_z, output_shape=(512,))([enc_m, enc_s])

    # Decoder layers (P(c|z,b))
    # z branch
    z_d1 = Dense(2048, activation='elu')
    z_resh1 = Reshape((512, 2, 2))
    z_up1 = UpSampling2D((2, 2), data_format='channels_first')
    z_conv1 = Conv2D(256, 3, activation='tanh', padding='same', input_shape=(512, 4, 4), data_format='channels_first')
    # 256*4*4

    z_up2 = UpSampling2D((2, 2), data_format='channels_first')
    z_conv2 = Conv2D(128, 3, activation='tanh', padding='same', input_shape=(256, 8, 8), data_format='channels_first')
    # 128*8*8

    z_up3 = UpSampling2D((2, 2), data_format='channels_first')
    z_conv3 = Conv2D(64, 3, activation='tanh', padding='same', input_shape=(128, 16, 16), data_format='channels_first')
    #dec_z_3 = UpSampling2D((2, 2), data_format='channels_first')(dec_z_3)
    # 64*16*16

    # Conditional border branch
    b_conv1 = Conv2D(64, 3, activation='tanh', padding='same', input_shape=(3, 64, 64), data_format='channels_first')
    b_mp1 = MaxPool2D((2, 2), padding='same', data_format='channels_first')
    # 64*32*32
    # dec2_1

    b_conv2 = Conv2D(128, 3, activation='tanh', padding='same', input_shape=(64, 32, 32), data_format='channels_first')
    b_mp2 = MaxPool2D((2, 2), padding='same', data_format='channels_first')
    # 128*16*16
    #dec2_2

    b_conv3 = Conv2D(256, 3, activation='tanh', padding='same', input_shape=(128, 16, 16), data_format='channels_first')
    b_mp3 = MaxPool2D((2, 2), padding='same', data_format='channels_first')
    # 256*8*8
    #dec2_3

    # Merged layers
    m_zp1 = ZeroPadding2D(padding=2, data_format='channels_first')
    m_mask1 = Lambda(Zero8CenterPadding, output_shape=(256, 8, 8))
    #m_add1 = layers.add([dec1, m_mask1])
    m_conv1 = Conv2D(256, 3, activation='tanh', padding='same', input_shape=(256, 8, 8), data_format='channels_first')
    m_up1 = UpSampling2D((2, 2), data_format='channels_first')
    m_conv2 = Conv2D(128, 3, activation='tanh', padding='same', input_shape=(256, 16, 16), data_format='channels_first')
    # 128*16*16

    m_zp2 = ZeroPadding2D(padding=4, data_format='channels_first')
    m_mask2 = Lambda(Zero16CenterPadding, output_shape=(128, 16, 16))
    #m_add2 = layers.add([dec2, m_mask2])
    #m_conc1 = layers.concatenate([dec1, dec2], axis=1)
    m_conv3 = Conv2D(128, 3, activation='tanh', padding='same', input_shape=(256, 16, 16), data_format='channels_first')
    m_up2 = UpSampling2D((2, 2), data_format='channels_first')
    m_conv4 = Conv2D(64, 3, activation='tanh', padding='same', input_shape=(128, 32, 32), data_format='channels_first')
    # 64*32*32

    m_zp3 = ZeroPadding2D(padding=8, data_format='channels_first')
    m_mask3 = Lambda(Zero32CenterPadding, output_shape=(64, 32, 32))
    #m_add3 = layers.add([dec3, m_mask3])
    #m_conc2 = layers.concatenate([dec2, dec3], axis=1)
    m_conv5 = Conv2D(64, 3, activation='tanh', padding='same', input_shape=(128, 32, 32), data_format='channels_first')
    m_up3 = UpSampling2D((2, 2), data_format='channels_first')
    m_conv6 = Conv2D(32, 3, activation='tanh', padding='same', input_shape=(64, 64, 64), data_format='channels_first')
    # 32*64*64

    m_conv7 = Conv2D(3, 3, activation='sigmoid', padding='same', input_shape=(32, 64, 64), data_format='channels_first')
    m_crop1 = Cropping2D(cropping=16, data_format='channels_first')
    # 3*32*32 --> 3 channels image

    def vae_loss(y_true, y_pred):

        """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        # E[log P(X|z,y)]
        recon =  original_dim * mean_squared_error(y_true, y_pred)

        # D_KL(Q(z|X,y) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kl = 0.5 * K.mean(K.exp(enc_s) + K.square(enc_m) - 1. - enc_s, axis=-1)

        return recon + 0.1 * kl

    def get_decoder_output(code, cond):

        # z branch
        dec_z_d1 = z_d1(code)
        dec_z_resh1 = z_resh1(dec_z_d1)
        dec_z_up1 = z_up1(dec_z_resh1)
        dec_z_conv1 = z_conv1(dec_z_up1)
        dec_z_up2 = z_up2(dec_z_conv1)
        dec_z_conv2 = z_conv2(dec_z_up2)
        dec_z_up3 = z_up3(dec_z_conv2)
        dec_z_conv3 = z_conv3(dec_z_up3)

        # b branch
        dec_b_conv1 = b_conv1(cond)
        dec_b_mp1 = b_mp1(dec_b_conv1)
        dec_b_conv2 = b_conv2(dec_b_mp1)
        dec_b_mp2 = b_mp2(dec_b_conv2)
        dec_b_conv3 = b_conv3(dec_b_mp2)
        dec_b_mp3 = b_mp3(dec_b_conv3)

        # merged branch
        dec_m_zp1 = m_zp1(dec_z_conv1)
        dec_m_mask1 = m_mask1(dec_b_mp3)
        dec_m_add1 = layers.add([dec_m_zp1, dec_m_mask1])
        dec_m_conv1 = m_conv1(dec_m_add1)
        dec_m_up1 = m_up1(dec_m_conv1)
        dec_m_conv2 = m_conv2(dec_m_up1)

        dec_m_zp2 = m_zp2(dec_z_conv2)
        dec_m_mask2 = m_mask2(dec_b_mp2)
        dec_m_add2 = layers.add([dec_m_zp2, dec_m_mask2])
        dec_m_conc1 = layers.concatenate([dec_m_add2, dec_m_conv2], axis=1)
        dec_m_conv3 = m_conv3(dec_m_conc1)
        dec_m_up2 = m_up2(dec_m_conv3)
        dec_m_conv4 = m_conv4(dec_m_up2)

        dec_m_zp3 = m_zp3(dec_z_conv3)
        dec_m_mask3 = m_mask3(dec_b_mp1)
        dec_m_add3 = layers.add([dec_m_zp3, dec_m_mask3])
        dec_m_conc2 = layers.concatenate([dec_m_add3, dec_m_conv4], axis=1)
        dec_m_conv5 = m_conv5(dec_m_conc2)
        dec_m_up3 = m_up3(dec_m_conv5)
        dec_m_conv6 = m_conv6(dec_m_up3)

        dec_m_conv7 = m_conv7(dec_m_conv6)
        dec_m_crop1 = m_crop1(dec_m_conv7)

        return dec_m_crop1

    # Construct vae model
    decoded_mean_center = get_decoder_output(z,cond)
    vae = Model(inputs=im, outputs=decoded_mean_center)
    vae.compile(optimizer='adam', loss=vae_loss)

    # Construct generator model
    decoder_input_code = Input(shape=(512,), name='decoder_input_code')
    decoder_input_cond = Input(shape=(3, 64, 64), name='decoder_input_cond')
    _decoded_mean_center = get_decoder_output(decoder_input_code, decoder_input_cond)
    generator = Model(inputs=[decoder_input_code, decoder_input_cond], outputs=_decoded_mean_center)

    # Construct encoder model
    encoder = Model(inputs=im, outputs=enc_m)

    return vae, generator, encoder

def sample_z(args):
    mu, log_sigma = args
    #eps = np.random.randn(200, 512).astype('float32')
    eps = K.random_normal(shape=(BATCH_SIZE, N_Z), mean=0.,stddev=1.0)
    return mu + K.exp(log_sigma / 2) * eps

