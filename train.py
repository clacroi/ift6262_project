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

projectAddress = "/home/ubuntu/project"

# load images filenames into list, load and plot the first image
print("Loading data...")
train_imgs_fn = listdir(projectAddress + '/Data/inpainting/train2014/')
val_imgs_fn = listdir(projectAddress + '/Data/inpainting/val2014/')
img=mpimg.imread(projectAddress + '/Data/inpainting/train2014/' + train_imgs_fn[0])
#imgplot = plt.imshow(img)

# Load 10% of train, val images into Python list
# Train images
train_images = []
for fn in train_imgs_fn:
    im = mpimg.imread(projectAddress + '/Data/inpainting/train2014/' + fn)
    if len(im.shape) == 3:
        train_images.append(im.transpose(2, 0, 1))
x_train = np.array(train_images)/255.0

# Validation images
val_images = []
for fn in val_imgs_fn:
    im = mpimg.imread(projectAddress + '/Data/inpainting/val2014/' + fn)
    if len(im.shape) == 3:
        val_images.append(im.transpose(2, 0, 1))
x_val = np.array(val_images)/255.0

# Convolutional Auto-Encoder v0.1
print("Constructing Keras model...")
autoencoder = models.Sequential()
# Encoder
autoencoder.add(Layer(input_shape=(3, 64, 64)))
autoencoder.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', input_shape=(3, 64, 64), dim_ordering='th'))
autoencoder.add(MaxPooling2D((2, 2), border_mode='same', dim_ordering='th'))
autoencoder.add(Convolution2D(16, 3, 3, activation='relu', border_mode='same', input_shape=(32, 32, 32), dim_ordering='th'))
autoencoder.add(MaxPooling2D((2, 2), border_mode='same', dim_ordering='th'))
autoencoder.add(Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='th'))
autoencoder.add(MaxPooling2D((2, 2), border_mode='same', dim_ordering='th'))
# Decoder
autoencoder.add(Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='th'))
autoencoder.add(UpSampling2D((2, 2), dim_ordering='th'))
autoencoder.add(Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='th'))
autoencoder.add(UpSampling2D((2, 2), dim_ordering='th'))
autoencoder.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering='th'))
autoencoder.add(UpSampling2D((2, 2), dim_ordering='th'))
autoencoder.add(Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same', dim_ordering='th'))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Fit model
print("Fitting model...")
autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_val, x_val))
