import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Deconvolution2D
from keras.callbacks import History
from keras.layers.normalization import BatchNormalization

import theano.tensor as T
from keras import backend as K

from models_v0 import *

PROJ_PATH = "/home/corentin/Documents/Polytechnique/Deep Learning/Projet"
BATCH_SIZE = 100
NB_EPOCH = 100
NB_SAMPLES_PER_EPOCH = 1000
FIT_STYLE = "classic"


def get_images_filenames(path):

    filenames_list = [path + img for img in listdir(path)]
    for i in range(0, len(filenames_list)):
        fn = filenames_list[i]
        im = mpimg.imread(fn)

        if len(im.shape) != 3:
            del filenames_list[i]

    return filenames_list


def load_and_transform_data(path, filenames, nb_images=None):
    if nb_images == None:
        nb_images == len(filenames)

    images_list = [mpimg.imread(path + fn).transpose(2, 0, 1) for fn in filenames[0:nb_images]]

    return np.array(images_list) / 255.0

def evaluate_model(model, fit_style, batch_size, nb_epoch,
                   x_train=None, x_val=None, y_train=None, y_val=None,
                   samples_generator=None, samples_per_epoch=None, train_fn_list=None, val_fn_list=None):

    train_history = History()

    if fit_style == "gen":
        model.fit_generator(samples_generator(train_fn_list, samples_per_epoch, batch_size),
                            samples_per_epoch=samples_per_epoch,
                            nb_epoch=nb_epoch,
                            validation_data=(x_val, y_val),
                            verbose=1,
                            callbacks=[train_history])
    else:
        model.fit(x_train, y_train,
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  shuffle=True,
                  validation_data=(x_val, y_val),
                  verbose=1,
                  callbacks=[train_history])


train_path = PROJ_PATH + '/Data/inpainting/train2014/'
val_path = PROJ_PATH + '/Data/inpainting/val2014/'

print("Loading data...")

# Training Data
with open("./Data/train_images_fn.pkl", 'rb') as input:
    train_fn = pickle.load(input)

x_train = load_and_transform_data(train_path, train_fn, 20)

# Validation Data
with open("./Data/val_images_fn.pkl", 'rb') as input:
    val_fn = pickle.load(input)

x_val = load_and_transform_data(train_path, train_fn, 20)
#val_fn = get_images_filenames(val_path)


# Convolutional Auto-Encoder v0.1
print("Compiling model...")
autoencoder = model_v01()

print("Fitting model...")
train_fn = [train_path + fn for fn in train_fn]
val_fn = [val_path + fn for fn in val_fn]

evaluate_model(autoencoder, "gen", BATCH_SIZE, NB_EPOCH,
               x_train=x_train[:,:,:,:], y_train=x_train[:,:,16:48,16:48],
               x_val=x_val[:,:,:,:], y_val=x_val[:,:,16:48,16:48],
               samples_generator=generate_samples_v01,
               samples_per_epoch=NB_SAMPLES_PER_EPOCH,
               train_fn_list=train_fn, val_fn_list=val_fn)