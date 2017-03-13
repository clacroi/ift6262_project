import numpy as np
from os import listdir
import pickle
import keras.models as models
from keras.callbacks import History
import theano.tensor as T
from keras import backend as K

from models_v0 import *
from preproc import *

PROJ_PATH = '/home/corentin/Documents/Polytechnique/Deep Learning/Projet'
BATCH_SIZE = 2
NB_EPOCH = 10
NB_SAMPLES_PER_EPOCH = 10
NB_VAL_SAMPLES = 10
FIT_STYLE = "gen"


def load_and_transform_data(path, filenames, nb_images=None):
    if nb_images == None:
        nb_images == len(filenames)

    images_list = [mpimg.imread(path + fn).transpose(2, 0, 1) for fn in filenames[0:nb_images-1]]

    return np.array(images_list) / 255.0


def load_data(path, filenames, nb_images=None):
    if nb_images == None:
        nb_images == len(filenames)

    images_list = [mpimg.imread(path + fn).transpose(2, 0, 1) for fn in filenames[0:nb_images]]

    return np.array(images_list)


def evaluate_model(model, fit_style, batch_size, nb_epoch, samples_per_epoch,
                   x_train=None, x_val=None, y_train=None, y_val=None,
                   samples_generator=None, generator_args=None):

    train_history = History()

    if fit_style == "gen":
        model.fit_generator(samples_generator(samples_per_epoch, batch_size, **generator_args),
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

    return train_history


train_path = PROJ_PATH + '/Data/inpainting/train2014/'
val_path = PROJ_PATH + '/Data/inpainting/val2014/'

print("Loading data...")

# Training Data

# Load valid validation images filenames
with open("./Data/train_images_fn.pkl", 'rb') as input:
    train_fn = pickle.load(input)

# Load python dict containing channel-wise means and stds
with open("./Data/train_meanStd_dict.pkl", 'rb') as input:
    train_meanStd_dict = pickle.load(input, encoding='latin1')

# Validation Data

# Load valid validation images filenames
with open("./Data/val_images_fn.pkl", 'rb') as input:
    val_fn = pickle.load(input)

# Load python dict containing channel-wise means and stds
with open("./Data/val_meanStd_dict.pkl", 'rb') as input:
    val_meanStd_dict = pickle.load(input, encoding='latin1')

x_val = load_data(val_path, val_fn, NB_VAL_SAMPLES) # load validation images
x_val = normalize_images(x_val, val_fn, val_meanStd_dict) # normalize validation images
y_val = x_val[:, :, 16:48, 16:48].copy() # construct y_val
x_val[:, :, 16:48, 16:48] = 0 # fill x_val central region with 0s

# Convolutional Auto-Encoder v0.1
model_name = "convautoencoder_v03"
print("Compiling model...")
autoencoder = model_v03()
autoencoder.summary()

print("Fitting model...")

generator_args = {'path':train_path, 'fn_list':train_fn, 'meanStd_dict':train_meanStd_dict}
autoencoder_train = evaluate_model(autoencoder, "gen", BATCH_SIZE, NB_EPOCH, NB_SAMPLES_PER_EPOCH,
               x_val=x_val, y_val=y_val,
               samples_generator=generate_samples_v02, generator_args=generator_args)

autoencoder.save_weights('./Results/' + model_name + '.h5')
print(autoencoder_train.history)
with open('./Results/Models_v0' + model_name + '_trainHistory.pkl', 'wb') as output:
    pickle.dump(autoencoder_train.history, output, pickle.HIGHEST_PROTOCOL)
