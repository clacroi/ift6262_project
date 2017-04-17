import pickle
from keras.callbacks import History, EarlyStopping
import theano.tensor as T
from keras import backend as K

from models_v0 import *
from models_v1 import *
from models_v2 import *
from preproc import *

PROJ_PATH = '/home/ec2-user/project'
BATCH_SIZE = 200
NB_EPOCH = 50
NB_SAMPLES_PER_EPOCH = 82610
NB_VAL_SAMPLES = 40438
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
                   samples_generator=None, generator_args=None,
                   val_gen=None, val_gen_args=None, nb_val_samples=None):

    train_history = History()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # generate train (validation) data with a generator
    if fit_style == "gen":

        # No validation data generator
        if val_gen == None:
            model.fit_generator(samples_generator(samples_per_epoch, batch_size, **generator_args),
                            samples_per_epoch=samples_per_epoch,
                            nb_epoch=nb_epoch,
                            validation_data=(x_val, y_val),
                            verbose=1,
                            callbacks=[train_history, early_stopping])

        # Validation data generator passed in arg
        else:
            model.fit_generator(samples_generator(samples_per_epoch, batch_size, **generator_args),
                                samples_per_epoch=samples_per_epoch,
                                nb_epoch=nb_epoch,
                                validation_data=val_gen(**val_gen_args),
                                nb_val_samples=nb_val_samples, 
                                verbose=1,
                                callbacks=[train_history, early_stopping])

    # No generator used to fit model
    else:
        model.fit(x_train, y_train,
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  shuffle=True,
                  validation_data=(x_val, y_val),
                  verbose=1,
                  callbacks=[train_history])

    return train_history

if __name__ == "__main__":

    train_path = PROJ_PATH + '/Data/inpainting/train2014/'
    val_path = PROJ_PATH + '/Data/inpainting/val2014/'

    print("Loading data...")

    # Load train images filenames
    with open("./Data/val_images_fn.pkl", 'rb') as input:
        train_fn = pickle.load(input)

    # Load valid validation images filenames
    with open("./Data/val_images_fn.pkl", 'rb') as input:
        val_fn = pickle.load(input)

    # Construct validation data (x_val, y_val)
    x_val = load_data(val_path, val_fn, NB_VAL_SAMPLES)/255.0 # load validation images
    y_val = x_val[:, :, 16:48, 16:48].copy() # construct y_val
    x_val[:, :, 16:48, 16:48] = 0 # fill x_val central region with 0s

    # Convolutional Auto-Encoder v1.0
    model_name = "convautoencoder_v13"
    print("Compiling model...")
    autoencoder = model_v13()
    autoencoder.summary()

    print("Fitting model...")

    generator_args = {'path':train_path, 'fn_list':train_fn}
    autoencoder_train = evaluate_model(autoencoder, "gen", BATCH_SIZE, NB_EPOCH, NB_SAMPLES_PER_EPOCH,
                   x_val=x_val, y_val=y_val,
                   samples_generator=generate_samples_v10, generator_args=generator_args)

    autoencoder.save('./Results/Models_v1/' + model_name + '.h5')
    print(autoencoder_train.history)
    with open('./Results/Models_v1/' + model_name + '_trainHistory.pkl', 'wb') as output:
        pickle.dump(autoencoder_train.history, output, pickle.HIGHEST_PROTOCOL)
