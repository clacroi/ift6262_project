import numpy as np
import matplotlib.image as mpimg


def generate_samples_v01(samples_per_epoch, batch_size, fn_list):
    while 1:
        for i in range(0, samples_per_epoch, batch_size):
            batch_images = []
            for fn in fn_list[i:i + batch_size]:
                im = mpimg.imread(fn)
                batch_images.append(im.transpose(2, 0, 1))

            batch_X = np.array(batch_images) / 255.0
            yield (batch_X, batch_X[:,:,16:48,16:48])


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

def generate_samples_v15(steps_per_epoch, batch_size, path, fn_list):

    samples_per_epoch = batch_size * steps_per_epoch
    while 1:
        for i in range(0, samples_per_epoch, batch_size):
            batch_images = []
            for fn in fn_list[i:i + batch_size]:
                im = mpimg.imread(path + fn)
                batch_images.append(im.transpose(2, 0, 1))

            batch_X = np.array(batch_images) / 255.0
            batch_Y = batch_X[:,:,16:48,16:48].copy()

            yield (batch_X, batch_Y)


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