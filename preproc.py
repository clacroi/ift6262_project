import numpy as np

def normalize_images(x, x_fn, meanStd_dict):
    n_images = x.shape[0]
    norm_x = np.zeros(x.shape)
    for i in range(n_images - 1):
        fn = x_fn[i]
        mean, std = meanStd_dict[fn]
        for k in range(3):
            norm_x[i, k, :, :] = (x[i, k, :, :] - mean[2 - k]) / std[2 - k]
    return norm_x


def reconstruct_images(norm_x, x_fn, meanStd_dict):
    n_images = norm_x.shape[0]
    x = np.zeros(norm_x.shape)
    for i in range(n_images - 1):
        fn = x_fn[i]
        mean, std = meanStd_dict[fn]
        for k in range(3):
            x[i, k, :, :] = (norm_x[i, k, :, :] * std[2 - k]) + mean[2 - k]
    return x.astype('uint8')