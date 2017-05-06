import numpy as np

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
    mask = np.ones((4,4), dtype='float32')
    mask[1:3,1:3] = np.zeros((2,2), dtype='float32')
    return x * mask

def Zero8CenterPadding(x):
    mask = np.ones((8,8), dtype='float32')
    mask[2:6,2:6] = np.zeros((4,4), dtype='float32')
    return x * mask

def Zero16CenterPadding(x):
    mask = np.ones((16,16), dtype='float32')
    mask[4:12,4:12] = np.zeros((8,8), dtype='float32')
    return x * mask

def Zero32CenterPadding(x):
    mask = np.ones((32,32), dtype='float32')
    mask[8:24,8:24] = np.zeros((16,16), dtype='float32')
    return x * mask

def Zero64CenterPadding(x):
    mask = np.ones((64,64), dtype='float32')
    mask[16:48,16:48] = np.zeros((32,32), dtype='float32')
    return x * mask