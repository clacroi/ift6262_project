import numpy as np

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