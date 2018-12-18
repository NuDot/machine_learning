# pylint: disable=R,C,E1101
#import torch
from keras import backend as K
import numpy as np

from functools import lru_cache
#from s2cnn.utils.decorator import show_running


def so3_integrate(x):
    """
    :param x: [..., beta, alpha, gamma] (..., 2b, 2b, 2b)
    :return y: [...] (...)
    """
    assert K.int_shape(x)[-1] == K.int_shape(x)[-2]
    assert K.int_shape(x)[-2] == K.int_shape(x)[-3]
 
    b = K.int_shape(x)[-1] // 2

    dtype = K.dtype(x)
    w = _setup_so3_integrate(b, dtype)

    x = K.sum(x, axis=-1, keepdims=False) 
    x = K.sum(x, axis=-1, keepdims=False)

    sz = K.int_shape(x)
    sz = [-1 if x is None else x for x in sz]
    x = K.reshape(x, [-1, 2 * b])
    w = K.reshape(w, [2 * b, 1])
    x = K.dot(x, w)
    x = K.reshape(x, [*sz[:-1]])
    return x


@lru_cache(maxsize=32)
#@show_running
#def _setup_so3_integrate(b, device_type, device_index):
def _setup_so3_integrate(b, dtype):
    import lie_learn.spaces.S3 as S3

    return K.constant(S3.quadrature_weights(b), dtype=dtype)


