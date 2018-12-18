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
    # print('This is the input of so3_integrate', x)
    #assert x.size(-1) == x.size(-2)
    assert K.int_shape(x)[-1] == K.int_shape(x)[-2]
    #assert x.size(-2) == x.size(-3)
    assert K.int_shape(x)[-2] == K.int_shape(x)[-3]
    

    #b = x.size(-1) // 2
    b = K.int_shape(x)[-1] // 2

    #w = _setup_so3_integrate(b, device_type=x.device.type, device_index=x.device.index)  # [beta]
    dtype = K.dtype(x)
    w = _setup_so3_integrate(b, dtype)

    #x = torch.sum(x, dim=-1).squeeze(-1)  # [..., beta, alpha]
    x = K.sum(x, axis=-1, keepdims=False) 
    #x = torch.sum(x, dim=-1).squeeze(-1)  # [..., beta]
    x = K.sum(x, axis=-1, keepdims=False)

    #sz = x.size()
    sz = K.int_shape(x)
    sz = [-1 if x is None else x for x in sz]
    #x = x.view(-1, 2 * b)
    x = K.reshape(x, [-1, 2 * b])
    #w = w.view(2 * b, 1)
    w = K.reshape(w, [2 * b, 1])
    #x = torch.mm(x, w).squeeze(-1)
    x = K.dot(x, w)
    #x = x.view(*sz[:-1])
    x = K.reshape(x, [*sz[:-1]])
    # print('This is the output of s03_integrate', x)
    return x


@lru_cache(maxsize=32)
#@show_running
#def _setup_so3_integrate(b, device_type, device_index):
def _setup_so3_integrate(b, dtype):
    import lie_learn.spaces.S3 as S3

    #return torch.tensor(S3.quadrature_weights(b), dtype=torch.float32, device=torch.device(device_type, device_index))  # (2b) [beta]  # pylint: disable=E1102
    return K.constant(S3.quadrature_weights(b), dtype=dtype)


