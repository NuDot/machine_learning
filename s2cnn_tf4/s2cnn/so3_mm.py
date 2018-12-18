# pylint: disable=R,C,E1101
import math
from functools import lru_cache
# import torch
# import torch.cuda
from keras import backend as K
import tensorflow as tf


def so3_mm(x, y):
    '''
    :param x: [l * m * n,   batch,    feature_in,  complex]
    :param y: [l * m * n, feature_in, feature_out, complex]
    :return:  [l * m * n,   batch,    feature_out, complex]
    '''
    import math

    nbatch = K.int_shape(x)[1]
    # print(nbatch)

    nfeature_in = K.int_shape(x)[2]
    nfeature_out = K.int_shape(y)[2]
    assert K.int_shape(y)[1]  == nfeature_in
    nspec = K.int_shape(x)[0]
    assert K.int_shape(y)[0] == nspec
    nl = math.ceil((3 / 4 * nspec) ** (1 / 3))
    assert nspec == nl * (4 * nl ** 2 - 1) // 3


    Fz_list = []
    begin = 0
    for l in range(nl):
        L = 2 * l + 1
        size = L ** 2

        Fx = x[begin:begin + size]  # [m * n,   batch,    feature_in,  complex]
        Fy = y[begin:begin + size]  # [m * n, feature_in, feature_out, complex]

        Fx = K.reshape(Fx, [L, L, -1, nfeature_in]) # [m, n, batch, feature_in]
        Fx = tf.transpose(Fx, perm=[2, 0, 3, 1])  # [batch, m, feature_in, n]
        Fx = K.reshape(Fx, [-1, nfeature_in * L])   # [batch * m, feature_in * n]

        Fy = K.reshape(Fy, [L, L, nfeature_in, nfeature_out])  # [m, n, feature_in, feature_out]
        Fy = tf.transpose(Fy, perm=[2, 1, 0, 3])   # [feature_in, n, m, feature_out]
        Fy = K.reshape(Fy, [nfeature_in * L, L * nfeature_out])  # [feature_in * n, m * feature_out]

        Fz = K.dot(Fx, tf.conj(Fy))  # [batch * m_x, m_y * feature_out] m_x -> m, m_y -> n
        Fz = K.reshape(Fz, [-1, L * L, nfeature_out])    # [batch, m * n, feature_out]
        Fz = tf.transpose(Fz, [1, 0, 2])    # [m * n, batch, feature_out]

        Fz_list.append(Fz)

        begin += size

    z = tf.concat(Fz_list, axis=0)  # [l * m * n, batch, feature_out]
    return z

