# pylint: disable=R,C,E1101
from functools import lru_cache
# import torch
# import torch.cuda
from keras import backend as K
import tensorflow as tf
from string import Template
import math

# TODO simplify the cuda code like it was done in SO3_mm using only one code for the kernel


def s2_mm(x, y):
    '''
    :param x: [l * m,     batch,      feature_in]
    :param y: [l * m,     feature_in, feature_out]
    :return:  [l * m * n, batch,      feature_out]
    '''
    nbatch = K.int_shape(x)[1]
    nfeature_in = K.int_shape(x)[2]
    nfeature_out = K.int_shape(y)[2]
    assert K.int_shape(y)[1] == nfeature_in
    nspec = K.int_shape(x)[0]
    assert K.int_shape(y)[0] == nspec
    nl = int(nspec**0.5)

    Fz_list = []
    begin = 0

    for l in range(nl):
        L = 2 * l + 1
        size = L

        Fx = x[begin:begin+size]  # [m, batch,      feature_in,  complex]
        Fy = y[begin:begin+size]  # [m, feature_in, feature_out, complex]

        Fx = K.reshape(Fx, [-1, nfeature_in])   # [m * batch, feature_in]

        Fy = tf.transpose(Fy, perm=[1, 0, 2])  # [feature_in, m, feature_out]
        Fy = K.reshape(Fy, [nfeature_in, L * nfeature_out])  # [feature_in, m * feature_out]

        Fz = K.dot(Fx, tf.conj(Fy))  # [m_x * batch, m_y * feature_out] m_x -> m, m_y -> n 
        Fz = K.reshape(Fz, [L, -1, L, nfeature_out])  # [m, batch, n, feature_out]
        Fz = tf.transpose(Fz, perm=[0, 2, 1, 3])  # [m, n, batch, feature_out]
        Fz = K.reshape(Fz, [L * L, -1, nfeature_out])  # [m * n, batch, feature_out]

        Fz_list.append(Fz)

        begin += size

    z = K.concatenate(Fz_list, axis=0)  # [l * m * n, batch, feature_out, complex]
    return z



