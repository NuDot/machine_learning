# pylint: disable=C,R,E1101
from keras import backend as K
import numpy as np

from s2cnn.soft.so3_fft import so3_rfft, so3_rifft
from functools import lru_cache
#from s2cnn.utils.decorator import cached_dirpklgz


def so3_rotation(x, alpha, beta, gamma):
    '''
    :param x: [..., beta, alpha, gamma] (..., 2b, 2b, 2b)
    '''
    #b = x.size()[-1] // 2
    #x_size = x.size()
    x_size = K.int_shape(x)
    b = x_size[-1] // 2

    Us = _setup_so3_rotation(b, alpha, beta, gamma)

    # fourier transform
    x = so3_rfft(x)  # [l * m * n, ...]
    x = K.eval(x)
    
    # rotated spectrum
    Fz_list = []
    begin = 0
    for l in range(b):
        L = 2 * l + 1
        size = L ** 2

        Fx = x[begin:begin+size]
        Fx = np.reshape(Fx, [L, -1]) # [m, n * batch]

        U = K.reshape(Us[l], [L, L]) # [m, n]
        U = K.eval(U)

        Fz = np.matmul(np.conj(U), Fx) # [m, n * batch]

        Fz = np.reshape(Fz, [size, -1])   # [m * n, batch]
        Fz_list.append(Fz)

        begin += size

    Fz = K.concatenate(Fz_list, 0) # [l * m * n, batch]
    z = so3_rifft(Fz)

    z = K.reshape(z, [*x_size])

    return z


#@cached_dirpklgz("cache/setup_so3_rotation")
def __setup_so3_rotation(b, alpha, beta, gamma):
    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix

    Us = [wigner_D_matrix(l, alpha, beta, gamma,
                          field='complex', normalization='quantum', order='centered', condon_shortley='cs')
          for l in range(b)]
    # Us[l][m, n] = exp(i m alpha) d^l_mn(beta) exp(i n gamma)

    #Us = [Us[l].astype(np.complex64).view(np.comlex64).reshape((2 * l + 1, 2 * l + 1)) for l in range(b)] ############################
	#Us = [np.reshape(Us[l], [2 * l + 1, 2 * l + 1]) for l in range(b)]

    return Us


@lru_cache(maxsize=32)
#def _setup_so3_rotation(b, alpha, beta, gamma, device_type, device_index):
def _setup_so3_rotation(b, alpha, beta, gamma):    
    Us = __setup_so3_rotation(b, alpha, beta, gamma)

    # covert to keras tensor
    Us = [K.constant(U, dtype='complex64') for U in Us]

    return Us
