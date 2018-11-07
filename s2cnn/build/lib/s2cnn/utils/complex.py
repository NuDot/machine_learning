# pylint: disable=C,R,E1101
from keras import backend as K
import numpy as np

def as_complex(x):
    """
    In pytorch, a complex array is represented as a real array with an extra length-2 axis at the end.
    This function takes a real-valued array x and adds complex axis where the real part is set to x and the imaginary part is set to 0.
    """
    #imaginary = torch.zeros_like(x)
    imaginary = K.zeros_like(x)
    #z = torch.stack((x, imaginary), dim=x.ndimension())
    z = K.stack((x, imaginary), axis=K.ndim(x))
    return z


def complex_mm(x, y, conj_x=False, conj_y=False):
    '''
    :param x: [i, k, complex] (M, K, 2)
    :param y: [k, j, complex] (K, N, 2)
    :return:  [i, j, complex] (M, N, 2)
    '''
    xr = x[:, :, 0]
    xi = x[:, :, 1]

    yr = y[:, :, 0]
    yi = y[:, :, 1]

    if not conj_x and not conj_y:
        zr = K.dot(xr, yr) - K.dot(xi, yi)
        zi = K.dot(xr, yi) + K.dot(xi, yr)
    if conj_x and not conj_y:
        zr = K.dot(xr, yr) + K.dot(xi, yi)
        zi = K.dot(xr, yi) - K.dot(xi, yr)
    if not conj_x and conj_y:
        zr = K.dot(xr, yr) + K.dot(xi, yi)
        zi = K.dot(xi, yr) - K.dot(xr, yi)
    if conj_x and conj_y:
        zr = K.dot(xr, yr) - K.dot(xi, yi)
        zi = - K.dot(xr, yi) - K.dot(xi, yr)

    return K.stack((zr, zi), 2)
