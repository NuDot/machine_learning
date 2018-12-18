# pylint: disable=R,C,E1101
import math
from functools import lru_cache
from keras import backend as K
import tensorflow as tf
import numpy as np


# inspired by https://gist.github.com/szagoruyko/89f83b6f5f4833d3c8adf81ee49f22a8

def so3_rfft_dumbb(x, for_grad=False, b_out=None):
    #:param x: [..., beta, alpha, gamma] in complex64
    #:return: [l * m * n, ...]

    assert K.dtype(x) == 'float32'
    b_in = K.int_shape(x)[-1] // 2
    assert K.int_shape(x)[-1] == 2 * b_in
    assert K.int_shape(x)[-2] == 2 * b_in
    assert K.int_shape(x)[-3] == 2 * b_in
    if b_out is None:
        b_out = b_in
    assert b_out <= b_in

    batch_size = K.int_shape(x)[:-3]
    batch_size = [-1 if x is None else x for x in batch_size]

    x = K.reshape(x, [-1, 2 * b_in, 2 * b_in, 2 * b_in])

    nspec = b_out * (4 * b_out ** 2 - 1) // 3
    nbatch = tf.shape(x)[0]

    w = _setup_wigner(b_in, nl=b_out, weighted=not for_grad)

    x = tf.spectral.fft2d(tf.complex(x, tf.zeros_like(x)))

    output = []

    for l in range(b_out):
        sz = l * (4 * l**2 - 1) // 3
        #s = slice(l * (4 * l**2 - 1) // 3, l * (4 * l**2 - 1) // 3 + (2 * l + 1) ** 2)
        s = slice(sz, sz + (2 * l + 1) ** 2)
        l1 = min(l, b_in - 1)  # if b_out > b_in, consider high frequencies as null

        xx = []

        AC, blank, BD = tf.split(x, [l1+1, 2*b_in-2*l1-1, l1], axis=2)
        A, blank, C = tf.split(AC, [l1+1, 2*b_in-2*l1-1, l1], axis=3)
        B, blank, D = tf.split(BD, [l1+1, 2*b_in-2*l1-1, l1], axis=3)

        DB = []
        DB.append(D)
        DB.append(B)
        DB = K.concatenate(DB, axis=3)

        CA= []
        CA.append(C)
        CA.append(A)
        CA = K.concatenate(CA, axis=3)

        xx.append(DB)
        xx.append(CA)
        xx = K.concatenate(xx, axis=2)

        ws = w[:, s]
        out = tf.einsum("bmn,zbmn->mnz", K.reshape(ws, [-1, 2 * l + 1, 2 * l + 1]), xx)
        out = K.reshape(out, [(2 * l + 1) ** 2, -1])

        output.append(out)

    output = K.concatenate(output, axis=0)
    output = K.reshape(output, [nspec, *batch_size])
    return output

def so3_rifft_dumbb(x, for_grad=False, b_out=None):
    #:param x: [l * m * n, ...]
    #:return: [..., beta, alpha, gamma] in complex64
    assert K.dtype(x) == 'complex64'
    nspec = K.int_shape(x)[0]
    b_in = int(round((3/4 * nspec)**(1/3)))
    if b_out is None:
        b_out = b_in
    assert b_out >= b_in

    nfeature_out = K.int_shape(x)[-1]

    x = K.reshape(x, [nspec, -1])  # [l * m * n, batch]
    nbatch = tf.shape(x)[1]

    w = _setup_wigner(b_out, nl=b_in, weighted=for_grad)

    output = tf.zeros([nbatch, 2 * b_out, 2 * b_out, 2 * b_out], dtype='complex64')

    for l in range(min(b_in, b_out)):
        s = slice(l * (4 * l**2 - 1) // 3, l * (4 * l**2 - 1) // 3 + (2 * l + 1) ** 2)
        xs = x[s]
        ws = w[:, s]
        out = tf.einsum("mnz,bmn->zbmn", tf.reshape(xs,[2 * l + 1, 2 * l + 1, -1]), tf.reshape(ws, [-1, 2 * l + 1, 2 * l + 1]))
        #print(out[:, :, l, l])
        l1 = min(l, b_out - 1)  # if b_out < b_in

        AB = []
        CD = []
        ABCD = []

        DB, CA = tf.split(out, [l1, l1+1], axis=2)
        D, B = tf.split(DB, [l1, l1+1], axis=3)
        C, A = tf.split(CA, [l1, l1+1], axis=3)

        AB.append(A)
        AB.append(tf.zeros([nbatch, 2*b_out, 2*b_out-2*l1-1, l1+1], dtype='complex64'))
        AB.append(B)
        AB = K.concatenate(AB, axis=2)

        CD.append(C)
        CD.append(tf.zeros([nbatch, 2*b_out, 2*b_out-2*l1-1, l1], dtype='complex64'))
        CD.append(D)
        CD = K.concatenate(CD, axis=2)

        ABCD.append(AB)
        ABCD.append(tf.zeros([nbatch, 2*b_out, 2*b_out, 2*b_out-2*l1-1], dtype='complex64'))
        ABCD.append(CD)
        ABCD = K.concatenate(ABCD, axis=3)

        output = tf.add(output, ABCD)

    output = K.reshape(output, [-1, nfeature_out, 2 * b_out, 2 * b_out, 2 * b_out])
    normal = K.int_shape(output)[-1]
    output, appendix = tf.split(output, [b_out+1, b_out-1], axis=4)
    output = tf.spectral.irfft2d(output) * normal ** 2  # [batch, beta, alpha, gamma] 

    return output


def so3_fft(x, for_grad=False, b_out=None):
    #:param x: [..., beta, alpha, gamma] in complex64
    #:return: [l * m * n, ...]

    # print("KMW", x.shape)
    assert K.dtype(x) == 'complex64'
    b_in = K.int_shape(x)[-1] // 2
    assert K.int_shape(x)[-1] == 2 * b_in
    assert K.int_shape(x)[-2] == 2 * b_in
    assert K.int_shape(x)[-3] == 2 * b_in
    if b_out is None:
        b_out = b_in
    assert b_out <= b_in

    batch_size = K.int_shape(x)[:-3]
    batch_size = [-1 if x is None else x for x in batch_size]
    #print("KMW", batch_size)

    x = K.reshape(x, [-1, 2 * b_in, 2 * b_in, 2 * b_in])

    nspec = b_out * (4 * b_out ** 2 - 1) // 3
    nbatch = K.int_shape(x)[0]

    w = _setup_wigner(b_in, nl=b_out, weighted=not for_grad)
    w = tf.cast(w, dtype='complex64')
    #w = K.eval(w)

    x = tf.spectral.fft2d(x)
    #x = K.eval(x)

    # output = np.zeros(shape=(nspec, nbatch))
    # output = K.constant(output, dtype='complex64')
    # output = K.eval(output)
    output = tf.Variable(tf.placeholder(dtype='complex64', shape=[nspec, nbatch]), validate_shape=False)

    # for l in range(b_out):
    #         s = slice(l * (4 * l**2 - 1) // 3, l * (4 * l**2 - 1) // 3 + (2 * l + 1) ** 2)
    #         l1 = min(l, b_in - 1)  # if b_out > b_in, consider high frequencies as null

    #         xx = np.zeros(shape=(np.shape(x)[0], np.shape(x)[1], 2 * l + 1, 2 * l + 1))
    #         xx = K.constant(xx, dtype='complex64')
    #         xx = K.eval(xx)
    #         xx[:, :, l: l + l1 + 1, l: l + l1 + 1] = x[:, :, :l1 + 1, :l1 + 1]
    #         if l1 > 0:
    #             xx[:, :, l - l1:l, l: l + l1 + 1] = x[:, :, -l1:, :l1 + 1]
    #             xx[:, :, l: l + l1 + 1, l - l1:l] = x[:, :, :l1 + 1, -l1:]
    #             xx[:, :, l - l1:l, l - l1:l] = x[:, :, -l1:, -l1:]

    #         ws = w[:, s]
    #         out = np.einsum("bmn,zbmn->mnz", np.reshape(ws, [-1, 2 * l + 1, 2 * l + 1]), xx)
    #         output[s] = np.reshape(out, [(2 * l + 1) ** 2, -1])

    for l in range(b_out):
        sz = l * (4 * l**2 - 1) // 3
        #s = slice(l * (4 * l**2 - 1) // 3, l * (4 * l**2 - 1) // 3 + (2 * l + 1) ** 2)
        s = slice(sz, sz + (2 * l + 1) ** 2)
        l1 = min(l, b_in - 1)  # if b_out > b_in, consider high frequencies as null

        #xx = np.zeros(shape=(np.shape(x)[0], np.shape(x)[1], 2 * l + 1, 2 * l + 1))
        #xx = tf.zeros(shape=(K.int_shape(x)[0], K.int_shape(x)[1], 2 * l + 1, 2 * l + 1))
        # xx = tf.Variable(tf.placeholder(dtype='complex64', shape=[K.int_shape(x)[0], K.int_shape(x)[1], 2 * l + 1, 2 * l + 1]), validate_shape=False)
        # xx = tf.transpose(xx, perm=[2,3,0,1])
        xx = tf.Variable(tf.placeholder(dtype='complex64', shape=[2 * l + 1, 2 * l + 1, K.int_shape(x)[0], K.int_shape(x)[1]]), validate_shape=False)

        for alpha in range(-l1, l1 + 1):
            for gamma in range(-l1, l1 + 1):
                xx = tf.scatter_nd_update(xx, [[alpha, gamma]], x[:, :, l + alpha, l + gamma])
        xx = tf.transpose(xx, perm=[2,3,0,1])

        ws = w[:, s]
        out = tf.einsum("bmn,zbmn->mnz", K.reshape(ws, [-1, 2 * l + 1, 2 * l + 1]), xx)
        #output[s] = K.reshape(out, [(2 * l + 1) ** 2, -1])
        out = K.reshape(out, [(2 * l + 1) ** 2, -1])
        #output = tf.scatter_nd_update(output, [[s]], K.reshape(out, [(2 * l + 1) ** 2, -1]))
        for i in range(sz, sz + (2 * l + 1) ** 2):
            output = tf.scatter_nd_update(output, [[i]], out[i-sz])
    output = K.reshape(output, [nspec, *batch_size])
    return output

def so3_rfft(x, for_grad=False, b_out=None):
    #:param x: [..., beta, alpha, gamma] in float32
    #:return: [l * m * n, ...]
    if K.dtype(x) == 'float32':
        x = tf.cast(x, dtype='complex64')
    output = so3_fft(x, for_grad=for_grad, b_out=b_out)
    return output

def so3_rfft_dumb(x, for_grad=False, b_out=None):
    #:param x: [..., beta, alpha, gamma] in float32
    #:return: [l * m * n, ...]
    if K.dtype(x) == 'float32':
        x = tf.cast(x, dtype='complex64')
    output = so3_fft(x, for_grad=for_grad, b_out=b_out)
    output = tf.real(output)
    return output


'''
def so3_rfft(x, for_grad=False, b_out=None):
    #:param x: [..., beta, alpha, gamma]
    #:return: [l * m * n, ...]

    assert K.dtype(x) == 'float32'
    x = K.eval(x)
    x = K.constant(x, dtype='complex64')

    b_in = K.int_shape(x)[-1] // 2
    assert K.int_shape(x)[-1] == 2 * b_in
    assert K.int_shape(x)[-2] == 2 * b_in
    assert K.int_shape(x)[-3] == 2 * b_in
    if b_out is None:
        b_out = b_in
    assert b_out <= b_in
    batch_size = K.int_shape(x)[:-3]

    x = K.reshape(x, [-1, 2 * b_in, 2 * b_in, 2 * b_in])

    nspec = b_out * (4 * b_out**2 - 1) // 3
    nbatch = K.int_shape(x)[0]

    w = _setup_wigner(b_in, nl=b_out, weighted=not for_grad)
    w = K.eval(w)
    w = K.constant(w, dtype='complex64')
    w = K.eval(w)

    x = tf.fft2d(x)
    x = K.eval(x)

    output = np.zeros(shape=(nspec, nbatch))
    output = K.constant(output, dtype='complex64')
    output = K.eval(output)

    for l in range(b_out):
            s = slice(l * (4 * l**2 - 1) // 3, l * (4 * l**2 - 1) // 3 + (2 * l + 1) ** 2)
            l1 = min(l, b_in - 1)  # if b_out > b_in, consider high frequencies as null

            xx = np.zeros(shape=(np.shape(x)[0], np.shape(x)[1], 2 * l + 1, 2 * l + 1))
            xx = K.constant(xx, dtype='complex64')
            xx = K.eval(xx)
            xx[:, :, l: l + l1 + 1, l: l + l1 + 1] = x[:, :, :l1 + 1, :l1 + 1]
            if l1 > 0:
                xx[:, :, l - l1:l, l: l + l1 + 1] = x[:, :, -l1:, :l1 + 1]
                xx[:, :, l: l + l1 + 1, l - l1:l] = x[:, :, :l1 + 1, -l1:]
                xx[:, :, l - l1:l, l - l1:l] = x[:, :, -l1:, -l1:]

            ws = w[:, s]
            out = np.einsum("bmn,zbmn->mnz", np.reshape(ws, [-1, 2 * l + 1, 2 * l + 1]), xx)
            output[s] = np.reshape(out, [(2 * l + 1) ** 2, -1])

    output = K.reshape(output, [-1, *batch_size])
    return output
'''



def so3_ifft(x, for_grad=False, b_out=None):
    #:param x: [l * m * n, ...]
    #:return: [..., beta, alpha, gamma] in complex64
 
    assert K.dtype(x) == 'complex64'
    nspec = K.int_shape(x)[0]
    #print('This is nspec', nspec)
    b_in = int((3/4 * nspec)**(1/3))
    # assert nspec == b_in * (4 * b_in**2 - 1) // 3
    if b_out is None:
        b_out = b_in
    assert b_out >= b_in

    #print("Abigail",K.int_shape(x))
    nfeature_out = K.int_shape(x)[-1]

    # batch_size = K.int_shape(x)[1:]
    x = K.reshape(x, [nspec, -1])  # [l * m * n, batch]

    w = _setup_wigner(b_out, nl=b_in, weighted=for_grad)
    w = K.cast(w, dtype='complex64')
    #[batch, beta, alpha, gamma]
    output = tf.Variable(tf.placeholder(dtype='complex64', shape=[None, 2 * b_out, 2 * b_out, 2 * b_out]), validate_shape=False)

    tf.transpose(output, perm=[2,3,0,1])
    # output = K.eval(output)

    for l in range(min(b_in, b_out)):
        s = slice(l * (4 * l**2 - 1) // 3, l * (4 * l**2 - 1) // 3 + (2 * l + 1) ** 2)
        xs = x[s]
        ws = w[:, s]
        out = tf.einsum("mnz,bmn->zbmn", tf.reshape(xs,[2 * l + 1, 2 * l + 1, -1]), tf.reshape(ws, [-1, 2 * l + 1, 2 * l + 1]))
        l1 = min(l, b_out - 1)  # if b_out < b_in
 
        for alpha in range(-l1, l1 + 1):
            for gamma in range(-l1, l1 + 1):
                #adding_out = tf.transpose(out[:, :, l + alpha, l + gamma], perm=[2,3,0,1])
                output = tf.scatter_nd_add(output, [[alpha, gamma]], out[:, :, l + alpha, l + gamma])
                # if alpha >=0 and gamma >= 0:
                #     output = tf.scatter_nd_add(output, [[alpha,gamma]], adding_out)
                # if l > 0:
                #     if alpha < 0 and gamma >= 0:
                #         #adding_out = tf.transpose(out[:, :, l + alpha, l: l + l1 + 1], perm=[2,3,0,1])
                #         output = tf.scatter_nd_add(output, [[-alpha,gamma]], adding_out)
                #     elif alpha >=0 and gamma < 0:
                #         #adding_out = tf.transpose(out[:, :, l: l + l1 + 1, l - l1: l], perm=[2,3,0,1])
                #         output = tf.scatter_nd_add(output, [[alpha, -gamma]], adding_out)
                #     elif alpha < 0 and gamma < 0:
                #         #adding_out = tf.transpose(out[:, :, l - l1: l, l - l1: l], perm=[2,3,0,1])
                #         output = tf.scatter_nd_add(output, [[-alpha,-gamma]], adding_out)

    # for l in range(min(b_in, b_out)):
    #         s = slice(l * (4 * l**2 - 1) // 3, l * (4 * l**2 - 1) // 3 + (2 * l + 1) ** 2)
    #         xs = x[s]
    #         ws = w[:, s]
    #         out = tf.einsum("mnz,bmn->zbmn", tf.reshape(xs,[2 * l + 1, 2 * l + 1, -1]), tf.reshape(ws, [-1, 2 * l + 1, 2 * l + 1]))
    #         print(out, output)
    #         l1 = min(l, b_out - 1)  # if b_out < b_in
    #         output[:, :, :l1 + 1, :l1 + 1] += out[:, :, l: l + l1 + 1, l: l + l1 + 1]
    #         if l > 0:
    #             output[:, :, -l1:, :l1 + 1] += out[:, :, l - l1: l, l: l + l1 + 1]
    #             output[:, :, :l1 + 1, -l1:] += out[:, :, l: l + l1 + 1, l - l1: l]
    #             output[:, :, -l1:, -l1:] += out[:, :, l - l1: l, l - l1: l]
    
    tf.transpose(output, perm=[2,3,0,1])
    normal = tf.cast(tf.shape(output)[-1], dtype='complex64')
    output = tf.spectral.ifft2d(output) * normal ** 2  # [batch, beta, alpha, gamma] 
    output = K.reshape(output, [-1, nfeature_out, 2 * b_out, 2 * b_out, 2 * b_out])
    #print(output.shape)

    return output


def so3_rifft(x, for_grad=False, b_out=None): 
    #:param x: [l * m * n, ...]
    #:return: [..., beta, alpha, gamma] in float32

    output = so3_ifft(x, for_grad=for_grad, b_out=b_out)
    output = K.cast(output, dtype='float32')
    return output

def so3_rifft_dumb(x, for_grad=False, b_out=None): 
    #:param x: [l * m * n, ...]
    #:return: [..., beta, alpha, gamma] in float32
    nspec = K.int_shape(x)[0]
    nfeature_out = K.int_shape(x)[-1]
    x = K.reshape(x, [nspec, -1])
    b_in = int((3/4 * nspec)**(1/3))
    if b_out is None:
        b_out = b_in
    assert b_out >= b_in

    print(x)

    dumb = tf.ones([nspec, 2*b_out, 2*b_out, 2*b_out], dtype='float32')
    dumb = tf.einsum("lz,lbag->zbag", x, dumb)
    dumb = K.reshape(dumb, [-1, nfeature_out, 2*b_out, 2*b_out, 2*b_out])
    #print('so3_rifft_dumb before doing fft', dumb.shape)
    #dumb = K.cast(dumb, dtype='float32') 
    #dumb = tf.spectral.irfft2d(dumb)
    #dumb = dumb + tf.conj(dumb)
    print('so3_rifft_dumb after doing fake fft', dumb)
    return dumb


'''
def so3_rifft(x, for_grad=False, b_out=None):
    #:param x: [l * m * n, ..., complex]

    assert K.dtype(x) == 'complex64'
    nspec = K.int_shape(x)[0]
    b_in = round((3/4 * nspec)**(1/3))
    assert nspec == b_in * (4 * b_in**2 - 1) // 3
    if b_out is None:
        b_out = b_in
    assert b_out >= b_in
    batch_size = K.int_shape(x)[1:-1]

    x = K.reshape(x, [nspec, -1])
    x = K.eval(x)

    nbatch = np.shape(x)[1]

    w = _setup_wigner(b_out, nl=b_in, weighted=for_grad)
    w = K.eval(w)
    w = K.constant(w, dtype='complex64')
    w = K.eval(w)

    output = np.zeros(shape=(nbatch, 2 * b_out, 2 * b_out, 2 * b_out))
    output = K.constant(output, dtype='complex64')
    output = K.eval(output)

    for l in range(min(b_in, b_out)):
            s = slice(l * (4 * l**2 - 1) // 3, l * (4 * l**2 - 1) // 3 + (2 * l + 1) ** 2)
            xs = x[s]
            ws = w[:, s]
            out = np.einsum("mnz,bmn->zbmn", np.reshape(xs,[2 * l + 1, 2 * l + 1, -1]), np.reshape(ws, [-1, 2 * l + 1, 2 * l + 1]))
            l1 = min(l, b_out - 1)  # if b_out < b_in
            output[:, :, :l1 + 1, :l1 + 1] += out[:, :, l: l + l1 + 1, l: l + l1 + 1]
            if l > 0:
                output[:, :, -l1:, :l1 + 1] += out[:, :, l - l1: l, l: l + l1 + 1]
                output[:, :, :l1 + 1, -l1:] += out[:, :, l: l + l1 + 1, l - l1: l]
                output[:, :, -l1:, -l1:] += out[:, :, l - l1: l, l - l1: l]

    output = K.reshape(output, [*batch_size, 2 * b_out, 2 * b_out, 2 * b_out])
    output = tf.ifft2d(output) * K.int_shape(output)[-1] ** 2
    output = K.eval(output)
    output = K.constant(output, dtype='float32')

    return output
'''


@lru_cache(maxsize=32)
def _setup_wigner(b, nl, weighted):
    dss = _setup_so3_fft(b, nl, weighted)
    dss = K.constant(dss, dtype='complex64')
    return dss

#@cached_dirpklgz("cache/setup_so3_fft")
def _setup_so3_fft(b, nl, weighted):
    from lie_learn.representations.SO3.wigner_d import wigner_d_matrix
    import lie_learn.spaces.S3 as S3
    import numpy as np
    import logging

    betas = (np.arange(2 * b) + 0.5) / (2 * b) * np.pi
    w = S3.quadrature_weights(b)
    assert len(w) == len(betas)

    logging.getLogger("trainer").info("Compute Wigner: b=%d nbeta=%d nl=%d nspec=%d", b, len(betas), nl, nl**2)

    dss = []
    for b, beta in enumerate(betas):
        ds = []
        for l in range(nl):
            d = wigner_d_matrix(l, beta,
                                field='complex', normalization='quantum', order='centered', condon_shortley='cs')
            d = d.reshape(((2 * l + 1)**2, ))

            if weighted:
                d *= w[b]
            else:
                d *= 2 * l + 1

            # d # [m * n]
            ds.append(d)
        ds = np.concatenate(ds)  # [l * m * n]
        dss.append(ds)
    dss = np.stack(dss)  # [beta, l * m * n]
    return dss

