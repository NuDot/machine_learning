# pylint: disable=R,C,E1101
import math
from functools import lru_cache
from keras import backend as K
import tensorflow as tf
import numpy as np

from string import Template


# inspired by https://gist.github.com/szagoruyko/89f83b6f5f4833d3c8adf81ee49f22a8

def s2_rfft_dumb(x, for_grad=False, b_out=None):
    '''
    :param x: [..., beta, alpha]
    :return:  [l * m, ...]
    '''
    b_in = x.shape[-1] // 2
    assert x.shape[-1] == 2 * b_in
    assert x.shape[-2] == 2 * b_in
    b_in = int(b_in)
    if b_out is None:
        b_out = b_in
    assert b_out <= b_in
    batch_size = x.shape[1:-2]

    x = K.reshape(x, [-1, 2 * b_in, 2 * b_in])  # [batch, beta, alpha]

    '''
    :param x: [batch, beta, alpha] (nbatch, 2 * b_in, 2 * b_in)
    :return: [l * m, batch] (b_out**2, nbatch)
    '''
    nspec = int(b_out) ** 2
    nbatch = K.int_shape(x)[0]

    w = _setup_wigner(b_in, nl=b_out, weighted=not for_grad)
    w = K.reshape(w, [2 * b_in, -1]) # [beta, l * m] (2 * b_in, nspec)

    x = tf.spectral.fft(tf.complex(x, tf.zeros_like(x)))

    output = []

    output_index = np.arange(b_out)

    for l in range(b_out):
            s = slice(l ** 2, l ** 2 + 2 * l + 1)
            xx = K.concatenate((x[:, :, -l:], x[:, :, :l+1]), axis=2) if l > 0 else x[:, :, :1]
            output.append(tf.einsum("bm,zbm->mz", w[:, s], xx))
            
    output = K.concatenate(output, axis=0)
    output = K.reshape(output, shape=[nspec, -1, *batch_size.as_list()])  # [l * m, ...] (nspec, ...)

    return output


def s2_fft(x, for_grad=False, b_out=None):
    '''
    :param x: [..., beta, alpha]
    :return:  [l * m, ...]
    '''

    #assert x.size(-1) == 2
    assert K.dtype(x) == 'complex64'#Changeme
    #b_in = x.size(-2) // 2
    #b_in = K.int_shape(x)[-1]  // 2
    b_in = x.shape[-1] // 2
    #assert x.size(-2) == 2 * b_in
    assert x.shape[-1] == 2 * b_in
    #assert x.size(-3) == 2 * b_in
    assert x.shape[-2] == 2 * b_in
    b_in = int(b_in)
    #print('This is b_in', b_in, 'This is b_out', b_out)
    if b_out is None:
        b_out = b_in
    assert b_out <= b_in
    #batch_size = x.size()[:-3]
    batch_size = x.shape[1:-2]

    #x = x.view(-1, 2 * b_in, 2 * b_in, 2)  # [batch, beta, alpha, complex]
    x = K.reshape(x, [-1, 2 * b_in, 2 * b_in])  # [batch, beta, alpha]

    '''
    :param x: [batch, beta, alpha, complex] (nbatch, 2 * b_in, 2 * b_in)
    :return: [l * m, batch, complex] (b_out**2, nbatch)
    '''
    nspec = int(b_out) ** 2
    #nbatch = x.size(0)
    nbatch = K.int_shape(x)[0]


    #wigner = _setup_wigner(b_in, nl=b_out, weighted=not for_grad, device_type=x.device.type, device_index=x.device.index)
    w = _setup_wigner(b_in, nl=b_out, weighted=not for_grad)
    w = tf.cast(w, dtype='complex64')
    #wigner = wigner.view(2 * b_in, -1)  # [beta, l * m] (2 * b_in, nspec)
    w = K.reshape(w, [2 * b_in, -1]) # [beta, l * m] (2 * b_in, nspec)

    #x = torch.fft(x, 1)  # [batch, beta, m, complex]
    x = tf.spectral.fft(x) # [batch, beta, m]

    output = tf.Variable(tf.placeholder(dtype='complex64', shape=[nspec, nbatch]), validate_shape=False)
    # output = tf.reshape(output, [nspec, -1])
    # print(output)
    output_index = np.arange(b_out)

    for l in range(b_out):
            s = slice(l ** 2, l ** 2 + 2 * l + 1)
            #xx = torch.cat((x[:, :, -l:], x[:, :, :l + 1]), dim=2) if l > 0 else x[:, :, :1]
            xx = K.concatenate((x[:, :, -l:], x[:, :, :l+1]), axis=2) if l > 0 else x[:, :, :1]
            #output[s] = torch.einsum("bm,zbmc->mzc", (wigner[:, s], xx))
            # if l == 0:
            #     output = tf.einsum("bm,zbm->mz", w[:, s], xx)
            # else:
            #     output = tf.concat([output, tf.einsum("bm,zbm->mz", w[:, s], xx)], 0)
            output = tf.scatter_nd_update(output, output_index[s], tf.einsum("bm,zbm->mz", w[:, s], xx))
            #print(output, output.shape)
            #K.eval(cache)
            #output = tf.fill(s, tf.einsum("bm,zbm->mz", w[:, s], xx))

    # #output = output.view(-1, *batch_size, 2)  # [l * m, ..., complex] (nspec, ..., 2)
    output = K.reshape(output, shape=[nspec, -1, *batch_size.as_list()])  # [l * m, ...] (nspec, ...)
    return output


def s2_ifft(x, for_grad=False, b_out=None):
    '''
    :param x: [l * m, ...]
    '''
    #assert x.size(-1) == 2
    assert K.dtype(x) == 'complex64'
    #nspec = x.size(0)
    nspec = K.int_shape(x)[0]
    b_in = int(nspec ** 0.5)
    assert nspec == b_in ** 2
    if b_out is None:
        b_out = b_in
    assert b_out >= b_in
    #batch_size = x.size()[1:-1]
    #batch_size = K.int_shape(x)[1:]

    #x = x.view(nspec, -1, 2)  # [l * m, batch, complex] (nspec, nbatch, 2)
    x = K.reshape(x, [nspec, -1])  # [l * m, batch] (nspec, nbatch)

    '''
    :param x: [l * m, batch, complex] (b_in**2, nbatch)
    :return: [batch, beta, alpha, complex] (nbatch, 2 b_out, 2 * b_out)
    '''
    #nbatch = x.size(1)
    nbatch = K.int_shape(x)[1]

    #wigner = _setup_wigner(b_out, nl=b_in, weighted=for_grad, device_type=x.device.type, device_index=x.device.index)
    w = _setup_wigner(b_out, nl=b_in, weighted=for_grad)
    w = tf.cast(w, dtype='complex64')
    #wigner = wigner.view(2 * b_out, -1)  # [beta, l * m] (2 * b_out, nspec)
    w = K.reshape(w, [2 * b_out, -1])  # [beta, l * m] (2 * b_out, nspec)

    #output = x.new_zeros((nbatch, 2 * b_out, 2 * b_out, 2))
    output = tf.Variable(tf.zeros([nbatch, 2 * b_out, 2 * b_out], dtype='complex64'), validate_shape=False)

    for l in range(b_in):
            s = slice(l ** 2, l ** 2 + 2 * l + 1)
            #out = torch.einsum("mzc,bm->zbmc", (x[s], wigner[:, s]))
            out = tf.einsum("mz,bm->zbm", x[s], w[:, s])
            #output[:, :, :l + 1] += out[:, :, -l - 1:]
            tf.transpose(output, perm=[2,0,1])
            for i in range(-l, l+1):
                output = tf.scatter_nd_add(output, [[i]], out[:, :, i-l-1])
            tf.transpose(output, perm=[1,2,0])
            # tf.assign_add(output[:, :, :l + 1], out[:, :, -l - 1:])
            # if l > 0:
            #     output[:, :, -l:] += out[:, :, :l]

    #output = torch.ifft(output, 1) * output.size(-2)  # [batch, beta, alpha, complex]
    #output = output.view(*batch_size, 2 * b_out, 2 * b_out, 2)
    #output = K.reshape(output, [*batch_size, 2 * b_out, 2 * b_out])
    output = K.reshape(output, [-1, 2 * b_out, 2 * b_out])
    output = tf.ifft(output) * K.int_shape(output)[-1]  # [batch, beta, alpha]
    return output

def s2_ifft_dumb(x, for_grad=False, b_out=None):
    '''
    :param x: [l * m, ...]
    '''
    assert K.dtype(x) == 'complex64'
    nspec = K.int_shape(x)[0]
    b_in = int(nspec ** 0.5)
    assert nspec == b_in ** 2
    if b_out is None:
        b_out = b_in
    assert b_out >= b_in

    x = K.reshape(x, [nspec, -1])  # [l * m, batch] (nspec, nbatch)
    '''
    :param x: [l * m, batch, complex] (b_in**2, nbatch)
    :return: [batch, beta, alpha, complex] (nbatch, 2 b_out, 2 * b_out)
    '''
    nbatch = K.int_shape(x)[1]

    w = _setup_wigner(b_out, nl=b_in, weighted=for_grad)
    w = tf.cast(w, dtype='complex64')
    w = K.reshape(w, [2 * b_out, -1])  # [beta, l * m] (2 * b_out, nspec)

    output = []

    # for l in range(b_in):
    #         s = slice(l ** 2, l ** 2 + 2 * l + 1)
    #         out = tf.einsum("mz,bm->zbm", x[s], w[:, s]) # [batch, beta, s]
    #         output.append(out)

    # output = K.concatenate(output, axis=-1)
    # output = K.reshape(output, [-1, 2 * b_out, 2 * b_out])
    # output = tf.ifft(output) * K.int_shape(output)[-1]  # [batch, beta, alpha]

    dumb = tf.ones([2*b_out, 2*b_out, nspec], dtype='complex64')
    dumb = tf.einsum("mz,bam->zba", x, dumb)
    #return output
    return dumb

# @lru_cache(maxsize=32)
# def _setup_wigner(b, nl, weighted, device_type, device_index):
#     dss = _setup_s2_fft(b, nl, weighted)
#     dss = torch.tensor(dss, dtype=torch.float32, device=torch.device(device_type, device_index))  # [beta, l * m] # pylint: disable=E1102
#     return dss.contiguous()

@lru_cache(maxsize=32)
def _setup_wigner(b, nl, weighted):
    dss = _setup_s2_fft(b, nl, weighted)
    dss = K.constant(dss, dtype='complex64')
    return dss

#@cached_dirpklgz("cache/setup_s2_fft")
def _setup_s2_fft(b, nl, weighted):
    from lie_learn.representations.SO3.wigner_d import wigner_d_matrix
    import lie_learn.spaces.S3 as S3
    import numpy as np
    import logging

    betas = (np.arange(2 * b) + 0.5) / (2 * b) * np.pi
    w = S3.quadrature_weights(b) * 2 * b
    assert len(w) == len(betas)

    logging.getLogger("trainer").info("Compute Wigner (only columns): b=%d nbeta=%d nl=%d nspec=%d", b, len(betas), nl,
                                      nl ** 2)

    dss = []
    for b, beta in enumerate(betas):
        ds = []
        for l in range(nl):
            d = wigner_d_matrix(l, beta,
                                field='complex', normalization='quantum', order='centered', condon_shortley='cs')
            d = d[:, l]  # d[m=:, n=0]

            if weighted:
                d *= w[b]
            else:
                d *= 2 * l + 1

            ds.append(d)  # [m]
        dss.append(np.concatenate(ds))  # [l * m]

    dss = np.stack(dss)  # [beta, l * m]
    return dss


'''
class S2_fft_real(torch.autograd.Function):
    def __init__(self, b_out=None):
        super(S2_fft_real, self).__init__()
        self.b_in = None
        self.b_out = b_out

    def forward(self, x):  # pylint: disable=W
        from s2cnn.utils.complex import as_complex
        self.b_in = x.size(-1) // 2
        return s2_fft(as_complex(x), b_out=self.b_out)

    def backward(self, grad_output):  # pylint: disable=W
        return s2_ifft(grad_output, for_grad=True, b_out=self.b_in)[..., 0]
'''
'''
class S2_ifft_real(torch.autograd.Function):
    def __init__(self, b_out=None):
        super(S2_ifft_real, self).__init__()
        self.b_in = None
        self.b_out = b_out

    def forward(self, x):  # pylint: disable=W
        nspec = x.size(0)
        self.b_in = round(nspec ** 0.5)
        return s2_ifft(x, b_out=self.b_out)[..., 0]

    def backward(self, grad_output):  # pylint: disable=W
        from s2cnn.utils.complex import as_complex
        return s2_fft(as_complex(grad_output), for_grad=True, b_out=self.b_in)
'''

'''
def test_s2fft_cuda_cpu():
    x = torch.rand(1, 2, 12, 12, 2)  # [..., beta, alpha, complex]
    z1 = s2_fft(x, b_out=5)
    z2 = s2_fft(x.cuda(), b_out=5).cpu()
    q = (z1 - z2).abs().max().item() / z1.std().item()
    print(q)
    assert q < 1e-4


def test_s2ifft_cuda_cpu():
    x = torch.rand(12 ** 2, 10, 2)  # [l * m, ..., complex]
    z1 = s2_ifft(x, b_out=13)
    z2 = s2_ifft(x.cuda(), b_out=13).cpu()
    q = (z1 - z2).abs().max().item() / z1.std().item()
    print(q)
    assert q < 1e-4


if __name__ == "__main__":
    test_s2fft_cuda_cpu()
    test_s2ifft_cuda_cpu()
'''


