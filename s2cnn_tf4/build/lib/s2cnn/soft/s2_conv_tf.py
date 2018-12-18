from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.topology import Layer

import math 
from s2cnn.soft.s2_fft import s2_fft, s2_rfft_dumb
from s2cnn.soft.so3_fft import so3_rifft, so3_rifft_dumb, so3_rifft_dumbb
from s2cnn.utils.complex import complex_mm
from functools import lru_cache
from s2cnn.s2_ft import s2_rft, s2_rft_dumb
from s2cnn.s2_mm import s2_mm

import tensorflow as tf

class S2Convolution(Layer):

    def __init__(self, output_dim, nfeature_in, b_in, b_out, grid,use_bias=False,
                 bias_initializer='zeros', bias_regularizer=None, bias_constraint=None, **kwargs):
        self.nfeature_in = nfeature_in
        self.output_dim = output_dim
        self.b_in = b_in
        self.b_out = b_out
        self.grid = grid
        #bias
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(S2Convolution, self).__init__(**kwargs)

    # def compute_output_shape(self, input_shape):
    #     if self.data_format == 'channels_last':
    #         space = input_shape[1:-1]
    #         new_space = []
    #         for i in range(len(space)):
    #             new_dim = conv_utils.conv_output_length(
    #                 space[i],
    #                 self.kernel_size[i],
    #                 padding=self.padding,
    #                 stride=self.strides[i],
    #                 dilation=self.dilation_rate[i])
    #             new_space.append(new_dim)
    #         return (input_shape[0],) + tuple(new_space) + (self.filters,)
    #     if self.data_format == 'channels_first':
    #         space = input_shape[2:]
    #         new_space = []
    #         for i in range(len(space)):
    #             new_dim = conv_utils.conv_output_length(
    #                 space[i],
    #                 self.kernel_size[i],
    #                 padding=self.padding,
    #                 stride=self.strides[i],
    #                 dilation=self.dilation_rate[i])
    #             new_space.append(new_dim)
    #         return (input_shape[0], self.filters) + tuple(new_space)

    def build(self, input_shape):
        #assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.nfeature_in, self.output_dim, len(self.grid)),
                                      initializer='uniform',
                                      trainable=True)
        #setup bias
        self.scaling = 1. / math.sqrt(len(self.grid) * self.nfeature_in * (self.b_out ** 3.) / (self.b_in ** 3.))
        if self.use_bias:
            self.bias = self.add_weight(shape=(1, output_dim, 1, 1, 1),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(S2Convolution, self).build(input_shape)  # Be sure to call this at the end

        #a, b = x
        #return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)]

    def compute_output_shape(self, input_shape):
        return(input_shape[0], self.output_dim, 2 * self.b_out, 2 * self.b_out, 2 * self.b_out)

    def call(self, x):
        #printcall(1)
        #print('This is b_out from s2_conv', self.b_out)
        #assert K.int_shape(x)[1] == self.nfeature_in
        # print('Input x', x.shape)
        assert K.int_shape(x)[-1] == 2 * self.b_in
        assert K.int_shape(x)[-2] == 2 * self.b_in

        # x = K.cast(x, 'complex64')
        #x = tf.complex(x, tf.zeros_like(x))
        #x = s2_fft(x, b_out=self.b_out)
        #x = s2_fft_dumb(x, b_out=self.b_out)
        # do rfft 
        x = s2_rfft_dumb(x, b_out=self.b_out)

        #y = s2_rft(self.kernel * self.scaling, self.b_out, self.grid)
        y = s2_rft_dumb(self.kernel * self.scaling, self.b_out, self.grid)
        # print('After s2_conv x', x)
        # print('After s2_conv y', y)
        #z = K.dot(x, y)  # [l * m * n, batch, feature_out, complex]
        #print('AAA',x,y,tf.shape(x), tf.shape(y))
        z = s2_mm(x, y)
        # print('s2_conv before doing so3_rifft_dumb z', z.shape)
        z = so3_rifft_dumbb(z)  # [batch, feature_out, beta, alpha, gamma]
        # print('After s2_conv z, need [batch, f_out, beta, alpha, gamma].', z)

        if self.use_bias:
            z = z + self.bias
        
        return z


        