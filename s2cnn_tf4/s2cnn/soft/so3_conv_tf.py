from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.topology import Layer

import math 
from s2cnn.soft.so3_fft import so3_rfft, so3_rfft_dumb, so3_rfft_dumbb, so3_rifft, so3_rifft_dumb, so3_rifft_dumbb
from functools import lru_cache
import numpy as np
import tensorflow as tf
from s2cnn.so3_ft import so3_rft, so3_rft_dumb
from s2cnn.so3_mm import so3_mm

class SO3Convolution(Layer):

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

        super(SO3Convolution, self).__init__(**kwargs)

    def build(self, input_shape):
        #assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.nfeature_in, self.output_dim, len(self.grid)),
                                      initializer='uniform',
                                      trainable=True)
        #setup bias
        if self.use_bias:
            self.bias = self.add_weight(shape=(1, output_dim, 1, 1, 1),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.scaling = 1. / math.sqrt(len(self.grid) * self.nfeature_in * (self.b_out ** 3.) / (self.b_in ** 3.))

        super(SO3Convolution, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        #print("KMW",x.shape)
        assert K.int_shape(x)[1] == self.nfeature_in
        assert K.int_shape(x)[2] == 2 * self.b_in
        assert K.int_shape(x)[3] == 2 * self.b_in
        assert K.int_shape(x)[4] == 2 * self.b_in

        print('Before so3_conv x', x)
        #x = so3_rfft(x, b_out=self.b_out) # [l * m * n, batch, feature_in]
        x = so3_rfft_dumbb(x, b_out=self.b_out)
        #y = so3_rft(self.kernel * self.scaling, self.b_out, self.grid)  # [l * m * n, feature_in, feature_out]
        y = so3_rft_dumb(self.kernel * self.scaling, self.b_out, self.grid)  # [l * m * n, feature_in, feature_out]

        assert K.int_shape(x)[0] == K.int_shape(y)[0]
        #print("Kiyohime", K.int_shape(x))
        assert K.int_shape(x)[2] == K.int_shape(y)[1]
        #z = K.dot(x, y)  # [l * m * n, batch, feature_out]
        print('After so3_conv x', x)
        print('After so3_conv y', y)
        z = so3_mm(x, y)
        assert K.int_shape(z)[0] == K.int_shape(x)[0]
        assert K.int_shape(z)[1] == K.int_shape(x)[1]
        assert K.int_shape(z)[2] == K.int_shape(y)[2]
        print('Before rifft z ', z)
        #z = so3_rifft(z)  # [batch, feature_out, beta, alpha, gamma]
        z = so3_rifft_dumbb(z)  # [batch, feature_out, beta, alpha, gamma]

        if self.use_bias:
            # z = K.eval(z)
            # z = z + K.eval(self.bias)
            # z = K.constant(z)
            z = z + self.bias
        print('After so3_conv z', z)
        return z

        #a, b = x
        #return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)]

    def compute_output_shape(self, input_shape):
        return(input_shape[0], self.output_dim, 2 * self.b_out, 2 * self.b_out, 2 * self.b_out)
    # def compute_output_shape(self, input_shape):
    #     assert isinstance(input_shape, list)
    #     shape_a, shape_b = input_shape
    #     return [(shape_a[0], self.output_dim), shape_b[:-1]]

    # def call(self, x):
    #         #print("KMW",x.shape)
    #         assert K.int_shape(x)[1] == self.nfeature_in
    #         assert K.int_shape(x)[2] == 2 * self.b_in
    #         assert K.int_shape(x)[3] == 2 * self.b_in
    #         assert K.int_shape(x)[4] == 2 * self.b_in

    #         x = so3_rfft(x, b_out=self.b_out) # [l * m * n, batch, feature_in]
    #         #y = so3_rft(self.kernel * self.scaling, self.b_out, self.grid)  # [l * m * n, feature_in, feature_out]
    #         y = so3_rft(self.kernel * self.scaling, self.b_out, self.grid)  # [l * m * n, feature_in, feature_out]

    #         assert K.int_shape(x)[0] == K.int_shape(y)[0]
    #         #print("Kiyohime", K.int_shape(x))
    #         assert K.int_shape(x)[2] == K.int_shape(y)[1]
    #         #z = K.dot(x, y)  # [l * m * n, batch, feature_out]
    #         z = so3_mm(x, y)
    #         assert K.int_shape(z)[0] == K.int_shape(x)[0]
    #         assert K.int_shape(z)[1] == K.int_shape(x)[1]
    #         assert K.int_shape(z)[2] == K.int_shape(y)[2]
    #         z = so3_rifft(z)  # [batch, feature_out, beta, alpha, gamma]

    #         if self.use_bias:
    #             # z = K.eval(z)
    #             # z = z + K.eval(self.bias)
    #             # z = K.constant(z)
    #             z = z + self.bias

    #         return z


