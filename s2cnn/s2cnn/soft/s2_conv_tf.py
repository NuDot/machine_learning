from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.topology import Layer

import math 
from s2cnn.soft.s2_fft import s2_fft
from s2cnn.soft.so3_fft import so3_rifft
from s2cnn.s2_ft import s2_rft

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

        super(S2Convolution, self).__init__(**kwargs)

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

        super(S2Convolution, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        #assert K.int_shape(x)[1] == self.nfeature_in
        assert K.int_shape(x)[1] == 2 * self.b_in
        assert K.int_shape(x)[2] == 2 * self.b_in
        print(x.shape)

        x = K.cast(x, 'complex64')
        x = s2_fft(x, self.b_out)
        y = s2_rft(self.kernel * self.scaling, self.b_out, self.grid)
        z = K.dot(x, y)  # [l * m * n, batch, feature_out, complex]
        z = so3_rifft(z)  # [batch, feature_out, beta, alpha, gamma]

        if self.use_bias:
            z = K.eval(z)
            z = z + self.bias
            z = K.constant(z)

        return z

        #a, b = x
        #return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)]

    # def compute_output_shape(self, input_shape):
    #     assert isinstance(input_shape, list)
    #     shape_a, shape_b = input_shape
    #     return [(shape_a[0], self.output_dim), shape_b[:-1]]


        