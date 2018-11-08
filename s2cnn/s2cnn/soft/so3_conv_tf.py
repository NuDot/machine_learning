from keras import backend as K
from keras.engine.topology import Layer

import math 
from so3_fft import so3_rfft, so3_rifft
from s2cnn import so3_rft

class SO3Convolution(Layer):

    def __init__(self, output_dim, nfeature_in, nfeature_out, b_in, b_out, grid,use_bias=True,
                 bias_initializer='zeros', bias_regularizer=None, bias_constraint=None, **kwargs):
        self.nfeature_in = nfeature_in
        self.nfeature_out = nfeature_out
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
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.nfeature_in, self.nfeature_out, len(self.grid)),
                                      initializer='uniform',
                                      trainable=True)
        #setup bias
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.scaling = 1. / math.sqrt(len(self.grid) * self.nfeature_in * (self.b_out ** 3.) / (self.b_in ** 3.))

        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert K.int_shape(x)[1] == self.nfeature_in
        assert K.int_shape(x)[2] == 2 * self.b_in
        assert K.int_shape(x)[3] == 2 * self.b_in
        assert K.int_shape(x)[4] == 2 * self.b_in

        x = so3_rfft(x, self.b_out)
        y = so3_rft(self.kernel * self.scaling, self.b_out, self.grid)

        assert K.int_shape(x)[0] == K.int_shape(y)[0]
        assert K.int_shape(x)[2] == K.int_shape(y)[1]
        z = K.dot(x, y)  # [l * m * n, batch, feature_out, complex]
        assert K.int_shape(z)[0] == K.int_shape(x)[0]
        assert K.int_shape(z)[1] == K.int_shape(x)[1]
        assert K.int_shape(z)[2] == K.int_shape(y)[2]
        z = so3_rifft(z)  # [batch, feature_out, beta, alpha, gamma]

        if self.use_bias:
            z = K.eval(z)
            z = z + self.bias
            z = K.constant(z)

        return z

        #a, b = x
        #return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]


