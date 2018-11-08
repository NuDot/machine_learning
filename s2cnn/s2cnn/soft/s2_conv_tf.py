from keras import backend as K
from keras.engine.topology import Layer

import math 
from s2_fft import s2_rfft
from so3_fft import so3_rifft
from s2cnn import s2_rft

class S2Convolution(Layer):

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

        super(S2Convolution, self).__init__(**kwargs)

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
        # assert K.int_shape(x)[1] == self.nfeature_in
        # assert K.int_shape(x)[2] == 2 * self.b_in
        # assert K.int_shape(x)[3] == 2 * self.b_in

        #x = s2_rfft(x, self.b_out)
        #y = s2_rft(self.kernel * self.scaling, self.b_out, self.grid)
        #z = K.dot(x, y)  # [l * m * n, batch, feature_out, complex]
        #z = s2_rifft(z)  # [batch, feature_out, beta, alpha, gamma]
        batch = K.int_shape(x)[0]
        z = np.random.radn(batch, self.feature_out, 2*b_in, 2*b_in, 2*b_in)  

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


        