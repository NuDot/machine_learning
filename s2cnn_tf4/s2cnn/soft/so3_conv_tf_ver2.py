from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.topology import Layer

import math 
from s2cnn.soft.so3_fft import so3_rfft, so3_rifft
from s2cnn import so3_mm
from s2cnn.so3_ft import so3_rft


class SO3Convolution(Layer):

	def __init__(self, nfeature_in, nfeature_out, b_in, b_out, grid, filters, activation=None, data_format=None, **kwargs): '''rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):'''
        super(SO3Convolution, self).__init__(**kwargs)
        #self.rank = rank
        self.filters = filters #number of filters we will use
        #self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.kernel_size = #############################################################################
        #self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.strides = conv_utils.normalize_tuple(1, 2, 'strides')
        #self.padding = conv_utils.normalize_padding(padding)
        self.padding = conv_utils.normalize_padding('valid')
        #self.data_format = K.normalize_data_format(data_format)
        self.data_format = K.normalize_data_format(None)
        #self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.dilation_rate = conv_utils.normalize_tuple(1, 2, 'dilation_rate')
        self.activation = activations.get(activation)
        #self.use_bias = use_bias
        #self.kernel_initializer = initializers.get(kernel_initializer)
        #self.bias_initializer = initializers.get(bias_initializer)
        #self.kernel_regularizer = regularizers.get(kernel_regularizer)
        #self.bias_regularizer = regularizers.get(bias_regularizer)
        #self.activity_regularizer = regularizers.get(activity_regularizer)
        #self.kernel_constraint = constraints.get(kernel_constraint)
        #self.bias_constraint = constraints.get(bias_constraint)
        #self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        # if self.data_format == 'channels_first':
        #     channel_axis = 1
        # else:
        #     channel_axis = -1
        # if input_shape[channel_axis] is None:
        #     raise ValueError('The channel dimension of the inputs '
        #                      'should be defined. Found `None`.')
        # input_dim = input_shape[channel_axis]
        #kernel_shape = self.kernel_size + (input_dim, self.filters)  ####################################3

        # self.kernel = self.add_weight(shape=kernel_shape,
        #                               initializer=self.kernel_initializer,
        #                               name='kernel',
        #                               regularizer=self.kernel_regularizer,
        #                               constraint=self.kernel_constraint)
        self.kernel = self.add_weight(shape=(nfeature_in, nfeature_out, len(grid)),  ##################################33
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      regularizer=None,
                                      constraint=None, 
                                      trainable=True)
        # if self.use_bias:
        #     self.bias = self.add_weight(shape=(self.filters,),
        #                                 initializer=self.bias_initializer,
        #                                 name='bias',
        #                                 regularizer=self.bias_regularizer,
        #                                 constraint=self.bias_constraint)
        
        self.bias = self.add_weight(shape=(1, nfeature_out, 1, 1, 1),
                                    initializer='zeros',
                                    name='bias',
                                    regularizer=None,
                                    constraint=None)
        # else:
        #     self.bias = None

        # Set input spec.
        # self.input_spec = InputSpec(ndim=self.rank + 2, ###############################################
        #                             axes={channel_axis: input_dim}) #################################
        self.scaling = 1. / math.sqrt(len(self.grid) * self.nfeature_in * (self.b_out ** 3.) / (self.b_in ** 3.))
        self.built = True
        super(SO3Convolution, self).build(input_shape) 

    def call(self, x): ##########################################################
        '''
        :x:      [batch, feature_in,  beta, alpha, gamma]
        :return: [batch, feature_out, beta, alpha, gamma]
        '''
        #assert K.int_shape(x)[1] == self.nfeature_in
        assert K.int_shape(x)[1] == 2 * self.b_in
        assert K.int_shape(x)[2] == 2 * self.b_in

        x = K.cast(x, 'float32')
        x = so3_rfft(x, self.b_out)	# [l * m * n, batch, feature_in]
        y = so3_rft(self.kernel * self.scaling, self.b_out, self.grid)	# [l * m * n, feature_in, feature_out]
        z = so3_mm(x, y)  # [l * m * n, batch, feature_out] 
        z = so3_rifft(z)  # [batch, feature_out, beta, alpha, gamma]

        z = z + self.bias

        return z


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

    # def get_config(self):
    #     config = {
    #         'rank': self.rank,
    #         'filters': self.filters,
    #         'kernel_size': self.kernel_size,
    #         'strides': self.strides,
    #         'padding': self.padding,
    #         'data_format': self.data_format,
    #         'dilation_rate': self.dilation_rate,
    #         'activation': activations.serialize(self.activation),
    #         'use_bias': self.use_bias,
    #         'kernel_initializer': initializers.serialize(self.kernel_initializer),
    #         'bias_initializer': initializers.serialize(self.bias_initializer),
    #         'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
    #         'bias_regularizer': regularizers.serialize(self.bias_regularizer),
    #         'activity_regularizer':
    #             regularizers.serialize(self.activity_regularizer),
    #         'kernel_constraint': constraints.serialize(self.kernel_constraint),
    #         'bias_constraint': constraints.serialize(self.bias_constraint)
    #     }
    #     base_config = super(_Conv, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))













