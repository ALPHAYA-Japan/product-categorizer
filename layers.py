import tensorflow as tf
import numpy as np

_LAYER_UIDS = {}                                                                # global unique layer ID dictionary for layer name assignment

''' ----------------------------------------------------------------------------
    ------------------------ Layer Identifier Helper ---------------------------
                    Helper function, assigns unique layer IDs
    ----------------------------------------------------------------------------'''
def get_layer_uid(layer_name=''):
    if layer_name not in _LAYER_UIDS:                                           # If it is a new layer ...
        _LAYER_UIDS[layer_name] = 1                                             # ... assign 1 to it
        return 1
    else:                                                                       #
        _LAYER_UIDS[layer_name] += 1                                            # else increase its id
        return _LAYER_UIDS[layer_name]

#-------------------------------------------------------------------------------
#                              Weights Initializers
#-------------------------------------------------------------------------------
def weight_initialization(input_dim, output_dim, name, mode = 0):
    if mode == 0:                                                               # Xavier Initialization
        val = 1 * np.sqrt(6.0 / (input_dim))                                    # You can use 4 for sigmoid or 1 for tanh activation
        initial = tf.random_uniform([input_dim, output_dim],                    #
                                     minval =-val,                              # Minimum Value
                                     maxval = val,                              # Maximum Value
                                     dtype  = tf.float32)
        return tf.Variable(initial, name = name)
        # initializer = tf.contrib.layers.variance_scaling_initializer(factor=6.0,
        #                                                              mode='FAN_AVG',
        #                                                              uniform=False,
        #                                                              seed=None,
        #                                                              dtype=tf.float32
        #                                                              )
        # initializer = tf.contrib.layers.xavier_initializer(False)
        # return tf.get_variable(name = name,
        #                        shape=[input_dim, output_dim],
        #                        initializer=initializer)
    elif mode == 1:                                                             # Uniform Initialization
        val = int(np.sqrt(6.0 / (input_dim)))
        initial = tf.random_uniform([input_dim, output_dim],
                                     minval =-val,                              # Minimum Value
                                     maxval = val,                              # Maximum Value
                                     dtype  = tf.float32)
        return tf.Variable(initial, name = name)
    elif mode == 2:                                                             # Gaussian Initialization
        val = 2. / (input_dim) #0.1
        initial = tf.random_normal([input_dim, output_dim],
                                    mean    = 0,                                # Mean Value
                                    stddev  = val,                              # Standard Deviation
                                    dtype   = tf.float32)
        return tf.Variable(initial, name = name)

#-------------------------------------------------------------------------------
#                               Bias Initializers
#-------------------------------------------------------------------------------
def bias_initialization(input_dim, output_dim, name, mode = 0):
    if mode == 0:
        initial = tf.constant(0.01, shape = [output_dim])             # Constant Initialization
        return tf.Variable(initial, name = name)
    elif mode == 1:
        initial = tf.random_normal(shape = [output_dim])             # Gaussian Initialization
        return tf.Variable(initial, name = name)
    elif mode == 2:                                                               # Xavier Initialization
        val = 1 * np.sqrt(6.0 / (input_dim))                                    # You can use 4 for sigmoid or 1 for tanh activation
        initial = tf.random_uniform([output_dim],                    #
                                     minval =-val,                              # Minimum Value
                                     maxval = val,                              # Maximum Value
                                     dtype  = tf.float32)
        return tf.Variable(initial, name = name)
    elif mode == 3:                                                             # Uniform Initialization
        val = int(np.sqrt(6.0 / (input_dim)))
        initial = tf.random_uniform([output_dim],
                                     minval =-val,                              # Minimum Value
                                     maxval = val,                              # Maximum Value
                                     dtype  = tf.float32)
        return tf.Variable(initial, name = name)
    elif mode == 4:                                                             # Gaussian Initialization
        val = 2. / (input_dim) #0.1
        initial = tf.random_normal([output_dim],
                                    mean    = 0,                                # Mean Value
                                    stddev  = val,                              # Standard Deviation
                                    dtype   = tf.float32)
        return tf.Variable(initial, name = name)

''' ----------------------------------------------------------------------------
    ------------------------------- Layer Class --------------------------------
    ----------------------------------------------------------------------------
     Base layer class. Defines basic API for all layer objects.
     @Properties:
           name: String, defines the variable scope of the layer.
     @Methods:
           _call(inputs): Defines computation graph of layer
                          (i.e. takes input, returns output)
           __call__(inputs): Wrapper for _call()
    ----------------------------------------------------------------------------'''
class Layer(object):
    def __init__(self,model_name, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        self.name = kwargs.get('name')                                          # Defining the name of the layer
        if not self.name:
            layer     = self.__class__.__name__.lower() + model_name
            self.name = layer + '_' + str(get_layer_uid(layer))

        self.vars     = {}                                                      # Defining the logging of the layer
        self.logging  = kwargs.get('logging', False)
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs

class Layer_1(Layer):
    def __init__(self, batch_size, model_name, input_dim, output_dim, mode, activation, **kwargs):

        # Step 1: Creation of the Layer.........................................
        super(Layer_1, self).__init__(model_name,**kwargs)                      # Calling the super class "Layer"
        self.output_dim = output_dim
        self.activation = activation
        self.batch_size = batch_size
        # self.keep_prob  = dropout

        # Step 2: Inialize the weights..........................................
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_initialization(input_dim,
                                                         output_dim,
                                                         name ="weights",
                                                         mode = mode)
            self.vars['biases' ] =  bias_initialization(input_dim,
                                                        output_dim,
                                                        name ="biases",
                                                        mode = 0)

    def _call(self, x):
        # if self.isBN == True:
        #     x = tf.contrib.layers.batch_norm(x,center=True, scale=True)
        # if self.keep_prob != 0.0:
        #     X = tf.nn.dropout(X, 1 - self.keep_prob)
        w = self.vars['weights']
        b = self.vars['biases' ]
        x = tf.add(tf.matmul(x, w),b)
        x = self.activation(x)
        return x

class Layer_2(Layer):
    def __init__(self, batch_size, model_name, input_dim, output_dim, mode, isBN, activation, **kwargs):

        # Step 1: Creation of the Layer.........................................
        super(Layer_2, self).__init__(model_name, **kwargs)                        # Calling the super class "Layer"
        self.output_dim = output_dim
        self.activation = activation
        self.isBN       = isBN
        self.batch_size = batch_size
        # self.keep_prob  = dropout

        # Step 2: Inialize the weights..........................................
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_initialization(input_dim,
                                                         output_dim,
                                                         name ="weights",
                                                         mode = mode)
            self.vars['biases' ] = bias_initialization( input_dim,
                                                        output_dim,
                                                        name ="biases",
                                                        mode = 0)

    def _call(self, x):
        # if self.keep_prob != 0.0:
        #     x = tf.nn.dropout(x, 1 - self.keep_prob)
        w = self.vars['weights']
        b = self.vars['biases']
        x = tf.matmul(x, w) + b
        x = self.activation(x)
        if self.isBN == True:
            x = tf.contrib.layers.batch_norm(x,center=True, scale=True)
        return x
