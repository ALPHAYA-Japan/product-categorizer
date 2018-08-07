import tensorflow as tf
import layers as layer

''' ----------------------------------------------------------------------------
    ------------------------------- Model Class --------------------------------
    ----------------------------------------------------------------------------
        Base Model class. Defines basic API for all Model objects.
        @Properties:
          name: String, defines the variable scope of the Model.
    ----------------------------------------------------------------------------'''
class Model(object):
    def __init__(self, model_name, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        self.name = kwargs.get('name')                                          # Defining the name of the Model
        if not self.name:
            self.name = self.__class__.__name__.lower() + model_name

        self.vars    = {}                                                       # Defining the logging of the Model
        self.logging = kwargs.get('logging', False)

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=self.name)
        self.vars= {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass

class MLP_Model(Model):
    def __init__(self, batch_size, model_name, inputs, input_dim, hiddens, output2_dim, mode, **kwargs):
        super(MLP_Model, self).__init__(model_name, **kwargs)

        self.input_dim  = input_dim
        self.hidden1    = hiddens[0]
        self.hidden2    = hiddens[1]
        self.output2_dim= output2_dim
        self.mode       = mode
        self.inputs     = inputs
        self.model_name = model_name
        self.batch_size = batch_size
        self.build()                                                            # Start Initialization

    def _build(self):
        self.h = self.inputs

        # self.h = tf.contrib.layers.batch_norm(self.h,center=True, scale=True) #,is_training = True)

        self.h = layer.Layer_1( input_dim   = self.input_dim,
                                output_dim  = self.hidden1,
                                mode        = self.mode,
                                activation  = lambda x : x,
                                # activation  = tf.nn.sigmoid,
                                model_name  = self.model_name,
                                batch_size  = self.batch_size,
                                logging     = self.logging
                                )(self.h)

        # self.h = tf.contrib.layers.fully_connected(self.h,
        #                                            self.hidden1,
        #                                            activation_fn = tf.nn.sigmoid,
        #                                            scope = 'dense')

        self.h = layer.Layer_1( input_dim   = self.hidden1,
                                output_dim  = self.hidden2,
                                mode        = self.mode,
                                # activation  = lambda x : x,
                                activation  = tf.nn.sigmoid,
                                model_name  = self.model_name,
                                batch_size  = self.batch_size,
                                logging     = self.logging
                                )(self.h)

        # Input Shape: [], Output Shape: []............
        self.yhat = layer.Layer_2( input_dim   = self.hidden2,
                                   output_dim  = self.output2_dim,
                                   mode        = self.mode,
                                   # activation  = lambda x : x,
                                   activation  = tf.nn.softmax,
                                   isBN        = False,
                                   batch_size  = self.batch_size,
                                   model_name  = self.model_name,
                                   logging     = self.logging
                                   )(self.h)
