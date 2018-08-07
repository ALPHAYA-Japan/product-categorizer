# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

#-------------------------------------------------------------------------------
#                           MLP Optimizers Function
#-------------------------------------------------------------------------------
def mlp_optimizer(lr, mode = 8):
    if mode == 0: return tf.train.FtrlOptimizer(learning_rate = lr)
    if mode == 1: return tf.train.AdamOptimizer(learning_rate = lr)
    if mode == 2: return tf.train.RMSPropOptimizer(learning_rate = lr)
    if mode == 3: return tf.train.AdagradOptimizer(learning_rate = lr)
    if mode == 4: return tf.train.AdadeltaOptimizer(learning_rate = lr)
    if mode == 5: return tf.train.ProximalAdagradOptimizer(learning_rate = lr)
    if mode == 6: return tf.train.GradientDescentOptimizer(learning_rate = lr)
    if mode==7:return tf.train.ProximalGradientDescentOptimizer(learning_rate=lr)
    if mode==8: return tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.5)

def correct_prediction(x,y):
    return tf.equal(tf.cast(x, tf.int32), tf.cast(y, tf.int32))                 # return x == y

#-------------------------------------------------------------------------------
#                                    Optimizer
#-------------------------------------------------------------------------------
class Optimizer(object):
    def __init__(self,lr,mode,model,y):
        # Forward propagation...............................
        # print(y," here")
        # y            = np.eye(36)[y]
        # self.yhat    = tf.transpose(model.yhat)
        self.yhat    = model.yhat
        # self.predict = tf.argmax(self.yhat, axis=1)
        # Backward propagation..............................
        self.learning_rate  = lr
        self.optimizer      = mlp_optimizer(lr,mode)
        # self.classification = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = self.yhat)
        self.classification = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = self.yhat)
        self.cost           = tf.reduce_mean(self.classification)
        self.opt_op         = self.optimizer.minimize(self.cost)

        self.predict      = tf.argmax(self.yhat, axis = 1)
        self.real_y       = tf.argmax(y, axis = 1)
        self.correct_pred = correct_prediction(self.predict, self.real_y)            # Get predicted values
        self.accuracy     = tf.reduce_mean(tf.cast(self.correct_pred,tf.float32))# Average Accuracy

#-------------------------------------------------------------------------------
#                                    Optimizer
#-------------------------------------------------------------------------------
class Generator(object):
    def __init__(self,model):
        self.yhat    = model.yhat
        self.predict = tf.argmax(self.yhat, axis = 1)
