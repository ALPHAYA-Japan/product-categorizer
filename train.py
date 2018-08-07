# -*- coding: utf-8 -*-
import os
import sys
import time
import random
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import input_data as loader
import model as models
import optimizer as optimizers

os.environ['CUDA_VISIBLE_DEVICES'] = "0"                                        # Train on CPU (hide GPU) due to memory constraints
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                                        # Remove the Warning : The TensorFlow library wasn't compiled
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

#-------------------------------------------------------------------------------
#                              Arabic Verbs Extractor
#-------------------------------------------------------------------------------
class Category_Classifier(object):
    def display(self):
        initial = ["Xavier", "Uniform", "Gaussian"]
        opt = ["Frtl",      "Adam",     "RMSProp",  "AdaGrad",
                "AdaDelta", "PrAdaGrad", "GD", "PrGD", "Momentum"]
        print("output_dim     :",self.output_dim)
        print("epochs         :",self.epochs)
        print("hiddens        :",self.hiddens)
        print("learning_rate  :",self.learning_rate)
        print("initialization :",initial[self.initialization])
        print("optimizer      :",opt[self.mlp_optimizer])
        print("acceptable_acc :",self.acceptable_acc)
        print("model_path     :",self.model_path)
        print("input_dim      :",self.input_dim)
        print("data_size      :",self.data_size)
        print("batch_size     :",self.batch_size)

    def __init__(self):
        self.output_dim     = 5     # Number of categories
        self.epochs         = 600
        self.hiddens        = [400,400]
        self.learning_rate  = 0.001
        self.initialization = 0     #xavier
        self.mlp_optimizer  = 1     #Adam
        self.acceptable_acc = 0.80
        self.batch_size     = 500
        self.model_path     = "./models/"

    def load_dataset(self, data_filename):
        outs = loader.load_data(data_filename)
        self.train_X = outs[0]
        self.test_X  = outs[1]
        self.train_Y = outs[2]
        self.test_Y  = outs[3]

        # Layer's sizes.....................................
        self.input_dim     = self.train_X.shape[1]
        self.data_size     = len(self.train_X)
        self.iterations    = 200 #int(self.data_size / self.batch_size)
        print(self.data_size,"/",self.batch_size,"=",self.iterations)

        self.display()

    def initialize(self, input_dim = 0, output_dim = 5, isPredict = False):
        # self.batch_size    = 64
        if isPredict == True:
            self.input_dim = input_dim
            self.output_dim = output_dim

        # Symbols...........................................
        self.X       = tf.placeholder(tf.float32, shape=[None, self.input_dim ])
        self.Y       = tf.placeholder(tf.float32, shape=[None, self.output_dim])

        # with tf.device('/device:GPU:0'):                                      # You can activate it for GPU version
        with tf.name_scope('model'):
            self.model =models.MLP_Model(inputs      = self.X,
                                         input_dim   = self.input_dim,
                                         hiddens     = self.hiddens,
                                         output2_dim = self.output_dim,
                                         batch_size  = self.batch_size,
                                         model_name  = "model_1",
                                         mode        = self.initialization)
        with tf.name_scope('optimizer'):
            self.optimizer = optimizers.Optimizer(lr   = self.learning_rate,
                                                  mode = self.mlp_optimizer,
                                                  model= self.model,
                                                  y    = self.Y)

        with tf.name_scope('generator'):
            self.generator = optimizers.Generator(model = self.model)

        # Session Initialization............................
        # self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()                                           # Add ops to save and restore all the variables.

    def validation(self):
        loss   = self.sess.run( self.optimizer.cost,
                                feed_dict ={self.X : self.test_X,
                                            self.Y : self.to_vectors(self.test_Y)})
        return np.mean(loss)

    def to_vectors(self,ys):
        return np.eye(self.output_dim)[ys]

    def train(self):
        from sklearn.metrics import f1_score
        import warnings
        warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

        self.initialize()
        # L = len(self.train_X[0])
        L = len(self.train_X)
        tenth_iter = int(self.iterations * 0.1)

        for epoch in range(self.epochs):
            t = time.time()
            try:
                print("Epoch %3d [" %(epoch+1),end=''); sys.stdout.flush()
                accuracy = [0. for _ in range(self.iterations)]
                loss     = [0. for _ in range(self.iterations)]
                k        = -1
                j_       = 0
                for i_ in range(self.iterations):                               # Train with each example
                    if j_ == tenth_iter: j_ = 0; print(end='='); sys.stdout.flush()         # Show the progress
                    else:                j_ += 1
                    i = random.randint(0, L - self.batch_size)
                    k += 1
                    feed_dict = {self.X: self.train_X[i: i + self.batch_size],
                                 self.Y: self.to_vectors(self.train_Y[i: i + self.batch_size])}
                    _ , accuracy[k], loss[k] , pred_y, real_y = self.sess.run([ self.optimizer.opt_op,
                                                                          self.optimizer.accuracy,
                                                                          self.optimizer.cost,
                                                                          self.optimizer.predict,
                                                                          self.optimizer.real_y],
                                                                          feed_dict = feed_dict)
                acc = np.mean(accuracy)
                cost = np.mean(loss)
                score_f1 = f1_score(real_y, pred_y, average='macro')

                if epoch % 10 != 0:
                    print("] acc : %.4f, f1-score : %.4f, loss_t : %.4f, loss_v : %.4f, time : %.4f sec" %(acc, score_f1, cost, self.validation(), time.time()-t)); sys.stdout.flush()
                else:
                    self.train_X, self.train_Y = shuffle(self.train_X, self.train_Y, random_state=0)
                    print("] acc : %.4f, f1-score : %.4f, loss_t : %.4f, loss_v : %.4f, time : %.4f sec + shuffling" %(acc, score_f1, cost, self.validation(), time.time()-t)); sys.stdout.flush()

                if acc > self.acceptable_acc:
                    self.acceptable_acc = acc
                    output_filename = self.model_path+str(epoch)+".ckpt"           # model'output file name
                    save_path = self.saver.save(self.sess, output_filename)         # we can save the model if we think it's good enough
                    print("Model saved in file: %s" % save_path)                    # `save` method will call `export_meta_graph` implicitly.
            except KeyboardInterrupt:# Just to avoid Python traceback
                print()
                sys.exit()

        self.sess.close()

    #----------------------------Pretrained Model Loading-----------------------
    def load_model(self,model_path):
        # self.initialize()
        self.saver.restore(self.sess,model_path)

    def prediction(self, features):
        return self.sess.run([self.generator.yhat,
                              self.generator.predict],
                              feed_dict = {self.X: features})

    def predict(self, features):
        outs = self.prediction([features])
        return outs

    def predict_all(self,filename):
        print(filename)
        X, Products = loader.load_test(filename)
        i = 0
        for i in range(len(X)):
            print(Products[i],":",self.predict(X[i])[1][0]) #,":",X[i])


if __name__ == '__main__':
    mymodel = Category_Classifier()

    # Training mode.............................................................
    if sys.argv[1] == 'train':
        mymodel.load_dataset(sys.argv[2])
        mymodel.train()

    # Pretraining mode..........................................................
    elif sys.argv[1] == 'predict':
        mymodel.initialize(input_dim = 5,output_dim = 5, isPredict = True)
        mymodel.load_model(sys.argv[3])
        mymodel.predict_all(sys.argv[2])
