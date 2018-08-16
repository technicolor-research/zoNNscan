# -*- coding: utf-8 -*-
# Copyright (c) 2015 – Thomson Licensing, SAS
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted (subject to the limitations in the disclaimer below) provided that
# the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#                                                                                                                     
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# * Neither the name of Thomson Licensing, or Technicolor, nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS
# LICENSE.  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.   

"""Core functions."""

import numpy as np
import scipy.stats as stat
from scipy.special import gammainc
import tensorflow as tf

class zoNNscan():

    ''' Analytic tool for measuring a model indecision in a zone around a given input
        Code: Adel Jaouen
        Paper: "zoNNscan: a boundary-entropy metric for zone inspection of neural models"
    '''

    def __init__(self, model_path, nb_classes=10):
        '''
        :param model_path: directory where to find .meta, .index, .data and checkpoint files.
        '''
        self.model_path = model_path
        self.nb_classes = nb_classes

    def load_model(self, name_input_placeholder='input_1:0', dict_name_Value={}):
        ''' Load the model from the path, must be called first
        :param name_input_placeholder: name of the input tensor to feed
        :param dict_name_Value: dictionary containing other tensors names and values corresponding if necessary to make predictions
        '''
        tf.reset_default_graph()
        def get_names(graph=tf.get_default_graph()):
            return [t.name for op in graph.get_operations() for t in op.values()]

        path = self.model_path

        sess = tf.Session()
        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        sess.run(init_op)
        new_saver = tf.train.import_meta_graph(path + '.meta')
        new_saver.restore(sess, path)
        x = tf.get_default_graph().get_tensor_by_name(name_input_placeholder)
        name_output = [name for name in get_names() if 'Softmax:0' in name]
        preds = tf.get_default_graph().get_tensor_by_name(name_output[-1])

        self.x = x
        self.inputs_shape = x.get_shape().as_list()
        self.input_shape_flatten = (1, np.prod(self.inputs_shape[1:]))
        self.inputs_shape[0] = -1
        self.preds = preds
        self.sess = sess
        dict = {}

        for key in dict_name_Value.keys() :
            dict[tf.get_default_graph().get_tensor_by_name(key)] = dict_name_Value[key]

        self.dictionary = dict

    def data_import(self, X_train, Y_train, X_test, Y_test):
        '''
        import the data
        :param X_train:
        :param Y_train:
        :param X_test:
        :param Y_test:
        '''
        self.X_train, self.Y_train, self.X_test, self.Y_test = X_train, Y_train, X_test, Y_test


    def change_point(self, point=None , random=False):
        '''
        change the point from which zoNNscan will be estimate
        :param point: the point assigned to the zoNNscan object
        :param random: if True, assign a random example from test-set loaded
        :param boundary: if True, assign a boundary point between point and another random test set point or between two random test set points if point is None
        :param eps: the step for the fgm method
        '''
        if random:
            try :
                self.X_test
            except AttributeError :
                print('no data load, try data_import first')
                return ;

            rnd = np.random.randint(len(self.X_test))
            point = self.X_train[rnd]

            self.index_point = rnd
            self.true_class = np.argmax(self.Y_test[rnd])

        self.point = point
        dict = {self.x: np.array(self.point).reshape(self.inputs_shape)}
        dict.update(self.dictionary)
        self.predict_class = np.argmax(self.sess.run(self.preds, feed_dict=dict))

    def sample_mmc(self, k, radius, ord=np.inf):
        '''
        sample k points around the self.point in the intersection of the ball (for the 2 or inf norm) and [0,1]^d
        :param k: length of the sample
        :param radius: 2 time the desired radius
        :param ord: order of the norm corresponding to the ball, (in {np.inf, 2})
        :return: the sample
        '''
        try :
            self.point
        except AttributeError :
            print('affect a point with change_point first!')
            return None

        #function to sample in the 2-norm ball
        def sample_ball(center, radius, k):
            r = radius
            ndim = center.size
            x = np.random.normal(size=(k, ndim))
            ssq = np.sum(x ** 2, axis=1)
            fr = r * gammainc(ndim / 2, ssq / 2) ** (1 / ndim) / np.sqrt(ssq)
            frtiled = np.tile(fr.reshape(k, 1), (1, ndim))
            return(center + np.multiply(x, frtiled))

        #function to sample in the inf-norm ball
        def sample_cube(center, radius, k_per_cube):
            ndim = center.size
            lim_inf_cube = center - np.ones(ndim)*radius
            lim_sup_cube = center + np.ones(ndim)*radius

            lims_sup = np.clip(lim_sup_cube,0,1)
            lims_inf = np.clip(lim_inf_cube,0,1)

            limites_sample = np.array((lims_inf, lims_sup))
            limites_sample = np.squeeze(limites_sample).T
            return(np.array([[np.random.uniform(liminf, limsup) for (liminf, limsup) in limites_sample]for k in range(k_per_cube)]))

        point = np.squeeze(np.asarray(self.point)).reshape(self.input_shape_flatten)
        assert ord in [2,np.inf] # 'ord must be np.inf or 2'
        if ord == np.inf :
            sample = sample_cube(point, radius, k)
        else: #2
            sample = sample_ball(point, radius, k)

        return(sample.reshape(self.inputs_shape))

    def predict(self, sample):
        '''
        returns the confidences array corresponding to the sample
        :param sample:
        :return:
        '''
        dict = {self.x : sample.reshape(self.inputs_shape)}
        dict.update(self.dictionary)

        return(self.sess.run(self.preds, feed_dict=dict))

    def H_line(self, predictions):
        '''
        returns a vector of entropies of each lines of a prediction array
        :param predictions:
        :return: vector of entropies
        '''
        entropies = [stat.entropy(probs, base=self.nb_classes) for probs in predictions]

        return(entropies)

    '''Core function for zoNNscan metric'''
    def zoNNscan(self, k, radius, ord=np.inf):
        sample = self.sample_mmc(k, radius, ord=ord)
        predictions = self.predict(sample)
        entropies = self.H_line(predictions)
        return(np.mean(entropies))

"""Example of use."""

from keras.datasets import mnist as data
from keras.utils import to_categorical

if __name__ == '__main__':
    # parameters
    path_mlp = './vars/mlp'
    nb_classes = 10
    
    (X_train, Y_train), (X_test, Y_test) = data.load_data()
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    X_train = X_train.astype('float32')
    X_train /= 255
    X_test = X_test.astype('float32')
    X_test /= 255
    
    k = 1000
    radius = 0.01
    ord = np.inf
    # preparing model
    mlp = zoNNscan(path_mlp, nb_classes)
    mlp.load_model(name_input_placeholder='Placeholder_1:0')
    mlp.data_import(X_train, Y_train, X_test, Y_test)

    # zoNNscan measure    
    mlp.change_point(random=True)
    zoNNscans_mlp = mlp.zoNNscan(k, radius, ord)

    print('zoNNscan value on the MLP model and a random test example: %f ' % zoNNscans_mlp)
