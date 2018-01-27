import numpy as np
import load_data
import tensorflow as tf
from scipy.stats import norm
from tensorflow import Session as sess
from keras.layers import Dense, Activation
from keras.engine import Layer
from keras.engine import InputSpec
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
from tensorflow.contrib.learn import KMeansClustering as tf_kmeans

def k_means(inputs, k_num):
    kms = tf_kmeans(k_num,initial_clusters='kmeans_plus_plus')
    def input_for_kmeans():
        data = tf.constant(inputs, dtype=tf.float32)
        return (data,None)
    kms.fit(input_fn=input_for_kmeans,max_steps=100)
    return kms.clusters()

def max_one(inputs):
    tf_inputs = tf.constant(inputs,dtype=tf.float32)
    the_max = tf.reduce_max(tf_inputs,axis=0)
    output = tf.where(tf.cast(the_max,dtype=tf.bool), the_max, tf.fill(the_max.shape,0.0000000001))
    return output


class rbfn(Layer):
    def __init__(self, units,
                 hidden_centers,
                 norm_mat,
                 delta = 0.1,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.units = units
        self.hidden_centers = hidden_centers
        self.delta = delta
        self.norm_mat = norm_mat
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        super(rbfn, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        center_num = self.hidden_centers.shape[0]
        self.kernel = self.add_weight(shape=(center_num, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def RBFunction(x, h_centers, delta, norm_mat):
        c = tf.divide(h_centers, norm_mat)
        x = tf.divide(x, norm_mat )
        e_c = tf.expand_dims(c, 0 )
        e_x = tf.expand_dims(x, 1 )
        return tf.exp( -delta * tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(e_c, e_x)), 2 )))

    def call(self, inputs):
        def RBFunction(x, h_centers, delta, norm_mat):
            c = tf.divide(h_centers, norm_mat)
            x = tf.divide(x, norm_mat )
            e_c = tf.expand_dims(c, 0 )
            e_x = tf.expand_dims(x, 1 )
            return tf.exp( -delta * tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(e_c, e_x)), 2 )))

        output = K.dot(RBFunction(inputs,self.hidden_centers,self.delta, self.norm_mat),self.kernel)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(rbf, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

