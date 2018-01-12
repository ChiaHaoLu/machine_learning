import numpy as np
import load_data
import tensorflow as tf
from tensorflow import Session as sess
from keras import Sequential
from keras import activations, initializers, regularizers, constraints

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Activation

def RBF_gauss_euclidean(x,c,sigma):
        #sample x number = N, center bumber = M, x dim = center dim = input_dim
        input_dim = x.shape[1]
        N = x.shape[0]
        M = c.shape[0]
	print x.shape
        M_for_eye = tf.cast(M, tf.int32)
        center_element_num = M*input_dim
	print M
        #(x-c)^2, output size:NxM
        #[(x1-c1)^2, (x1-c2)^2, ...]
        #[(x2-c1)^2, (x2-c2)^2, ...]
        c = tf.reshape(c,[1,center_element_num])
	print c
        c_mat = tf.tile(c,[N,1])
        x_mat = tf.tile(x,[1,M])
        element_x_sub_c_square_mat = tf.square(x_mat-c_mat)

        I_mat = tf.eye(M_for_eye)
        element_merge_mat = tf.tile(I_mat,[1,input_dim])
        element_merge_mat = tf.reshape(element_merge_mat,[input_dim*M,M])
        merge_x_sub_c_square_mat = K.dot(element_x_sub_c_square_mat,element_merge_mat)
        dist_mat = tf.sqrt(merge_x_sub_c_square_mat)
        gauss = pdf.Normal(loc=0., scale=sigma)
        output = gauss.prob(dist_mat)
        return output

class rbf(Layer):
	def __init__(self, units, activation=None,
		kernel_initializer='glorot_uniform',
		kernel_regularizer=None, activity_regularizer=None,
		kernel_constraint=None,
		**kwargs):
		self.kernel_initializer = initializers.get(kernel_initializer)
                self.kernel_regularizer = regularizers.get(kernel_regularizer)
                self.activity_regularizer = regularizers.get(activity_regularizer)
                self.kernel_constraint = constraints.get(kernel_constraint)
		self.units = units
		#self.input_shape = input_shape
		self.center_num = units
		super(rbf, self).__init__(**kwargs)
	def build(self, input_shape):
        	# Create a trainable weight variable for this layer.
		input_dim = input_shape[-1]
		print input_shape
		
		self.centers = self.add_weight(shape=(self.center_num, input_dim),
				initializer=self.kernel_initializer,
				name='centers',
				regularizer=self.kernel_regularizer,
				constraint=self.kernel_constraint)

		self.sigmas = self.add_weight(shape=(1, self.center_num),
                                initializer=self.kernel_initializer,
                                name='sigmas',
                                regularizer=self.kernel_regularizer,
                                constraint=self.kernel_constraint)

		super(rbf, self).build(input_shape)
		# Be sure to call this somewhere!

	def call(self, inputs):
		output = RBF_gauss_euclidean(inputs,self.centers,self.sigmas)
		return output

	def compute_output_shape(self, input_shape):
		output_shape = list(input_shape)
		output_shape[-1] = self.output_dim
		#sess.run(output_shape)
		print output_shape
		return tuple(output_shape)	
