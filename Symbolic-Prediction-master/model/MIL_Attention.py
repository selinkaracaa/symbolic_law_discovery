import h5py
from keras import backend as K
from keras import initializers, regularizers
from keras.layers import Input, Layer, multiply
from keras.models import Model
from keras.regularizers import l2
import numpy as np
import tensorflow as tf

from common.preset import *
from model.CNN import VGG16


class Attention_Layer(Layer):

	def __init__(
		self, L_dim, kernel_initializer='glorot_uniform', 
		kernel_regularizer=None, use_bias=True, use_gated=True, **kwargs
	):
		self.L_dim = L_dim
		self.use_bias = use_bias
		self.use_gated = use_gated

		self.v_init = initializers.get(kernel_initializer)
		self.w_init = initializers.get(kernel_initializer)
		self.u_init = initializers.get(kernel_initializer)


		self.v_regularizer = regularizers.get(kernel_regularizer)
		self.w_regularizer = regularizers.get(kernel_regularizer)
		self.u_regularizer = regularizers.get(kernel_regularizer)

		super(Attention_Layer, self).__init__(**kwargs)

	def build(self, input_shape):

		assert len(input_shape) == 2
		input_dim = input_shape[1]

		self.V = self.add_weight(
			shape=(input_dim, self.L_dim),
			initializer=self.v_init,
			name='v',
			regularizer=self.v_regularizer,
			trainable=True
		)


		self.w = self.add_weight(
			shape=(self.L_dim, 1),
			initializer=self.w_init,
			name='w',
			regularizer=self.w_regularizer,
			trainable=True
		)


		if self.use_gated:
			self.U = self.add_weight(
				shape=(input_dim, self.L_dim),
				initializer=self.u_init,
				name='U',
				regularizer=self.u_regularizer,
				trainable=True
			)
		else:
			self.U = None

		self.input_built = True


	def call(self, x, mask=None):
		n, d = x.shape
		original_x = x

		x = K.tanh(K.dot(x, self.V))

		if self.use_gated:
			gate_x = K.sigmoid(K.dot(original_x, self.U))
			ac_x = x * gate_x
		else:
			ac_x = x

		soft_x = K.dot(ac_x, self.w)
		alpha = K.softmax(K.transpose(soft_x))
		alpha = K.transpose(alpha)
		return alpha

	def compute_output_shape(self, input_shape):
		shape = list(input_shape)
		assert len(shape) == 2
		shape[1] = 1
		return tuple(shape)

	def get_config(self):
		config = {
			'v_initializer': initializers.serialize(self.V.initializer),
			'w_initializer': initializers.serialize(self.w.initializer),
			'v_regularizer': regularizers.serialize(self.v_regularizer),
			'w_regularizer': regularizers.serialize(self.w_regularizer),
			'use_bias': self.use_bias
		}
		base_config = super(Attention_Layer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class Last_Sigmoid_Layer(Layer):

	def __init__(
		self, output_dim, kernel_initializer='glorot_uniform', 
		bias_initializer='zeros', kernel_regularizer=None, 
		bias_regularizer=None, use_bias=True, **kwargs
	):
		self.output_dim = output_dim

		self.kernel_initializer = initializers.get(kernel_initializer)
		self.bias_initializer = initializers.get(bias_initializer)
		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.bias_regularizer = regularizers.get(bias_regularizer)

		self.use_bias = use_bias
		super(Last_Sigmoid_Layer, self).__init__(**kwargs)

	def build(self, input_shape):
		assert len(input_shape) == 2
		input_dim = input_shape[1]

		self.kernel = self.add_weight(
			shape=(input_dim, self.output_dim),
			initializer=self.kernel_initializer,
			name='kernel',
			regularizer=self.kernel_regularizer
		)

		if self.use_bias:
			self.bias = self.add_weight(
				shape=(self.output_dim,),
				initializer=self.bias_initializer,
				name='bias',
				regularizer=self.bias_regularizer
		)

		else:
			self.bias = None

		self.input_built = True

	def call(self, x, mask=None):
		n, d = x.shape
		x = K.sum(x, axis=0, keepdims=True)
		x = K.dot(x, self.kernel)
		if self.use_bias:
			x = K.bias_add(x, self.bias)

		out = K.sigmoid(x)
		return out

	def compute_output_shape(self, input_shape):
		shape = list(input_shape)
		assert len(shape) == 2
		shape[1] = self.output_dim
		return tuple(shape)

	def get_config(self):
		config = {
			'output_dim': self.output_dim,
			'kernel_initializer': initializers.serialize(self.kernel.initializer),
			'bias_initializer': initializers.serialize(self.bias_initializer),
			'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
			'bias_regularizer': regularizers.serialize(self.bias_regularizer),
			'use_bias': self.use_bias
		}
		base_config = super(Last_Sigmoid_Layer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


def MIL_Attention():
	attention_regularizer = l2(ATTENTION_REGULARIZATION)
	data_input = Input(shape=(*HIGH_RESOLUTION,1))
	encoding = VGG16(data_input)

	alpha = Attention_Layer(
		L_dim=ATTENTION_DIM, 
		kernel_regularizer=attention_regularizer
	)(encoding)

	x_mul = multiply([alpha, encoding])
	output = Last_Sigmoid_Layer(output_dim=1)(x_mul)

	return Model(data_input, output)
