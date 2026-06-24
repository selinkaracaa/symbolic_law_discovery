import keras
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.regularizers import l2
import tensorflow as tf

from common.preset import *


def Conv_Layer(inp, filters):
	return Conv2D(
			filters=filters,
			kernel_size=(3,3),
			padding="same",
			activation="elu", 
			kernel_regularizer=l2(CNN_REGULARIZATION)
		)(inp)


def Max_Pooling_Layer(inp):
	return MaxPooling2D(pool_size=(2,2),strides=(2,2))(inp)


def Dense_Layer(inp, units):
	return Dense(
			units=units, 
			activation="elu", 
			kernel_regularizer=l2(CNN_REGULARIZATION)
		)(inp)


def VGG_Conv_Block(inp, filters):
	conv = Conv_Layer(inp, filters)
	conv = Conv_Layer(conv, filters)
	oup = Max_Pooling_Layer(conv)
	return oup


def VGG16(data_input):
	conv = VGG_Conv_Block(data_input, 64)
	conv = VGG_Conv_Block(conv, 128)
	conv = VGG_Conv_Block(conv, 256)
	conv = VGG_Conv_Block(conv, 512)
	conv = VGG_Conv_Block(conv, 512)

	encoding = Flatten()(conv)
	encoding = Dense_Layer(encoding, 4096)
	encoding = Dropout(0.5)(encoding)
	encoding = Dense_Layer(encoding, 4096)
	encoding = Dropout(0.5)(encoding)
	encoding = Dense_Layer(encoding, 1000)

	return encoding


def CNN():
	data_input = Input(shape=(*HIGH_RESOLUTION,1))
	encoding = VGG16(data_input)
	output = Dense(units=1, activation="sigmoid")(encoding)
	return Model(data_input, output)