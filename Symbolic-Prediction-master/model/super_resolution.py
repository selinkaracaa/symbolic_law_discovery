from keras.layers import Conv2D, Lambda, Input, Add
from keras.models import Model
import tensorflow as tf

def pixel_shuffle(scale):
	return lambda x: tf.nn.depth_to_space(x, scale)


def upsample(x, num_filters):

	def upsample_1(x, factor, **kwargs):
		x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
		return Lambda(pixel_shuffle(scale=factor))(x)

	x = upsample_1(x, 2, name='conv2d_1_scale_2')
	x = upsample_1(x, 2, name='conv2d_2_scale_2')
	return x


def res_block(x_in, filters):
	x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
	x = Conv2D(filters, 3, padding='same')(x)
	return x


def EDSR(num_filters=64, num_res_blocks=8):
	x_in = Input(shape=(None, None, 1))
	x = b = Conv2D(num_filters, 3, padding='same')(x_in)

	for i in range(num_res_blocks):
		b = res_block(b, num_filters)

	b = Conv2D(num_filters, 3, padding='same')(b)
	x = Add()([x, b])
	x = upsample(x, num_filters)
	x = Conv2D(1, 3, padding='same')(x)

	return Model(x_in, x, name="edsr")