from enum import Enum
import math
import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
from tensor_ops import blur, minibatch_stddev, pixel_norm, upsample, downsample
from typing import List


class PixelNorm(tf.keras.layers.Layer):
    def call(self, x: tf.Tensor) -> tf.Tensor:
        return pixel_norm(x)


class Upsample(tf.keras.layers.Layer):
    def call(self, x: tf.Tensor) -> tf.Tensor:
        return upsample(x)


class Downsample(tf.keras.layers.Layer):
    def call(self, x: tf.Tensor) -> tf.Tensor:
        return downsample(x)


class MinibatchStddev(tf.keras.layers.Layer):
    def call(self, x: tf.Tensor) -> tf.Tensor:
        return minibatch_stddev(x)


class ScaledLeakyRelu(tf.keras.layers.Layer):
    def __init__(self, alpha: float = 0.2, gain: float = math.sqrt(2.), **kwargs):
        super(ScaledLeakyRelu, self).__init__(**kwargs)
        self.alpha = alpha
        self.gain = gain

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return tf.nn.leaky_relu(x, self.alpha) * self.gain

    def get_config(self):
        config = super(ScaledLeakyRelu, self).get_config()
        config.update({
            'alpha': self.alpha,
            'gain': self.gain
        })
        return config


class ImageConversionMode(Enum):
    TENSORFLOW_TO_MODEL = 0  # [0, 1] to [-1, 1]
    MODEL_TO_TENSORFLOW = 1  # [-1, 1] to [0, 1]


class ImageConversion(tf.keras.layers.Layer):
    def __init__(self, conversion_mode: ImageConversionMode, **kwargs):
        super(ImageConversion, self).__init__(**kwargs)
        self.conversion_mode = ImageConversionMode(conversion_mode)

    def call(self, image: tf.Tensor) -> tf.Tensor:
        if self.conversion_mode == ImageConversionMode.TENSORFLOW_TO_MODEL:
            return image * 2. - 1.  # [0, 1] to [-1, 1]
        elif self.conversion_mode == ImageConversionMode.MODEL_TO_TENSORFLOW:
            return image * 0.5 + 0.5  # [-1, 1] to [0, 1]
        else:
            assert False, f'Unknown conversion mode: {self.conversion_mode}'

    def get_config(self):
        config = super(ImageConversion, self).get_config()
        config.update({
            'conversion_mode': self.conversion_mode.value
        })
        return config


class ScaledAdd(tf.keras.layers.Layer):
    def __init__(self, scale: float = 1. / math.sqrt(2.), **kwargs):
        super(ScaledAdd, self).__init__(**kwargs)
        self.scale_value = scale

    def build(self, input_shapes):
        assert len(input_shapes) == 2
        a_shape, b_shape = input_shapes
        assert a_shape[1:] == b_shape[1:], f'{a_shape} != {b_shape}'
        self.scale = self.add_weight(
            name='scale',
            initializer=tf.keras.initializers.Constant(value=self.scale_value),
            trainable=False)

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        return (inputs[0] + inputs[1]) * self.scale

    def get_config(self):
        config = super(ScaledAdd, self).get_config()
        config.update({
            'scale': self.scale_value
        })
        return config


class ScaledConv2d(tf.keras.layers.Layer):
    def __init__(
            self,
            channel_count: int,
            kernel_size: int,
            strides: int = 1,
            padding: str = 'valid',
            pre_blur: bool = False,
            **kwargs):
        super(ScaledConv2d, self).__init__(**kwargs)
        self.rank = 2
        self.channel_count = channel_count
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.pre_blur = pre_blur

    def build(self, input_shape: List[int]) -> None:
        assert len(input_shape) == self.rank + 2
        in_channel_count = input_shape[-1]
        kernel_shape = self.kernel_size + (in_channel_count, self.channel_count)
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=tf.keras.initializers.random_normal(mean=0., stddev=1.),
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(self.channel_count,),
            initializer=tf.keras.initializers.zeros(),
            trainable=True)
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=tf.keras.initializers.constant(
                1. / tf.sqrt(tf.reduce_prod(tf.cast(kernel_shape[:-1], tf.float32)))),
            trainable=False)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        y = x
        if self.pre_blur:
            y = blur(y)
        y = tf.nn.conv2d(y, self.kernel * self.scale, strides=self.strides, padding=self.padding.upper())
        y = tf.nn.bias_add(y, self.bias)
        return y

    def get_config(self):
        config = super(ScaledConv2d, self).get_config()
        config.update({
            'channel_count': self.channel_count,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'pre_blur': self.pre_blur
        })
        return config


'''Replaces an upscale2d + conv2d sequence

Based on progan/stylegan code, this is "Faster and uses less memory than performing the operations
separately."

This is designed to upsample by 2x using a 3x3 convolution and does not accept any parameters other
than the channel count
'''
class UpsampleConv2d(tf.keras.layers.Layer):
    def __init__(
            self,
            channel_count: int,
            **kwargs):
        super(UpsampleConv2d, self).__init__(**kwargs)
        self.rank = 2
        self.channel_count = channel_count
        self.kernel_size = conv_utils.normalize_tuple(3, self.rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(2, self.rank, 'strides')
        self.padding = conv_utils.normalize_padding('same')

    def build(self, input_shape: List[int]) -> None:
        assert len(input_shape) == self.rank + 2
        in_channel_count = input_shape[-1]
        kernel_shape = self.kernel_size + (self.channel_count, in_channel_count)
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=tf.keras.initializers.random_normal(mean=0., stddev=1.),
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(self.channel_count,),
            initializer=tf.keras.initializers.zeros(),
            trainable=True)
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=tf.keras.initializers.constant(
                1. / tf.sqrt(tf.cast(tf.reduce_prod(kernel_shape) // self.channel_count, tf.float32))),
            trainable=False)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([
            input_shape[0],
            input_shape[1] * 2,
            input_shape[2] * 2,
            self.channel_count])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        input_shape = tf.shape(x)
        batch_size, in_height, in_width = input_shape[0], input_shape[1], input_shape[2]
        output_shape = (batch_size, in_height * 2, in_width * 2, self.channel_count)

        y = tf.nn.conv2d_transpose(
            x,
            self.kernel * self.scale,
            output_shape,
            self.strides,
            padding=self.padding.upper())

        if not tf.executing_eagerly():
            y.set_shape(self.compute_output_shape(x.shape))

        y = tf.nn.bias_add(y, self.bias)
        y = blur(y)
        return y

    def get_config(self):
        config = super(UpsampleConv2d, self).get_config()
        config.update({
            'channel_count': self.channel_count
        })
        return config


class ScaledDense(tf.keras.layers.Layer):
    def __init__(
            self,
            output_count: int,
            **kwargs):
        super(ScaledDense, self).__init__(**kwargs)
        self.output_count = output_count

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.kernel = self.add_weight(
            name='kernel',
            shape=[input_shape[-1], self.output_count],
            initializer=tf.keras.initializers.random_normal(mean=0., stddev=1.),
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(self.output_count,),
            initializer=tf.keras.initializers.zeros(),
            trainable=True)
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=tf.keras.initializers.constant(1. / math.sqrt(input_shape[-1])),
            trainable=False)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        y = tf.matmul(x, self.kernel * self.scale)
        return tf.nn.bias_add(y, self.bias)

    def get_config(self):
        config = super(ScaledDense, self).get_config()
        config.update({
            'output_count': self.output_count
        })
        return config
