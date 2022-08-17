import tensorflow as tf


def create_blur_filter(dtype):
    return tf.constant(
        [[0.015625, 0.046875, 0.046875, 0.015625],
         [0.046875, 0.140625, 0.140625, 0.046875],
         [0.046875, 0.140625, 0.140625, 0.046875],
         [0.015625, 0.046875, 0.046875, 0.015625]],
        dtype=dtype)


def blur(x, strides=1):
    channel_count = x.shape[3]
    filter = create_blur_filter(x.dtype)[:, :, tf.newaxis, tf.newaxis]
    filter = tf.tile(filter, [1, 1, channel_count, 1])

    return tf.nn.depthwise_conv2d(x, filter, strides=[1, strides, strides, 1], padding='SAME')


def upsample(x):
    def upsample_with_zeros(x):
        in_height = x.shape[1]
        in_width = x.shape[2]
        channel_count = x.shape[3]

        out_height = in_height * 2
        out_width = in_width * 2
        x = tf.reshape(x, [-1, in_height, 1, in_width, 1, channel_count])
        x = tf.pad(x, [[0, 0], [0, 0], [0, 1], [0, 0], [0, 1], [0, 0]])
        return tf.reshape(x, [-1, out_height, out_width, channel_count])

    return blur(upsample_with_zeros(x*4.))


def downsample(x):
    return blur(x, strides=2)


def pixel_norm(x: tf.Tensor, epsilon: float = 1e-7) -> tf.Tensor:
    original_dtype = x.dtype
    x = tf.cast(x, tf.float32)
    normalized = x / tf.math.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon)
    return tf.cast(normalized, original_dtype)


def lerp(start: tf.Tensor, end: tf.Tensor, factor: tf.Tensor) -> tf.Tensor:
    return start + (end - start) * factor


def reduce_std_nan_safe(x, axis=None, keepdims=False, epsilon=1e-7):
    y = tf.cast(x, tf.float32)
    mean = tf.reduce_mean(y, axis=axis, keepdims=True)
    variance = tf.reduce_mean(tf.square(y - mean), axis=axis, keepdims=keepdims)
    sqrt = tf.sqrt(variance + epsilon)
    return tf.cast(sqrt, x.dtype)


def minibatch_stddev(x: tf.Tensor, group_size=4) -> tf.Tensor:
    original_shape = tf.shape(x)
    original_dtype = x.dtype
    global_sample_count = original_shape[0]
    group_size = tf.minimum(group_size, global_sample_count)
    group_count = global_sample_count // group_size
    tf.Assert(
        group_size * group_count == global_sample_count,
        ['Sample count was not divisible by group size'])
    # Shape definitions:
    # N = global sample count
    # G = group count
    # M = sample count within group
    # H = height
    # W = width
    # C = channel count
                                                                # [NHWC] Input shape
    y = tf.reshape(
        x,
        tf.concat([[-1, group_size], original_shape[1:]], 0))   # [GMHWC] Split into groups
    y = tf.cast(y, tf.float32)
    stddevs = reduce_std_nan_safe(y, axis=1, keepdims=True)     # [G1HWC]
    avg = tf.reduce_mean(
        stddevs,
        axis=tf.range(1, tf.rank(stddevs)),
        keepdims=True)                                          # [G1111]
    new_feature_shape = tf.concat([tf.shape(y)[:-1], [1]], 0)   # [GMHW1]
    new_feature = tf.broadcast_to(avg, new_feature_shape)
    y = tf.concat([y, new_feature], axis=-1)
    y = tf.reshape(
		y,
		tf.concat([[-1], tf.shape(y)[2:]], 0))
    y = tf.cast(y, original_dtype)
    return y
