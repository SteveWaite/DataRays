from layers import ImageConversion, ImageConversionMode, MinibatchStddev, PixelNorm, ScaledAdd, ScaledConv2d, ScaledDense, Upsample, Downsample, ScaledLeakyRelu, UpsampleConv2d
import tensorflow as tf
from typing import Dict, Optional, Tuple


def validate_resolution(resolution: int) -> None:
    assert resolution in [4, 8, 16, 32, 64, 128, 256, 512, 1024]


def activate(
        x,
        activation = lambda: ScaledLeakyRelu()):
    if activation is not None:
        x = activation()(x)
    return x


def conv_2d(
        x,
        channel_count: int,
        kernel_size: int = 3,
        activation = lambda: ScaledLeakyRelu(),
        padding: str = 'same',
        strides: int = 1,
        pre_blur: bool = False,
        name: Optional[str] = None):
    x = ScaledConv2d(
        channel_count,
        kernel_size,
        padding=padding,
        strides=strides,
        pre_blur=pre_blur,
        name=name)(x)
    return activate(x, activation=activation)


def create_generator_body(resolution, latent_vector):
    def to_rgb(x):
        rgb = conv_2d(x, 3, 1, activation=None, name=f'to_rgb_{resolution}x{resolution}')
        return rgb

    resolution_to_channel_counts = {
        4: 512, 8: 512, 16: 512, 32: 512, 64: 512, 128: 256, 256: 128, 512: 64, 1024: 32}
    channel_count = resolution_to_channel_counts[resolution]
    if resolution == 4:
        block = ScaledDense(
            channel_count*4*4,
            name='latent_to_4x4')(latent_vector)
        block = tf.keras.layers.Reshape((4, 4, channel_count))(block)
        block = activate(block)
        block = conv_2d(block, channel_count, name='conv_4x4')
        rgb = to_rgb(block)
        return rgb, block
    else:
        lower_res_rgb, lower_res_block = create_generator_body(resolution // 2, latent_vector)
        name_base = f'conv_{resolution}x{resolution}'

        block = lower_res_block
        block = activate(UpsampleConv2d(channel_count, name=f'{name_base}_1')(block))
        block = conv_2d(block, channel_count, name=f'{name_base}_2')
        rgb = to_rgb(block)

        lower_res_rgb = Upsample()(lower_res_rgb)
        rgb = tf.keras.layers.Add()([lower_res_rgb, rgb])
        return rgb, block


def create_generator(output_resolution: int, latent_vector_size: int ) -> tf.keras.Model:
    validate_resolution(output_resolution)

    latent_vector = tf.keras.layers.Input((latent_vector_size,))
    normalized_latent_vector = PixelNorm()(latent_vector)

    model_rgb, _ = create_generator_body(output_resolution, normalized_latent_vector)
    tensorflow_rgb = ImageConversion(ImageConversionMode.MODEL_TO_TENSORFLOW)(model_rgb)

    # Cast the output to float32. Needed when using mixed_float16.
    if tensorflow_rgb.dtype != tf.float32:
        tensorflow_rgb = tf.keras.layers.Activation(None, dtype=tf.float32)(tensorflow_rgb)

    return tf.keras.Model(inputs=latent_vector, outputs=tensorflow_rgb)


def make_discriminator_body(tf_format_image, resolution):
    resolution_to_feature_counts: Dict[int, Tuple[int, int]] = {
        1024: (32, 64),
        512: (64, 128),
        256: (128, 256),
        128: (256, 512),
        64: (512, 512),
        32: (512, 512),
        16: (512, 512),
        8: (512, 512),
        4: (512, 512)}
    feature_counts = resolution_to_feature_counts[resolution]

    incoming_block = None
    if tf_format_image.shape[1] > resolution:
        incoming_block = make_discriminator_body(tf_format_image, resolution*2)
    else:
        model_format_image = ImageConversion(ImageConversionMode.TENSORFLOW_TO_MODEL)(tf_format_image)
        incoming_block = conv_2d(
            model_format_image,
            feature_counts[0],
            kernel_size=1,
            name=f'from_rgb_{resolution}x{resolution}')

    if resolution == 4:
        block = incoming_block
        block = MinibatchStddev()(block)
        block = conv_2d(block, feature_counts[0], name=f'conv_4x4_1')
        block = conv_2d(block, feature_counts[1], kernel_size=4, padding='valid', name=f'conv_4x4_2')
        return block
    else:
        name_base = f'conv_{resolution}x{resolution}'
        block = incoming_block
        block = conv_2d(block, feature_counts[0], name=f'{name_base}_1')
        block = conv_2d(block, feature_counts[1], strides=2, pre_blur=True, name=f'{name_base}_2')

        shortcut = Downsample()(incoming_block)
        if shortcut.shape[-1] != block.shape[-1]:
            shortcut = conv_2d(
                shortcut,
                block.shape[-1],
                kernel_size=1,
                activation=None,
                name=f'shortcut_{resolution}x{resolution}')
        block = ScaledAdd()([block, shortcut])
        return block


def create_discriminator(input_resolution: int) -> tf.keras.Model:
    validate_resolution(input_resolution)

    image = tf.keras.layers.Input((input_resolution, input_resolution, 3))

    x = make_discriminator_body(image, 4)
    x = tf.keras.layers.Flatten()(x)
    classification = ScaledDense(
            1,
            name='to_classification')(x)
    if classification.dtype != tf.float32:
        classification = tf.keras.layers.Activation(None, dtype=tf.float32)(classification)

    return tf.keras.Model(inputs=image, outputs=classification)
