import argparse
from create_datetime_str import create_datetime_str
from default_to import default_to
from limit_gpu_memory_usage import limit_gpu_memory_usage
from checkpointer import Checkpointer
import math
import os
from perceptual_difference import create_perceptual_difference_model
from save_image import save_image
from serialize import deserialize
import tensorflow as tf
from tensor_ops import pixel_norm, lerp
from train import TrainingState
from typing import List, Optional


def load_generator(checkpoint_folder_path: os.PathLike, checkpoint_i: Optional[int] = None):
    checkpointer = Checkpointer(
        os.path.join(checkpoint_folder_path, '{checkpoint_i}.checkpoint'))
    checkpoint_i = default_to(checkpoint_i, max(checkpointer.list_checkpoints()))
    training_state: TrainingState = deserialize(checkpointer.load_checkpoint(checkpoint_i))
    return training_state.visualization_generator


def calc_magnitude(v):
    return tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1, keepdims=True))


# Normalize batch of vectors.
def normalize(v, magnitude=1.0):
    return v * magnitude / tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1, keepdims=True))


# Spherical interpolation of a batch of vectors.
def slerp(a, b, t):
    norm_mag = lerp(calc_magnitude(a), calc_magnitude(b), t)
    a = normalize(a)
    b = normalize(b)
    d = tf.reduce_sum(a * b, axis=-1, keepdims=True)
    p = t * tf.math.acos(d)
    c = normalize(b - d * a)
    d = a * tf.math.cos(p) + c * tf.math.sin(p)
    return normalize(d, norm_mag)


@tf.function
def sample_path_lengths_from(
        generator: tf.keras.Model,
        perceptual_differencer: tf.keras.Model,
        origin_noise: tf.Tensor,
        origin_images: tf.Tensor) -> tf.Tensor:
    def resize_and_crop(images: tf.Tensor) -> tf.Tensor:
        differ_image_size = perceptual_differencer.input_shape[0][1:3]
        # Resize the images to be slightly larger than the differ's input size then crop the center
        # to reduce the amount of background in the image and focus on the face.
        images = tf.image.resize(
            images,
            (256 * differ_image_size[0] // 224, 256 * differ_image_size[1] // 224))
        images = tf.image.resize_with_crop_or_pad(images, differ_image_size[0], differ_image_size[1])
        return images

    target_noises = slerp(origin_noise, tf.random.normal(origin_noise.shape), 0.1)
    target_images = generator(target_noises)
    origin_images, target_images = map(resize_and_crop, (origin_images, target_images))
    return perceptual_differencer([origin_images, target_images])


def calculate_perceptual_path_length(
        generator: tf.keras.Model,
        perceptual_differencer: tf.keras.Model,
        noise: tf.Tensor,
        images: tf.Tensor,
        sample_count: int = 10):
    diffs = [
        sample_path_lengths_from(generator, perceptual_differencer, noise, images)
        for _ in range(sample_count)]
    avg_diff = tf.reduce_max(tf.square(tf.stack(diffs, axis=-1)), axis=-1)
    return avg_diff


def generate_random_images(
        generator: tf.keras.Model,
        args: argparse.Namespace):
    perceptual_differencer = create_perceptual_difference_model()
    batch_size = 4
    noise_shape = (batch_size, generator.input_shape[-1])

    prog_bar = tf.keras.utils.Progbar(args.sample_count)
    generated_sample_count = 0
    while generated_sample_count < args.sample_count:
        noise = pixel_norm(tf.random.normal(noise_shape))
        images = generator(noise)
        ppls = (
            calculate_perceptual_path_length(generator, perceptual_differencer, noise, images)
            if args.threshold > 0 else
            tf.zeros((batch_size,)))

        for (image, score) in zip(images, ppls):
            if args.threshold > 0 and score > args.threshold:
                continue
            file_name = os.path.join(args.out, f'{create_datetime_str()}_{score}.png')
            save_image(file_name, image)
            generated_sample_count += 1
            prog_bar.add(1)
            if generated_sample_count == args.sample_count:
                break


def generate_interpolation(
        generator: tf.keras.Model,
        start_noise: tf.Tensor,
        end_noise: tf.Tensor) -> List[tf.Tensor]:
    def generate_image(factor):
        noise = slerp(start_noise, end_noise, factor)
        return generator(noise)[0]
    return list(map(generate_image, tf.linspace(0.0, 1.0, 100)))


def generate_random_interpolations(
        generator: tf.keras.Model,
        args: argparse.Namespace):
    def generate_random_interpolation():
        noise_shape = (1, generator.input_shape[-1])
        start_noise = pixel_norm(tf.random.normal(noise_shape))
        end_noise = pixel_norm(tf.random.normal(noise_shape))
        return generate_interpolation(generator, start_noise, end_noise)

    prog_bar = tf.keras.utils.Progbar(args.sample_count)
    for _ in range(args.sample_count):
        interpolation_folder = os.path.join(args.out, create_datetime_str())
        images = generate_random_interpolation()
        for image_i, image in enumerate(images):
            file_path = interpolation_folder / f'{image_i}.png'
            save_image(file_path, image)
        prog_bar.add(1)


def generate_circular_interpolation(
        generator: tf.keras.Model,
        noise_origin: tf.Tensor,
        noise_pass_through_point: tf.Tensor,
        step_count: int = 500) -> List[tf.Tensor]:
    assert noise_origin.shape[0] == 1
    assert noise_pass_through_point.shape[0] == 1

    dot_product = tf.reduce_sum(noise_origin * noise_pass_through_point, axis=-1)[0]
    angular_distance = tf.math.acos(dot_product)
    assert tf.abs(angular_distance) > 1e-7

    full_revolution_factor = 2. * math.pi / angular_distance

    images = []
    for factor in tf.linspace(0.0, full_revolution_factor, step_count):
        noise = slerp(noise_origin, noise_pass_through_point, factor)
        images.append(generator(noise)[0])
    return images


def generate_random_circular_interpolations(
        generator: tf.keras.Model,
        args: argparse.Namespace):
    def generate_random_circular_interpolation():
        noise_shape = (1, generator.input_shape[-1])
        noise_origin = normalize(tf.random.normal(noise_shape))
        noise_pass_through_point = normalize(tf.random.normal(noise_shape))
        return generate_circular_interpolation(generator, noise_origin, noise_pass_through_point)

    prog_bar = tf.keras.utils.Progbar(args.sample_count)
    for _ in range(args.sample_count):
        interpolation_folder = os.path.join(args.out, create_datetime_str())
        images = generate_random_circular_interpolation()
        for image_i, image in enumerate(images):
            file_path = interpolation_folder / f'{image_i}.png'
            save_image(file_path, image)
        prog_bar.add(1)


def main():
    limit_gpu_memory_usage(4*1024)

    parser = argparse.ArgumentParser(
        description='Sample images from a trained generator',
        fromfile_prefix_chars='@')

    subparsers = parser.add_subparsers()

    images_parser = subparsers.add_parser('images', help='Sample random images from the generator')
    images_parser.add_argument(
        '--threshold',
        type=float,
        default=0.3,
        help='Threshold for max perceptual path length filter. 0 for unlimited.')
    images_parser.set_defaults(action=generate_random_images)

    interpolation_parser = subparsers.add_parser(
        'interpolations',
        help='Sample interpolations from randomly chosen start and end points in the the latent ' +
            'space')
    interpolation_parser.set_defaults(action=generate_random_interpolations)

    circular_parser = subparsers.add_parser(
        'circular-interpolations',
        help='Sample interpolations from a randomly chosen start point in the latent space that ' +
            'travel in a random direction in a circle around the latent space back to the start ' +
            'point.')
    circular_parser.set_defaults(action=generate_random_circular_interpolations)

    for subparser in [images_parser, interpolation_parser, circular_parser]:
        subparser.add_argument('--sample_count', help='number of samples to generator', type=int, default=1)
        subparser.add_argument('--checkpoint', help='checkpoint to load. Defaults highest checkpoint number.', type=int)
        subparser.add_argument('checkpoint_folder', help='folder containing checkpoints')
        subparser.add_argument('out', help='root output folder path')

    args = parser.parse_args()

    generator = load_generator(args.checkpoint_folder, args.checkpoint)
    args.action(generator, args)


if __name__ == '__main__':
    main()
