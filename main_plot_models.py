from typing import Callable
from ensure_dir import ensure_dir_for_file
from models import create_discriminator, create_generator
import os
import tensorflow as tf
from limit_gpu_memory_usage import limit_gpu_memory_usage


def plot_models(create_model: Callable[[int], tf.keras.Model], model_name: str):
    for resolution in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        print(resolution)
        model = create_model(resolution)
        file_path = os.path.join('output', 'model_plots', f'{model_name}_{resolution}x{resolution}.png')
        ensure_dir_for_file(file_path)
        tf.keras.utils.plot_model(model, file_path, show_shapes=True, show_dtype=True)


def main():
    limit_gpu_memory_usage(1024)
    plot_models(
        lambda resolution: create_generator(resolution, 512),
        'generator')
    plot_models(
        create_discriminator,
        'discriminator')


if __name__ == '__main__':
    main()
