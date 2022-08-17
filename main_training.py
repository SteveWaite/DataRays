import argparse
from typing import Callable, List, Optional
from create_datetime_str import create_datetime_str
from default_to import default_to
from dummy_strategy import DummyStrategy
import functools
from images_to_gridded_image import images_to_gridded_image, spherically_project_images_to_grid
from limit_gpu_memory_usage import limit_gpu_memory_usage
from checkpointer import Checkpointer
import os
from save_image import save_image
from serialize import deserialize
import tensorflow as tf
from train import TrainingOptions, TrainingState, train

def save_visualization(
        visualization_folder_path: os.PathLike,
        should_spherically_project: bool,
        epoch_i: int,
        images: tf.Tensor,
        visualization_callback: Optional[Callable[[tf.Tensor], None]] = None):
    image_file_name = f'{epoch_i+1}.png'

    unprojected_image = images_to_gridded_image(images)
    unprojected_folder_path = (
        os.path.join(visualization_folder_path, 'unprojected')
        if should_spherically_project else
        visualization_folder_path)
    save_image(os.path.join(unprojected_folder_path, image_file_name), unprojected_image)

    if should_spherically_project:
        projected_image = spherically_project_images_to_grid(images)
        save_image(os.path.join(visualization_folder_path, 'projected', image_file_name), projected_image)

    if visualization_callback:
        visualization_callback(projected_image if should_spherically_project else unprojected_image)


def create_visualization_callback(
        root_output_path: os.PathLike,
        latent_size: int,
        visualization_callback: Optional[Callable[[tf.Tensor], None]]):
    visualization_path = os.path.join(root_output_path, 'visualizations')
    should_spherically_project = latent_size == 3
    return functools.partial(
        save_visualization,
        visualization_path,
        should_spherically_project,
        visualization_callback=visualization_callback)


def create_checkpointer(output_root: os.PathLike):
    checkpointer_path = os.path.join(output_root, 'checkpoints', '{checkpoint_i}.checkpoint')
    return Checkpointer(checkpointer_path)


def init_training(
        args: argparse.Namespace,
        visualization_callback: Optional[Callable[[tf.Tensor], None]],
        strategy: Optional[tf.distribute.Strategy]):
    output_root = args.out
    if args.create_unique_id:
        output_root = os.path.join(output_root, create_datetime_str())

    strategy = default_to(strategy, DummyStrategy())

    checkpointer = create_checkpointer(output_root)
    training_options = TrainingOptions(
        tuple(args.visualization_grid_size),
        args.resolution,
        args.replica_batch_size,
        epoch_sample_count=args.epoch_sample_count,
        total_sample_count=args.total_sample_count,
        real_images_file_pattern=args.dataset_file_pattern,
        latent_size=args.latent_size,
        checkpoint_interval=args.checkpoint_interval,
        visualization_smoothing_sample_count=args.visualization_smoothing_sample_count,
        randomly_flip_data=(not args.disable_horizontal_flip_data_augmentation))
    training_state = TrainingState(training_options)

    visualization_callback = create_visualization_callback(
        output_root,
        training_state.options.latent_size,
        visualization_callback)

    train(
        strategy,
        checkpointer,
        training_state,
        on_visualization_callbacks=[visualization_callback])

def resume_training(
        args: argparse.Namespace,
        visualization_callback: Optional[Callable[[tf.Tensor], None]],
        strategy: Optional[tf.distribute.Strategy]):
    strategy = default_to(strategy, DummyStrategy())
    checkpointer = create_checkpointer(args.out)
    checkpoint_i = default_to(args.checkpoint, max(checkpointer.list_checkpoints()))

    print(f'Resuming training from checkpoint {checkpoint_i}')

    with strategy.scope():
        training_state = deserialize(checkpointer.load_checkpoint(checkpoint_i))

    visualization_callback = create_visualization_callback(
        args.out,
        training_state.options.latent_size,
        visualization_callback)

    train(
        strategy,
        checkpointer,
        training_state,
        on_visualization_callbacks=[visualization_callback])


def main(
        raw_arguments: Optional[List[str]] = None,
        visualization_callback: Optional[Callable[[tf.Tensor], None]] = None,
        strategy: Optional[tf.distribute.Strategy] = None):
    parser = argparse.ArgumentParser(
        description='Train a GAN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@')
    subparsers = parser.add_subparsers()

    init_parser = subparsers.add_parser(
        'init',
        help='initialize and begin training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    init_parser.set_defaults(func=init_training)

    init_parser.add_argument(
        '--create-unique-id',
        action='store_true',
        help='if true, a unique id will be created from the date/time and appended to the output path')

    init_parser.add_argument(
        '--visualization-grid-size',
        type=int,
        nargs=2,
        help='height and width of the visualization grid. Ex: ... --visualization_grid_size 30 60 ...',
        default=[4, 8])

    init_parser.add_argument(
        '--resolution',
        type=int,
        help='resolution of generated images. Must match the resolution of the dataset',
        default=512)

    init_parser.add_argument(
        '--replica-batch-size',
        type=int,
        help='size of batch per replica',
        default=8)

    init_parser.add_argument(
        '--epoch-sample-count',
        type=int,
        help='number of samples per epoch. Must divide evenly into total-sample-count',
        default=16*1024)

    init_parser.add_argument(
        '--total-sample-count',
        type=int,
        help='number of total samples to train on. Must be divisible by epoch-sample-count',
        default=25*1024*1024)

    init_parser.add_argument(
        '--latent-size',
        type=int,
        help='size of the generator\'s latent vector',
        default=512)

    init_parser.add_argument(
        '--checkpoint-interval',
        type=int,
        help='interval of epochs to save a checkpoint',
        default=10)

    init_parser.add_argument(
        '--visualization-smoothing-sample-count',
        type=float,
        help='the factor by which to decay the visualization weights. If 0, no smoothing will be applied.',
        default=10000)

    init_parser.add_argument(
        '--disable-horizontal-flip-data-augmentation',
        action='store_true',
        help='including this option will disable horizontal flip data augmentation. Use when ' +
            'horizontally flipping an image changes its semantics, ex., mnist digits.')

    init_parser.add_argument(
        'dataset_file_pattern',
        help='GLOB pattern for the dataset files. Ex: \'D:/datasets/ffhq/1024x1024/*.tfrecord\'')

    resume_parser = subparsers.add_parser('resume', help='resume training from a checkpoint')
    resume_parser.set_defaults(func=resume_training)

    for subparser in [init_parser, resume_parser]:
        subparser.add_argument(
            '--gpu-mem-limit',
            help='maximum amount of memory to consume on the gpu in GB. 0 for unlimited',
            type=int,
            default=0)
        subparser.add_argument('out', help='root output folder')

    resume_parser.add_argument(
        '--checkpoint',
        help='checkpoint epoch to resume from. Defaults to the largest checkpointed epoch',
        type=int)

    args = parser.parse_args(args=raw_arguments)

    if args.gpu_mem_limit != 0:
        limit_gpu_memory_usage(args.gpu_mem_limit * 1024)

    args.func(args, visualization_callback, strategy)


if __name__ == '__main__':
    main()


