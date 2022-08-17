import argparse
from default_to import default_to
from ensure_dir import ensure_dir_for_file
from limit_gpu_memory_usage import limit_gpu_memory_usage
import os
from save_image import save_image
import tensorflow as tf
import tensorflow_datasets as tfds
from tf_image_records import decode_record_image, encode_record_image
from typing import Optional, Tuple

DEFAULT_RECORD_COUNT = 8

def slice_dataset(dataset: tf.data.Dataset, start_index: int, increment: int):
    def is_in_increment_group(index, value):
        return index % increment == start_index

    def unenumerate(index, value):
        return value

    return dataset.enumerate(
        ).filter(is_in_increment_group
        ).map(unenumerate)


def resize_image(image: tf.Tensor, size: Tuple[int, int]) -> tf.Tensor:
    original_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32, saturate=True)
    # Computer Color is Broken - minutephysics
    # https://youtu.be/LKnqECcg6Gw
    image = tf.image.adjust_gamma(image, 2.2) # Gamma decode the image assuming a gamma of 2.2
    image = tf.image.resize(image, size)
    image = tf.image.adjust_gamma(image, 1./2.2) # Re-gamma-encode the image
    image = tf.image.convert_image_dtype(image, original_dtype, saturate=True)
    return image


def create_record(dataset: tf.data.Dataset, output_file_name: os.PathLike):
    ensure_dir_for_file(output_file_name)
    with tf.io.TFRecordWriter(str(output_file_name)) as file_writer:
        for image in dataset:
            file_writer.write(encode_record_image(image))


def create_records_from_dataset(
        dataset: tf.data.Dataset,
        prepare,
        output_resolution: Optional[int],
        output_dir: os.PathLike,
        record_count: int):
    prog_bar = tf.keras.utils.Progbar(record_count)
    for record_i in range(record_count):
        record_dataset = slice_dataset(dataset, record_i, record_count)
        if prepare is not None:
            record_dataset = record_dataset.map(prepare)

        if output_resolution is not None:
            record_dataset = record_dataset.map(
                lambda image: resize_image(image, (output_resolution, output_resolution)))

        record_dataset = record_dataset.prefetch(tf.data.AUTOTUNE)

        record_file_path = os.path.join(output_dir, f'{record_i}.tfrecord')
        create_record(record_dataset, record_file_path)
        prog_bar.add(1)


def dataset_from_image_files(input_file_pattern: str):
    def prepare(file_name: os.PathLike):
        return tf.io.decode_image(tf.io.read_file(file_name))
    input_file_names = tf.io.gfile.glob(input_file_pattern)
    dataset = tf.data.Dataset.from_tensor_slices(input_file_names)
    return dataset, prepare


def dataset_from_tfrecords(record_file_pattern: str):
    record_file_names = tf.io.gfile.glob(record_file_pattern)
    dataset = tf.data.TFRecordDataset(record_file_names)
    return dataset, decode_record_image


def extract_record(record_name, output_dir, group_i: int = 0, group_count: int = 1):
    dataset = tf.data.TFRecordDataset([record_name]
        ).map(decode_record_image
        ).prefetch(tf.data.AUTOTUNE)
    for i, image in enumerate(dataset):
        file_name = os.path.join(output_dir, f'{i*group_count+group_i}.png')
        save_image(file_name, image)


def load_mnist_data() -> tf.data.Dataset:
    def prepare(sample):
        image = sample['image']
        image = tf.broadcast_to(image, image.shape[:2] + (3,))
        return image
    return tfds.load('mnist', split='train').map(prepare)


def create_records_from_args(args: argparse.Namespace):
    def gather_args():
        if args.source.lower() == 'mnist':
            print('Creating dataset from MNIST')
            return (
                load_mnist_data(),
                None,
                default_to(args.resolution, 32),
                default_to(args.record_count, DEFAULT_RECORD_COUNT))

        if args.source.lower().endswith('tfrecord'):
            print(f'Creating dataset from records: {args.source}')
            dataset, prepare = dataset_from_tfrecords(args.source)
            return (
                dataset,
                prepare,
                args.resolution,
                default_to(args.record_count, len(tf.io.gfile.glob(args.source))))

        print(f'Creating dataset from images: {args.source}')
        dataset, prepare = dataset_from_image_files(args.source)
        return (
            dataset,
            prepare,
            args.resolution,
            default_to(args.record_count, DEFAULT_RECORD_COUNT))

    source_dataset, prepare, resolution, record_count = gather_args()
    create_records_from_dataset(
        source_dataset,
        prepare,
        resolution,
        args.output_folder_name,
        record_count)


def extract_records(args: argparse.Namespace):
    print(f'Extracting records from {args.source}')
    record_paths = tf.io.gfile.glob(args.source)
    record_count = len(record_paths)

    prog_bar = tf.keras.utils.Progbar(record_count)
    for record_i, record_path in enumerate(record_paths):
        extract_record(
            record_path,
            args.output_folder_name,
            group_i=record_i,
            group_count=record_count)
        prog_bar.add(1)


def main():
    limit_gpu_memory_usage()

    parser = argparse.ArgumentParser(
        description='Create, resize, and extract tfrecords',
        fromfile_prefix_chars='@')
    subparsers = parser.add_subparsers()

    create_parser = subparsers.add_parser(
        'create',
        help='create a set of tf records')
    create_parser.set_defaults(action=create_records_from_args)

    create_parser.add_argument(
        '--resolution',
        help='resolution to resize the images to. By default the images will remain their ' +
            'pre-existing size with the exception of "mnist", which will be resized to 32 by '
            'default.',
        type=int)

    create_parser.add_argument(
        '--record-count',
        help='number of tf records to split the data into. If the source is a set of tfrecords, ' +
            'this will default to the number of records in the source. Otherwise, it will ' +
            f'default to {DEFAULT_RECORD_COUNT}.',
        type=int)

    create_parser.add_argument(
        'source',
        help='a file glob pattern for images or records (*.tfrecord) to pack into the ' +
            'destination records. You can pass the special value of "mnist" to download ' +
            'the mnist dataset and pack it into the records.')

    create_parser.add_argument(
        'output_folder_name',
        help='output directory to write the tf records')

    extract_parser = subparsers.add_parser(
        'extract',
        help='extract the images from a set of tf records',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    extract_parser.set_defaults(action=extract_records)

    extract_parser.add_argument(
        'source',
        help='a file glob pattern for the tf records to be extracted')

    extract_parser.add_argument(
        'output_folder_name',
        help='output directory to write the images')

    args = parser.parse_args()
    args.action(args)


if __name__ == '__main__':
    main()