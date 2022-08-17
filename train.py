import functools

from checkpointer import Checkpointer
from default_to import default_to
from generator_visualizer_callback import GeneratorVisualizer
import math
from models import create_discriminator, create_generator
from serialize import deserialize_model, serialize, serialize_model
import tensorflow as tf
from tf_image_records import decode_record_image
from typing import List, Optional, Callable, Tuple
from training_loop import training_loop
from update_visualization_generator_callback import UpdateVisualizationGeneratorCallback


def make_real_image_dataset(
        batch_size: int,
        file_pattern: str = 'gs://bk-ffhq/512x512/*.tfrecord',
        randomly_flip: bool = True
        ) -> tf.data.Dataset:
    file_names = tf.io.gfile.glob(file_pattern)

    def apply_flip(x):
        if randomly_flip and tf.random.uniform((), minval=0., maxval=1., dtype=tf.float32) > 0.5:
            x = x[:, ::-1, :]
        return x

    return tf.data.TFRecordDataset(file_names
        ).map(decode_record_image
        ).map(lambda image: tf.image.convert_image_dtype(image, tf.float32, saturate=True)
        ).shuffle(1000
        ).repeat(
        ).map(apply_flip
        ).batch(batch_size
        ).prefetch(tf.data.AUTOTUNE)


def create_visualizer_noise(
        visualization_grid_size: Tuple[int, int],
        latent_size: int) -> tf.Tensor:
    if latent_size == 3:
        lat_domain = tf.linspace(-math.pi / 2., math.pi / 2., visualization_grid_size[0])
        lon_domain = tf.linspace(-math.pi, math.pi, visualization_grid_size[1])
        lons, lats = tf.meshgrid(lon_domain, lat_domain)
        lons, lats = tf.reshape(lons, [-1]), tf.reshape(lats, [-1])
        xs = tf.cos(lats) * tf.cos(lons)
        ys = tf.cos(lats) * tf.sin(lons)
        zs = tf.sin(lats)
        return tf.stack([xs, ys, zs], axis=-1)

    return tf.random.normal((visualization_grid_size[0] * visualization_grid_size[1], latent_size))


class TrainingOptions:
    def __init__(
            self,
            visualization_grid_size: Tuple[int, int],
            resolution: int,
            replica_batch_size: int,
            epoch_sample_count: int = 1024 * 16,
            total_sample_count: int = 1024 * 800,
            learning_rate: float = 0.002,
            real_images_file_pattern: str = 'gs://bk-ffhq/1024x1024/*.tfrecord',
            latent_size = 512,
            randomly_flip_data: bool = True,
            checkpoint_interval: int = 10,
            visualizer_noise: Optional[tf.Tensor] = None,
            visualization_smoothing_sample_count: int = 10000
            ):
        assert epoch_sample_count % replica_batch_size == 0
        assert total_sample_count % epoch_sample_count == 0

        self.visualization_grid_size = visualization_grid_size
        self.resolution = resolution
        self.replica_batch_size = replica_batch_size
        self.epoch_sample_count = epoch_sample_count
        self.total_sample_count = total_sample_count
        self.learning_rate = learning_rate
        self.real_images_file_pattern = real_images_file_pattern
        self.latent_size = latent_size
        self.randomly_flip_data = randomly_flip_data
        self.checkpoint_interval = checkpoint_interval
        self.visualizer_noise = default_to(
            visualizer_noise,
            create_visualizer_noise(visualization_grid_size, latent_size))
        expected_noise_shape = (visualization_grid_size[0] * visualization_grid_size[1], latent_size)
        assert self.visualizer_noise.shape == expected_noise_shape
        self.visualization_smoothing_sample_count = visualization_smoothing_sample_count

    @property
    def epoch_count(self):
        return self.total_sample_count // self.epoch_sample_count


class TrainingState:
    def __init__(
            self,
            options: TrainingOptions,
            generator: Optional[tf.keras.Model] = None,
            visualization_generator: Optional[tf.keras.Model] = None,
            discriminator: Optional[tf.keras.Model] = None,
            epoch_i: int = 0):
        self.options = options
        self.generator = generator
        self.visualization_generator = visualization_generator
        self.discriminator = discriminator
        self.epoch_i = epoch_i

    def training_is_done(self) -> bool:
        return self.epoch_i * self.options.epoch_sample_count >= self.options.total_sample_count

    def __getstate__(self):
        state = self.__dict__.copy()
        state['generator'] = serialize_model(self.generator)
        state['visualization_generator'] = serialize_model(self.visualization_generator)
        state['discriminator'] = serialize_model(self.discriminator)
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        self.generator = deserialize_model(
            self.generator,
            functools.partial(create_generator, self.options.resolution, self.options.latent_size))
        self.visualization_generator = deserialize_model(
            self.visualization_generator,
            functools.partial(create_generator, self.options.resolution, self.options.latent_size))
        self.discriminator = deserialize_model(
            self.discriminator,
            functools.partial(create_discriminator, self.options.resolution))


class CheckpointStateCallback(tf.keras.callbacks.Callback):
    def __init__(
            self,
            state: TrainingState,
            checkpointer: Checkpointer):
        self.state = state
        self.checkpointer = checkpointer
        super().__init__()

    def on_epoch_end(self, epoch_i: int, logs=None) -> None:
        self.state.epoch_i = epoch_i + 1
        if self.state.epoch_i % self.state.options.checkpoint_interval == 0:
            self.checkpointer.save_checkpoint(self.state.epoch_i, serialize(self.state))


def train(
        strategy: tf.distribute.Strategy,
        checkpointer: Checkpointer,
        state: TrainingState,
        on_visualization_callbacks: List[Callable[[int, tf.Tensor], None]] = list()
        ) -> None:
    options = state.options
    def create_visualizer() -> GeneratorVisualizer:
        replica_batch_size = min(options.replica_batch_size, 8)
        visualization_image_count = options.visualization_grid_size[0] * options.visualization_grid_size[1]
        assert visualization_image_count % strategy.num_replicas_in_sync == 0
        max_replica_batch_size = visualization_image_count // strategy.num_replicas_in_sync
        if (replica_batch_size > max_replica_batch_size or
            visualization_image_count % (replica_batch_size * strategy.num_replicas_in_sync) != 0):
            replica_batch_size = max_replica_batch_size

        return GeneratorVisualizer(
            strategy,
            options.visualization_grid_size,
            state.visualization_generator,
            options.visualizer_noise,
            update_interval=1,
            on_image_callbacks=on_visualization_callbacks,
            replica_batch_size=replica_batch_size)

    checkpoint_callback = CheckpointStateCallback(state, checkpointer)

    if state.generator is None:
        global_batch_size = options.replica_batch_size * strategy.num_replicas_in_sync
        with strategy.scope():
            state.generator = create_generator(options.resolution, options.latent_size)
            state.visualization_generator = create_generator(
                options.resolution,
                options.latent_size)
            state.discriminator = create_discriminator(options.resolution)

        @tf.function
        def zero_vars(var):
            for v in var:
                v.assign(tf.zeros_like(v))
        strategy.run(zero_vars, args=(state.visualization_generator.trainable_variables,))

    global_batch_size = options.replica_batch_size * strategy.num_replicas_in_sync

    visualization_weight_decay = (
        0.5 ** (global_batch_size / options.visualization_smoothing_sample_count)
        if options.visualization_smoothing_sample_count > 0 else
        0.0)
    update_visualization_generator_callback = UpdateVisualizationGeneratorCallback(
        strategy,
        state.generator,
        state.visualization_generator,
        visualization_weight_decay)

    visualizer = create_visualizer()


    image_dataset = strategy.experimental_distribute_dataset(
        make_real_image_dataset(
            global_batch_size,
            file_pattern=options.real_images_file_pattern,
            randomly_flip=options.randomly_flip_data))

    state.epoch_i = training_loop(
        strategy,
        state.generator,
        state.discriminator,
        image_dataset,
        state.epoch_i,
        options.epoch_count,
        options.replica_batch_size,
        options.epoch_sample_count,
        learning_rate=options.learning_rate,
        callbacks=[update_visualization_generator_callback, visualizer, checkpoint_callback])
    checkpointer.save_checkpoint(state.epoch_i, serialize(state))
