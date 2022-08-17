import math
import tensorflow as tf
from typing import List, Callable, Tuple


class GeneratorVisualizer(tf.keras.callbacks.Callback):
    def __init__(
            self,
            strategy: tf.distribute.Strategy,
            grid_size: Tuple[int, int],
            generator: tf.keras.Model,
            noise: tf.Tensor,
            on_image_callbacks: List[Callable[[int, tf.Tensor], None]] = list(),
            update_interval: int = 1,
            replica_batch_size: int = 8
            ):
        self.generator = generator
        self.grid_size = grid_size
        self.image_count = grid_size[0] * grid_size[1]
        self.strategy = strategy
        self.replica_batch_size = replica_batch_size
        self.noise = noise
        assert self.noise.shape[0] == grid_size[0] * grid_size[1]
        self.update_interval = update_interval
        self.on_image_callbacks = on_image_callbacks

    @tf.function
    def predict(self, samples):
        return self.generator(samples, training=False)

    def predict_in_batches(self) -> tf.Tensor:
        global_batch_size = self.replica_batch_size * self.strategy.num_replicas_in_sync
        sample_count = self.noise.shape[0]
        batch_count = math.ceil(sample_count / global_batch_size)
        padded_sample_count = batch_count * global_batch_size
        padded_input = tf.concat(
            [self.noise,
            tf.zeros((padded_sample_count - sample_count,) + self.noise.shape[1:], dtype=self.noise.dtype)],
            0)

        dataset = self.strategy.experimental_distribute_dataset(
            tf.data.Dataset.from_tensor_slices(padded_input).batch(global_batch_size))

        batch_outputs = []

        for batch_samples in dataset:
            batch_outputs.append(
                self.strategy.gather(
                    self.strategy.run(self.predict, args=(batch_samples,)),
                    0))

        padded_output = tf.concat(batch_outputs, 0)
        return padded_output[:sample_count]

    def generate_image(self) -> tf.Tensor:
        # Note: calling self.generator.predict(self.noise, batch_size=self.batch_size)
        # caused the next iterations of the training loop to generate NaNs with a
        # high probability, especially on larger resolutions. I didn't find an
        # obvious reason for that. And since it's incredibly tangential to the
        # course topic, I went with a simple work-around.
        images = self.predict_in_batches()
        images = tf.image.convert_image_dtype(images, tf.uint8, saturate=True)

        return tf.reshape(
            images,
            tf.concat(
                [self.grid_size, tf.shape(images)[1:]],
                0))

    def on_epoch_end(self, epoch_i: int, logs=None) -> None:
        if epoch_i % self.update_interval != 0:
            return

        image = self.generate_image()

        for callback in self.on_image_callbacks:
          callback(epoch_i, image)
