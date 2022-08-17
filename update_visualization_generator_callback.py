import tensorflow as tf
from tensor_ops import lerp

class UpdateVisualizationGeneratorCallback(tf.keras.callbacks.Callback):
    def __init__(
            self,
            strategy: tf.distribute.Strategy,
            generator: tf.keras.Model,
            visualization_generator: tf.keras.Model,
            visualization_weight_decay: float):
        self.strategy = strategy
        self.generator = generator
        self.visualization_generator = visualization_generator
        self.visualization_weight_decay = visualization_weight_decay

    @tf.function
    def update_weight_mas(self):
        for ma, v in zip(
                self.visualization_generator.trainable_weights,
                self.generator.trainable_weights):
            ma.assign(lerp(ma, v, 1. - self.visualization_weight_decay))

    def on_train_batch_end(self, batch_i: int, logs=None) -> None:
        self.strategy.run(self.update_weight_mas)