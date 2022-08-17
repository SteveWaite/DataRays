from default_to import default_to_fn
import tensorflow as tf

def try_create_tpu_strategy(tpu_name=''):
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu=tpu_name)
        return tf.distribute.TPUStrategy(tpu)
    except ValueError:
        return None


def create_strategy(tpu_name=''):
    return default_to_fn(
        try_create_tpu_strategy(tpu_name=tpu_name),
        tf.distribute.MirroredStrategy)
