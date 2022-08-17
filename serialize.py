import io
from typing import Callable
import numpy as np
import pickle
import tensorflow as tf


def serialize(unserialized: object) -> bytes:
    return pickle.dumps(unserialized)


def deserialize(serialized: bytes):
    return pickle.loads(serialized)


'''
    I'm using np.save() over tf.io.serialize_tensor because parse_tensor
    (https://www.tensorflow.org/api_docs/python/tf/io/parse_tensor) requires the caller to know
    the correct dtype whereas np.save() does not.

    Adapted from:
    https://stackoverflow.com/questions/30698004/how-can-i-serialize-a-numpy-array-while-preserving-matrix-dimensions
'''
def serialize_array(arr: np.ndarray) -> bytes:
    mem_file = io.BytesIO()

    np.save(mem_file, arr)

    mem_file.seek(0)
    return mem_file.read()


def deserialize_array(serialized: bytes) -> np.ndarray:
    mem_file = io.BytesIO()

    mem_file.write(serialized)

    mem_file.seek(0)
    return np.load(mem_file)


def serialize_tensor(t: tf.Tensor) -> bytes:
    return serialize_array(t.numpy())


def serialize_variable(v: tf.Variable) -> bytes:
    return serialize_tensor(v)


def deserialize_tensor(serialized: bytes) -> tf.Tensor:
    return tf.constant(deserialize_array(serialized))


def deserialize_variable(serialized: bytes) -> tf.Variable:
    return tf.Variable(deserialize_array(serialized))


'''
    Annoyingly, keras does not provide a way to serialize a model directly to a bytes object. The
    save()/load_model() API requires a local filesystem, which is commonly not available cloud TPUs.
    The to_json() API does not store weights and stores intermediate dtype information so that a
    model trained on bfloat16 cannot be loaded on a machine that doesn't support bfloat16. For now,
    we'll just stored the weights and assume the deserializer knows how to create the model
    architecture.
'''
def serialize_model(model: tf.keras.Model) -> bytes:
    return serialize({
        'model_weights': list(map(serialize_array, model.get_weights()))
    })


def deserialize_model(serialized: bytes, create_model: Callable[[], tf.keras.Model]) -> tf.keras.Model:
    serializable = deserialize(serialized)
    model = create_model()
    model.set_weights(list(map(deserialize_array, serializable['model_weights'])))
    return model
