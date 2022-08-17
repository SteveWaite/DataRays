import bk_io
import os
import tensorflow as tf

def save_image(path: os.PathLike, image: tf.Tensor) -> None:
    image = tf.convert_to_tensor(image)
    if image.dtype != tf.uint8:
        image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
    bk_io.write_binary_file(path, tf.io.encode_png(image).numpy())
