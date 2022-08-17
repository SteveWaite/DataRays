import tensorflow as tf

def encode_record_image(image: tf.Tensor):
    image_shape = image.shape.as_list()
    assert image.dtype == tf.uint8
    assert len(image_shape) == 3
    for x in image_shape:
        assert x is not None
        assert type(x) == int
    image_bytes = tf.io.encode_png(image).numpy()
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'image_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image_shape)),
                'image_bytes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
                })
        ).SerializeToString()


def decode_record_image(record_bytes):
    schema = {
        'image_shape': tf.io.FixedLenFeature([3], dtype=tf.int64),
        'image_bytes': tf.io.FixedLenFeature([], dtype=tf.string)
        }
    example = tf.io.parse_single_example(record_bytes, schema)
    image = tf.io.decode_image(example['image_bytes'])
    image = tf.reshape(image, tf.cast(example['image_shape'], tf.int32))
    return image
