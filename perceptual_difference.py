from limit_gpu_memory_usage import limit_gpu_memory_usage
import tensorflow as tf


def normalize_channels(x):
    return x / tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) + 1e-7)


def calculate_feature_pair_difference(a, b):
    a = normalize_channels(a)
    b = normalize_channels(b)

    diffs = tf.linalg.norm(a - b, axis=-1)
    return tf.reduce_mean(diffs, axis=[1, 2])

def calculate_perceptual_difference(feature_pairs):
    difference = None
    for a, b in feature_pairs:
        pair_diff = calculate_feature_pair_difference(a, b)
        difference = pair_diff if difference is None else difference + pair_diff
    difference /= len(feature_pairs)
    return difference


def create_feature_extractor():
    def create_vgg_feature_extractor():
        layer_names = [
            # 'block1_conv2',
            # 'block2_conv2',
            # 'block3_conv3',
            'block4_conv3',
            'block5_conv3',
            ]

        vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layer_names]

        return tf.keras.Model([vgg.input], outputs)

    vgg_features = create_vgg_feature_extractor()
    image = tf.keras.layers.Input((224, 224, 3))

    def preprocess(image):
        return tf.keras.applications.vgg16.preprocess_input(image * 255.)

    preprocessed = tf.keras.layers.Lambda(preprocess)(image)
    features = vgg_features(preprocessed)
    return tf.keras.Model(image, features)


def create_perceptual_difference_model():
    feature_extractor = create_feature_extractor()

    input_shape = feature_extractor.input_shape[1:]

    image_a = tf.keras.layers.Input(input_shape)
    image_b = tf.keras.layers.Input(input_shape)

    features_a = feature_extractor(image_a)
    features_b = feature_extractor(image_b)

    if not isinstance(features_a, list):
        features_a = [features_a]
        features_b = [features_b]

    diff = tf.keras.layers.Lambda(calculate_perceptual_difference)(list(zip(features_a, features_b)))

    return tf.keras.Model([image_a, image_b], diff)
