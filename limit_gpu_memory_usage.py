import os
import tensorflow as tf

def for_each_physical_gpu(fn):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                fn(gpu)
        except RuntimeError as e:
            print(e)


def limit_gpu_memory_usage(memory_limit_mb=1024*2):
    def limit_gpu(gpu):
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mb)])
    for_each_physical_gpu(limit_gpu)
