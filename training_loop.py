import tensorflow as tf
from typing import Dict, List


def training_loop(
        strategy: tf.distribute.Strategy,
        generator: tf.keras.Model,
        discriminator: tf.keras.Model,
        real_image_dataset: tf.distribute.DistributedDataset,
        epoch_i: int,
        end_epoch_i: int,
        replica_batch_size: int,
        epoch_sample_count: int,
        learning_rate: float = 0.002,
        beta_1: float = 0.0,
        beta_2: float = 0.99,
        d_regularization_interval: int = 16,
        callbacks: List[tf.keras.callbacks.Callback] = []
        ) -> int:
    global_batch_size = replica_batch_size * strategy.num_replicas_in_sync
    assert epoch_sample_count % global_batch_size == 0
    epoch_batch_count = epoch_sample_count // global_batch_size

    noise_size = generator.inputs[0].shape[-1]

    d_stat_names = ['d_loss', 'd_real', 'd_fake']
    g_stat_names = ['g_loss']

    progbar_callback = tf.keras.callbacks.ProgbarLogger(count_mode='steps')
    progbar_callback.target = epoch_batch_count
    callback_list = tf.keras.callbacks.CallbackList(
        callbacks=[progbar_callback] + callbacks,
        model=generator)

    with strategy.scope():
        generator.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2)

        lazy_ratio = d_regularization_interval / (d_regularization_interval + 1)
        discriminator.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate*lazy_ratio,
            beta_1=beta_1**lazy_ratio,
            beta_2=beta_2**lazy_ratio)

    def reduce_across_batch(x: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(x) / global_batch_size

    @tf.function
    def take_g_step() -> Dict[str, tf.Tensor]:
        noise = tf.random.normal(shape=(replica_batch_size, noise_size))
        fake_images = generator(noise, training=True)
        fake_classifications = discriminator(fake_images, training=False)
        loss = reduce_across_batch(tf.nn.softplus(-fake_classifications))

        grads = tf.gradients(loss, generator.trainable_variables)

        assert len(generator.trainable_variables) == len(grads)
        generator.optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        return {'g_loss': loss}

    @tf.function
    def take_d_classification_step(real_images) -> Dict[str, tf.Tensor]:
        noise = tf.random.normal(shape=(replica_batch_size, noise_size))
        fake_images = generator(noise, training=False)
        real_classifications = discriminator(real_images, training=True)
        fake_classifications = discriminator(fake_images, training=True)

        real_loss = reduce_across_batch(tf.nn.softplus(-real_classifications))
        fake_loss = reduce_across_batch(tf.nn.softplus(fake_classifications))
        d_loss = real_loss + fake_loss
        d_grads = tf.gradients(d_loss, discriminator.trainable_variables)
        assert len(d_grads) == len(discriminator.trainable_variables)
        discriminator.optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

        stats = [d_loss, real_loss, fake_loss]
        assert len(stats) == len(d_stat_names)
        stat_dict = dict(zip(d_stat_names, stats))

        return stat_dict

    @tf.function
    def take_d_reg_step(real_images) -> Dict[str, tf.Tensor]:
        real_classifications = discriminator(real_images, training=True)
        real_grads = tf.gradients(tf.reduce_sum(real_classifications), real_images)
        gradient_loss = reduce_across_batch(tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3]))
        gradient_penalty_strength = 10. * 0.5 * d_regularization_interval
        gradient_penalty = gradient_loss * gradient_penalty_strength
        reg_grads = tf.gradients(gradient_penalty, discriminator.trainable_variables)
        assert len(reg_grads) == len(discriminator.trainable_variables)

        # The final bias addition has a second derivative of 0 which tf.gradients reports as
        # None. To prevent apply_gradients() from warning about this, we just insert a tensor
        # full of zeros.
        assert reg_grads[-1] is None
        reg_grads[-1] = tf.zeros_like(discriminator.trainable_variables[-1])
        discriminator.optimizer.apply_gradients(zip(reg_grads, discriminator.trainable_variables))

        return {'d_grad_reg': gradient_penalty}

    def tensor_dict_to_numpy(tensor_dict: Dict[str, tf.Tensor]) -> Dict[str, float]:
        return {key: strategy.reduce('sum', value, axis=None).numpy() for key, value in tensor_dict.items()}

    real_image_iter = iter(real_image_dataset)
    callback_list.on_train_begin()
    while epoch_i < end_epoch_i:
        callback_list.on_epoch_begin(epoch_i)
        all_stat_names = g_stat_names + d_stat_names + ['d_grad_reg', 'epoch']
        epoch_stats = {key: 0. for key in all_stat_names}
        epoch_stat_counts = epoch_stats.copy()
        for batch_i in range(0, epoch_batch_count):
            callback_list.on_train_batch_begin(batch_i)

            batch_stats = {}
            d_stats = strategy.run(take_d_classification_step, args=(next(real_image_iter),))
            batch_stats.update(tensor_dict_to_numpy(d_stats))

            global_batch_i = epoch_i * epoch_batch_count + batch_i
            if global_batch_i % d_regularization_interval == 0:
                d_reg_stats = strategy.run(take_d_reg_step, args=(next(real_image_iter),))
                batch_stats.update(tensor_dict_to_numpy(d_reg_stats))

            g_stats = strategy.run(take_g_step)
            batch_stats.update(tensor_dict_to_numpy(g_stats))

            batch_stats['epoch'] = epoch_i

            for name, stat in batch_stats.items():
                epoch_stats[name] = ((epoch_stats[name] * epoch_stat_counts[name] + stat) /
                    (epoch_stat_counts[name] + 1))
                epoch_stat_counts[name] += 1

            callback_list.on_train_batch_end(batch_i, logs=batch_stats)

        callback_list.on_epoch_end(epoch_i, logs=epoch_stats)
        try:
            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            print(f'Memory usage: {mem_info}')
        except:
            pass
        epoch_i += 1
    callback_list.on_train_end()
    return epoch_i
