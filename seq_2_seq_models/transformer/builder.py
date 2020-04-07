import tensorflow as tf
from seq_2_seq_models.transformer.transformer import Transformer


def get_model(opts, input_seq_len, input_vocab_size, target_seq_len, target_vocab_size):
    optimizer = tf.keras.optimizers.Adam(opts.lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real, pred):
        targets_real = real[:, 1:]  # shift for targets real
        mask = tf.math.logical_not(tf.math.equal(targets_real, 0))
        loss_ = loss_object(targets_real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    transformer = Transformer(input_seq_len, input_vocab_size, target_seq_len - 1, target_vocab_size, opts) # -1 bc targets will be shifted by 1

    transformer.compile(optimizer=optimizer, loss=loss_function, run_eagerly=True)
    return transformer
