import tensorflow as tf

import numpy as np
import os
import time


def Loss(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    targets_real = real[:, 1:]  # shift for targets real
    mask = tf.math.logical_not(tf.math.equal(targets_real, 0))
    loss_ = loss_object(targets_real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)/tf.reduce_sum(mask)


class embedding_warmer(tf.keras.callbacks.Callback):
    def __init__(self, start_train_epoch=1):
        super(warmer_callback, self).__init__()
        self.start_train_epoch = start_train_epoch

    def on_epoch_begin(self, epoch, logs=None):
        for i in range(len(self.model.layers)):
            if self.model.layers[i].name[:9] == 'embedding':
                if epoch >= self.start_train_epoch:
                    self.model.layers[i].trainable = True
                else:
                    self.model.layers[i].trainable = False


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=False,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])

    return model
