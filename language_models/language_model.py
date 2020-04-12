import tensorflow as tf
import numpy as np
import os
import time

from utils.gensim_embeddings import load_and_create
from utils.data import swap_dict_key_value


def Loss(targets_real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(targets_real, 0))
    loss_ = loss_object(targets_real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


class embedding_warmer(tf.keras.callbacks.Callback):
    def __init__(self, start_train_epoch=1):
        super(embedding_warmer, self).__init__()
        self.start_train_epoch = start_train_epoch

    def on_epoch_begin(self, epoch, logs=None):
        for i in range(len(self.model.layers)):
            if self.model.layers[i].name[:9] == 'embedding':
                if epoch >= self.start_train_epoch:
                    self.model.layers[i].trainable = True
                else:
                    self.model.layers[i].trainable = False
                break


class embedding_loader(tf.keras.callbacks.Callback):
    def __init__(self, fasttext_path=None, v2id=None, id2v=None):
        super(embedding_loader, self).__init__()
        self.fasttext_path = fasttext_path
        if fasttext_path is not None:
            if id2v is None:
                self.id2v = swap_dict_key_value(v2id)
            else:
                self.id2v = id2v
            self.emb_matrix = load_and_create(fasttext_path, self.id2v)

    def on_train_begin(self, epoch, logs=None):
        if self.fasttext_path is not None:
            for i in range(len(self.model.layers)):
                if self.model.layers[i].name[:9] == 'embedding':
                    self.model.layers[i].build((None,))
                    self.model.layers[i].set_weights([self.emb_matrix])
                    self.model.layers[i].trainable = False
                    break


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
