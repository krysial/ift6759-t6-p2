import numpy as np
import tensorflow as tf


# From: https://www.tensorflow.org/tutorials/text/transformer#positional_encoding
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                 np.arange(d_model)[np.newaxis, :],
                                 d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# From: https://www.tensorflow.org/tutorials/text/transformer#positional_encoding
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


# From: https://www.tensorflow.org/tutorials/text/transformer#masking
def create_padding_mask(seqs):  # seqs (batch_size * seq_len)
    seqs = tf.cast(tf.math.equal(seqs, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seqs[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# From: https://www.tensorflow.org/tutorials/text/transformer#masking
def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


# From: https://www.tensorflow.org/tutorials/text/transformer#point_wise_feed_forward_network
def point_wise_feed_forward_network(atten_dim, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(atten_dim)  # (batch_size, seq_len, d_model)
    ])
