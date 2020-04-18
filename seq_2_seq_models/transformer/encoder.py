import tensorflow as tf
import numpy as np


from seq_2_seq_models.transformer.utils import positional_encoding, point_wise_feed_forward_network
from seq_2_seq_models.transformer.multihead_attention import MultiHeadAttention


class Encoder(tf.keras.layers.Layer):
    # https://www.tensorflow.org/tutorials/text/transformer#encoder
    def __init__(self, vocab_size, max_input_seq_len, opts):
        super().__init__()

        self.num_layers = opts.num_layers
        self.atten_dim = opts.atten_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, opts.atten_dim)  # mask_zero=True
        self.pos_encoding = positional_encoding(max_input_seq_len, opts.atten_dim)
        self.dropout = tf.keras.layers.Dropout(opts.embed_dr)

        self.enc_layers = [EncoderLayer(opts) for _ in range(opts.num_layers)]

    def call(self, inputs, input_pad_mask, training=True):  # (batch_size, input_seq_len, d_model)
        seq_len = tf.shape(inputs)[1]
        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.atten_dim, tf.float32))  # TODO: check why
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, input_pad_mask, training=training)

        return x


class EncoderLayer(tf.keras.layers.Layer):
    # https://www.tensorflow.org/tutorials/text/transformer#encoder_layer
    def __init__(self, opts):  # num_heads, atten_dim, ff_dim
        super().__init__()
        self.multihead_attention = MultiHeadAttention(opts.num_heads, opts.atten_dim)
        self.ffn = point_wise_feed_forward_network(opts.atten_dim, opts.ff_dim)

        self.dropout1 = tf.keras.layers.Dropout(opts.ff_dr)
        self.dropout2 = tf.keras.layers.Dropout(opts.ff_dr)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, input_pad_mask, training=True):
        att_output, atten_weights = self.multihead_attention(inputs, inputs, inputs, input_pad_mask)  # (batch_size, input_seq_len, d_model)
        att_output = self.dropout1(att_output, training=training)
        # residual connection + layer normalization
        att_output = self.layernorm1(att_output + inputs)

        ff_output = self.ffn(att_output)
        ff_output = self.dropout2(ff_output)
        # residual connection + layer normalization
        x = self.layernorm2(ff_output + att_output)  # (batch_size, input_seq_len, d_model)

        return x
