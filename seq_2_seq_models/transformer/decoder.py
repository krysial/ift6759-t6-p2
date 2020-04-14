import tensorflow as tf

from seq_2_seq_models.transformer.multihead_attention import MultiHeadAttention
from seq_2_seq_models.transformer.utils import point_wise_feed_forward_network, positional_encoding
from utils.seeder import SEED

# From https://www.tensorflow.org/tutorials/text/transformer with small modifications
# Specific sections are referenced in the code


class Decoder(tf.keras.layers.Layer):
    # https://www.tensorflow.org/tutorials/text/transformer#decoder
    def __init__(self, vocab_size, max_target_seq_len, opts, rate=0.1):
        super().__init__()

        SEED(S=123)

        self.num_layers = opts.num_layers
        self.atten_dim = opts.atten_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, opts.atten_dim)  # mask_zero=True
        self.pos_encoding = positional_encoding(max_target_seq_len, opts.atten_dim)
        self.dropout = tf.keras.layers.Dropout(rate)

        self.dec_layers = [DecoderLayer(opts) for _ in range(opts.num_layers)]

    def call(self, targets, enc_output, enc_mask_pad, target_look_ahead_pad, training=True):
        seq_len = tf.shape(targets)[1]
        x = self.embedding(targets)
        x *= tf.math.sqrt(tf.cast(self.atten_dim, tf.float32))  # TODO: check why
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, enc_mask_pad, target_look_ahead_pad, training)

        return x


class DecoderLayer(tf.keras.layers.Layer):
    # https://www.tensorflow.org/tutorials/text/transformer#decoder_layer
    def __init__(self, opts, rate=0.1):
        super().__init__()

        SEED(S=123)
        
        self.masked_multihead_attention = MultiHeadAttention(opts.num_heads, opts.atten_dim)
        self.multihead_attention = MultiHeadAttention(opts.num_heads, opts.atten_dim)
        self.ffnn =  point_wise_feed_forward_network(opts.atten_dim, opts.ff_dim)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, targets, enc_output, enc_mask_pad, target_look_ahead_mask, training=True):
        q, _ = self.masked_multihead_attention(targets, targets, targets, target_look_ahead_mask)
        q = self.dropout1(q, training=training)
        # residual connection + layer normalization
        q = self.layernorm1(q + targets)

        att_output, _ = self.multihead_attention(q, enc_output, enc_output, enc_mask_pad)  # q, k, v
        att_output = self.dropout2(att_output, training=training)
        # residual connection + layer normalization
        att_output = self.layernorm2(att_output + q)

        ff_output = self.ffnn(att_output)
        ff_output = self.dropout3(ff_output, training=training)
        # residual connection + layer normalization
        ff_output = self.layernorm3(ff_output + att_output)

        return ff_output
