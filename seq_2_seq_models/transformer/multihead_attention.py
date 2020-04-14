import tensorflow as tf

from utils.seeder import SEED
# From https://www.tensorflow.org/tutorials/text/transformer with small modifications
# Specific sections are referenced in the code


class MultiHeadAttention(tf.keras.layers.Layer):
    # https://www.tensorflow.org/tutorials/text/transformer#multi-head_attention
    def __init__(self, num_heads, atten_dim):
        super().__init__()

        SEED(S=123)

        self.atten_dim = atten_dim
        self.num_heads = num_heads
        assert atten_dim % num_heads == 0, "Attention layer dimensionality (atten_dim) should be a factor of num_heads"
        self.head_dim = atten_dim // num_heads

        self.wq = tf.keras.layers.Dense(atten_dim)
        self.wk = tf.keras.layers.Dense(atten_dim)
        self.wv = tf.keras.layers.Dense(atten_dim)

        self.dense = tf.keras.layers.Dense(atten_dim)

    def call(self, Q, K, V, pad_mask):
        q = self.wq(Q)
        k = self.wq(K)
        v = self.wq(V)

        batch_size = tf.shape(q)[0]

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        z, atten_weights = self.scaled_dot_product_attention(q, k, v, pad_mask)

        # From:
        scaled_attention = tf.transpose(z, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, head_dim)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.atten_dim))  # (batch_size, seq_len_q, atten_dim)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, atten_dim)
        return output, atten_weights

    def split_heads(self, atten_term, batch_size):  # atten_term: q, k, v -> (batch_size, seq_len, atten_dim)
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        atten_term = tf.reshape(atten_term, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(atten_term, perm=[0, 2, 1, 3])

    # From: https://www.tensorflow.org/tutorials/text/transformer#scaled_dot_product_attention
    def scaled_dot_product_attention(self, q, k, v, mask):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
          q: query shape == (..., seq_len_q, depth)
          k: key shape == (..., seq_len_k, depth)
          v: value shape == (..., seq_len_v, depth_v)
          mask: Float tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
          output, attention_weights
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights
