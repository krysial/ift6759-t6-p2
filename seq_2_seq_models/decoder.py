import tensorflow as tf
from seq_2_seq_models.attention import BahdanauAttention
from language_models.language_model import Loss


class Decoder_GRU(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim,
                 dec_units, lang_model=None):
        super(Decoder_GRU, self).__init__()
        self.dec_units = dec_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.lang_model = lang_model
        if lang_model is not None:
            model = tf.keras.models.load_model(
                lang_model, custom_objects={'loss': Loss}
            )

            for i in range(len(model.layers)):
                if model.layers[i].name[:9] == 'embedding':
                    self.embedding = model.layers[i]
                if model.layers[i].name[:3] == 'gru':
                    self.gru = model.layers[i]
                # if model.layers[i].name[:5] == 'dense':
                    # self.fc = model.layers[i]

        # used for attention
        self.attn_align_shape = 16
        self.attention = BahdanauAttention(self.attn_align_shape)

        self.attention_fc = tf.keras.layers.Dense(self.dec_units)

    # @tf.function(experimental_compile=True)
    def call(self, x, hidden, enc_output, dec_init):
        # enc_output shape == (batch_size, max_length, hidden_size)
        # context_vector, attention_weights
        #   = self.attention(hidden, enc_output)

        # x shape after passing through embedding:
        # (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # passing the embedding vector to the GRU
        # dec_gru_output shape == (batch_size, embedding_dim)

        if self.lang_model is None:
            dec_gru_output, dec_state = self.gru(x, initial_state=dec_init)
        else:
            dec_gru_output = self.gru(x, initial_state=dec_init)
            dec_state = dec_gru_output[:, -1, :]

        # pass dec_hidden output and enc_output to attention
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # shape after concatenation:
        # (batch_size, 1, embedding_dim + hidden_size)
        dec_attn_output = tf.concat(
            [tf.expand_dims(context_vector, 1), dec_gru_output],
            axis=-1
        )

        # dec_output shape == (batch_size * 1, hidden_size + embedding_dim)
        dec_attn_output = tf.reshape(
            dec_attn_output,
            (-1, dec_attn_output.shape[2])
        )

        dec_output = self.attention_fc(dec_attn_output)

        # x shape == (batch_size, vocab)
        x = self.fc(dec_output)

        return x, dec_state, attention_weights
