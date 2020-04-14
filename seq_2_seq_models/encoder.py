import tensorflow as tf

from language_models.language_model import Loss


class Encoder_GRU(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units,
                 lang_model=None):
        super(Encoder_GRU, self).__init__()
        self.enc_units = enc_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

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

    # @tf.function(experimental_compile=True)
    def call(self, x, hidden):
        x = self.embedding(x)
        if self.lang_model is None:
            output, state = self.gru(x, initial_state=hidden)
        else:
            output = self.gru(x, initial_state=hidden)
            state = output[:, -1, :]
        return output, state

    def initialize_hidden_state(self, bs):
        return tf.zeros((bs, self.enc_units))
