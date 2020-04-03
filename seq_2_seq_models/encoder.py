import tensorflow as tf


class Encoder_GRU(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units,
                 batch_sz, lang_model=None):
        super(Encoder_GRU, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        if lang_model is not None:
            model = tf.keras.models.load_model(lang_model)

        for i in range(len(model.layers)):
            if model.layers[i].name[:9] == 'embedding':
                self.embedding = model.layers[i]
            if model.layers[i].name[:3] == 'gru':
                self.gru = model.layers[i]

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))
