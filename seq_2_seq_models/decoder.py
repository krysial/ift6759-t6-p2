import tensorflow as tf
from seq_2_seq_models.attention import BahdanauAttention

class Decoder_GRU(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, lang_model=None):
    super(Decoder_GRU, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units


    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)


    if lang_model is not None:
        model = tf.keras.models.load_model(lang_model)

        for i in range(len(model.layers)):
            if model.layers[i].name[:9] == 'embedding':
                self.embedding = model.layers[i]
            if model.layers[i].name[:3] == 'gru':
                self.gru = model.layers[i]
            if model.layers[i].name[:5] == 'dense':
                self.fc = model.layers[i]


    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights