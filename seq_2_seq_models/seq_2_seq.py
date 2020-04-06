import tensorflow as tf

from seq_2_seq_models.encoder import Encoder_GRU
from seq_2_seq_models.decoder import Decoder_GRU


def Loss(labels, logits):
    return tf.keras.losses.SparseCategoricalCrossentropy(
        labels, logits, from_logits=True, reduction='none'
    )


class seq_2_seq_GRU(tf.keras.Model):
    def __init__(self, vocab_inp_size, encoder_embedding_dim, encoder_units,
                 vocab_tar_size, decoder_embedding_dim, decoder_units,
                 decoder_v2id, targ_seq_len, BATCH_SIZE,
                 encoder_lang_model=None, decoder_lang_model=None):

        super(seq_2_seq_GRU, self).__init__()

        self.encoder = Encoder_GRU(vocab_size=vocab_inp_size,
                                   embedding_dim=encoder_embedding_dim,
                                   enc_units=encoder_units,
                                   batch_sz=BATCH_SIZE,
                                   lang_model=encoder_lang_model)
        self.decoder = Decoder_GRU(vocab_size=vocab_tar_size,
                                   embedding_dim=decoder_embedding_dim,
                                   dec_units=decoder_units,
                                   batch_sz=BATCH_SIZE,
                                   lang_model=decoder_lang_model)
        self.v2id = decoder_v2id
        self.targ_seq_len = targ_seq_len
        self.BATCH_SIZE = BATCH_SIZE

    # @tf.function(experimental_compile=True)
    def call(self, tup, targ=None, training=False):

        inp, targ = tup

        enc_hidden = self.encoder.initialize_hidden_state(self.BATCH_SIZE)

        enc_output, enc_hidden = self.encoder(inp, enc_hidden)
        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([self.v2id['<SOS>']] * self.BATCH_SIZE, 1)

        Predictions = []
        Predictions.append(
            tf.one_hot(
                [self.v2id['<SOS>']] * self.BATCH_SIZE, len(self.v2id),
                dtype=tf.float32
            )
        )

        # Feeding the target as the next input
        for t in range(1, self.targ_seq_len):
            # passing enc_output to the decoder

            if t == 1:
                prediction, dec_hidden, _ = self.decoder(
                    dec_input, dec_hidden, enc_output, None
                )

            else:
                prediction, dec_hidden, _ = self.decoder(
                    dec_input, dec_hidden, enc_output, dec_hidden
                )

            Prediction = tf.argmax(prediction, -1)
            Predictions.append(prediction)

            if training:
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
            else:
                # passing prev prediction
                dec_input = tf.expand_dims(Prediction, 1)

        # timestep, batch_size, vocab_size -> batch_size, timestep, vocab_size
        return tf.transpose(
            tf.convert_to_tensor(Predictions, dtype=tf.float32),
            perm=[1, 0, 2]
        )
