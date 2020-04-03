import tensorflow as tf

from seq_2_seq_models.encoder import Encoder_GRU
from seq_2_seq_models.decoder import Decoder_GRU

class seq_2_seq_GRU(tf.keras.Model):
    def __init__(self, vocab_inp_size, encoder_embedding_dim, encoder_units,
                vocab_tar_size, decoder_embedding_dim, decoder_units, v2id,
                targ_seq_len, BATCH_SIZE):
        super(seq_2_seq_GRU, self).__init__()
        self.encoder = Encoder_GRU(vocab_inp_size, encoder_embedding_dim, encoder_units, BATCH_SIZE)
        self.decoder = Decoder_GRU(vocab_tar_size, decoder_embedding_dim, decoder_units, BATCH_SIZE)
        self.v2id = v2id
        self.targ_seq_len = targ_seq_len

    def __call__(self, inp, targ=None, training=False):

        enc_hidden = encoder.initialize_hidden_state()

        enc_output, enc_hidden = self.encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        
        dec_input = tf.expand_dims([self.v2id['<SOS>']] * BATCH_SIZE, 1)

        Predictions = []
        # Feeding the target as the next input
        for t in range(1, targ_seq_len):
            # passing enc_output to the decoder
            prediction, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)

            Prediction = tf.argmax(prediction, 1)
            Predictions.append(Prediction)

            if training:
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
            else:
                # passing prev prediction
                dec_input = tf.expand_dims(Prediction, 1)

        return Predictions
