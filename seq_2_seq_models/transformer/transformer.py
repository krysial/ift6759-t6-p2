import tensorflow as tf
from seq_2_seq_models.transformer.encoder import Encoder
from seq_2_seq_models.transformer.decoder import Decoder
from seq_2_seq_models.transformer.utils import create_padding_mask, create_combined_mask


class Transformer(tf.keras.Model):
    def __init__(self, input_seq_len, input_vocab_size, target_seq_len, target_vocab_size, opts):
        super(Transformer, self).__init__()
        self.encoder = Encoder(input_vocab_size, input_seq_len, opts)  # opts: batch_size, atten_dim, num_heads, ff_dim
        self.decoder = Decoder(target_vocab_size, target_seq_len, opts)  # opts: batch_s, atten_dim, num_heads, ff_dim
        self.dense = tf.keras.layers.Dense(target_vocab_size)

    def call(self, input_target, training=True):  # TODO: pass masks
        inputs, targets = input_target
        targets_train = targets[:, :-1]  # shifting targets for training
        input_pad_mask = create_padding_mask(inputs)
        target_look_ahead_mask = create_combined_mask(targets_train)
        enc_outputs = self.encoder(inputs, input_pad_mask, training=training)
        dec_outputs = self.decoder(targets_train, enc_outputs, input_pad_mask, target_look_ahead_mask, training=training)  # input_pad_mask = enc_pad_mask
        dense_outputs = self.dense(dec_outputs)
        return dense_outputs
