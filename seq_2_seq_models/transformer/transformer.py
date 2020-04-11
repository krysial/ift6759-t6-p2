import tensorflow as tf
from seq_2_seq_models.transformer.encoder import Encoder
from seq_2_seq_models.transformer.decoder import Decoder
from seq_2_seq_models.transformer.utils import create_padding_mask, create_combined_mask


class Transformer(tf.keras.Model):
    def __init__(self, max_input_seq_len, input_vocab_size, max_target_seq_len, target_vocab_size, output_SOS_id, opts):
        super(Transformer, self).__init__()
        self.output_SOS_id = output_SOS_id
        self.max_target_seq_len = max_target_seq_len
        self.encoder = Encoder(input_vocab_size, max_input_seq_len, opts)  # opts: batch_size, atten_dim, num_heads, ff_dim
        self.decoder = Decoder(target_vocab_size, max_target_seq_len, opts)  # opts: batch_s, atten_dim, num_heads, ff_dim
        self.dense = tf.keras.layers.Dense(target_vocab_size)

    def call(self, input_target, training=True):  # TODO: pass masks
        inputs, targets = input_target
        targets_train = targets[:, :-1]  # shifting targets for training

        input_pad_mask = create_padding_mask(inputs)
        enc_outputs = self.encoder(inputs, input_pad_mask, training=training)

        if training:
            target_look_ahead_mask = create_combined_mask(targets_train)
            dec_outputs = self.decoder(targets_train, enc_outputs, input_pad_mask, target_look_ahead_mask,
                                       training=training)  # input_pad_mask = enc_pad_mask
            dense_outputs = self.dense(dec_outputs)
            outputs = dense_outputs
        else:
            targets_train = tf.fill((tf.shape(targets_train)[0], 1), self.output_SOS_id)  # output_SOS_id
            for seq_step in range(1, self.max_target_seq_len):
                target_look_ahead_mask = create_combined_mask(targets_train)
                dec_outputs = self.decoder(targets_train, enc_outputs, input_pad_mask, target_look_ahead_mask,
                                           training=training)  # input_pad_mask = enc_pad_mask
                predictions = self.dense(dec_outputs)
                outputs = predictions
                predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
                predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
                targets_train = tf.concat([targets_train, predicted_id], axis=-1)

        return outputs
