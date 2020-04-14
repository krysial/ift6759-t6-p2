import tensorflow as tf
import datetime

from utils.gensim_embeddings import load_and_create
from utils.data import swap_dict_key_value
from seq_2_seq_models.encoder import Encoder_GRU
from seq_2_seq_models.decoder import Decoder_GRU


class checkpointer(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super(checkpointer, self).__init__()
        self.filepath = filepath

    def on_train_begin(self, logs=None):
        self.ckpt = tf.train.Checkpoint(
            model=self.model, optimizer=self.model.optimizer
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt,
            self.filepath,
            max_to_keep=5
        )

    def on_epoch_end(self, epoch, logs=None):
        self.ckpt_manager.save()


class embedding_loader(tf.keras.callbacks.Callback):
    def __init__(self, enc_fasttext_path=None, dec_fasttext_path=None,
                 enc_v2id=None, enc_id2v=None, dec_v2id=None, dec_id2v=None):
        super(embedding_loader, self).__init__()
        self.enc_fasttext_path = enc_fasttext_path
        self.dec_fasttext_path = dec_fasttext_path

        if enc_fasttext_path is not None:
            self.enc_id2v = self.get_id2v(enc_id2v, enc_v2id)
            self.enc_emb_matrix = load_and_create(
                enc_fasttext_path, self.enc_id2v)
        if dec_fasttext_path is not None:
            self.dec_id2v = self.get_id2v(dec_id2v, dec_v2id)
            self.dec_emb_matrix = load_and_create(
                dec_fasttext_path, self.dec_id2v)

    def get_id2v(self, id2v=None, v2id=None):
        if id2v is None:
            return swap_dict_key_value(v2id)
        else:
            return id2v

    def on_train_begin(self, epoch, logs=None):
        if self.enc_fasttext_path is not None:
            self.model.encoder.embedding.build((None,))
            self.model.encoder.embedding.set_weights([self.enc_emb_matrix])
            self.model.encoder.embedding.trainable = False
        if self.dec_fasttext_path is not None:
            self.model.decoder.embedding.build((None,))
            self.model.decoder.embedding.set_weights([self.dec_emb_matrix])
            self.model.decoder.embedding.trainable = False


class embedding_warmer(tf.keras.callbacks.Callback):
    def __init__(self, enc_embd_start_train_epoch=1, dec_embd_start_train_epoch=1):
        super(embedding_warmer, self).__init__()
        self.enc_embd_start_train_epoch = enc_embd_start_train_epoch
        self.dec_embd_start_train_epoch = dec_embd_start_train_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.enc_embd_start_train_epoch:
            self.model.encoder.embedding.trainable = True
        else:
            self.model.encoder.embedding.trainable = False
        if epoch >= self.dec_embd_start_train_epoch:
            self.model.decoder.embedding.trainable = True
        else:
            self.model.decoder.embedding.trainable = False


class GRU_attn_warmer(tf.keras.callbacks.Callback):
    def __init__(self, enc_gru_start_train_epoch=1, dec_gru_start_train_epoch=1):
        super(GRU_attn_warmer, self).__init__()
        self.enc_gru_start_train_epoch = enc_gru_start_train_epoch
        self.dec_gru_start_train_epoch = dec_gru_start_train_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.enc_gru_start_train_epoch:
            self.model.encoder.embedding.trainable = True
            self.model.encoder.gru.trainable = True
        else:
            self.model.encoder.embedding.trainable = False
            self.model.encoder.gru.trainable = False
        if epoch >= self.dec_gru_start_train_epoch:
            self.model.decoder.embedding.trainable = True
            self.model.decoder.gru.trainable = True
        else:
            self.model.decoder.embedding.trainable = False
            self.model.decoder.gru.trainable = False


class seq_2_seq_GRU(tf.keras.Model):
    def __init__(self, vocab_inp_size, encoder_embedding_dim, encoder_units,
                 vocab_tar_size, decoder_embedding_dim, decoder_units,
                 decoder_v2id, targ_seq_len,
                 encoder_lang_model=None, decoder_lang_model=None):

        super(seq_2_seq_GRU, self).__init__()

        self.encoder = Encoder_GRU(vocab_size=vocab_inp_size,
                                   embedding_dim=encoder_embedding_dim,
                                   enc_units=encoder_units,
                                   lang_model=encoder_lang_model)
        self.decoder = Decoder_GRU(vocab_size=vocab_tar_size,
                                   embedding_dim=decoder_embedding_dim,
                                   dec_units=decoder_units,
                                   lang_model=decoder_lang_model)
        self.v2id = decoder_v2id
        self.targ_seq_len = targ_seq_len

    # @tf.function(experimental_compile=True)
    def call(self, tup, targ=None, training=False):

        inp, targ = tup

        BATCH_SIZE = inp.shape[0]
        enc_hidden = self.encoder.initialize_hidden_state(BATCH_SIZE)

        enc_output, enc_hidden = self.encoder(inp, enc_hidden)
        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([self.v2id['<SOS>']] * BATCH_SIZE, 1)

        Predictions = []
        Predictions.append(
            tf.one_hot(
                [self.v2id['<SOS>']] * BATCH_SIZE, len(self.v2id),
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
