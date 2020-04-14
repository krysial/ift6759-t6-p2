import tensorflow as tf
import os
from types import SimpleNamespace

from seq_2_seq_models.transformer.transformer import Transformer
from seq_2_seq_models.seq_2_seq import seq_2_seq_GRU


def loss_function_GRU(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    targets_real = real[:, :]  # shift for targets real
    mask = tf.math.logical_not(tf.math.equal(targets_real, 0))
    loss_ = loss_object(targets_real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def loss_function_Transformer(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    targets_real = real[:, 1:]  # shift for targets real
    mask = tf.math.logical_not(tf.math.equal(targets_real, 0))
    loss_ = loss_object(targets_real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def get_model(model_name, train_opts, seq_model_opts,
              encoder_lang_config, decoder_lang_config):

    if model_name == "Transformer":
        get = get_model_Transformer
    elif model_name == "GRU":
        get = get_model_GRU

    model = get(
        model_name=model_name,
        seq_model_opts=seq_model_opts,
        train_opts=train_opts,
        encoder_lang_config=encoder_lang_config,
        decoder_lang_config=decoder_lang_config,
    )

    if model_name == "Transformer":
        loss_function = loss_function_Transformer
    elif model_name == "GRU":
        loss_function = loss_function_GRU

    optimizer = tf.keras.optimizers.Adam(
        train_opts['lr'], beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    model.compile(optimizer=optimizer,
                  loss=loss_function, run_eagerly=True)
    return model


def get_model_Transformer(model_name, seq_model_opts, train_opts,
                          encoder_lang_config, decoder_lang_config):

    opts = SimpleNamespace(**seq_model_opts)
    opts.teacher_forcing_ratio = train_opts['teacher_forcing_ratio']

    transformer = Transformer(
        max_input_seq_len=encoder_lang_config["max_seq"],
        input_vocab_size=encoder_lang_config["vocab_size"],
        max_target_seq_len=decoder_lang_config["max_seq"],
        target_vocab_size=decoder_lang_config["vocab_size"],
        output_SOS_id=seq_model_opts["decoder_v2id"]["<SOS>"],
        opts=opts
    )

    return transformer


def get_model_GRU(model_name, seq_model_opts, train_opts,
                  encoder_lang_config, decoder_lang_config):

    seq_model_opts['encoder_config']['lang_model_checkpointer'] = os.path.join(
        "language_models", train_opts['encoder_lang_model_task'],
        model_name +
        "_{}.h5".format(seq_model_opts['encoder_config']['checkpoint_epoch'])
    )

    seq_model_opts['decoder_config']['lang_model_checkpointer'] = os.path.join(
        "language_models", train_opts['decoder_lang_model_task'],
        model_name +
        "_{}.h5".format(seq_model_opts['decoder_config']['checkpoint_epoch'])
    )

    GRU_seq_2_seq = seq_2_seq_GRU(
        vocab_inp_size=encoder_lang_config['vocab_size'],
        encoder_embedding_dim=encoder_lang_config['embedding_dim'],
        encoder_units=seq_model_opts['encoder_config']['units'],
        vocab_tar_size=decoder_lang_config['vocab_size'],
        decoder_embedding_dim=decoder_lang_config['embedding_dim'],
        decoder_units=seq_model_opts['decoder_config']['units'],
        decoder_v2id=seq_model_opts['decoder_v2id'],
        targ_seq_len=decoder_lang_config['max_seq'],
        encoder_lang_model=seq_model_opts['encoder_config']['lang_model_checkpointer'],
        decoder_lang_model=seq_model_opts['decoder_config']['lang_model_checkpointer'],
        teacher_forcing_ratio=train_opts['teacher_forcing_ratio']
    )

    return GRU_seq_2_seq
