from dataloader.dataloader import get_dataset_eval
from seq_2_seq_models.builder import get_model
from utils.data import postprocessing

import os
import tensorflow as tf


def Model1(input_file, translator_DT, output_file):
    # Complete Translation Task: w2w sequence model
    Complete_Translation = seq2seq_block(
        DT=translator_DT,
        model_name="Transformer",
        input_file=input_file,
        encoder_lang_model_task='unformated_en_w2w',
        decoder_lang_model_task='formated_fr_w2w',
        output_file=output_file)

    return Complete_Translation


def Model2(input_file, translator_DT, punctuator_DT, output_file):
    # Translation Task: w2w sequence model
    Translation = seq2seq_block(
        DT=translator_DT,
        model_name="Transformer",
        input_file=input_file,
        encoder_lang_model_task='unformated_en_w2w',
        decoder_lang_model_task='unformated_fr_w2w',
        output_file=None)

    # Punctuation Task: c2c sequence model
    Punctuated_Translation = seq2seq_block(
        DT=punctuator_DT,
        model_name="GRU",
        input_file=Translation,
        encoder_lang_model_task='unformated_fr_c2c',
        decoder_lang_model_task='formated_fr_c2c',
        output_file=output_file)

    return Punctuation_Translation


def seq2seq_block(DT, model_name, encoder_lang_model_task,
                  input_file, decoder_lang_model_task, output_file
):

    (
        lang_model_opts,
        seq_model_opts,
        train_opts,
        dataset,
        encoder_v2id
    ) = get_dataset_eval(
        DT, model_name,
        input_file,
        encoder_lang_model_task,
        decoder_lang_model_task
    )

    model = get_model(
        model_name,
        train_opts=train_opts,
        seq_model_opts=seq_model_opts,
        encoder_lang_config=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ],
        decoder_lang_config=lang_model_opts[
            train_opts['decoder_lang_model_task']
        ]
    )

    # Directory where the checkpoints will be loaded from
    checkpoint_dir = os.path.join(
        'seq_2_seq_models',
        train_opts['encoder_lang_model_task'][:-4] + "_2_" +
        train_opts['decoder_lang_model_task'][:-4] + "_" +
        train_opts['encoder_lang_model_task'][-1] + "2" +
        train_opts['decoder_lang_model_task'][-1],
        train_opts['model_name'], DT
    )

    # ckpt = tf.train.Checkpoint(
    #     model=model, optimizer=model.optimizer)

    # ckpt_manager = tf.train.CheckpointManager(
    #     ckpt, checkpoint_dir, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    # if ckpt_manager.latest_checkpoint:
    #     ckpt.restore(ckpt_manager.latest_checkpoint)
    model.predict(dataset.take(1))
    model.load_weights(checkpoint_dir + '/transformer_weights.h5')

    # if a checkpoint exists, restore the latest checkpoint.
    #if ckpt_manager.latest_checkpoint:
    #    ckpt.restore(ckpt_manager.latest_checkpoint)

    predictions = tf.argmax(model.predict(dataset, verbose=1), axis=-1)

    processed_predicitons = postprocessing(
        dec_data=predictions,
        dec_v2id=seq_model_opts['decoder_v2id'],
        Print=True,
        output=output_file,
        tokenize_type=lang_model_opts[
            train_opts['decoder_lang_model_task']
        ]['tokenize_type'],
        fasttext_model=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['fasttext_model'],
        enc_data=input_file,
        threshold=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['threshold'],
        remove_punctuation=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['remove_punctuation'],
        lower=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['lower'],
        CAP=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['CAP'],
        NUM=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['NUM'],
        ALNUM=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['ALNUM'],
        UPPER=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['UPPER'],
        enc_v2id=encoder_v2id
    )

    return processed_predicitons
