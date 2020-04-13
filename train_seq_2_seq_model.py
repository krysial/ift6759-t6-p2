from sklearn.model_selection import train_test_split
from seq_2_seq_models.builder import get_model
from dataloader.dataloader import get_dataset_train
from seq_2_seq_models.seq_2_seq import checkpointer, embedding_loader, embedding_warmer, GRU_attn_warmer
from utils.data import load_json

import tensorflow as tf
import numpy as np
import os
import time
import click
import json
import logging
import datetime


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logger = tf.get_logger()
logger.setLevel(logging.FATAL)
logging.disable(logging.CRITICAL)


@click.command()
@click.option('--encoder_lang_model_task', default=None)
@click.option('--decoder_lang_model_task', default=None)
@click.option('--batch_size', default=None, type=int)
@click.option('--epochs', default=None, type=int)
@click.option('--lr', default=None, type=float)
@click.option('--dr', default=None, type=float)
@click.option('--enc_checkpoint_epoch', default=None, type=int)
@click.option('--dec_checkpoint_epoch', default=None, type=int)
@click.option('--enc_embd_start_train_epoch', default=None, type=int)
@click.option('--dec_embd_start_train_epoch', default=None, type=int)
@click.option('--enc_gru_start_train_epoch', default=None, type=int)
@click.option('--dec_gru_start_train_epoch', default=None, type=int)
@click.option('--train_split_ratio', default=None, type=float)
@click.option('--steps_per_epoch', default=None, type=int)
@click.option('--model_name', default=None)
@click.option('--load_embedding', is_flag=True)
@click.option('--lang_model_opts_path', default='config/language_models.json')
@click.option('--seq_model_opts_path', default='config/seq_2_seq_model.json')
@click.option('--train_opts_path', default='config/train_seq_2_seq_config.json')
def train(
    encoder_lang_model_task, decoder_lang_model_task,
    batch_size, epochs, lr, dr, train_split_ratio,
    enc_checkpoint_epoch, dec_checkpoint_epoch, model_name,
    enc_embd_start_train_epoch, dec_embd_start_train_epoch,
    load_embedding, enc_gru_start_train_epoch,
    dec_gru_start_train_epoch, steps_per_epoch,
    lang_model_opts_path, seq_model_opts_path, train_opts_path,
):
    DT = datetime.datetime.now().strftime("%d-%H-%M-%S")

    lang_model_opts = load_json(lang_model_opts_path)
    train_opts = load_json(train_opts_path)
    seq_model_opts = load_json(seq_model_opts_path)

    if model_name is not None:
        train_opts['model_name'] = model_name
    seq_model_opts = seq_model_opts[train_opts['model_name']]

    if encoder_lang_model_task is not None:
        train_opts['encoder_lang_model_task'] = encoder_lang_model_task

    if decoder_lang_model_task is not None:
        train_opts['decoder_lang_model_task'] = decoder_lang_model_task

    if batch_size is not None:
        train_opts['batch_size'] = batch_size

    if epochs is not None:
        train_opts['epochs'] = epochs

    if lr is not None:
        train_opts['lr'] = lr

    if dr is not None:
        train_opts['dr'] = dr

    if load_embedding:
        train_opts['load_embedding'] = load_embedding

    if enc_checkpoint_epoch is not None:
        seq_model_opts['encoder_config']['enc_checkpoint_epoch'] = enc_checkpoint_epoch

    if dec_checkpoint_epoch is not None:
        seq_model_opts['decoder_config']['dec_checkpoint_epoch'] = dec_checkpoint_epoch

    if enc_embd_start_train_epoch is not None:
        seq_model_opts['encoder_config']['enc_embd_start_train_epoch'] = enc_embd_start_train_epoch

    if dec_embd_start_train_epoch is not None:
        seq_model_opts['decoder_config']['dec_embd_start_train_epoch'] = dec_embd_start_train_epoch

    if enc_embd_start_train_epoch is not None:
        seq_model_opts['encoder_config']['enc_gru_start_train_epoch'] = enc_gru_start_train_epoch

    if dec_embd_start_train_epoch is not None:
        seq_model_opts['decoder_config']['dec_gru_start_train_epoch'] = dec_gru_start_train_epoch

    if train_split_ratio is not None:
        train_opts['train_split_ratio'] = train_split_ratio

    if steps_per_epoch is not None:
        train_opts['steps_per_epoch'] = steps_per_epoch

    # Directory where the checkpoints will be saved
    root_dir = os.path.join(
        'seq_2_seq_models',
        train_opts['encoder_lang_model_task'][:-4] + "_2_" +
        train_opts['decoder_lang_model_task'][:-4] + "_" +
        train_opts['encoder_lang_model_task'][-1] + "2" +
        train_opts['decoder_lang_model_task'][-1],
        train_opts['model_name']
    )
    checkpoint_dir = os.path.join(root_dir, DT)

    ##########
    # ENCODER-DECODER LANG CONFIG
    ##########
    '''
    Encoder-Decoder Config Loading:
        max_seq, vocab_size, remove_punctuation, tokenize_type,
        data_file, embedding_dim, units, lang_model_checkpointer
    '''
    lang_model_opts[train_opts['encoder_lang_model_task']
                    ]['tokenize_type'] = list(train_opts['encoder_lang_model_task'])[-1]
    lang_model_opts[train_opts['decoder_lang_model_task']
                    ]['tokenize_type'] = list(train_opts['decoder_lang_model_task'])[-1]

    lang_model_opts[train_opts['encoder_lang_model_task']]['data_file'] = os.path.join(
        "data",
        "aligned_" + train_opts['encoder_lang_model_task'].split("_")[0] + "_" +
        train_opts['encoder_lang_model_task'].split("_")[1]
    )
    lang_model_opts[train_opts['decoder_lang_model_task']]['data_file'] = os.path.join(
        "data",
        "aligned_" + train_opts['decoder_lang_model_task'].split("_")[0] +
        "_" + train_opts['decoder_lang_model_task'].split("_")[1]
    )

    ###########

    (
        lang_model_opts,
        train_opts,
        seq_model_opts,
        dataset_train,
        dataset_valid,
    ) = get_dataset_train(
        model_name=train_opts['model_name'],
        encoder_lang_model_task=train_opts['encoder_lang_model_task'],
        decoder_lang_model_task=train_opts['decoder_lang_model_task'],
        lang_model_opts=lang_model_opts,
        train_opts=train_opts,
        seq_model_opts=seq_model_opts,
        DT=DT
    )

    # Callbacks
    callbacks = []

    checkpoint_callback = checkpointer(filepath=checkpoint_dir)
    callbacks.append(checkpoint_callback)

    csv_logger_callback = tf.keras.callbacks.CSVLogger(
        os.path.join(checkpoint_dir, DT + '.log'))
    callbacks.append(csv_logger_callback)

    tb_path = os.path.join(root_dir, tensorboard, DT)
    os.makedirs(os.path.dirname(tb_path), exist_ok=True)
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tb_path,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        embeddings_freq=2,
    )
    callbacks.append(tb_callback)

    if train_opts['load_embedding'] and \
            lang_model_opts[encoder_lang_model_task]['fasttext_model'] is not None and \
            lang_model_opts[decoder_lang_model_task]['fasttext_model'] is not None:
        embedding_loader_callback = embedding_loader(
            enc_fasttext_path=lang_model_opts[encoder_lang_model_task]['fasttext_model'],
            dec_fasttext_path=lang_model_opts[decoder_lang_model_task]['fasttext_model'],
            enc_v2id=seq_model_opts['encoder_v2id'],
            dec_v2id=seq_model_opts['decoder_v2id'])
        callbacks.append(embedding_loader_callback)

    embedding_warmer_callback = embedding_warmer(
        enc_embd_start_train_epoch=seq_model_opts['encoder_config']['enc_embd_start_train_epoch'],
        dec_embd_start_train_epoch=seq_model_opts['decoder_config']['dec_embd_start_train_epoch'])
    callbacks.append(embedding_warmer_callback)

    if train_opts['model_name'] == "GRU":
        GRU_attn_warmer_callback = GRU_attn_warmer(
            enc_gru_start_train_epoch=seq_model_opts['encoder_config']['enc_gru_start_train_epoch'],
            dec_gru_start_train_epoch=seq_model_opts['decoder_config']['dec_gru_start_train_epoch'])
        callbacks.append(GRU_attn_warmer_callback)

    model = get_model(
        model_name=train_opts['model_name'],
        train_opts=train_opts,
        seq_model_opts=seq_model_opts,
        encoder_lang_config=lang_model_opts[train_opts['encoder_lang_model_task']],
        decoder_lang_config=lang_model_opts[train_opts['decoder_lang_model_task']],
    )

    print("#### Model Loaded ####")

    history = model.fit(
        dataset_train,
        epochs=train_opts['epochs'],
        callbacks=callbacks,
        verbose=1,
        steps_per_epoch=train_opts['steps_per_epoch'],
        shuffle=True,
        validation_data=dataset_valid,
        validation_steps=train_opts['steps_per_epoch']
    )


if __name__ == '__main__':
    train()
