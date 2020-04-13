from sklearn.model_selection import train_test_split
from seq_2_seq_models.builder import get_model
from dataloader.dataloader import get_dataset_train
from utils.checkpoint_callback import CheckpointCallback

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
@click.option('--train_split_ratio', default=None, type=float)
@click.option('--steps_per_epoch', default=None, type=int)
@click.option('--model_name', default=None)
@click.option('--lang_model_opts_path', default='config/language_models.json')
@click.option('--seq_model_opts_path', default='config/seq_2_seq_model.json')
@click.option('--train_opts_path', default='config/train_seq_2_seq_config.json')
def train(
    encoder_lang_model_task, decoder_lang_model_task,
    batch_size, epochs, lr, dr, train_split_ratio, steps_per_epoch,
    enc_checkpoint_epoch, dec_checkpoint_epoch, model_name,
    lang_model_opts_path, seq_model_opts_path, train_opts_path,
):
    DT = datetime.datetime.now().strftime("%d-%H-%M-%S")

    with open(lang_model_opts_path, "r") as fd:
        lang_model_opts = json.load(fd)

    with open(train_opts_path, "r") as fd:
        train_opts = json.load(fd)

    if model_name is not None:
        train_opts['model_name'] = model_name

    with open(seq_model_opts_path, "r") as fd:
        seq_model_opts = json.load(fd)
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

    if enc_checkpoint_epoch is not None:
        seq_model_opts['encoder_config']['enc_checkpoint_epoch'] = enc_checkpoint_epoch

    if dec_checkpoint_epoch is not None:
        seq_model_opts['decoder_config']['dec_checkpoint_epoch'] = dec_checkpoint_epoch

    if train_split_ratio is not None:
        train_opts['train_split_ratio'] = train_split_ratio

    if steps_per_epoch is not None:
        train_opts['steps_per_epoch'] = steps_per_epoch

    # Directory where the checkpoints will be saved
    checkpoint_dir = os.path.join(
        os.getcwd(),
        'seq_2_seq_models',
        train_opts['encoder_lang_model_task'][:-4] + "_2_" +
        train_opts['decoder_lang_model_task'][:-4] + "_" +
        train_opts['encoder_lang_model_task'][-1] + "2" +
        train_opts['decoder_lang_model_task'][-1]
    )
    checkpoint_prefix = os.path.join(
        os.getcwd(),
        checkpoint_dir,
        train_opts['model_name'],
        DT
    )

    # Mahmoud
    checkpoint_callback = CheckpointCallback(
        filepath=checkpoint_prefix
    )

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
        callbacks=[checkpoint_callback],
        verbose=1,
        steps_per_epoch=train_opts['steps_per_epoch'],
        shuffle=True,
        validation_data=dataset_valid,
        validation_steps=train_opts['steps_per_epoch']
    )


if __name__ == '__main__':
    train()
