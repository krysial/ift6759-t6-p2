from sklearn.model_selection import train_test_split
from utils.data import preprocess_v2id
from seq_2_seq_models.builder import get_model

import tensorflow as tf
import numpy as np
import os
import time
import click
import json
import logging


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logger = tf.get_logger()
logger.setLevel(logging.FATAL)
logging.disable(logging.CRITICAL)


@click.command()
@click.option('--lang_model_opts_path', default='config/language_models.json')
@click.option('--seq_model_opts_path', default='config/seq_2_seq_model.json')
@click.option('--train_opts_path', default='config/train_config.json')
def train(
    lang_model_opts_path, seq_model_opts_path, train_opts_path,
):

    # lang_model_opts[encoder_lang_model_task] = encoder_lang_config = lang_model_opts[train_opts['encoder_lang_model_task']]
    # lang_model_opts[decoder_lang_model_task] = decoder_lang_config = lang_model_opts[train_opts['decoder_lang_model_task']]

    with open(lang_model_opts_path, "r") as fd:
        lang_model_opts = json.load(fd)

    with open(train_opts_path, "r") as fd:
        train_opts = json.load(fd)

    with open(seq_model_opts_path, "r") as fd:
        seq_model_opts = json.load(fd)
        seq_model_opts = seq_model_opts[train_opts['model_name']]

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
        train_opts['model_name'] + "_{epoch}.h5"
    )

    # Mahmoud
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=False)
    ###

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
        lang_model_opts,  # = LOAD CONFIG/LANGUAGE_MODELS.JSON
        dataset_train,
        dataset_valid,
        steps_per_epoch
    ) = get_dataset_train(
        batch_size=train_opts['batch_size'],
        train_split_ratio=train_opts['train_split_ratio'],
        steps_per_epoch=train_opts['steps_per_epoch'],
        model_name=train_opts['model_name'],
        encoder_lang_model_task=train_opts['encoder_lang_model_task'],
        decoder_lang_model_task=train_opts['decoder_lang_model_task'],
        lang_model_opts=lang_model_opts,
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
