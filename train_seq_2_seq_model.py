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
@click.option('--encoder_lang_model_task', default='unformated_en_w2w')
@click.option('--decoder_lang_model_task', default='unformated_fr_w2w')
@click.option('--lang_model_config_path', default='config/language_models.json')
@click.option('--seq_model_config_path', default='config/seq_2_seq_model.json')
@click.option('--train_config_path', default='config/train_config.json')
def train(encoder_lang_model_task, decoder_lang_model_task,
          lang_model_config_path, seq_model_config_path, train_config_path,
          ):

    with open(lang_model_config_path, "r") as fd:
        lang_model_config = json.load(fd)

    with open(train_config_path, "r") as fd:
        train_config = json.load(fd)

    with open(seq_model_config_path, "r") as fd:
        seq_model_config[train_config['model_name']] = json.load(fd)

    # Directory where the checkpoints will be saved
    checkpoint_dir = os.path.join(
        os.getcwd(),
        'seq_2_seq_models',
        encoder_lang_model_task[:-4] + "_2_" +
        decoder_lang_model_task[:-4] + "_" +
        encoder_lang_model_task[-1] + "2" +
        decoder_lang_model_task[-1]
    )
    checkpoint_prefix = os.path.join(
        os.getcwd(),
        checkpoint_dir,
        train_config['model_name'] + "_{epoch}.h5"
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
    encoder_lang_config = lang_model_config[encoder_lang_model_task]
    decoder_lang_config = lang_model_config[decoder_lang_model_task]
    encoder_lang_config['tokenize_type'] = list(encoder_lang_model_task)[-1]
    decoder_lang_config['tokenize_type'] = list(decoder_lang_model_task)[-1]

    encoder_lang_config['data_file'] = os.path.join(
        "data",
        "aligned_" + encoder_lang_model_task.split("_")[0] + "_" +
        encoder_lang_model_task.split("_")[1]
    )
    decoder_lang_config['data_file'] = os.path.join(
        "data",
        "aligned_" + decoder_lang_model_task.split("_")[0] +
        "_" + decoder_lang_model_task.split("_")[1]
    )

    ###########

    (
        train_opts,  # = LOAD CONFIG/CONFIG.JSON
        model_opts,  # = LOAD CONFIG/SEQ_2_SEQ_MODEL.JSON
        dataset_train,
        dataset_valid
    ) = get_dataset_train(encoder_lang_config=encoder_lang_config,
                          decoder_lang_config=decoder_lang_config,
                          batch_size=train_config['batch_size'],
                          train_split_ratio=train_config['train_split_ratio'],
                          steps_per_epoch=train_config['steps_per_epoch'],
                          model_name=train_config['model_name'],
                          encoder_lang_model_task=encoder_lang_model_task,
                          decoder_lang_model_task=decoder_lang_model_task,
                          )

    model = get_model(
        model_name=train_config['model_name'],
        train_opts=train_opts,
        model_opts=model_opts,
        encoder_lang_config=encoder_lang_config,
        decoder_lang_config=decoder_lang_config,
    )

    print("#### Model Loaded ####")

    history = model.fit(
        dataset_train,
        epochs=train_config['epochs'],
        callbacks=[checkpoint_callback],
        verbose=1,
        steps_per_epoch=train_config['steps_per_epoch'],
        shuffle=True,
        validation_data=dataset_valid,
        validation_steps=train_config['steps_per_epoch']
    )


if __name__ == '__main__':
    train()
