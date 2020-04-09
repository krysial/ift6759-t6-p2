from sklearn.model_selection import train_test_split
from utils.data import preprocess_v2id
from seq_2_seq_models.seq_2_seq import seq_2_seq_GRU, Loss

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
@click.option('--config_path', default='config/language_models.json')
@click.option('--batch_size', default=64)
@click.option('--epochs', default=20)
@click.option('--checkpoint_epoch', default=30)
@click.option('--train_split_ratio', default=0.2)
@click.option('--steps_per_epoch', default=None)
@click.option('--model_name', default='GRU')
@click.option('--fasttext_model', default=None)
def train(encoder_lang_model_task, decoder_lang_model_task, config_path,
          batch_size, epochs, checkpoint_epoch, train_split_ratio,
          steps_per_epoch, model_name, fasttext_model):

    with open(config_path, "r") as fd:
        config = json.load(fd)

    BATCH_SIZE = batch_size

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
        model_name + "_{epoch}.h5"
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
    encoder_lang_config = config[encoder_lang_model_task]
    decoder_lang_config = config[decoder_lang_model_task]
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
        encoder_lang_config,
        decoder_lang_config,
        dataset_train,
        dataset_valid
    ) = get_dataset_train(encoder_lang_config,
                          decoder_lang_config,
                          BATCH_SIZE,
                          train_split_ratio,
                          steps_per_epoch,
                          model_name,
                          fasttext_model,
                          encoder_lang_model_task,
                          decoder_lang_model_task,
                          )

    def get_model():

        # encoder_config
        # decoder_config

        encoder_checkpoint_file = model_name + \
            "_{}.h5".format(checkpoint_epoch)
        encoder_config['lang_model_checkpointer'] = os.path.join(
            "language_models",
            encoder_lang_model_task,
            encoder_checkpoint_file
        )

        decoder_checkpoint_file = model_name + \
            "_{}.h5".format(checkpoint_epoch)
        decoder_config['lang_model_checkpointer'] = os.path.join(
            "language_models",
            decoder_lang_model_task,
            decoder_checkpoint_file
        )

        seq_2_seq_model = seq_2_seq_GRU(
            vocab_inp_size=vocab_inp_size,
            encoder_embedding_dim=encoder_lang_config['embedding_dim'],
            # encoder_units=encoder_config['units'],
            vocab_tar_size=vocab_tar_size,
            decoder_embedding_dim=decoder_lang_config['embedding_dim'],
            # decoder_units=decoder_config['units'],
            decoder_v2id=decoder_v2id,
            targ_seq_len=decoder_lang_config['max_seq'],
            encoder_lang_model=encoder_config['lang_model_checkpointer'],
            decoder_lang_model=decoder_config['lang_model_checkpointer']
        )

        optimizer = tf.keras.optimizers.Adam()

        seq_2_seq_model.compile(
            optimizer=optimizer, loss=Loss, run_eagerly=True
        )

        return seq_2_seq_model

    seq_2_seq_model = get_model()

    print("#### Model Loaded ####")

    history = seq_2_seq_model.fit(
        dataset_train,
        epochs=epochs,
        callbacks=[checkpoint_callback],
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        shuffle=True,
        validation_data=dataset_valid,
        validation_steps=steps_per_epoch
    )


if __name__ == '__main__':
    train()
