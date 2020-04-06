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
def train(encoder_lang_model_task, decoder_lang_model_task, config_path,
          batch_size, epochs, checkpoint_epoch, train_split_ratio,
          steps_per_epoch, model_name):

    with open(config_path, "r") as fd:
        config = json.load(fd)

    BATCH_SIZE = batch_size
    encoder_checkpoint_file = model_name + "_{}.h5".format(checkpoint_epoch)
    decoder_checkpoint_file = model_name + "_{}.h5".format(checkpoint_epoch)

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
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=False)

    '''
    Encoder-Decoder Config Loading:
        max_seq, vocab_size, remove_punctuation, tokenize_type,
        data_file, embedding_dim, units, lang_model_checkpointer
    '''
    encoder_config = config[encoder_lang_model_task]
    decoder_config = config[decoder_lang_model_task]

    encoder_config['tokenize_type'] = list(encoder_lang_model_task)[-1]
    encoder_config['data_file'] = os.path.join(
        "data",
        "aligned_" + encoder_lang_model_task.split("_")[0] + "_" +
        encoder_lang_model_task.split("_")[1]
    )
    encoder_config['lang_model_checkpointer'] = os.path.join(
        "language_models",
        encoder_lang_model_task,
        encoder_checkpoint_file
    )

    decoder_config['tokenize_type'] = list(decoder_lang_model_task)[-1]
    decoder_config['data_file'] = os.path.join(
        "data",
        "aligned_" + decoder_lang_model_task.split("_")[0] +
        "_" + decoder_lang_model_task.split("_")[1]
    )
    decoder_config['lang_model_checkpointer'] = os.path.join(
        "language_models",
        decoder_lang_model_task,
        decoder_checkpoint_file
    )

    # dataset
    encoder_v2id, encoder_dataset = preprocess_v2id(
        data=os.path.join(os.getcwd(), encoder_config['data_file']),
        v2id=os.path.join(
            os.getcwd(),
            "language_models",
            encoder_lang_model_task,
            "v2id.json"
        ),
        tokenize_type=encoder_config['tokenize_type'],
        max_seq=encoder_config['max_seq'],
        remove_punctuation=encoder_config['remove_punctuation']
    )

    decoder_v2id, decoder_dataset = preprocess_v2id(
        data=os.path.join(os.getcwd(), decoder_config['data_file']),
        v2id=os.path.join(
            os.getcwd(),
            "language_models",
            decoder_lang_model_task,
            "v2id.json"
        ),
        tokenize_type=decoder_config['tokenize_type'],
        max_seq=decoder_config['max_seq'],
        remove_punctuation=decoder_config['remove_punctuation']
    )

    (
        input_tensor_train,
        input_tensor_valid,
        target_tensor_train,
        target_tensor_valid
    ) = train_test_split(
        encoder_dataset, decoder_dataset, test_size=train_split_ratio)

    BUFFER_SIZE = len(input_tensor_train)
    if steps_per_epoch is None:
        steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
    vocab_inp_size = len(encoder_v2id)
    vocab_tar_size = len(decoder_v2id)

    encoder_config['max_seq'] = encoder_dataset.shape[-1]
    decoder_config['max_seq'] = decoder_dataset.shape[-1]

    encoder_config['vocab_size'] = vocab_inp_size
    decoder_config['vocab_size'] = vocab_tar_size

    print("#### ENC-DEC DATA Preprocessed ####")
    print("Enc:", encoder_config)
    print("Dec:", decoder_config)

    dataset_train = tf.data.Dataset.from_tensor_slices(
        ((input_tensor_train, target_tensor_train), target_tensor_train)
    ).shuffle(BUFFER_SIZE)
    dataset_train = dataset_train.batch(
        BATCH_SIZE,
        drop_remainder=True).repeat()

    dataset_valid = tf.data.Dataset.from_tensor_slices(
        ((input_tensor_valid, target_tensor_valid), target_tensor_valid)
    ).shuffle(BUFFER_SIZE)
    dataset_valid = dataset_valid.batch(
        BATCH_SIZE, drop_remainder=True
    ).repeat()

    print("#### Datasets Loaded ####")
    print(dataset_train, dataset_valid)

    def get_model():
        seq_2_seq_model = seq_2_seq_GRU(
            vocab_inp_size=vocab_inp_size,
            encoder_embedding_dim=encoder_config['embedding_dim'],
            encoder_units=encoder_config['units'],
            vocab_tar_size=vocab_tar_size,
            decoder_embedding_dim=decoder_config['embedding_dim'],
            decoder_units=decoder_config['units'],
            decoder_v2id=decoder_v2id,
            targ_seq_len=decoder_config['max_seq'],
            BATCH_SIZE=BATCH_SIZE,
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
