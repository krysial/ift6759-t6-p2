from utils.data import preprocess_v2id

import tensorflow as tf
import numpy as np
import os
import time
import click
import json

from sklearn.model_selection import train_test_split


# Config File for Language Model
config_path = 'config/language_models.json'
with open(config_path, "r") as fd:
    config = json.load(fd)

model = 'GRU'
BATCH_SIZE = 64
encoder_lang_model_task = 'unformated_en_w2w'
decoder_lang_model_task = 'unformated_fr_w2w'
encoder_checkpoint_file = model + "_5.h5"
decoder_checkpoint_file = model + "_5.h5"

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
    "aligned_" +
    encoder_lang_model_task.split("_")[0] +
    "_" +
    encoder_lang_model_task.split("_")[1]
)
encoder_config['lang_model_checkpointer'] = \
    os.path.join(
        "language_models",
        encoder_lang_model_task,
        encoder_checkpoint_file
    )

decoder_config['tokenize_type'] = list(decoder_lang_model_task)[-1]
decoder_config['data_file'] = os.path.join(
    "data",
    "aligned_" +
    decoder_lang_model_task.split("_")[0] +
    "_" +
    decoder_lang_model_task.split("_")[1]
)
decoder_config['lang_model_checkpointer'] = os.path.join(
    "language_models",
    decoder_lang_model_task,
    decoder_checkpoint_file
)

# dataset
encoder_v2id, encoder_dataset = preprocessing(
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

decoder_v2id, decoder_dataset = preprocessing(
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

input_tensor_train,
input_tensor_val,
target_tensor_train,
target_tensor_val = train_test_split(
    encoder_dataset,
    decoder_dataset,
    test_size=0.8
)

BUFFER_SIZE = len(input_tensor_train)
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
vocab_inp_size = len(encoder_v2id) + 1
vocab_tar_size = len(decoder_v2id) + 1

dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train)
).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

encoder = Encoder(
    vocab_inp_size,
    encoder_config['embedding_dim'],
    encoder_config['units'],
    BATCH_SIZE
)
decoder = Decoder(
    vocab_tar_size,
    decoder_config['embedding_dim'],
    decoder_config['units'],
    BATCH_SIZE
)
