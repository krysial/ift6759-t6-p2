import tensorflow as tf

import numpy as np
import os
import time
import click
import json
from sklearn.model_selection import train_test_split

from language_models.language_model import build_model, embedding_warmer, embedding_loader, Loss
from utils.data import preprocessing


###########################
#       TPU setup         #
###########################
# %tensorflow_version 2.x
import tensorflow as tf
print("Tensorflow version " + tf.__version__)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
    pass


@click.command()
@click.option('--config_path', default='config/language_models.json')
@click.option('--model_name', default='GRU')
@click.option('--task', default="unformated_fr_w2w")
@click.option('--batch_size', default=64)
@click.option('--train_split_ratio', default=0.30)
@click.option('--epochs', default=20)
@click.option('--units', default=128)
@click.option('--lr', default=0.001)
@click.option('--dr', default=0.1)
@click.option('--embedding_warmer_epoch', default=1)
@click.option('--steps_per_epoch', default=500)
@click.option('--embedding_dim', default=None)
def train(task, config_path, units, lr, dr, model_name, train_split_ratio,
          batch_size, epochs, steps_per_epoch, embedding_warmer_epoch,
          embedding_dim):
    with open(config_path, "r") as fd:
        config = json.load(fd)

    config = config[task]
    config['model_name'] = model_name
    config['task'] = task
    config['batch_size'] = batch_size
    config['epochs'] = epochs
    config['units'] = units
    config['lr'] = lr
    config['dr'] = dr
    config['steps_per_epoch'] = steps_per_epoch
    config['embedding_warmer_epoch'] = embedding_warmer_epoch
    config['train_split_ratio'] = train_split_ratio

    if embedding_dim is not None:
        config['embedding_dim'] = embedding_dim

    if config['fasttext_model'] is not None:
        config['fasttext_model'] = "embeddings/" + task + "/" + \
            str(config["embedding_dim"]) + "/" + config['fasttext_model']

    data_file = "data/unaligned_" + task.split("_")[0] + "_" + \
        task.split("_")[1]
    tokenize_type = list(task)[-1]

    print("data_file:{}, tokenize_type:{}, rm_punc:{}".format(
        data_file,
        tokenize_type,
        config['remove_punctuation']))

    # Directory where the checkpoints will be saved
    checkpoint_dir = 'language_models/' + task
    checkpoint_prefix = os.path.join(
        os.getcwd(),
        checkpoint_dir,
        model_name + "_{epoch}.h5"
    )

    # dataset
    id2v, v2id, train_dataset = preprocessing(
        data=os.path.join(os.getcwd(), data_file),
        tokenize_type=tokenize_type,
        max_seq=config['max_seq'],
        vocab_size=config['vocab_size'],
        remove_punctuation=config['remove_punctuation'],
        lower=config['lower'],
        CAP=config['CAP'],
        NUM=config['NUM'],
        ALNUM=config['ALNUM'],
        UPPER=config['UPPER'],
        save_v2id_path=os.path.join(os.getcwd(), checkpoint_dir,
                                    "v2id.json"),
        fasttext_model=config['fasttext_model'],
        threshold=config['threshold']
    )

    config['vocab_size'] = len(id2v)
    config['max_seq'] = train_dataset.shape[1]

    # Callbacks
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=False)

    embedding_warmer_callback = embedding_warmer(
        start_train_epoch=config['embedding_warmer_epoch'])

    embedding_loader_callback = embedding_loader(
        fasttext_path=config['fasttext_model'], v2id=None, id2v=id2v)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    (
        train_dataset,
        valid_dataset,
    ) = train_test_split(train_dataset, test_size=config['train_split_ratio'])

    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    train_dataset = train_dataset.map(split_input_target)
    train_dataset = train_dataset.shuffle(1000).batch(
        batch_size, drop_remainder=True).repeat()

    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_dataset)
    valid_dataset = valid_dataset.map(split_input_target)
    valid_dataset = valid_dataset.shuffle(1000).batch(
        batch_size, drop_remainder=True).repeat()

    # creating the model in the TPUStrategy scope means we will
    # train the model on the TPU
    with tpu_strategy.scope():
        model = build_model(
            config['vocab_size'],
            config['embedding_dim'],
            config['units'],
            batch_size
        )

        optimizer = tf.keras.optimizers.Adam(config['lr'])
        model.compile(optimizer=optimizer, loss=Loss, run_eagerly=False)

        history = model.fit(
            train_dataset,
            epochs=epochs,
            callbacks=[checkpoint_callback,
                       embedding_warmer_callback, embedding_loader_callback],
            verbose=1,
            steps_per_epoch=steps_per_epoch,
            shuffle=True,
            validation_data=valid_dataset,
            validation_steps=steps_per_epoch
        )


if __name__ == '__main__':
    train()
