import tensorflow as tf

import numpy as np
import os
import time
import click
import json

from language_models.language_model import build_model
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
@click.option('--task', default='unformated_fr_w2w')
@click.option('--config_path', default='config/language_models.json')
@click.option('--batch_size', default=64)
@click.option('--epochs', default=20)
@click.option('--steps_per_epoch', default=500)
@click.option('--model_name', default='GRU')
def train(task, config_path, batch_size, epochs, steps_per_epoch, model_name):
    with open(config_path, "r") as fd:
        config = json.load(fd)

    data_file = "data/unaligned_" + task.split("_")[0] + "_" + \
        task.split("_")[1]
    tokenize_type = list(task)[-1]

    print("data_file:{}, tokenize_type:{}, rm_punc:{}".format(
        data_file,
        tokenize_type,
        config[task]['remove_punctuation']))

    # Directory where the checkpoints will be saved
    checkpoint_dir = 'language_models/' + task
    checkpoint_prefix = os.path.join(
        os.getcwd(),
        checkpoint_dir,
        model_name + "_{epoch}.h5"
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=checkpoint_prefix,
                        save_weights_only=False)

    # dataset
    id2v, v2id, train_dataset = preprocessing(
        os.path.join(os.getcwd(), data_file),
        tokenize_type=tokenize_type,
        max_seq=config[task]['max_seq'],
        vocab_size=config[task]['vocab_size'],
        remove_punctuation=config[task]['remove_punctuation'],
        save_v2id_path=os.path.join(os.getcwd(), checkpoint_dir,
                                    "v2id.json")
    )

    config[task]['vocab_size'] = len(id2v)
    config[task]['max_seq'] = train_dataset.shape[1]

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    train_dataset = train_dataset.map(split_input_target)
    train_dataset = train_dataset.shuffle(1000).batch(
                        batch_size,
                        drop_remainder=True
                    ).repeat()

    # creating the model in the TPUStrategy scope means we will
    # train the model on the TPU
    with tpu_strategy.scope():
        model = build_model(
            config[task]['vocab_size'],
            config[task]['embedding_dim'],
            config[task]['units'],
            batch_size
        )

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.sparse_categorical_crossentropy(
                labels, logits, from_logits=True
            ),
            run_eagerly=False
        )

        history = model.fit(
            train_dataset,
            epochs=epochs,
            callbacks=[checkpoint_callback],
            verbose=1,
            steps_per_epoch=steps_per_epoch,
            shuffle=True,
            validation_data=train_dataset,
            validation_steps=steps_per_epoch
        )


if __name__ == '__main__':
    train()
