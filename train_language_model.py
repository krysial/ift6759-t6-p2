import tensorflow as tf

import numpy as np
import os
import time

from language_models.language_model import build_model
from utils.data import preprocessing

BATCH_SIZE = 32

# Directory where the checkpoints will be saved
checkpoint_dir = './unformated_eng_w2w/gru'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# dataset
id2v, v2id, dataset = preprocessing('data/unaligned.en')


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = tf.data.Dataset.from_tensor_slices(dataset)
dataset = dataset.map(split_input_target)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

model = build_model(
    len(id2v),
    256,
    1024,
    BATCH_SIZE
)

optimizer = tf.keras.optimizers.Adam()


@tf.function
def train_step(inp, target):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target, predictions, from_logits=True))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


# Training step
EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()

    # initializing the hidden state at the start of every epoch
    # initally hidden is None
    hidden = model.reset_states()

    for (batch_n, (inp, target)) in enumerate(dataset):
        loss = train_step(inp, target)

        if batch_n % 100 == 0:
            template = 'Epoch {} Batch {} Loss {}'
            print(template.format(epoch+1, batch_n, loss))

    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))

    print('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    model.save_weights(checkpoint_prefix.format(epoch=epoch))
