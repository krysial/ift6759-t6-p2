import tensorflow as tf

import numpy as np
import os
import time

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
  print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
  raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

############################


BATCH_SIZE = 64
epoch = 20
steps_per_epoch = 500
Task = "formated_fr_c2c"
data_file = "data/unaligned_" + Task.split("_")[0] + "_" + Task.split("_")[1]
tokenize_type = list(Task)[-1]
max_seq = None
vocab_size = None
remove_punctuation=False

print("data_file:",data_file, ", tokenize_type:", tokenize_type, ", remove_punctuation:", remove_punctuation)


# Directory where the checkpoints will be saved
checkpoint_dir = 'language_models/' + Task
checkpoint_prefix = os.path.join(os.getcwd(),checkpoint_dir,"GRU_{epoch}.h5")
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_prefix,
                    save_weights_only=False)

# dataset
id2v, v2id, train_dataset = preprocessing(os.path.join(os.getcwd(),data_file),
                                          tokenize_type=tokenize_type,
                                          max_seq=max_seq,
                                          vocab_size=vocab_size,
                                          remove_punctuation=remove_punctuation)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
train_dataset = train_dataset.map(split_input_target)
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE, drop_remainder=True).repeat()

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

with tpu_strategy.scope(): # creating the model in the TPUStrategy scope means we will train the model on the TPU
  model = build_model(
      len(id2v),
      256,
      1024,
      BATCH_SIZE
  )

  model.compile(optimizer='adam', loss=loss, run_eagerly=False)

  history = model.fit(train_dataset, epochs=epoch, callbacks=[checkpoint_callback], verbose=1,
                    steps_per_epoch=steps_per_epoch, shuffle=True, validation_data=train_dataset, validation_steps=steps_per_epoch)

