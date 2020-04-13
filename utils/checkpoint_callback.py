import tensorflow as tf
import datetime


class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super(CheckpointCallback, self).__init__()
        self.filepath = filepath

    def on_train_begin(self, logs=None):
        self.ckpt = tf.train.Checkpoint(
            model=self.model, optimizer=self.model.optimizer
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt,
            self.filepath,
            max_to_keep=5
        )

    def on_epoch_end(self, epoch, logs=None):
        self.ckpt_manager.save()
