import tensorflow as tf
import os

from seq_2_seq_models.transformer.utils import create_padding_mask, create_look_ahead_mask
from seq_2_seq_models.transformer.transformer import Transformer
from dataloader import dataloader

tf.random.set_seed(12345)


class Model:
    def __init__(self, opts):
        self.opts = opts

        self.optimizer = tf.keras.optimizers.Adam(self.opts.lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.train_loss = tf.keras.metrics.Mean(name='mean_metric')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    def train(self):
        # data
        dataset, en_stats, fr_stats = dataloader.get_transformer_dataset(batch_size=self.opts.batch_size)
        en_seq_len = en_stats[2]
        en_vocab_size = len(en_stats[0]) + 1  # +1 bc 0 is padding
        fr_seq_len = fr_stats[2]
        fr_vocab_size = len(fr_stats[0]) + 1  # +1 bc 0 is padding

        # model
        model = Transformer(en_seq_len, en_vocab_size, fr_seq_len - 1, fr_vocab_size,
                            self.opts)  # -1 bc targets will be shifted by 1

        # checkpoints
        chkpt = tf.train.Checkpoint(model=model, optimizer=self.optimizer)
        chkpt_path = os.path.join(self.opts.checkpoints_path, self.opts.model_name)
        chkpt_manager = tf.train.CheckpointManager(chkpt, chkpt_path, max_to_keep=self.opts.max_to_keep)

        # if a checkpoint exists, restore the latest checkpoint.
        if chkpt_manager.latest_checkpoint:
            chkpt.restore(chkpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

        # epochs
        for epoch in range(self.opts.epochs):
            print(f"--------------------- Epoch: {epoch} ---------------------")
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            # batches
            for batch, (inputs, targets) in enumerate(dataset.take(1)):  # TODO: remove take 1 when training the model
                targets_input = targets[:, :-1]
                targets_real = targets[:, 1:]

                # TODO: use @tf.function
                input_pad_mask = create_padding_mask(inputs)
                target_pad_mask = create_padding_mask(targets_input)
                target_look_ahead_mask = create_look_ahead_mask(tf.shape(targets_input)[-1])  # seq_len of targets
                combined_mask = tf.maximum(target_pad_mask, target_look_ahead_mask)  # target_look_ahead_mask in decoding

                with tf.GradientTape() as tape:
                    predictions = model(inputs, targets_input, input_pad_mask, combined_mask, training=True)
                    loss = self.loss_function(targets_real, predictions)

                gradients = tape.gradient(loss, model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                self.train_loss(loss)
                self.train_accuracy(targets_real, predictions)

            if epoch % self.opts.after_num_epochs == 0:
                ckpt_save_path = chkpt_manager.save()
                print(f"Epoch: {epoch}: loss:{self.train_loss.result()} accuracy: {self.train_accuracy.result()} chkp_path: {ckpt_save_path}")

    # https://www.tensorflow.org/tutorials/text/transformer#loss_and_metrics
    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def evaluate(self):
        pass
