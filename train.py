import argparse
import os
from types import SimpleNamespace
import tensorflow as tf
import json

from seq_2_seq_models.transformer import builder as transformer_builder
from dataloader import dataloader


PATH_DATA = 'config'
OPTIONS_CONF_FILE = os.path.join(PATH_DATA, 'config.json')


def main(arguments):
    """
        Handles application arguments
    """
    options = {}
    if OPTIONS_CONF_FILE:
        assert os.path.isfile(
            OPTIONS_CONF_FILE), f"invalid config file: {OPTIONS_CONF_FILE}"
        with open(OPTIONS_CONF_FILE, "r") as f:
            options = json.load(f)
    options.update(arguments)
    opts = SimpleNamespace(**options)

# --------------- MODEL LOADING AND TRAINING ----------------------
    # Checkpointer
    CHKPT_FOLDER = os.path.join(opts.checkpoints_path, opts.model_name)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHKPT_FOLDER, save_weights_only=False)

    # Dataset
    dataset_train, en_stats, fr_stats = dataloader.get_transformer_dataset(batch_size=opts.batch_size)
    input_seq_len = en_stats[2]
    input_vocab_size = len(en_stats[0]) + 1  # +1 bc 0 is padding
    target_seq_len = fr_stats[2]
    target_vocab_size = len(fr_stats[0]) + 1  # +1 bc 0 is padding

    # Model
    transformer = transformer_builder.get_model(opts, input_seq_len, input_vocab_size, target_seq_len, target_vocab_size)

    # Model fit
    transformer.fit(
        dataset_train,
        epochs=opts.epochs,
        callbacks=[checkpoint_callback],
        verbose=1,
        steps_per_epoch=opts.steps_per_epoch,
        shuffle=True,
        # validation_data=dataset_valid,
        # validation_steps=steps_per_epoch
    )


if __name__ == '__main__':
    print("--- Running transformer script ---")
    parser = argparse.ArgumentParser('Script for training a model.')
    parser.add_argument('--model_name', help='model_name')  # required=True
    parser.add_argument('--best_model_path', help='candidate models path')
    parser.add_argument('--checkpoints_path', help='checkpoints path')
    parser.add_argument('--after_num_epochs', help='checkpoint after this num of epochs', type=int)
    parser.add_argument('--batch_size', help='batch size', type=int)
    parser.add_argument('--lr', help='learning rate', type=float)
    parser.add_argument('--dr', help='dropout rate', type=float)
    parser.add_argument('--epochs', help='number of epochs', type=int)
    parser.add_argument('--atten_dim', help='attention model dimensionality', type=int)
    parser.add_argument('--num_heads', help='number of head for attention', type=int)
    parser.add_argument('--ff_dim', help='fully connected dimensionality', type=int)

    args = vars(parser.parse_args())
    args = {key: value for key, value in args.items() if value is not None}
    main(args)
