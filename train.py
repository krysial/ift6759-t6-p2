import argparse
import os
from types import SimpleNamespace
import tensorflow as tf
import json

from seq_2_seq_models.transformer import builder as transformer_builder
from dataloader import dataloader

def main(arguments):
    """
        Handles application arguments
    """

    PATH_DATA = 'config'
    OPTIONS_CONF_FILE = os.path.join(PATH_DATA, 'config.json')
    
    options = {}
    if OPTIONS_CONF_FILE:
        assert os.path.isfile(
            OPTIONS_CONF_FILE), f"invalid config file: {OPTIONS_CONF_FILE}"
        with open(OPTIONS_CONF_FILE, "r") as f:
            options = json.load(f)
    options.update(arguments)
    opts = SimpleNamespace(**options)


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
