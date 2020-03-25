import argparse
import os
import json

from src.main import Application

PATH_DATA = 'data'
OPTIONS_CONF_FILE = os.path.join(PATH_DATA, 'config.json')


def main(args):
    """
        Handles application arguments
    """

    options = {}
    if OPTIONS_CONF_FILE:
        assert os.path.isfile(OPTIONS_CONF_FILE), f"invalid config file: {OPTIONS_CONF_FILE}"
        with open(OPTIONS_CONF_FILE, "r") as f:
            options = json.load(f)

    options.update(args)
    app = Application(options)
    app.train()


if __name__ == '__main__':
    print("--- Running training script ---")
    parser = argparse.ArgumentParser('Script for training a model.')
    parser.add_argument('--model', help='model_name', required=True)
    parser.add_argument('--batch_size', help='batch size', type=int)
    parser.add_argument('--lr', help='learning rate', type=float)
    parser.add_argument('--epochs', help='number of epochs', type=int)

    args = vars(parser.parse_args())
    args = {key: value for key, value in args.items() if value is not None}
    main(args)
