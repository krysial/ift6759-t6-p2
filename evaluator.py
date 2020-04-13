import argparse
import subprocess
import tempfile
import os
import tensorflow as tf

from dataloader.dataloader import get_dataset_eval
from seq_2_seq_models.builder import get_model
from utils.data import postprocessing


def generate_predictions(input_file_path: str, pred_file_path: str):
    """Generates predictions for the machine translation task (EN->FR).

    You are allowed to modify this function as needed, but one again, you cannot
    modify any other part of this file. We will be importing only this function
    in our final evaluation script. Since you will most definitely need to import
    modules for your code, you must import these inside the function itself.

    Args:
        input_file_path: the file path that contains the input data.
        pred_file_path: the file path where to store the predictions.

    Returns: None

    """
    (
        lang_model_opts,
        seq_model_opts,
        train_opts,
        dataset,
        encoder_v2id
    ) = get_dataset_eval(
        '12-03-08-12',
        input_file_path, 'unformated_en_w2w', 'unformated_fr_w2w'
    )

    w_2_w_model = get_model(
        'GRU',
        train_opts=train_opts,
        seq_model_opts=seq_model_opts,
        encoder_lang_config=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ],
        decoder_lang_config=lang_model_opts[
            train_opts['decoder_lang_model_task']
        ]
    )

    # Directory where the checkpoints will be saved
    checkpoint_dir = os.path.join(
        os.getcwd(),
        'seq_2_seq_models',
        train_opts['encoder_lang_model_task'][:-4] + "_2_" +
        train_opts['decoder_lang_model_task'][:-4] + "_" +
        train_opts['encoder_lang_model_task'][-1] + "2" +
        train_opts['decoder_lang_model_task'][-1]
    )
    checkpoint_prefix = os.path.join(
        os.getcwd(),
        checkpoint_dir,
        train_opts['model_name'] + '_{epoch}.h5'
    )

    ckpt = tf.train.Checkpoint(model=w_2_w_model,
                               optimizer=w_2_w_model.optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_prefix, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)

    unformated_translation = tf.argmax(w_2_w_model.predict(dataset), axis=-1)
    processed_unformated_translation = postprocessing(
        dec_data=unformated_translation,
        dec_v2id=seq_model_opts['decoder_v2id'],
        output=None,
        tokenize_type=lang_model_opts[
            train_opts['decoder_lang_model_task']
        ]['tokenize_type'],
        fasttext_model=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['fasttext_model'],
        enc_data=input_file_path,
        remove_punctuation=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['remove_punctuation'],
        lower=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['lower'],
        CAP=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['CAP'],
        NUM=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['NUM'],
        ALNUM=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['ALNUM'],
        UPPER=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['UPPER'],
        enc_v2id=encoder_v2id
    )

    (
        lang_model_opts,
        seq_model_opts,
        train_opts,
        dataset,
        encoder_v2id
    ) = get_dataset_eval(
        '13-00-37-09',
        processed_unformated_translation,
        'unformated_fr_c2c', 'formated_fr_c2c'
    )

    c_2_c_model = get_model(
        'GRU',
        train_opts=train_opts,
        seq_model_opts=seq_model_opts,
        encoder_lang_config=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ],
        decoder_lang_config=lang_model_opts[
            train_opts['decoder_lang_model_task']
        ]
    )

    formated_translation = tf.argmax(c_2_c_model.predict(
        dataset
    ), axis=-1)
    processed_formated_translation = postprocessing(
        dec_data=formated_translation,
        dec_v2id=seq_model_opts['decoder_v2id'],
        output=None,
        tokenize_type=lang_model_opts[
            train_opts['decoder_lang_model_task']
        ]['tokenize_type'],
        fasttext_model=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['fasttext_model'],
        enc_data=input_file_path,
        remove_punctuation=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['remove_punctuation'],
        lower=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['lower'],
        CAP=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['CAP'],
        NUM=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['NUM'],
        ALNUM=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['ALNUM'],
        UPPER=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['UPPER'],
        enc_v2id=encoder_v2id
    )

    return processed_formated_translation


def compute_bleu(pred_file_path: str, target_file_path: str, print_all_scores: bool):
    """

    Args:
        pred_file_path: the file path that contains the predictions.
        target_file_path: the file path that contains the targets (also called references).
        print_all_scores: if True, will print one score per example.

    Returns: None

    """
    out = subprocess.run(["sacrebleu", "--input", pred_file_path, target_file_path, '--tokenize',
                          'none', '--sentence-level', '--score-only'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = out.stdout.split('\n')
    if print_all_scores:
        print('\n'.join(lines[:-1]))
    else:
        scores = [float(x) for x in lines[:-1]]
        print('final avg bleu score: {:.2f}'.format(sum(scores) / len(scores)))


def main():
    parser = argparse.ArgumentParser('script for evaluating a model.')
    parser.add_argument('--target-file-path',
                        help='path to target (reference) file', required=True)
    parser.add_argument('--input-file-path',
                        help='path to input file', required=True)
    parser.add_argument('--print-all-scores', help='will print one score per sentence',
                        action='store_true')
    parser.add_argument('--do-not-run-model',
                        help='will use --input-file-path as predictions, instead of running the '
                             'model on it',
                        action='store_true')

    args = parser.parse_args()

    if args.do_not_run_model:
        compute_bleu(args.input_file_path, args.target_file_path,
                     args.print_all_scores)
    else:
        _, pred_file_path = tempfile.mkstemp()
        generate_predictions(args.input_file_path, pred_file_path)
        compute_bleu(pred_file_path, args.target_file_path,
                     args.print_all_scores)


if __name__ == '__main__':
    main()
