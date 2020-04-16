import argparse
import subprocess
import tempfile
import os
import tensorflow as tf

from dataloader.dataloader import get_dataset_eval
from seq_2_seq_models.builder import get_model
from utils.data import postprocessing


def seq2seq_block(DT, model_name, encoder_lang_model_task,
                  input_file, decoder_lang_model_task):

    (
        lang_model_opts,
        seq_model_opts,
        train_opts,
        dataset,
        encoder_v2id
    ) = get_dataset_eval(
        DT, model_name,
        input_file,
        encoder_lang_model_task,
        decoder_lang_model_task
    )

    model = get_model(
        model_name,
        train_opts=train_opts,
        seq_model_opts=seq_model_opts,
        encoder_lang_config=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ],
        decoder_lang_config=lang_model_opts[
            train_opts['decoder_lang_model_task']
        ]
    )

    # Directory where the checkpoints will be loaded from
    checkpoint_dir = os.path.join(
        'seq_2_seq_models',
        train_opts['encoder_lang_model_task'][:-4] + "_2_" +
        train_opts['decoder_lang_model_task'][:-4] + "_" +
        train_opts['encoder_lang_model_task'][-1] + "2" +
        train_opts['decoder_lang_model_task'][-1],
        train_opts['model_name'], DT
    )

    ckpt = tf.train.Checkpoint(
        model=model, optimizer=model.optimizer)

    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_dir, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)

    predictions = tf.argmax(model.predict(dataset), axis=-1)

    processed_predicitons = postprocessing(
        dec_data=predictions,
        dec_v2id=seq_model_opts['decoder_v2id'],
        Print=True,
        tokenize_type=lang_model_opts[
            train_opts['decoder_lang_model_task']
        ]['tokenize_type'],
        fasttext_model=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['fasttext_model'],
        enc_data=input_file,
        threshold=lang_model_opts[
            train_opts['encoder_lang_model_task']
        ]['threshold'],
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

    return processed_predicitons


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

    # Translation Task: w2w sequence model
    Translation = seq2seq_block(
        DT='14-20-50-00',
        model_name="GRU",
        input_file=input_file_path,
        encoder_lang_model_task='unformated_en_w2w',
        decoder_lang_model_task='unformated_fr_w2w')

    # Punctuation Task: c2c sequence model
    Punctuated_Translation = seq2seq_block(
        DT='15-00-11-46',
        model_name="GRU",
        input_file=Translation,
        encoder_lang_model_task='unformated_fr_c2c',
        decoder_lang_model_task='formated_fr_c2c')

    return Punctuated_Translation


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
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
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
