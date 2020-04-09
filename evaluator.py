import argparse
import subprocess
import tempfile
import os


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
    with open(config_path, "r") as fd:
        config = json.load(fd)

    model_path = os.path.join(
        os.getcwd(),
        'seq_2_seq_models',
        'unformated_en_2_unformated_fr_w2w'
    )
    model_path = os.path.join(model_path, 'GRU_2.h5')

    enc_v2id_path = os.path.join(
        os.getcwd(),
        "language_models",
        encoder_lang_model_task,
        "v2id.json"
    )

    w_2_w_model = seq_2_seq_GRU(
        vocab_inp_size=vocab_inp_size,
        encoder_embedding_dim=encoder_config['embedding_dim'],
        encoder_units=encoder_config['units'],
        vocab_tar_size=vocab_tar_size,
        decoder_embedding_dim=decoder_config['embedding_dim'],
        decoder_units=decoder_config['units'],
        decoder_v2id=decoder_v2id,
        targ_seq_len=decoder_config['max_seq'],
        BATCH_SIZE=BATCH_SIZE,
        encoder_lang_model=encoder_config['lang_model_checkpointer'],
        decoder_lang_model=decoder_config['lang_model_checkpointer']
    )

    c_2_c_model = seq_2_seq_GRU(
        vocab_inp_size=vocab_inp_size,
        encoder_embedding_dim=encoder_config['embedding_dim'],
        encoder_units=encoder_config['units'],
        vocab_tar_size=vocab_tar_size,
        decoder_embedding_dim=decoder_config['embedding_dim'],
        decoder_units=decoder_config['units'],
        decoder_v2id=decoder_v2id,
        targ_seq_len=decoder_config['max_seq'],
        BATCH_SIZE=BATCH_SIZE,
        encoder_lang_model=encoder_config['lang_model_checkpointer'],
        decoder_lang_model=decoder_config['lang_model_checkpointer']
    )

    processed_sentence = None

    batched_unformated_french_w2w = np.argmax(w_2_w_model.predict(processed_sentence), axis=-1)

    batched_unformated_french_w2w > french_w2w.txt

    data = process(french_w2w.txt)
    c_2c_model.predict(data)


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
