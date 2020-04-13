from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import json

from utils import data
from utils.data import preprocess_v2id


def generate_config_path(root_path, dt, config_name):
    return os.path.join(
        root_path,
        'configs',
        dt,
        '{}.json'.format(config_name)
    )


def get_dataset_train(
    model_name,
    encoder_lang_model_task,
    decoder_lang_model_task,
    lang_model_opts,
    train_opts,
    seq_model_opts,
    DT
):
    '''


    '''

    ##########
    # dataset
    ##########
    encoder_v2id, encoder_dataset = preprocess_v2id(
        data=os.path.join(
            os.getcwd(), lang_model_opts[encoder_lang_model_task]['data_file']),
        v2id=os.path.join(
            os.getcwd(),
            "language_models",
            encoder_lang_model_task,
            "v2id.json"
        ),
        tokenize_type=lang_model_opts[encoder_lang_model_task]['tokenize_type'],
        max_seq=lang_model_opts[encoder_lang_model_task]['max_seq'],
        remove_punctuation=lang_model_opts[encoder_lang_model_task]['remove_punctuation'],
        lower=lang_model_opts[encoder_lang_model_task]['lower'],
        threshold=lang_model_opts[encoder_lang_model_task]['threshold'],
        CAP=lang_model_opts[encoder_lang_model_task]['CAP'],
        NUM=lang_model_opts[encoder_lang_model_task]['NUM'],
        ALNUM=lang_model_opts[encoder_lang_model_task]['ALNUM'],
        UPPER=lang_model_opts[encoder_lang_model_task]['UPPER'],
        fasttext_model=lang_model_opts[encoder_lang_model_task]['fasttext_model'],
    )

    decoder_v2id, decoder_dataset = preprocess_v2id(
        data=os.path.join(
            os.getcwd(), lang_model_opts[decoder_lang_model_task]['data_file']),
        v2id=os.path.join(
            os.getcwd(),
            "language_models",
            decoder_lang_model_task,
            "v2id.json"
        ),
        tokenize_type=lang_model_opts[decoder_lang_model_task]['tokenize_type'],
        max_seq=lang_model_opts[decoder_lang_model_task]['max_seq'],
        remove_punctuation=lang_model_opts[decoder_lang_model_task]['remove_punctuation'],
        lower=lang_model_opts[decoder_lang_model_task]['lower'],
        threshold=lang_model_opts[decoder_lang_model_task]['threshold'],
        CAP=lang_model_opts[decoder_lang_model_task]['CAP'],
        NUM=lang_model_opts[decoder_lang_model_task]['NUM'],
        ALNUM=lang_model_opts[decoder_lang_model_task]['ALNUM'],
        UPPER=lang_model_opts[decoder_lang_model_task]['UPPER'],
        fasttext_model=lang_model_opts[decoder_lang_model_task]['fasttext_model'],
    )

    seq_model_opts['decoder_v2id'] = decoder_v2id

    ##########
    # SPLIT TRAIN-VALID
    ##########

    (
        input_tensor_train,
        input_tensor_valid,
        target_tensor_train,
        target_tensor_valid
    ) = train_test_split(
        encoder_dataset, decoder_dataset, test_size=train_opts['train_split_ratio'])

    ##########

    BUFFER_SIZE = len(input_tensor_train)

    if train_opts['steps_per_epoch'] is None:
        train_opts['steps_per_epoch'] = len(
            input_tensor_train)//train_opts['batch_size']

    lang_model_opts[encoder_lang_model_task]['max_seq'] = encoder_dataset.shape[-1]
    lang_model_opts[decoder_lang_model_task]['max_seq'] = decoder_dataset.shape[-1]

    lang_model_opts[encoder_lang_model_task]['vocab_size'] = len(encoder_v2id)
    lang_model_opts[decoder_lang_model_task]['vocab_size'] = len(decoder_v2id)

    print("#### ENC-DEC DATA Preprocessed ####")
    print("Enc:", lang_model_opts[encoder_lang_model_task])
    print("Dec:", lang_model_opts[decoder_lang_model_task])

    ##########
    # TF.DATA.DATASET
    ##########

    dataset_train = tf.data.Dataset.from_tensor_slices(
        ((input_tensor_train, target_tensor_train), target_tensor_train)
    ).shuffle(BUFFER_SIZE)
    dataset_train = dataset_train.batch(
        train_opts['batch_size'],
        drop_remainder=True).repeat()

    dataset_valid = tf.data.Dataset.from_tensor_slices(
        ((input_tensor_valid, target_tensor_valid), target_tensor_valid)
    ).shuffle(BUFFER_SIZE)
    dataset_valid = dataset_valid.batch(
        train_opts['batch_size'], drop_remainder=True
    ).repeat()

    print("#### Datasets Loaded ####")
    print(dataset_train, dataset_valid)

    ##########

    root_path = os.path.join(
        os.getcwd(),
        'seq_2_seq_models',
        train_opts['encoder_lang_model_task'][:-4] + "_2_" +
        train_opts['decoder_lang_model_task'][:-4] + "_" +
        train_opts['encoder_lang_model_task'][-1] + "2" +
        train_opts['decoder_lang_model_task'][-1]
    )

    def save_json(dt, data, config_name):
        path = generate_config_path(root_path, dt, config_name)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as json_file:
            json.dump(data, json_file)

    save_json(DT, lang_model_opts, 'lang_model_opts')
    save_json(DT, train_opts, 'train_opts')
    save_json(DT, seq_model_opts, 'seq_model_opts')

    return (
        lang_model_opts,
        train_opts,
        seq_model_opts,
        dataset_train,
        dataset_valid
    )


def get_dataset_eval(
    DT,
    encoder_file_path,
    encoder_lang_model_task,
    decoder_lang_model_task
):
    '''


    '''
    root_path = os.path.join(
        os.getcwd(),
        'seq_2_seq_models',
        encoder_lang_model_task[:-4] + "_2_" +
        decoder_lang_model_task[:-4] + "_" +
        encoder_lang_model_task[-1] + "2" +
        decoder_lang_model_task[-1]
    )

    def load_json(dt, config_name):
        path = generate_config_path(
            root_path, dt, config_name
        )

        with open(path, 'r') as f:
            return json.load(f)

    lang_model_opts = load_json(DT, 'lang_model_opts')
    seq_model_opts = load_json(DT, 'seq_model_opts')
    train_opts = load_json(DT, 'train_opts')

    encoder_v2id, encoder_dataset = preprocess_v2id(
        data=encoder_file_path,
        v2id=os.path.join(
            os.getcwd(),
            "language_models",
            encoder_lang_model_task,
            "v2id.json"
        ),
        tokenize_type=lang_model_opts[encoder_lang_model_task]['tokenize_type'],
        max_seq=lang_model_opts[encoder_lang_model_task]['max_seq'],
        remove_punctuation=lang_model_opts[encoder_lang_model_task]['remove_punctuation'],
        lower=lang_model_opts[encoder_lang_model_task]['lower'],
        threshold=lang_model_opts[encoder_lang_model_task]['threshold'],
        CAP=lang_model_opts[encoder_lang_model_task]['CAP'],
        NUM=lang_model_opts[encoder_lang_model_task]['NUM'],
        ALNUM=lang_model_opts[encoder_lang_model_task]['ALNUM'],
        UPPER=lang_model_opts[encoder_lang_model_task]['UPPER'],
        fasttext_model=lang_model_opts[encoder_lang_model_task]['fasttext_model'],
    )

    lang_model_opts[encoder_lang_model_task]['max_seq'] = encoder_dataset.shape[-1]

    print("#### ENC-DEC DATA Preprocessed ####")

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            (encoder_dataset, tf.zeros_like(encoder_dataset)),
            tf.zeros_like(encoder_dataset)
        )
    ).batch(train_opts['batch_size'], drop_remainder=False)

    print("#### Datasets Loaded ####")

    return (
        lang_model_opts,
        seq_model_opts,
        train_opts,
        dataset,
        encoder_v2id
    )
