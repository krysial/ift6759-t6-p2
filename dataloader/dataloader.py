from utils import data
import tensorflow as tf
import os

ALIGNED_FORMATTED_FRENCH = 'aligned_formated_fr'
ALIGNED_UNFORMATTED_ENG = 'aligned_unformated_en'


def get_transformer_dataset(batch_size, dataset='train'):
    token_type = 'w'
    dataset_path = 'data/Train'

    if dataset == 'val':
        dataset_path = 'data/Valid'

    fr_id2v, fr_v2id, fr_lines = data.preprocessing(
        os.path.join(dataset_path, ALIGNED_FORMATTED_FRENCH),
        tokenize_type=token_type,
        add_start=False,
        add_end=False,
        padding='post'
    )

    en_id2v, en_v2id, en_lines = data.preprocessing(
        os.path.join(dataset_path, ALIGNED_UNFORMATTED_ENG),
        tokenize_type=token_type,
        padding='post'
    )

    fr_max_seq_len = fr_lines.shape[1]
    en_max_seq_len = en_lines.shape[1]

    dataset = tf.data.Dataset.from_tensor_slices(
        ((en_lines, fr_lines), fr_lines))  # adding targets to the inputs
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset, \
        {'id2w': en_id2v, 'w2id': en_v2id, 'max_seq_len': en_max_seq_len},\
        {'id2w': fr_id2v, 'w2id': fr_v2id, 'max_seq_len': fr_max_seq_len}


def get_dataset_train(
    BATCH_SIZE,
    train_split_ratio,
    steps_per_epoch,
    model_name,
    encoder_lang_model_task,
    decoder_lang_model_task,
    lang_model_opts
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
        fasttext_model=lang_model_opts[decoder_lang_model_task]['fasttext_model'],
    )

    ##########
    # SPLIT TRAIN-VALID
    ##########

    (
        input_tensor_train,
        input_tensor_valid,
        target_tensor_train,
        target_tensor_valid
    ) = train_test_split(
        encoder_dataset, decoder_dataset, test_size=train_split_ratio)

    ##########

    BUFFER_SIZE = len(input_tensor_train)

    if steps_per_epoch is None:
        steps_per_epoch = len(input_tensor_train)//BATCH_SIZE

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
        BATCH_SIZE,
        drop_remainder=True).repeat()

    dataset_valid = tf.data.Dataset.from_tensor_slices(
        ((input_tensor_valid, target_tensor_valid), target_tensor_valid)
    ).shuffle(BUFFER_SIZE)
    dataset_valid = dataset_valid.batch(
        BATCH_SIZE, drop_remainder=True
    ).repeat()

    print("#### Datasets Loaded ####")
    print(dataset_train, dataset_valid)

    ##########

    return (
        lang_model_opts,
        dataset_train,
        dataset_valid,
        steps_per_epoch
    )


def get_dataset_eval():
    pass
