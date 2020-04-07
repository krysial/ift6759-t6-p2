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
        padding_value=0,
        padding='post'
    )

    en_id2v, en_v2id, en_lines = data.preprocessing(
        os.path.join(dataset_path, ALIGNED_UNFORMATTED_ENG),
        tokenize_type=token_type,
        padding='post',
        padding_value=0
    )

    fr_max_seq_len = fr_lines.shape[1]
    en_max_seq_len = en_lines.shape[1]

    dataset = tf.data.Dataset.from_tensor_slices(((en_lines, fr_lines), fr_lines))  # adding targets to the inputs
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset, (en_id2v, en_v2id, en_max_seq_len), (fr_id2v, fr_v2id, fr_max_seq_len)


