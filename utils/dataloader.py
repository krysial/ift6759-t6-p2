from utils.data import preprocess_v2id, postprocessing
from utils.data import swap_dict_key_value
import os


def encoder_preprocess(data):
    encoder_v2id, encoder_dataset = preprocess_v2id(
            data=data,
            v2id=os.path.join("language_models", "unformated_en_w2w", "v2id.json"),
            tokenize_type="w",
            max_seq=None,
            remove_punctuation=True,
            lower=False,
            threshold=0.85,
            CAP=True,
            NUM=True,
            ALNUM=True,
            UPPER=True,
            fasttext_model="embeddings/unformated_en_w2w/128/unaligned_unformated_en",
    )
    encoder_id2v = swap_dict_key_value(encoder_v2id)
    return encoder_dataset, encoder_v2id, encoder_id2v


def decoder_preprocess(data):
    decoder_v2id, decoder_dataset = preprocess_v2id(
            data=data,
            v2id=os.path.join("language_models","formated_fr_w2w", "v2id.json"),
            tokenize_type="w",
            max_seq=None,
            remove_punctuation=False,
            lower=False,
            threshold=0.85,
            CAP=True,
            NUM=True,
            ALNUM=True,
            UPPER=True,
            fasttext_model="embeddings/formated_fr_w2w/128/unaligned_formated_fr",
    )
    decoder_id2v = swap_dict_key_value(decoder_v2id)
    return decoder_dataset, decoder_v2id, decoder_id2v
