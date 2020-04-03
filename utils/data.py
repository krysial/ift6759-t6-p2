from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np
import json
from tqdm import tqdm


def preprocess_v2id(data, v2id, max_seq=None, add_start=True,
                    add_end=True, remove_punctuation=True,
                    tokenize_type="w", padding='post'):
    """
    Gets tokenized and integer encoded format of given
    text corpus using given v2id mapping

    Arguments:
        data: string or list. Path of input text file when string and
              data (batch of sentences expected) itself when List(list).
        v2id: string or dict, Mapping of vocabulary to integer encoding.
        max_seq: int or Nonr, default None. Sequence length to retain.
        add_start: boolean, default True.
                   If True, adds start-of-sequence (<SOS>)
        add_end: boolean, default True.
                 If True, adds end-of-sequence (<EOS>) tokens.
        remove_punctuation: boolean, default True.
                            If True, removes punctuation symbols.
        tokenize_type: "w" or "c", default "w".
                       Defines token type as word(w) or char(c).
        padding: 'post' or 'pre', default 'post'.
                 Defines where to pad, begining('pre') or end('post').

    Returns:
        Dictionaries ``v2id`` and ``id2v`` for encoding and
                decoding vocabulary as id
        Numpy array ``encoded_lines`` of shape (batch_size, max_seq)
    """

    # Read data as list of samples
    lines = checkout_data(data)

    # Handle punctuations if required
    if remove_punctuation:
        lines = handle_punctuation(lines)

    # Get v2id dicitionary
    if isinstance(v2id, str):
        v2id = load_v2id(v2id)

    # Get data as list(list(token))
    lines = handle_tokenizer(lines, tokenize_type)

    # Add <SOS> / <EOS>
    lines = handle_sos_eos(lines, add_start, add_end)

    # Encode text data using v2id
    lines = handle_encoding_v2id(lines, v2id)

    # Padd Padd Padd ..
    lines = handle_padding(lines, padding, max_seq, v2id)

    return v2id, lines


def preprocessing(data, max_seq=None, vocab_size=None, add_start=True,
                  add_end=True, remove_punctuation=True, tokenize_type="w",
                  padding='post', save_v2id_path=None):
    """
    Gets tokenized and integer encoded format of given text corpus.

    Arguments:
        data: string or list. Path of input text file when string and
              data (batch of sentences expected) itself when List(list).
        max_seq: int or Nonr, default None. Sequence length to retain.
        vocab_size: int or None, default None. Number of words to retain,
                    sorted by most common. If None, retain all words.
        add_start: boolean, default True.
                   If True, adds start-of-sequence (<SOS>)
        add_end: boolean, default True.
                 If True, adds end-of-sequence (<EOS>) tokens.
        remove_punctuation: boolean, default True.
                            If True, removes punctuation symbols.
        tokenize_type: "w" or "c", default "w".
                       Defines token type as word(w) or char(c).
        padding: 'post' or 'pre', default 'post'.
                 Defines where to pad, begining('pre') or end('post').
        save_v2id_path: string, default is None.
                        Path of saving v2id dictionary mapping.

    Returns:
        Dictionaries ``v2id`` and ``id2v`` for encoding and
                        decoding vocabulary as id
        Numpy array ``encoded_lines`` of shape (batch_size, max_seq)
    """

    # Read data as list of samples
    lines = checkout_data(data)

    # Handle punctuations if required
    if remove_punctuation:
        lines = handle_punctuation(lines)

    # Get corpus tokens
    id2v, v2id = handle_vocab(lines, tokenize_type, vocab_size)

    # Get data as list(list(token))
    lines = handle_tokenizer(lines, tokenize_type)

    # Add <SOS> / <EOS>
    lines = handle_sos_eos(lines, add_start, add_end)

    # Encode text data using v2id
    lines = handle_encoding_v2id(lines, v2id)

    # Padd Padd Padd ..
    lines = handle_padding(lines, padding, max_seq, v2id)

    # Save v2id
    if save_v2id_path is not None:
        save_v2id(v2id, path=save_v2id_path)

    return id2v, v2id, lines


# DATA UTILS:

def handle_punctuation(lines):
    # Symbols to be removed, from punctuation_remover.py
    PUNCTUATION = {",", ";", ":", "!", "?", ".", "'", '"',
                   "(", ")", "...", "[", "]", "{", "}", "’"}
    # Remove punctuation
    split_lines = [
        [x for x in line.split() if x not in PUNCTUATION] for line in lines
    ]
    lines = [' '.join(line) for line in split_lines]
    return lines


def checkout_data(data):
    # DATA is FILE
    if isinstance(data, str):
        # Read lines from file
        with open(data) as f:
            lines = f.readlines()
    # DATA is list(sentences)
    elif isinstance(data, list):
        lines = data
    # Strip newline
    lines = [line.strip() for line in lines]
    return lines


def handle_tokenizer(lines, tokenize_type):
    # Tokenize input dataset
    if tokenize_type == "w":
        lines = [line.split() for line in lines]
    if tokenize_type == "c":
        lines = [list(line) for line in lines]
    return lines


def handle_vocab(lines, tokenize_type, vocab_size):
    # Token is word
    if tokenize_type == "w":
        corpus = ' '.join(lines).split()
    # Token is char
    if tokenize_type == "c":
        corpus = list(' '.join(lines))
    # Get most common words
    vocab = Counter(corpus).most_common(vocab_size) + \
        [('<SOS>', 0), ('<EOS>', 0)]
    # Create id to vocabulary dictionary
    id2v = dict(pd.DataFrame(vocab, columns=['tokens', 'count'])['tokens'])
    v2id = swap_dict_key_value(id2v)
    return id2v, v2id


def swap_dict_key_value(k2v):
    v2k = dict([(value, key) for key, value in k2v.items()])
    return v2k


def handle_sos_eos(lines, add_start, add_end):
    # Start sentences with <SOS>
    if add_start:
        lines = [['<SOS>'] + line for line in lines]
    # End sentences with <EOS>
    if add_end:
        lines = [line + ['<EOS>'] for line in lines]
    return lines


def handle_encoding_v2id(lines, v2id):
    encoded_lines = [list(map(v2id.get, line)) for line in tqdm(lines)]
    return encoded_lines


def handle_padding(lines, padding, max_seq, v2id):
    if padding == 'pre':
        padder = '<SOS>'
    elif padding == 'post':
        padder = '<EOS>'
    encoded_lines = pad_sequences(
        lines,
        maxlen=max_seq,
        padding=padding,
        value=v2id[padder]
    )
    return encoded_lines


def save_v2id(v2id, path):
    with open(path, 'w') as f:
        json.dump(v2id, f)
    print("\n# v2id dictionary saved at: {}".format(path))


def load_v2id(path):
    with open(path, 'r') as f:
        return json.load(f)
