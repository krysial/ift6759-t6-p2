from collections import Counter
from tqdm import tqdm
from copy import deepcopy as copy
from tensorflow.keras.preprocessing.sequence import pad_sequences

import re
import pandas as pd
import numpy as np
import os
import json
import gensim.models


def preprocess_v2id(
        data,
        v2id,
        fasttext_model=None,
        max_seq=None,
        add_start=True,
        add_end=True,
        remove_punctuation=True,
        lower=True,
        threshold=0.5,
        CAP=False,
        NUM=False,
        ALNUM=False,
        UPPER=False,
        tokenize_type="w",
        padding='post',
        post_process_usage=False):
    """
    Gets tokenized and integer encoded format of given
    text corpus using given v2id mapping

    Arguments:
        data: string or list. Path of input text file when string and
            data (batch of sentences expected) itself when List(list).
        v2id: string or dict, Mapping of vocabulary to integer encoding.
        fasttext_model: str, default None. Fasttext Model path
        max_seq: int, default None. Sequence length to retain.
        add_start: boolean, default True.
            If True, adds start-of-sequence (<SOS>)
        add_end: boolean, default True.
            If True, adds end-of-sequence (<EOS>) tokens.
        remove_punctuation: boolean, default True.
            If True, removes punctuation symbols.
        lower: boolean, default True. To condition case folding.
        threshold: float. default is 0.5.
            To threshold fasttext similarity indexing.
        CAP: boolean, default False. To condition special token <CAP>
        NUM: boolean, default False. To condition special token <NUM>
        ALNUM: boolean, default False. To condition special token <ALNUM>
        UPPER: boolean, default False. To condition special token <UPPER>
        tokenize_type: "w" or "c", default "w".
            Defines token type as word(w) or char(c).
        padding: 'post' or 'pre', default 'post'.
            Defines where to pad, begining('pre') or end('post').
        post_process_usage: boolean, default is False.
            Defines the usage of this function.

    Returns:
        Dictionaries ``v2id`` and ``id2v`` for encoding and decoding
            vocabulary as id
        Numpy array ``encoded_lines`` of shape (batch_size, max_seq)
    """

    # Read data as list of samples
    lines = checkout_data(data)

    # Convert to lowercase
    if lower:
        lines = [line.lower() for line in lines]

    # Handle punctuations if required
    if remove_punctuation:
        lines = handle_punctuation(lines)

    # Apply regex constraints
    if CAP or NUM or ALNUM or UPPER:
        lines = handle_regex(lines, CAP=CAP, NUM=NUM, ALNUM=ALNUM, UPPER=UPPER)

    # Get v2id dicitionary
    if isinstance(v2id, str):
        v2id = load_json(v2id)

    # Get data as list(list(token))
    lines = handle_tokenizer(lines, tokenize_type)

    # Add <SOS> / <EOS>
    lines = handle_sos_eos(lines, add_start, add_end)

    # If called by postprocessing, end function and return List(list(words))
    if post_process_usage:
        return lines

    # Setup fasttext model from path
    if fasttext_model is not None and isinstance(fasttext_model, str):
        fasttext_model = gensim.models.FastText.load(fasttext_model)

    # Encode text data using v2id
    lines = handle_encoding_v2id(lines, v2id, fasttext_model, threshold)

    # Padd Padd Padd ..
    lines = handle_padding(lines, padding, max_seq, v2id)

    return v2id, lines


def preprocessing(data, max_seq=None, vocab_size=None, add_start=True,
                  add_end=True, remove_punctuation=True, lower=True,
                  CAP=False, NUM=False, ALNUM=False, UPPER=False,
                  tokenize_type="w", padding='post', save_v2id_path=None,
                  fasttext_model=None, threshold=0.5):
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
        lower: boolean, default True. To condition case folding.
        CAP: boolean, default False. To condition special token <CAP>
        NUM: boolean, default False. To condition special token <NUM>
        ALNUM: boolean, default False. To condition special token <ALNUM>
        UPPER: boolean, default False. To condition special token <UPPER>
        tokenize_type: "w" or "c", default "w".
            Defines token type as word(w) or char(c).
        padding: 'post' or 'pre', default 'post'.
            Defines where to pad, begining('pre') or end('post').
        save_v2id_path: string, default is None.
            Path of saving v2id dictionary mapping.
        fasttext_model: str, default None. Fasttext Model path
        threshold: float. default is 0.5.
            To threshold fasttext similarity indexing.

    Returns:
        Dictionaries ``v2id`` and ``id2v`` for encoding and
            decoding vocabulary as id
        Numpy array ``encoded_lines`` of shape (batch_size, max_seq)
    """

    # Read data as list of samples
    lines = checkout_data(data)

    # Convert to lowercase
    if lower:
        lines = [line.lower() for line in lines]

    # Handle punctuations if required
    if remove_punctuation:
        lines = handle_punctuation(lines)

    # Get corpus tokens
    id2v, v2id = handle_vocab(lines, tokenize_type, vocab_size)

    # Apply regex constraints
    if CAP or NUM or ALNUM or UPPER:
        lines = handle_regex(lines, CAP=CAP, NUM=NUM, ALNUM=ALNUM, UPPER=UPPER)

    # Get data as list(list(token))
    lines = handle_tokenizer(lines, tokenize_type)

    # Add <SOS> / <EOS>
    lines = handle_sos_eos(lines, add_start, add_end)

    # Setup fasttext model from path
    if fasttext_model is not None and isinstance(fasttext_model, str):
        fasttext_model = gensim.models.FastText.load(fasttext_model)

    # Encode text data using v2id
    lines = handle_encoding_v2id(lines, v2id, fasttext_model, threshold)

    # Padd Padd Padd ..
    lines = handle_padding(lines, padding, max_seq, v2id)

    # Save v2id
    if save_v2id_path is not None:
        save_json(data=v2id, path=save_v2id_path)

    return id2v, v2id, lines


def postprocessing(
        dec_data,
        dec_v2id,
        dec_id2v=None,
        output=None,
        tokenize_type='w',
        fasttext_model=None,
        enc_data=None,
        add_start=True,
        add_end=True,
        remove_punctuation=True,
        lower=True,
        enc_v2id=None,
        Print=False,
        CAP=False,
        NUM=False,
        ALNUM=False,
        UPPER=False,
        threshold=0.5):
    """
    Decodes given integer token lists to text corpus using given
    v2id or id2v mapping.

    Arguments:
        dec_data: List(list(tokens)) or list(tokens) or np.array(shape=(batch_size,))
            Tokens are in int format.
        dec_v2id: string or dict, Mapping of vocabulary to integer encoding.
        dec_id2v: string or dict. defalt None. Mapping of int to vocab encoding
        output: string or None, default is None. If None, prints the decoded
            sentence list. Else saves at given output path.
        token_type: 'w' or 'c', default is 'w'. 'w' represents word tokens & 'c'
            represents char tokens. This is for encoder.
        fasttext_model: str, default None. Fasttext Model path. For encoder
        enc_data: string or list. Path of input text file when string and
            data (batch of sentences expected) itself when List(list).
        add_start: boolean, default True.
            If True, adds start-of-sequence (<SOS>). For encoder
        add_end: boolean, default True.
            If True, adds end-of-sequence (<EOS>) tokens. For encoder
        remove_punctuation: boolean, default True.
            If True, removes punctuation symbols. For encoder.
        lower: boolean, default True. To condition case folding. For encoder.
        enc_v2id: string or dict, Mapping of vocabulary to integer encoding.
        CAP: boolean, default False. To condition special token <CAP>. For enc.
        NUM: boolean, default False. To condition special token <NUM>. For enc.
        ALNUM: boolean, default False. Condition special token <ALNUM>. For enc.
        UPPER: boolean, default False. Condition special token <UPPER>. For enc.
        threshold: float. default is 0.5.
            To threshold fasttext similarity indexing. For encoder.

    Returns:
        dec_data: List(sentences) of shape (batch_size). Decoded sentences.
    """

    # Get given dec_data in 2D numpy array format
    dec_data = checkout_predictions(dec_data)

    # Get v2id and id2v dicitionary
    if isinstance(enc_v2id, str):
        enc_v2id = load_json(enc_v2id)
    if isinstance(dec_v2id, str):
        dec_v2id = load_json(dec_v2id)
    if dec_id2v is None:
        dec_id2v = swap_dict_key_value(dec_v2id)

    # Now dec_data is in format list(List)
    dec_data = clear_after_eos(dec_data, dec_v2id)

    # Clean list from padding elements
    dec_data = remove_padding(dec_data, dec_v2id)

    # Clean list from sos and eos elements
    dec_data = remove_sos_eos(dec_data, dec_v2id)

    # Decode integer tokens to words/chars
    dec_data = decode_token_2_words(dec_data, dec_id2v, tokenize_type)

    # Clean list of words from <CAP> elements
    if CAP:
        dec_data = remove_cap(dec_data)

    # Clean list of words from <UPPER> elements
    if UPPER:
        dec_data = remove_upper(dec_data)

    # Setup fasttext model from path
    if fasttext_model is not None and isinstance(fasttext_model, str):
        fasttext_model = gensim.models.FastText.load(fasttext_model)

    # Replacing <UNK> by mapping <UNK> created at input to encoder
    if fasttext_model is not None and tokenize_type == 'w':

        # Get enc_data in format List(list(int)) exactly as input to enc
        _, enc_data_int = preprocess_v2id(data=copy(enc_data),
                                          v2id=enc_v2id,
                                          fasttext_model=fasttext_model,
                                          add_start=add_start,
                                          add_end=add_end,
                                          remove_punctuation=remove_punctuation,
                                          lower=lower,
                                          threshold=threshold,
                                          CAP=False,
                                          NUM=NUM,
                                          ALNUM=ALNUM,
                                          UPPER=False,
                                          tokenize_type=tokenize_type,
                                          post_process_usage=False)

        # Get enc_data in format List(list(words))
        enc_data_words = preprocess_v2id(data=copy(enc_data),
                                         v2id=enc_v2id,
                                         fasttext_model=fasttext_model,
                                         add_start=add_start,
                                         add_end=add_end,
                                         remove_punctuation=remove_punctuation,
                                         lower=lower,
                                         threshold=threshold,
                                         tokenize_type=tokenize_type,
                                         post_process_usage=True,
                                         CAP=False,
                                         NUM=False,
                                         ALNUM=False,
                                         UPPER=False)

        # Map predicted <UNK> with word in input
        dec_data = deal_with_special_token(
            dec_data, enc_data_int, enc_v2id, enc_data_words, "<UNK>")
        # Map predicted <NUM> with word in input
        if NUM:
            dec_data = deal_with_special_token(
                dec_data, enc_data_int, enc_v2id, enc_data_words, "<NUM>")
        # Map predicted <ALNUM> with word in input
        if ALNUM:
            dec_data = deal_with_special_token(
                dec_data, enc_data_int, enc_v2id, enc_data_words, "<ALNUM>")

    # Get tokens as part of sentence from
    # dec_data(List(list(tokens))->list(sentence))
    dec_data = [" ".join(line) for line in dec_data]

    if Print:
        print("\nPredictions are as follows:\n")
        _ = [print(f"({i+1})", D) for i, D in enumerate(dec_data)]
    if output is not None:
        write_file(dec_data, output)
        print("Predictions written to file at {}".format(output))
    if output is None:
        return dec_data


def oversample(data_1, data_2, n):
    """
    Randomly samples from a pair of aligned datasets with replacement. Implementation of increasing sampling by sampling

    Arguments:
        data_1: string or list of sentences. First dataset.
        data_2: string or list of sentences. Second dataset.
        n: integer or float. If integer: number of samples. If float: number of samples relative to input dataset size.

    Returns:
        Two lists of sampled sentences.
    """

    # Read lines from datasets
    lines_1 = checkout_data(data_1)
    lines_2 = checkout_data(data_2)
    assert len(lines_1) == len(
        lines_2), 'Oversampled datasets should contain the same number of sentences.'

    # Initialize variables
    samples_1 = []
    samples_2 = []
    length = len(lines_1)

    # If n is an integer, randomly sample until n is reached
    if isinstance(n, int):
        for _ in range(n):
            sample = np.random.randint(length)
            samples_1.append(lines_1[sample])
            samples_2.append(lines_2[sample])

    # If n is a float, randomly sample until len(lines_1)*n is reached
    if isinstance(n, float):
        while len(samples_1) < len(lines_1) * n:
            sample = np.random.randint(length)
            samples_1.append(lines_1[sample])
            samples_2.append(lines_2[sample])

    return samples_1, samples_2


# DATA UTILS:

def handle_regex(lines, CAP, NUM, ALNUM, UPPER):
    # Uses regex to find uppercase, capitalized, numeric, and alphanumeric
    # tokens
    uppercase = re.compile(r'^[À-ÜA-Z]+[À-ÜA-Z]*$')
    capitalized = re.compile(r'^[À-ÜA-Z]')
    numeric = re.compile(r'^[0-9]*$')
    alphanumeric = re.compile(r'.*[0-9].*')

    new_lines = [[] for _ in lines]

    for idx, line in enumerate(lines):
        line = line.split()
        for word in line:
            if numeric.match(word):
                if NUM:
                    new_lines[idx].append('<NUM>')
            elif alphanumeric.match(word):
                if ALNUM:
                    new_lines[idx].append('<ALNUM>')
            elif len(word) > 1 and uppercase.match(word):
                if UPPER:
                    new_lines[idx].extend(['<UPPER>', word.lower()])
            elif len(word) > 1 and capitalized.match(word):
                if CAP:
                    new_lines[idx].extend(['<CAP>', word.lower()])
            else:
                new_lines[idx].append(word)
        new_lines[idx] = ' '.join(new_lines[idx])

    return new_lines


def handle_punctuation(lines):
    # Symbols to be removed, from punctuation_remover.py
    PUNCTUATION = {",", ";", ":", "!", "?", ".", "'",
                   '"', "(", ")", "...", "[", "]", "{", "}", "’"}
    # Remove punctuation
    split_lines = [[x for x in line.split() if x not in PUNCTUATION]
                   for line in lines]
    lines = [' '.join(line) for line in split_lines]
    return lines


def checkout_predictions(data):
    # Make any format data to numpy
    data = np.array(data)
    data = np.reshape(data, (data.shape[0], -1))
    return data


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
    if vocab_size is not None:
        vocab_size -= 8
    vocab = [('<PAD>', 0), ('<SOS>', 0), ('<EOS>', 0), ('<UNK>', 0), ('<NUM>', 0),
             ('<ALNUM>', 0), ('<CAP>', 0), ('<UPPER>', 0)] + Counter(corpus).most_common(vocab_size)
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


def handle_encoding_v2id(lines, v2id, fasttext_model=None, threshold=0.5):

    vocab2id = v2id.copy()

    if fasttext_model is not None:
        encoded_lines = [[handle_oov(l, vocab2id, fasttext_model, 10, threshold)
                          for l in line] for line in tqdm(lines)]
    else:
        encoded_lines = [[vocab2id.setdefault(l, vocab2id['<UNK>'])
                          for l in line] for line in tqdm(lines)]
    return encoded_lines


def handle_oov(token, v2id, fasttext_model, num_similar=10, threshold=0.5):
    if token in v2id:
        return v2id[token]
    try:
        similar_words = fasttext_model.wv.similar_by_word(token, num_similar)
    except KeyError:
        num_similar = 0
    for i in range(num_similar):
        w = similar_words[i]
        if w[1] < threshold:
            return v2id['<UNK>']
        if w[0] in v2id:
            return v2id[w[0]]
    return v2id['<UNK>']


def handle_padding(lines, padding, max_seq, v2id):
    encoded_lines = pad_sequences(
        lines, maxlen=max_seq, padding=padding, value=v2id['<PAD>'])
    return encoded_lines


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def decode_token_2_words(data, id2v, tokenize_type):
    if tokenize_type == 'w':
        sep = " "
    elif tokenize_type == 'c':
        sep = ""
    data = [[id2v[token] for token in line] for line in data]
    data = [sep.join(line) for line in data]
    return data


def deal_with_special_token(
        dec_data,
        enc_data_int,
        enc_v2id,
        enc_data_words,
        token):
    token_id = enc_v2id[token]
    MAP = []
    for i, line in enumerate(enc_data_int):
        MAPLine = []
        for j, word in enumerate(line):
            if word == token_id:
                MAPLine.append(enc_data_words[i][j])
        MAP.append(MAPLine)

    Lines = []
    for i, line in enumerate(dec_data):
        Line = []
        for j, word in enumerate(line):
            if word == token:
                try:
                    word = MAP[i][j]
                    Line.append(word)
                except IndexError:
                    pass
            else:
                Line.append(word)
        Lines.append(Line)

    return Lines


def write_file(data, output):
    Lines = ""
    for line in tqdm(data):
        Lines += line + "\n"
    with open(output, 'w') as stream:
        stream.write(Lines)


def clear_after_eos(Data, v2id):
    eos = v2id['<EOS>']
    data_without_eos = [[d for d in data[:None if eos not in data else np.where(
        data == eos)[0][0]]] for data in Data]
    return data_without_eos


def remove_padding(Data, v2id):
    pad = v2id['<PAD>']
    data_without_pad = [[d for d in data if d != pad] for data in Data]
    return data_without_pad


def remove_sos_eos(Data, v2id):
    sos = v2id['<SOS>']
    eos = v2id['<EOS>']
    data_without_sos_eos = [[d for d in data if (
        d != sos) and (d != eos)] for data in Data]
    return data_without_sos_eos


def remove_cap(dec_data):
    cap = "<CAP>"
    Lines = []
    for line in dec_data:
        Line = []
        line = line.split(" ")
        for i, word in enumerate(line):
            if word == cap:
                try:
                    line[i + 1] = line[i + 1][0].upper() + line[i + 1][1:]
                except IndexError:
                    pass
            else:
                Line.append(word)
        Lines.append(Line)
    return Lines


def remove_upper(dec_data):
    UP = "<UPPER>"
    Lines = []
    for line in dec_data:
        Line = []
        for i, word in enumerate(line):
            if word == UP:
                try:
                    line[i + 1] = line[i + 1].upper()
                except IndexError:
                    pass
            else:
                Line.append(word)
        Lines.append(Line)
    return Lines
