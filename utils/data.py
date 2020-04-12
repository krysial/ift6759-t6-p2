from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np
import json
from tqdm import tqdm


def preprocess_v2id(data, v2id, fasttext_model=None, max_seq=None, add_start=True,
                    add_end=True, remove_punctuation=True, lower=True,
                    tokenize_type="w", padding='post', post_process_usage=False):
    """
    Gets tokenized and integer encoded format of given
    text corpus using given v2id mapping

    Arguments:
        data: string or list. Path of input text file when string and
            data (batch of sentences expected) itself when List(list).
        v2id: string or dict, Mapping of vocabulary to integer encoding.
        fasttext_model: str, default None. Fasttext Model reference
        max_seq: int, default None. Sequence length to retain.
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

    # Encode text data using v2id
    lines = handle_encoding_v2id(lines, v2id, fasttext_model)

    # Padd Padd Padd ..
    lines = handle_padding(lines, padding, max_seq, v2id)

    return v2id, lines


def preprocessing(data, max_seq=None, vocab_size=None, add_start=True,
                  add_end=True, remove_punctuation=True, lower=True,
                  tokenize_type="w", padding='post', save_v2id_path=None):
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
        tokenize_type: "w" or "c", default "w".
            Defines token type as word(w) or char(c).
        padding: 'post' or 'pre', default 'post'.
            Defines where to pad, begining('pre') or end('post').
        save_v2id_path: string, default is None.
            Path of saving v2id dictionary mapping.
        post_process_usage: Bool, default is False.
            Condition allowing usage of function during post_processing.

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


def postprocessing(dec_data, dec_v2id, dec_id2v=None, output=None, tokenize_type='w',
                   fasttext_model=None, enc_data=None, add_start=True, add_end=True,
                   remove_punctuation=True, lower=True, enc_v2id=None):
    """
    Decodes given integer token lists to text corpus using given 
    v2id or id2v mapping.

    Arguments:
        dec_data: List(list(tokens)) or list(tokens) or np.array(shape=(batch_size,)) 
            Tokens are in int format.
        dec_v2id: string or dict, Mapping of vocabulary to integer encoding.
        output: string or None, default is None. If None, prints the decoded 
            sentence list. Else saves at given output path.
        token_type: 'w' or 'c', default is 'w'. 'w' represents word tokens & 'c'
            represents char tokens.
        fasttext_model: str, default None. Fasttext Model reference
        enc_data: string or list. Path of input text file when string and
            data (batch of sentences expected) itself when List(list).
        add_start: boolean, default True.
            If True, adds start-of-sequence (<SOS>)
        add_end: boolean, default True.
            If True, adds end-of-sequence (<EOS>) tokens.
        remove_punctuation: boolean, default True.
            If True, removes punctuation symbols.
        lower: boolean, default True. To condition case folding.
        enc_v2id: string or dict, Mapping of vocabulary to integer encoding.

    Returns:
        dec_data: List(sentences) of shape (batch_size). Decoded sentences.
    """

    # Get given dec_data in 2D numpy array format
    dec_data = chekout_predictions(dec_data)

    # Get v2id dicitionary
    if isinstance(v2id, str):
        v2id = load_json(v2id)

    # Now dec_data is in format list(List)
    dec_data = clear_after_eos(dec_data, dec_v2id)

    # Clean list from padding elements
    dec_data = remove_padding(dec_data, dec_v2id)

    # Decode integer tokens to words/chars
    dec_data = decode_token_2_words(dec_data, id2v, tokenize_type)

    # Replacing <UNK> by mapping <UNK> created at input to encoder
    if fasttext_model is not None and tokenize_type == 'w':
        # Get enc_data in format List(list(words))
        enc_data = preprocessing(data=enc_data,
                                 v2id=enc_v2id,
                                 add_start=add_start,
                                 add_end=add_end,
                                 remove_punctuation=remove_punctuation,
                                 lower=lower,
                                 tokenize_type=tokenize_type,
                                 post_process_usage=True)
        # Map predicted <UNK> with word in input
        dec_data = deal_with_UNK_pred(dec_data, enc_data, enc_v2id, dec_v2id,
                                      fasttext_model)

    # Get tokens as
    if output is not None:
        write_file(dec_data, output)
        print("Predictions written to file at {}".format(output))
    else:
        print("\nPredictions are as follows:\n")
        _ = [print(f"({i+1})", D) for i, D in enumerate(dec_data)]

Implementation for increase in data sampling
def oversample(data_1, data_2, n):
    """
    Randomly samples from a pair of aligned datasets with replacement.

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
    assert len(lines_1) == len(lines_2), 'Oversampled datasets should contain the same number of sentences.'

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
    vocab = [('<PAD>', 0)] + Counter(corpus).most_common(vocab_size) + \
        [('<SOS>', 0), ('<EOS>', 0), ('<UNK>', 0)]
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


def handle_encoding_v2id(lines, v2id, fasttext_model=None):
    if fasttext_model is not None:
        encoded_lines = [[handle_oov(l, v2id, fasttext_model)
                          for l in line] for line in tqdm(lines)]
    else:
        encoded_lines = [[v2id.setdefault(l, v2id['<UNK>'])
                          for l in line] for line in tqdm(lines)]
    return encoded_lines


def handle_oov(token, v2id, fasttext_model, num_similar=10, threshold=0.5):
    if token in v2id:
        return v2id[token]
    similar_words = fasttext_model.wv.similar_by_word(token, num_similar)
    for i in range(num_similar):
        w = similar_words[i]
        if w[1] < threshold:
            return v2id['<UNK>']
        if w[0] in v2id:
            return v2id[w[0]]
    return v2id['<UNK>']


def handle_padding(lines, padding, max_seq, v2id):
    encoded_lines = pad_sequences(lines, maxlen=max_seq, padding=padding, value=v2id['<PAD>'])
    return encoded_lines


def save_v2id(v2id, path):
    with open(path, 'w') as f:
        json.dump(v2id, f)
    print("\n# v2id dictionary saved at: {}".format(path))


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


def deal_with_UNK_pred(dec_data, enc_data, enc_v2id, dec_v2id, fasttext_model):
    """
    dec_data: List(list(words))
    enc_data: List(list(words))
    """
    # Get id of <UNK> in both encoder and decoder ends
    enc_UNK = enc_v2id['<UNK>']
    dec_UNK = dec_v2id['<UNK>']
    return dec_data


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
