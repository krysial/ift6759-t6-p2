from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np
from tqdm import tqdm

def processing(data_path, max_seq=None, vocab_size=None, add_start=True, add_end=True, remove_punctuation=True, tokenize_type="w", padding='post'):
    """
    Gets the samples and the vocabulary of a text file, with word counts.

    Arguments:
        data_path: string. Path of input text file.
        vocab_size: int or None, default None. Number of words to retain, sorted by most common. If None, retain all words.
        add_start_end: boolean, default True. If True, adds start-of-sequence (<SOS>) and end-of-sequence (<EOS>) tokens.
        remove_punctuation: boolean, default True. If True, removes punctuation symbols.
    
    Returns:
        Dictionaries v2id and id2v for encoding and decoding vocabulary as id
        Numpy array of shape (batch_size, max_seq)
    """

    # Symbols to be removed, from punctuation_remover.py
    PUNCTUATION = {",", ";", ":", "!", "?", ".", "'", '"', "(", ")", "...", "[", "]", "{", "}", "’"}

    # Read lines from file
    with open(data_path) as f:
        lines = f.readlines()
    
    # Strip newline
    lines = [line.strip() for line in lines]

    # Remove punctuation
    if remove_punctuation:
        split_lines = [[x for x in line.split() if x not in PUNCTUATION] for line in lines]
        lines = [' '.join(line) for line in split_lines]

    # Get corpus words
    corpus = ' '.join(lines).split()
    # Get most common words
    vocab = Counter(corpus).most_common(vocab_size) + [('<SOS>', 0), ('<EOS>', 0)]
    # Create id to vocabulary dictionary
    id2v = dict(pd.DataFrame(vocab, columns=['tokens', 'count'])['tokens'])
    v2id = dict([(value, key) for key, value in id2v.items()])

    # Tokenize input dataset
    if tokenize_type == "w":
        lines = [line.split() for line in lines]
    if tokenize_type == "c":
        lines = [list(line) for line in lines]

    # Add <SOS> / <EOS>
    if add_start:
        lines = [['<SOS>'] + line for line in lines]
    if add_end:
        lines = [line + ['<EOS>'] for line in lines]

    # Encode text data using v2id
    encoded_lines = [list(map(v2id.get,line)) for line in lines]
    if padding == 'pre':
        padder = '<SOS>'
    elif padding == 'post':
        padder = '<EOS>'
    encoded_lines = pad_sequences(encoded_lines, maxlen=max_seq, padding=padding, value=v2id[padder])

    return id2v, v2id, encoded_lines
