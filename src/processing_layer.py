from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np
from tqdm import tqdm

def processing(data, max_seq=None, vocab_size=None, add_start=True, add_end=True, remove_punctuation=True, tokenize_type="w", padding='post'):
    """
    Gets tokenized and integer encoded format of given text corpus.

    Arguments:
        data: string or list. Path of input text file when string and data (batch of sentences expected) itself when List(list).
        max_seq: int or Nonr, default None. Sequence length to retain.
        vocab_size: int or None, default None. Number of words to retain, sorted by most common. If None, retain all words.
        add_start: boolean, default True. If True, adds start-of-sequence (<SOS>) 
        add_end: boolean, default True. If True, adds end-of-sequence (<EOS>) tokens.
        remove_punctuation: boolean, default True. If True, removes punctuation symbols.
        tokenize_type: "w" or "c", default "w". Defines token type as word(w) or char(c).
        padding: 'post' or 'pre', default 'post'. Defines where to pad, begining('pre') or end('post').
    
    Returns:
        Dictionaries ``v2id`` and ``id2v`` for encoding and decoding vocabulary as id
        Numpy array ``encoded_lines`` of shape (batch_size, max_seq)
    """

    # Symbols to be removed, from punctuation_remover.py
    PUNCTUATION = {",", ";", ":", "!", "?", ".", "'", '"', "(", ")", "...", "[", "]", "{", "}", "’"}


    if isinstance(data, str):
        # Read lines from file
        with open(data) as f:
            lines = f.readlines()
    elif isinstance(data, list):
        lines = data
    
    # Strip newline
    lines = [line.strip() for line in lines]

    # Remove punctuation
    if remove_punctuation:
        split_lines = [[x for x in line.split() if x not in PUNCTUATION] for line in lines]
        lines = [' '.join(line) for line in split_lines]

    # Get corpus tokens
    if tokenize_type == "w":
        corpus = ' '.join(lines).split()
    if tokenize_type == "c":
        corpus = list(' '.join(lines))
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
    encoded_lines = [list(map(v2id.get,line)) for line in tqdm(lines)]
    if padding == 'pre':
        padder = '<SOS>'
    elif padding == 'post':
        padder = '<EOS>'
    encoded_lines = pad_sequences(encoded_lines, maxlen=max_seq, padding=padding, value=v2id[padder])

    return id2v, v2id, encoded_lines
