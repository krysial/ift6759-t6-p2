from collections import Counter

import pandas as pd


def get_vocabulary(data_path, vocab_size=None, add_start_end=True, remove_punctuation=True):
    """
    Gets the samples and the vocabulary of a text file, with word counts.

    Arguments:
        data_path: string. Path of input text file.
        vocab_size: int or None, default None. Number of words to retain, sorted by most common. If None, retain all words.
        add_start_end: boolean, default True. If True, adds start-of-sequence (<SOS>) and end-of-sequence (<EOS>) tokens.
        remove_punctuation: boolean, default True. If True, removes punctuation symbols.
    
    Returns:
        A pandas DataFrame "data" with integer indices and column ['sample'].
        A pandas DataFrame "vocab" with integer indices and columns ['word', 'count'].
    """

    # Symbols to be removed, from punctuation_remover.py
    PUNCTUATION = {",", ";", ":", "!", "?", ".", "'", '"', "(", ")", "...", "[", "]", "{", "}"}
    # Add apostrophe
    PUNCTUATION.add("’")

    # Read lines from file, strip newline, and add <SOS> / <EOS>
    with open(data_path) as f:
        lines = f.readlines()
        if add_start_end:
            lines = ['<SOS> ' + line.strip() + ' <EOS>' for line in lines]
        else:
            lines = [line.strip() for line in lines]
    
    # Remove punctuation
    if remove_punctuation:
        split_lines = [[x for x in line.split() if x not in PUNCTUATION] for line in lines]
        lines = [' '.join(line) for line in split_lines]

    # Get list of words
    corpus = ' '.join(lines).split()
    # Get most common words
    vocab = Counter(corpus).most_common(vocab_size)
    # Create vocabulary DataFrame
    vocab = pd.DataFrame(vocab, columns=['word', 'count'])
    # Create sentence DataDrame
    data = pd.DataFrame(lines, columns=['sample'])

    return data, vocab
