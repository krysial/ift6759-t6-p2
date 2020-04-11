import numpy as np
from gensim import utils
import gensim.models

from utils.data import preprocessing


# Iterator that yields sentences given a text file
# Adapted from: https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
class MyCorpus(object):
    """An iterator that yields sentences (lists of str)."""
    def __init__(self, data_path):
        self.data_path = data_path

    def __iter__(self):
        for line in open(self.data_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


def create_model(algorithm, data_path, save_path=None, **kwargs):
    """Creates a gensim model from the given text file."""
    sentences = MyCorpus(data_path)
    model = algorithm(sentences=sentences, **kwargs)
    if save_path is not None:
        model.save(save_path)
    return model


def create_matrix(id2v, wv, oov=None):
    """
    Creates an embedding matrix for the given vocabulary.

    Arguments:
        id2v: dictionary. Id-to-vocabulary dictionary created by processing.py.
        wv: KeyedVectors object. Word embeddings from a gensim model.
        oov: function. Takes no input and outputs a vector of length (embedding_size). 
            Handles out-of-vocabulary (OOV) tokens. If None, OOV embeddings are zero vectors.
    
    Returns:
        A NumPy array of shape (vocabulary_size, embedding_size).
    """

    # Get vocabulary size and embedding size
    vocab_size = len(id2v)
    emb_size = len(wv[id2v[0]])

    # Initialize embedding matrix
    emb_mat = np.zeros((vocab_size, emb_size))

    # Iterate through tokens in vocabulary
    for idx, word in id2v.items():
        try:
            # Get token embedding if possible
            emb_mat[idx] = wv[word]
        except:
            # If token is OOV, use OOV function
            if oov is not None:
                emb_mat[idx] = oov()
            # If no OOV function, leave embedding as zero vector
            else:
                pass

    return emb_mat


def load_and_create(model_path, data, oov=None, algorithm=gensim.models.FastText, **kwargs):
    """
    Loads a model from disk and creates an embedding matrix from the given data.

    Arguments:
        model_path: string. Path to saved model file.
        data: string or dict. Path to vocabulary text file or id-to-vocabulary dictionary.
        oov: function. Takes no input and outputs a vector of length (embedding_size).
            Handles out-of-vocabulary (OOV) tokens. If None, OOV embeddings are zero vectors.
        algorithm: gensim model used, e.g. Word2Vec or FastText.
        **kwargs: keyword arguments passed to the preprocessing function.
    
    Returns:
        A NumPy array of shape (vocabulary_size, embedding_size).
    """
    
    model = algorithm.load(model_path)
    if isinstance(data, str):
        data, _, _ = preprocessing(data, **kwargs)
    return create_matrix(data, model.wv, oov)


# Example usage
if __name__ == '__main__':
    # Set embedding size
    emb_size = 8
    # Create function to handle out-of-vocabulary tokens
    oov = lambda:np.random.normal(0, 1, emb_size)
    # Create model from text file
    model = create_model(gensim.models.Word2Vec, 'data/train.lang1', size=emb_size)
    # Get id-to-vector dictionary
    id2v = {0:'the', 1:'be', 2:'to', 3:'of', 4:'and', 5:'out-of-vocabulary-token'}
    # Create embedding matrix
    emb_mat = create_matrix(id2v, model.wv, oov)

    print(emb_mat)
