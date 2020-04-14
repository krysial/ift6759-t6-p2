import os
import numpy as np
import tensorflow as tf
import random


def SEED(S=123):
    os.environ['PYTHONHASHSEED'] = str(S)
    random.seed(S)
    np.random.seed(S)
    tf.random.set_seed(S)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
