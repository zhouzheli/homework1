import pickle
from pathlib import Path
import numpy as np
import gzip
def data(path):
    with gzip.open(Path(path).as_posix(), 'rb') as f:
        ((train_X, train_y), (test_X, test_y), _) = pickle.load(f, encoding='latin-1')
    # n = train_y.shape[0]
    # train_y_vector = np.zeros((n, 10))
    # for i in range(n):
    #     train_y_vector[i][train_y[i]] = 1
    return train_X, train_y, test_X, test_y
def _accuracy(y_true, y_pred):
    return len(np.where(y_true==y_pred)[0]) / len(y_true)
def mini_batch(n, batch_size):
    batch_step = np.arange(0, n, batch_size)
    indices = np.arange(n, dtype=np.int64)
    np.random.shuffle(indices)
    batches = [indices[i: i + batch_size] for i in batch_step]
    return batches
