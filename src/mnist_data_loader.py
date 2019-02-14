import cPickle
import gzip
import numpy as np

def load_data():
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    raw_tr_d, raw_va_d, raw_te_d = cPickle.load(f)
    f.close()
    tr_d_inp = [np.reshape(x, (784, 1)) for x in raw_tr_d[0]]
    exp_tr_d_out = [__vectorize_result__(y) for y in raw_tr_d[1]]
    tr_d = zip(tr_d_inp, exp_tr_d_out)
    va_d_inp = [np.reshape(x, (784, 1)) for x in raw_va_d[0]]
    va_d = zip(va_d_inp, raw_va_d[1])
    te_d_inp = [np.reshape(x, (784, 1)) for x in raw_te_d[0]]
    te_d = zip(te_d_inp, raw_te_d[1])
    return tr_d, va_d, te_d

def __vectorize_result__(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
