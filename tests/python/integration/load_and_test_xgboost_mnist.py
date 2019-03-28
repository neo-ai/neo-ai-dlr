from sklearn.datasets import load_svmlight_file
import numpy as np
import dlr
import os

def _sparse_to_dense(csr_matrix):
    out = np.full(shape=csr_matrix.shape, fill_value=np.nan, dtype=np.float32)
    rowind = np.repeat(np.arange(csr_matrix.shape[0]), np.diff(csr_matrix.indptr))
    out[rowind, csr_matrix.indices] = csr_matrix.data
    return out

def test_mnist():
    model_dir = os.path.join(os.path.dirname(__file__), 'xgboost-mnist', 'model')
    data_file = os.path.join(os.path.dirname(__file__), 'xgboost-mnist', 'mnist.libsvm')
    model = dlr.DLRModel(model_dir, 'cpu', 0)

    X, _ = load_svmlight_file(data_file, zero_based=True)
    for _ in range(100):   # Should not crash
        assert model.run(_sparse_to_dense(X))[0] == 7.0
