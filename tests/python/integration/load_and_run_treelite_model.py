from __future__ import print_function
from dlr import DLRModel
import numpy as np
import os
from test_utils import get_arch, get_models
from sklearn.datasets import load_svmlight_file

def todense(csr_matrix):
    out = np.full(shape=csr_matrix.shape, fill_value=np.nan, dtype=np.float32)
    rowind = np.repeat(np.arange(csr_matrix.shape[0]), np.diff(csr_matrix.indptr))
    out[rowind, csr_matrix.indices] = csr_matrix.data
    return out

def test_mnist():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xgboost-mnist')
    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xgboost', 'mnist.libsvm')
    model = DLRModel(model_path, 'cpu', 0)

    X, _ = load_svmlight_file(data_file, zero_based=True)
    print('Testing inference on XGBoost MNIST...')
    assert model.run(_sparse_to_dense(X))[0] == 7.0

def test_iris():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xgboost-iris')
    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xgboost', 'iris.libsvm')
    model = DLRModel(model_path, 'cpu', 0)

    X, _ = load_svmlight_file(data_file, zero_based=True)
    expected = np.array([2.159504452720284462e-03, 9.946205615997314453e-01, 3.219985403120517731e-03])
    print('Testing inference on XGBoost Iris...')
    assert np.allclose(model.run(_sparse_to_dense(X))[0], expected)

def test_letor():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xgboost-letor')
    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xgboost', 'letor.libsvm')
    model = DLRModel(model_path, 'cpu', 0)

    X, _ = load_svmlight_file(data_file, zero_based=True)
    expected = np.array([1.372033834457397461e+00, -2.448803186416625977e+00, 8.579480648040771484e-01,
                         1.369985580444335938e+00, -7.058695554733276367e-01, 4.134958684444427490e-01,
                         -2.247941017150878906e+00, -2.461995363235473633e+00, -2.394921064376831055e+00,
                         -1.191793322563171387e+00, 9.672126173973083496e-02, 2.687671184539794922e-01,
                         1.417675256729125977e+00, -1.832636356353759766e+00, -5.582004785537719727e-02,
                         -9.497703313827514648e-01, -1.219825387001037598e+00, 1.512521862983703613e+00,
                         -1.179921030998229980e-01, -2.383430719375610352e+00, -9.094548225402832031e-01])
    print('Testing inference on XGBoost LETOR...')
    assert np.allclose(model.run(_sparse_to_dense(X))[0], expected)


if __name__ == '__main__':
    arch = get_arch()
    model_names = ['xgboost-mnist', 'xgboost-iris', 'xgboost-letor']
    for model_name in model_names:
        get_models(model_name, arch, kind='treelite')
    test_mnist()
    test_iris()
    test_letor()
    print('All tests passed!')
