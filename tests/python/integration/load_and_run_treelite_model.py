from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, log_loss
import dlr
import tarfile
import contextlib
import os
import numpy as np

def todense(csr_matrix):
  out = np.full(shape=csr_matrix.shape, fill_value=np.nan, dtype=np.float32)
  rowind = np.repeat(np.arange(csr_matrix.shape[0]), np.diff(csr_matrix.indptr))
  out[rowind, csr_matrix.indices] = csr_matrix.data
  return out

def load_tar(tar_path):
  query_dims = None
  with contextlib.closing(tarfile.open(tar_path, 'r:bz2')) as tar:
    for file_info in tar.getmembers():
      if file_info.name.endswith('libsvm'):
        f = tar.extractfile(file_info)
        X, y = load_svmlight_file(f, zero_based=True)
      elif file_info.name.endswith('group'):
        f = tar.extractfile(file_info)
        query_dims = []
        for l in f:
          query_dims.append(int(l))
        query_dims = np.array(query_dims)
  return todense(X), y, query_dims

def run_model(model_dir, X, y):
  model = dlr.DLRModel(model_dir)
  output = np.concatenate(tuple(model.run(X[i:i+1,:])[0] for i in range(X.shape[0]))).squeeze()
  return output

def ndcg(y_true, y_score, query_sessions):
  def calc_dcg(query):
    return np.sum((2**query - 1) / np.log2(np.arange(query.shape[0]) + 2))

  nquery = len(query_sessions)
  rowptr = np.concatenate(([0], np.cumsum(query_sessions)))
  sum_ndcg = 0.0
  for query_id in range(nquery):
    # Sort documents in the current query by ranking scores, in descending order
    # List relevance judgments in the same order as sorted documents
    query_score = y_score[rowptr[query_id]:rowptr[query_id+1]]
    query_relevance = y_true[rowptr[query_id]:rowptr[query_id+1]]
    query_relevance = query_relevance[(-query_score).argsort(kind='mergesort')]
    # Compute DCG
    dcg = calc_dcg(query_relevance)
    # Compute ideal DCG, with ideal scenario where documents are
    #   perfectly sorted by relevance judgments
    idcg = calc_dcg(-np.sort(-query_relevance))
    sum_ndcg += (dcg / idcg if idcg != 0 else 1)

  return sum_ndcg / nquery

def test_iris():
  model_path = os.path.join(os.getcwd(), 'iris_amd64')
  xgb_pred_path = os.path.join(model_path, 'iris_test.pred.gz')  # prediction from XGBoost
  data_path = os.path.join(model_path, 'iris_test.tar.bz2')

  X, y, _ = load_tar(data_path)
  output = run_model(model_path, X, y)
  assert accuracy_score(y, output.argmax(axis=1)) >= 0.95
  y_indicator = np.zeros((y.shape[0], 3))
  y_indicator[np.arange(y.shape[0]), y.astype(int)] = 1.0
  assert log_loss(y_indicator, output) < 0.01

  # Compare with XGBoost prediction
  assert np.allclose(output, np.loadtxt(xgb_pred_path, delimiter=','))

def test_letor():
  model_path = os.path.join(os.getcwd(), 'letor_amd64')
  xgb_pred_path = os.path.join(model_path, 'letor_test.pred.gz')  # prediction from XGBoost
  data_path = os.path.join(model_path, 'letor_test.tar.bz2')

  X, y, query_dims = load_tar(data_path)
  output = run_model(model_path, X, y)
  assert ndcg(y, output, query_dims) >= 0.83

  # Compare with XGBoost prediction
  assert np.allclose(output, np.loadtxt(xgb_pred_path, delimiter=','))

def main():
  test_iris()
  test_letor()
  print "All tests passed!"

if __name__ == '__main__':
  main()
