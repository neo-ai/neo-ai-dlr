import dlr

def test_mnist():
    model = dlr.DLRModel('./model', 'cpu', 0)
