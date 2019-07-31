from __future__ import print_function
from dlr import DLRModel
import numpy as np
import os
from test_utils import get_arch, get_models

def test_resnet():
    # Load the model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            'resnet18_v1')
    classes = 1000
    device = 'cpu'
    model = DLRModel(model_path, device)

    # Run the model
    image = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dog.npy')).astype(np.float32)
    #flatten within a input array
    input_data = {'data': image}
    print('Testing inference on resnet18...')
    probabilities = model.run(input_data) #need to be a list of input arrays matching input names
    assert probabilities[0].argmax() == 151

def test_heterogeneous_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            'hetero')
    model = DLRModel(model_path)

    shape = (4,)

    tensor_a = np.random.uniform(size=shape).astype(np.float32)
    tensor_b = np.random.uniform(size=shape).astype(np.float32)
    tensor_c = np.random.uniform(size=shape).astype(np.float32)
    tensor_d = np.random.uniform(size=shape).astype(np.float32)
    
    input_dict = {'A': tensor_a, 'B': tensor_b, 'C': tensor_c, 'D': tensor_d}
    print('Testing inference on heterogeneouse model...')
    output = model.run(input_dict)

    np.testing.assert_equal(
                output[0], tensor_a + tensor_b - tensor_c + tensor_d)

def test_multi_input_multi_output():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '4in2out')
    device = 'cpu'
    model = DLRModel(model_path, device)

    assert model._impl._get_output_size_dim(0) == (2, 1)
    assert model._impl._get_output_size_dim(1) == (3, 1)

    input1 = np.asarray([1., 2.])
    input2 = np.asarray([3., 4.])
    input3 = np.asarray([5., 6., 7])
    input4 = np.asarray([8., 9., 10])
    input_map = {'data1': input1, 'data2': input2, 'data3': input3, 'data4': input4}
    print('Testing multi_input/multi_output support...')
    outputs = model.run(input_map)

    assert outputs[0].tolist() == [4, 6]
    assert outputs[1].tolist() == [13, 15, 17]


def test_assign_op():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            'assign_op')
    device = 'cpu'
    model = DLRModel(model_path, device)

    print('Testing _assign() operator...')
    # Example from https://github.com/dmlc/tvm/blob/bb87f044099ba61ba4782d17dd9127b869936373/nnvm/tests/python/compiler/test_top_assign.py
    np.random.seed(seed=0)
    input1 = np.random.random(size=(5, 3, 18, 18))
    model.run({'w': input1})
    input1_next = model.get_input('w2', shape=(5, 3, 18, 18))
    assert np.allclose(input1_next, input1 + 2)

    model.run({})
    input1_next = model.get_input('w2', shape=(5, 3, 18, 18))
    assert np.allclose(input1_next, input1 + 3)

if __name__ == '__main__':
    arch = get_arch()
    model_names = ['resnet18_v1', '4in2out', 'assign_op']
    for model_name in model_names:
        get_models(model_name, arch, kind='tvm')
    test_resnet()
    test_multi_input_multi_output()
    test_assign_op()
    test_heterogeneous_model()
    print('All tests passed!')