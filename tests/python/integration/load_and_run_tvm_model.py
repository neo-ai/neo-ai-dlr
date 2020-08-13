from __future__ import print_function
from dlr import DLRModel
import numpy as np
import os
from test_utils import get_arch, get_models

def test_resnet():
    # Load the model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            'resnet18_v1')
    device = 'cpu'
    model = DLRModel(model_path, device)

    # Run the model
    image = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dog.npy')).astype(np.float32)
    #flatten within a input array
    input_data = {'data': image}
    print('Testing inference on resnet18...')
    probabilities = model.run(input_data) #need to be a list of input arrays matching input names
    assert probabilities[0].argmax() == 151
    assert model.get_input_names() == ["data"]
    assert model.get_input_dtypes() == ["float32"]
    assert model.get_output_dtypes() == ["float32"]
    assert model.get_input_dtype(0) == "float32"
    assert model.get_output_dtype(0) == "float32"


def test_mobilenet_v1_0_75_224_quant():
    # Load the model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'mobilenet_v1_0.75_224_quant')
    device = 'cpu'
    model = DLRModel(model_path, device)
    # load image (dtype: uint8)
    image = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cat_224_uint8.npy'))
    print('Testing inference on mobilenet_v1_0.75_224_quant...')
    input_data = {'input': image}
    probabilities = model.run(input_data)
    assert probabilities[0].argmax() == 282
    assert model.get_input_names() == ["input"]
    assert model.get_input_dtypes() == ["uint8"]
    assert model.get_output_dtypes() == ["uint8"]
    assert model.get_input_dtype(0) == "uint8"
    assert model.get_output_dtype(0) == "uint8"
    input2 = model.get_input("input")
    assert input2.dtype == 'uint8'
    assert input2.shape == (1, 224, 224, 3)
    assert (input2 == image).all()


def test_mobilenet_v1_0_75_224_quant_wrong_input_type():
    # Load the model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'mobilenet_v1_0.75_224_quant')
    device = 'cpu'
    model = DLRModel(model_path, device)
    # load image (dtype: float32)
    image = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dog.npy'))
    print('Testing inference on mobilenet_v1_0.75_224_quant with float32 input...')
    try:
        model.run({'input': image})
        assert False, "ValueError is expected"
    except ValueError as e:
        assert str(e) == "input data with name input should have dtype uint8"


def test_multi_input_multi_output():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '4in2out')
    device = 'cpu'
    model = DLRModel(model_path, device)

    assert (model.get_output_size(0), model.get_output_dim(0)) == (2, 1)
    assert (model.get_output_size(1), model.get_output_dim(1)) == (3, 1)
    import pdb; pdb.set_trace();
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
    model_names = ['resnet18_v1', 'mobilenet_v1_0.75_224_quant', '4in2out', 'assign_op']
    for model_name in model_names:
        get_models(model_name, arch, kind='tvm')
    test_resnet()
    test_mobilenet_v1_0_75_224_quant()
    test_mobilenet_v1_0_75_224_quant_wrong_input_type()
    # test_multi_input_multi_output()
    # test_assign_op()
    print('All tests passed!')
