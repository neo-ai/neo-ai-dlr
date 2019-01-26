from dlr import DLRModel
import numpy as np
import os

def test_resnet():
    # Load the model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            'resnet18')
    classes = 1000
    device = 'cpu'
    model = DLRModel(model_path, device)

    # Run the model
    image = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dog.npy')).astype(np.float32)
    #flatten within a input array
    input_data = {'data': image}
    probabilities = model.run(input_data) #need to be a list of input arrays matching input names
    assert probabilities[0].argmax() == 111

def test_multi_input():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '2in1out')
    device = 'cpu'
    model = DLRModel(model_path, device)

    input1 = np.asarray([1., 2.])
    input2 = np.asarray([3., 4.])
    input_map = {'data1': input1, 'data2': input2}
    outputs = model.run(input_map)
    assert outputs[0].tolist() == [4, 6]

def test_multi_input_multi_output():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '4in2out')
    device = 'cpu'
    model = DLRModel(model_path, device)

    input1 = np.asarray([1., 2.])
    input2 = np.asarray([3., 4.])
    input3 = np.asarray([5., 6.])
    input4 = np.asarray([7., 8.])
    input_map = {'data1': input1, 'data2': input2, 'data3': input3, 'data4': input4}
    outputs = model.run(input_map)

    assert outputs[0].tolist() == [4, 6]
    assert outputs[1].tolist() == [12, 14]


if __name__ == '__main__':
    test_resnet()
    test_multi_input()
    test_multi_input_multi_output()
    print "All tests passed!"
