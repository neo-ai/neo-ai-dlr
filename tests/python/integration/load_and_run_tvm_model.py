from __future__ import print_function
from dlr import DLRModel
import numpy as np
import os
import platform
import logging
import urllib

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


def get_arch():
    machine_type = platform.machine()
    if machine_type == 'x86_64':
        return 'x86_64'
    elif machine_type == 'aarch64':
        return 'ec2_a1'
    elif machine_type == 'armv7l':
        return 'rasp3b'
    else:
        raise ValueError('Unsupported platform, please supply matching model')

def get_models(model_name, arch):
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            model_name)
    if not os.path.exists(model_path):
        try:  
            os.mkdir(model_path)
        except OSError:  
            raise ValueError("Creation of the directory %s failed" % path)
    
    s3_bucket = 'https://s3-us-west-2.amazonaws.com/neo-ai-dlr-test-artifacts'
    artifact_extensions = ['.json', '.params', '.so']
    print("Preparing model artifacts for %s ..." % model_name)
    for extension in artifact_extensions:
        s3_path = s3_bucket + '/' + model_name + '/' + arch + extension
        local_path = os.path.join(model_path, model_name + '_' + arch + extension)
        if not os.path.exists(local_path):
            try:
                urlretrieve(s3_path, local_path) 
            except urllib.error.URLError or urllib.error.HTTPError:
                raise ValueError('Downloading of model artifacts from %s failed' % s3_path)
                
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


def test_multi_input_multi_output():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '4in2out')
    device = 'cpu'
    model = DLRModel(model_path, device)

    assert model._get_output_size_dim(0) == (2, 1)
    assert model._get_output_size_dim(1) == (3, 1)

    input1 = np.asarray([1., 2.])
    input2 = np.asarray([3., 4.])
    input3 = np.asarray([5., 6., 7])
    input4 = np.asarray([8., 9., 10])
    input_map = {'data1': input1, 'data2': input2, 'data3': input3, 'data4': input4}
    print('Testing multi_input/multi_output support...')
    outputs = model.run(input_map)

    assert outputs[0].tolist() == [4, 6]
    assert outputs[1].tolist() == [13, 15, 17]


if __name__ == '__main__':
    logging.basicConfig(filename='test-dlr.log',level=logging.INFO)
    arch = get_arch()
    model_names = ['resnet18_v1', '4in2out']
    for model_name in model_names:
        get_models(model_name, arch)
    test_resnet()
    test_multi_input_multi_output()
    print('All tests passed!')
