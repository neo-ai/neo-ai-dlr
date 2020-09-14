from __future__ import print_function
from dlr import DLRModel
import numpy as np
import os
from test_utils import get_arch, get_models
import time
import cProfile


def sample_data():
    batch_size = 1
    data_shape = (batch_size, 3, 224, 224)
    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
    return data


def load_test():
    # Load the model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'resnet18_v1')
    classes = 1000
    device = 'cpu'
    model = DLRModel(model_path, device)

    results = []
    for i in range(0, 100):
        # Run the model
        image = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dog.npy')).astype(np.float32)
        # flatten within a input array
        input_data = {'data': image}
        print('Testing inference on resnet18...')

        start = time.time()
        probabilities = model.run(input_data)  # need to be a list of input arrays matching input names
        end = time.time()
        results.append(1000 * (end - start))

        assert probabilities[0].argmax() == 151


    results = np.array(results)
    print(results.min(), results.mean(), results.max())


def set_up():
    arch = get_arch()
    model_names = ['resnet18_v1', '4in2out', 'assign_op']
    for model_name in model_names:
        get_models(model_name, arch, kind='tvm')

set_up()
cProfile.run('load_test()')
print("test done")