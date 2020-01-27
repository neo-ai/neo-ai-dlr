import io
import sys
import pytest

from dlr import DLRModel
import numpy as np
import os
from test_utils import get_arch, get_models


# def test_notification(capsys):
#     model_path = None
#     device = None
#
#     with pytest.raises(Exception) as e:
#         DLRModel(model_path, device)
#
#     captured = capsys.readouterr()
#     print(captured)
#     assert captured.out == ''


def test_notification(capsys):
    # setup purpose
    setup_mock_dlr()

    # integration model mirror load_and_run_tvm_model.py
    # load the model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resnet18_v1')
    classes = 1000
    device = 'cpu'
    model = DLRModel(model_path, device)

    # test the notification capture
    captured = capsys.readouterr()
    print(captured)
    assert captured.out is not ''

    # run the model
    image = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dog.npy')).astype(np.float32)
    # flatten within a input array
    input_data = {'data': image}
    print('Testing inference on resnet18...')
    probabilities = model.run(input_data)  # need to be a list of input arrays matching input names

    assert probabilities[0].argmax() == 151


@pytest.fixture(scope="module")
def setup_mock_dlr():
    print("set up mock dlr")
    
    arch = get_arch()
    model_names = ['resnet18_v1']
    for model_name in model_names:
        get_models(model_name, arch, kind='tvm')

    # test_notification()
    # print('All tests passed!')
