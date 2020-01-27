import numpy as np
import os

from test_utils import get_arch, get_models


def setup_mock_dlr():
    """setup function mirror load_and_run_tvm_model.py
    """
    print("set up dlr")

    arch = get_arch()
    model_names = ['resnet18_v1']
    for model_name in model_names:
        get_models(model_name, arch, kind='tvm')


def test_notification(capsys):
    """integration test for phone-home mechanism

    This mirrors load_and_run_tvm_model.py to simulate proper usage of the DLR.
    To run this, just pytest -s test_counter_mgr.py
    """
    from dlr import DLRModel
    from dlr.counter.config import CALL_HOME_USR_NOTIFICATION

    # test the notification capture
    captured = capsys.readouterr()
    print('captured output:', captured.out)
    assert captured.out is not ''
    assert captured.out.find(CALL_HOME_USR_NOTIFICATION) >= 0

    # # setup
    # setup_mock_dlr()
    #
    # # mirror load_and_run_tvm_model.py for integration test
    # # load the model
    # model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resnet18_v1')
    # classes = 1000
    # device = 'cpu'
    # model = DLRModel(model_path, device)
    #
    # # run the model
    # image = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dog.npy')).astype(np.float32)
    # # flatten within a input array
    # input_data = {'data': image}
    # print('Testing inference on resnet18...')
    # probabilities = model.run(input_data)  # need to be a list of input arrays matching input names
    #
    # assert probabilities[0].argmax() == 151
