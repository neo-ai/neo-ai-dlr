from __future__ import print_function
from dlr import DLRModel
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'integration'))

from test_utils import get_arch, get_models

def test_get_set_input():
    model_path = get_models(model_name='4in2out', arch=get_arch(), kind='tvm')
    device = 'cpu'
    model = DLRModel(model_path, device)

    input1 = np.asarray([1., 2.])
    input2 = np.asarray([3., 4.])
    input3 = np.asarray([5., 6., 7])
    input4 = np.asarray([8., 9., 10])
    
    model.run({'data1': input1, 'data2': input2, 'data3': input3, 'data4': input4})

    np.array_equal(model.get_input('data1'), input1)
    np.array_equal(model.get_input('data2'), input2)
    np.array_equal(model.get_input('data3'), input3)
    np.array_equal(model.get_input('data4'), input4)
