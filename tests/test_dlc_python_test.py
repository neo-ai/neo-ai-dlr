import pytest

import dlr
import sys
import os
import tvm
import topi
import nnvm
import treelite

def test_that_you_wrote_tests():
    print("testing that test ran")
    """
    x = tvm.const(1)
    print(x.dtype)
    assert x.dtype == tvm.int32
    print(x)
    assert isinstance(x, tvm.expr.IntImm)
    print("testing that test finished")
    """
"""def test_nnvm_from_mxnet():
    import mxnet as mx
    from mxnet.gluon.model_zoo.vision import get_model
    from mxnet.gluon.utils import download
    block = get_model('resnet18_v1', pretrained=True)
    sym, params = nnvm.frontend.from_mxnet(block)
    sym = nnvm.sym.softmax(sym)
    #import nnvm.compiler"""
