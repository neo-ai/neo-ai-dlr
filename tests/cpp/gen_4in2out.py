from os import path, mkdir
import numpy as np
import sys

import nnvm.symbol as symbol
import nnvm.graph as graph
import nnvm.compiler
from nnvm.frontend import from_mxnet

import tvm

data1 = symbol.Variable(name="data1")
data2 = symbol.Variable(name="data2")
net1 = data1 + data2
data_shape1 = (2,)
data_shape2 = (3,)
shape_dict = {"data1": data_shape1, "data2": data_shape1}
params = {}
params["data1"] = data1 = np.random.uniform(-1, 1, size=data_shape1).astype("float32")
params["data2"] = data2 = np.random.uniform(-1, 1, size=data_shape1).astype("float32")

data3 = symbol.Variable(name="data3")
data4 = symbol.Variable(name="data4")
net2 = data3 + data4
shape_dict.update({"data3": data_shape2, "data4": data_shape2})
params["data3"] = data3 = np.random.uniform(-1, 1, size=data_shape2).astype("float32")
params["data4"] = data4 = np.random.uniform(-1, 1, size=data_shape2).astype("float32")

net = symbol.Group([net1, net2])

deploy_graph, lib, params = nnvm.compiler.build(
    net, target="llvm", shape=shape_dict, dtype="float32", params=params)

temp = path.join(path.curdir, "4in2out")
if not path.exists(temp):
    mkdir(temp)
path_lib = temp
if sys.platform == "darwin":
    path_lib = path.join(temp, "deploy.dylib")
else:
    path_lib = path.join(temp, "deploy.so")
lib.export_library(path_lib)
with open(path.join(temp, "deploy.json"), "w") as fo:
    fo.write(deploy_graph.json())
with open(path.join(temp, "deploy.params"), "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))