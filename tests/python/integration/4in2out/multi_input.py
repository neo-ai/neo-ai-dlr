from os import path
import numpy as np

import nnvm.symbol as symbol
import nnvm.graph as graph
import nnvm.compiler.graph_util as graph_util
import nnvm.compiler
from nnvm.frontend import from_mxnet
from nnvm.testing import utils

import tvm
from tvm.contrib import graph_runtime, util

data1 = symbol.Variable(name="data1")
data2 = symbol.Variable(name="data2")
net1 = data1 + data2
data_shape = (2,)
shape_dict = {"data1": data_shape, "data2": data_shape}
params = {}
params["data1"] = data1 = np.random.uniform(-1, 1, size=data_shape).astype("float32")
params["data2"] = data2 = np.random.uniform(-1, 1, size=data_shape).astype("float32")

data3 = symbol.Variable(name="data3")
data4 = symbol.Variable(name="data4")
net2 = data3 + data4
shape_dict.update({"data3": data_shape, "data4": data_shape})
params["data3"] = data3 = np.random.uniform(-1, 1, size=data_shape).astype("float32")
params["data4"] = data4 = np.random.uniform(-1, 1, size=data_shape).astype("float32")

net = symbol.Group([net1, net2])

deploy_graph, lib, params = nnvm.compiler.build(
    net, target="llvm", shape=shape_dict, dtype="float32", params=params)

temp = path.curdir
path_lib = path.join(temp, "deploy.so")
lib.export_library(path_lib)
with open(path.join(temp, "deploy.json"), "w") as fo:
    fo.write(deploy_graph.json())
with open(path.join(temp, "deploy.params"), "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))

loaded_lib = tvm.module.load(path_lib)
loaded_json = open(path.join(temp, "deploy.json")).read()
loaded_json = graph.load_json(loaded_json)
loaded_params = bytearray(
    open(path.join(temp, "deploy.params"), "rb").read())

module = graph_runtime.create(loaded_json, loaded_lib, tvm.cpu(0))
loaded_params = nnvm.compiler.load_param_dict(loaded_params)
module.set_input(**loaded_params)
module.run()
_, oshape = graph_util.infer_shape(loaded_json)
out1 = module.get_output(0, out=tvm.nd.empty(oshape[0], "float32"))
assert np.allclose(data1 + data2, out1.asnumpy())

out2 = module.get_output(1, out=tvm.nd.empty(oshape[1], "float32"))
assert np.allclose(data3 + data4, out2.asnumpy())
