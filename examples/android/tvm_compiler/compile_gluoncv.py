#!/usr/bin/env python3

from tvm import relay
from mxnet.gluon.model_zoo.vision import get_model
from tvm_compiler_utils import tvm_compile

model_name, dlr_model_name = "mobilenetv2_0.75", "dlr_gluoncv_mobilenet_v2_075"
#model_name, dlr_model_name = "mobilenetv2_1.0", "dlr_gluoncv_mobilenet_v2_100"
#model_name, dlr_model_name = "resnet18_v2", "dlr_gluoncv_resnet18_v2"
#model_name, dlr_model_name = "resnet50_v2", "dlr_gluoncv_resnet50_v2"
shape_dict = {'data': (1, 3, 224, 224)}
dtype='float32'

print("Model:", model_name, ", shape_dict:", shape_dict)
block = get_model(model_name, pretrained=True)

for arch in ["arm64-v8a", "armeabi-v7a", "x86_64", "x86"]:
  sym, params = relay.frontend.from_mxnet(block, shape=shape_dict, dtype=dtype)
  func = sym["main"]
  func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)
  tvm_compile(func, params, arch, dlr_model_name)
