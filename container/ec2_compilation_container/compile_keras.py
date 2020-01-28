#!/usr/bin/env python3

from tvm import relay
from keras.applications import mobilenet_v2
from tvm_ec2_compiler_utils import tvm_compile

keras_model = mobilenet_v2.MobileNetV2()
#keras_model.summary()
shape_dict = {'input_1': (1, 3, 224, 224)}
dlr_model_name = "dlr_keras_mobilenet_v2"

print("Model:", keras_model.name, ", shape_dict:", shape_dict)

for arch in ["c4", "c5", "p3"]:
  sym, params = relay.frontend.from_keras(keras_model, shape_dict)
  func = sym["main"]
  tvm_compile(func, params, arch, dlr_model_name)
