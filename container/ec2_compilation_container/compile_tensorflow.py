#!/usr/bin/env python3

from tvm import relay
import tensorflow as tf
from tvm_ec2_compiler_utils import tvm_compile

frozen_pb_file = "mobilenet_v1_1.0_224_frozen.pb"
shape_dict = {'input': (1, 224, 224, 3)}
dlr_model_name = "dlr_tf_mobilenet_v1_100"

with tf.compat.v1.gfile.GFile(frozen_pb_file, 'rb') as f:
  graph_def = tf.compat.v1.GraphDef()
  graph_def.ParseFromString(f.read())

print("Model:", frozen_pb_file, ", shape_dict:", shape_dict)

for arch in ["c4", "c5", "p3"]:
  sym, params = relay.frontend.from_tensorflow(graph_def, layout='NCHW', shape=shape_dict)
  func = sym["main"]
  tvm_compile(func, params, arch, dlr_model_name)
