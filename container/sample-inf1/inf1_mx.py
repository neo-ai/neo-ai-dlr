import mxnet as mx
#import neomxnet
import os
import json
import numpy as np
from collections import namedtuple
import os
dtype='float32'
Batch = namedtuple('Batch', ['data'])
ctx = mx.neuron()
is_gpu = False

def model_fn(model_dir):
  print("param {}".format(os.environ.get('MODEL_NAME_CUSTOM')))
  print("ctx {}".format(ctx))
  sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(model_dir, os.environ.get('MODEL_NAME_CUSTOM')), 0)
  mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
  for arg in arg_params:
      arg_params[arg] = arg_params[arg].astype(dtype)

  for arg in aux_params:
      aux_params[arg] = aux_params[arg].astype(dtype)

  exe = mod.bind(for_training=False,
             data_shapes=[('data', (1,3,224,224))],
             label_shapes=mod._label_shapes)

  mod.set_params(arg_params, aux_params, allow_missing=True)
  return mod

def transform_fn(mod, img, input_content_type, output_content_type):
  '''
  stream = os.popen('/opt/aws/neuron/bin/neuron-cli list-model')
  output = stream.read()
  print(output)
  stream = os.popen('/opt/aws/neuron/bin/neuron-cli list-ncg')
  output = stream.read()
  print(output)
  '''
  image = mx.image.imdecode(img)
  resized = mx.image.resize_short(image, 224)  # minimum 224x224 images
  cropped, crop_info = mx.image.center_crop(resized, (224, 224))
  normalized = mx.image.color_normalize(cropped.astype(np.float32) / 255,
                                        mean=mx.nd.array([0.485, 0.456, 0.406]),
                                        std=mx.nd.array([0.229, 0.224, 0.225]))
  # the network expect batches of the form (N,3,224,224)
  transposed = normalized.transpose((2, 0, 1))  # Transposing from (224, 224, 3) to (3, 224, 224)
  batchified = transposed.expand_dims(axis=0)  # change the shape from (3, 224, 224) to (1, 3, 224, 224)
  image = batchified.astype(dtype='float32')
  mod.forward(Batch([image]))
  prob = mod.get_outputs()[0].asnumpy().tolist()
  prob_json = json.dumps(prob)
  return prob_json, output_content_type
