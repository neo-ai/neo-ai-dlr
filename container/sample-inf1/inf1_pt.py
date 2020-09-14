import argparse
import logging

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import io
import os
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
  model_files = []
  for f in os.listdir(model_dir):
      if os.path.isfile(f):
          name, ext = os.path.splitext(f)
          if ext == ".pt" or ext == ".pth":
              model_files.append(f)
  if len(model_files) != 1:
      raise ValueError("Exactly one .pth or .pt file is required for PyTorch models: {}".format(model_files))
  return torch.jit.load(model_files[0])

def input_fn(request_body, request_content_type):
  '''
  stream = os.popen('/opt/aws/neuron/bin/neuron-cli list-model')
  output = stream.read()
  print(output)
  stream = os.popen('/opt/aws/neuron/bin/neuron-cli list-ncg')
  output = stream.read()
  print(output)
  '''
  f = io.BytesIO(request_body)
  input_image = Image.open(f).convert('RGB')
  preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(299),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  input_tensor = preprocess(input_image)
  input_batch = input_tensor.unsqueeze(0)
  return input_batch