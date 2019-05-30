import tvm
from tvm.contrib import graph_runtime
import numpy as np
import time
ms = lambda: int(round(time.time() * 1000))

voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def run(m, x, ctx):
    tvm_input = tvm.nd.array(x, ctx=ctx)
    m.run(data=tvm_input)
    class_IDs, scores, bounding_boxs = m.get_output(0), m.get_output(1), m.get_output(2)
    return class_IDs, scores, bounding_boxs


######################################################################
# Load pre-processed demo image
x = np.load('street_small.jpg.npy')

path_lib = "./models/yolov3_darknet53/deploy_lib.so"
path_graph = "./models/yolov3_darknet53/deploy_graph.json"
path_param = "./models/yolov3_darknet53/deploy_param.params"

graph = open(path_graph).read()
params = bytearray(open(path_param, "rb").read())
lib = tvm.module.load(path_lib)

dshape = (1, 3, 300, 300)
dtype = "float32"

ctx = tvm.opencl(0)
m = graph_runtime.create(graph, lib, ctx)
m.load_params(params)

# warmup
run(m, x, ctx)

# run
N = 10
for i in range(N):
  t1 = ms()
  class_IDs, scores, bboxes = run(m, x, ctx)
  t2 = ms()
  print("Inference time: {:,} ms".format(t2 - t1))

######################################################################
# Dump results
for i, cl in enumerate(scores.asnumpy()[0]):
  prop = cl[0]
  if prop < 0.5:
    continue
  cl_id = int(class_IDs.asnumpy()[0][i][0])
  bbox = bboxes.asnumpy()[0][i]
  print(i, cl_id, voc_classes[cl_id], prop, bbox)


