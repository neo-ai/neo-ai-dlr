import tvm

from matplotlib import pyplot as plt
from tvm import relay
from tvm.contrib import graph_runtime
from gluoncv import data, utils

voc_classes = =['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def run(graph, lib, params, ctx):
    # Build TVM runtime
    m = graph_runtime.create(graph, lib, ctx)
    tvm_input = tvm.nd.array(x.asnumpy(), ctx=ctx)
    m.set_input('data', tvm_input)
    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    class_id = m.get_output(0)
    class_IDs, scores, bounding_boxs = m.get_output(0), m.get_output(1), m.get_output(2)
    return class_IDs, scores, bounding_boxs


######################################################################
# Download and pre-process demo image

im_fname = 'dog.jpg'
x, img = data.transforms.presets.ssd.load_test(im_fname, short=300)

path_lib = "./models/yolov3_darknet53/deploy_lib.so"
path_graph = "./models/yolov3_darknet53/deploy_graph.json"
path_param = "./models/yolov3_darknet53/deploy_param.params"

graph = open(path_graph).read()
params = relay.load_param_dict(bytearray(open(path_param, "rb").read()))
lib = tvm.module.load(path_lib)

dshape = (1, 3, 300, 300)
dtype = "float32"

ctx = tvm.opencl(0)
class_IDs, scores, bounding_boxs = run(graph, lib, params, ctx)

######################################################################
# Display result

ax = utils.viz.plot_bbox(img, bounding_boxs.asnumpy()[0], scores.asnumpy()[0],
                         class_IDs.asnumpy()[0], class_names=voc_classes)
plt.show()

