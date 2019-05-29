import tvm

from matplotlib import pyplot as plt
from tvm.contrib import graph_runtime
from gluoncv import data, utils

voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def run(m, x, ctx):
    # Build TVM runtime
    tvm_input = tvm.nd.array(x.asnumpy(), ctx=ctx)
    m.set_input('data', tvm_input)
    # execute
    m.run()
    # to show the performance test if needed
    num_warmup = 5
    num_test   = 10
    # perform some warm up runs
    print("warm up..")
    warm_up_timer = m.module.time_evaluator("run", ctx, num_warmup)
    warmup = warm_up_timer()
    print("cost per image: %.4fs" % warmup.mean)
    # test
    print("test..")
    ftimer = m.module.time_evaluator("run", ctx, num_test)
    prof_res = ftimer()
    print("cost per image: %.4fs" % prof_res.mean)
    # get outputs
    class_IDs, scores, bounding_boxs = m.get_output(0), m.get_output(1), m.get_output(2)
    return class_IDs, scores, bounding_boxs


######################################################################
# Download and pre-process demo image

im_fname = 'street_small.jpg'
x, img = data.transforms.presets.ssd.load_test(im_fname, short=300)

path_lib = "./models/ssd_mobilenet1.0/deploy_lib.so"
path_graph = "./models/ssd_mobilenet1.0/deploy_graph.json"
path_param = "./models/ssd_mobilenet1.0/deploy_param.params"

graph = open(path_graph).read()
params = bytearray(open(path_param, "rb").read())
lib = tvm.module.load(path_lib)

dshape = (1, 3, 300, 300)
dtype = "float32"

ctx = tvm.opencl(0)
m = graph_runtime.create(graph, lib, ctx)
m.load_params(params)

class_IDs, scores, bounding_boxs = run(m, x, ctx)

######################################################################
# Display result

ax = utils.viz.plot_bbox(img, bounding_boxs.asnumpy()[0], scores.asnumpy()[0],
                         class_IDs.asnumpy()[0], class_names=voc_classes)
plt.show()
