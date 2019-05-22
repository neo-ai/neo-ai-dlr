#!/usr/bin/env python3

import dlr
import numpy as np
import time
import os
import psutil
from statistics import mean, median, mode, stdev

from coco import image_classes

ms = lambda: int(round(time.time() * 1000))

#model_path, input_tensor_name= "models/ssd_mobilenet_v1_coco_2018_01_28_frozen.pb", "import/image_tensor:0"
model_path, input_tensor_name = "models/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tflite", "normalized_input_image_tensor"


def mem_usage():
    process = psutil.Process(os.getpid())
    print("Memory RSS: {:,}".format(process.memory_info().rss))


def get_input(img_files, out_size):
    from PIL import Image
    res = []
    for f in img_files:
        img = np.array(Image.open(f).resize(out_size))
        res.append(img)
    return np.array(res)


def print_output(inp_files, res):
    #for o in res:
    #  print(o.shape)
    boxes,classes,scores,num_det=res

    for i, fname in enumerate(inp_files):
      n_obj = int(num_det[i])

      print("{} - found objects:".format(fname))
      for j in range(n_obj):
        cl_id = int(classes[i][j]) + 1
        label = image_classes[cl_id]
        score = scores[i][j]
        if score < 0.5:
            continue
        box = boxes[i][j]
        print("  ", cl_id, label, score, box)


inp_files = ['dogs.jpg']
inp = get_input(inp_files, (300, 300))

print("model:", model_path)
print(inp.shape, inp.dtype)

mem_usage()

m = dlr.DLRModel(model_path)
mem_usage()
print("input names:", m.get_input_names())
print("output names:", m.get_output_names())
# dryrun
print("dryrun...")
m.run({input_tensor_name: inp})
mem_usage()

N = 10
durations = []
for i in range(N):
    print(i+1, "m.run...")
    t1 = ms()
    res = m.run({input_tensor_name: inp})
    t2 = ms()
    mem_usage()
    durations.append(t2-t1)
    print("m.run done, duration {:,} ms".format(t2-t1))
    print_output(inp_files, res)

print("Avg: {:,} ms, Median {:,} ms (stddev: {:,})".format(mean(durations), median(durations), stdev(durations)))  
mem_usage()
