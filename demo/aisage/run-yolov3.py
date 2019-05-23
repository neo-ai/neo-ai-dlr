#!/usr/bin/env python3

import dlr
import numpy as np
import time
import os
import psutil
from statistics import mean, median, mode, stdev

from coco_list import image_classes

ms = lambda: int(round(time.time() * 1000))

model_path, input_tensor_name, model_res = "models/yolov3.pb", "import/input_data:0", (416, 416)


def mem_usage():
    process = psutil.Process(os.getpid())
    print("Memory RSS: {:,}".format(process.memory_info().rss))


def get_input(f, out_size):
    import cv2
    img_ori = cv2.imread(f)
    orig_res = img_ori.shape[:2]
    orig_res = [orig_res,]
    img = cv2.resize(img_ori, out_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.
    return img, orig_res

def resize_boxes(boxes, orig_res):
    for i, (width_ori, height_ori) in enumerate(orig_res):
        boxes[i, 0] *= (width_ori/float(model_res[0]))
        boxes[i, 2] *= (width_ori/float(model_res[0]))
        boxes[i, 1] *= (height_ori/float(model_res[1]))
        boxes[i, 3] *= (height_ori/float(model_res[1]))
    return boxes


def print_output(res, orig_res):
    boxes,scores,classes=res
    boxes = resize_boxes(boxes, orig_res)
    n_obj = len(classes)

    print("{} - objects:".format(inp_file))
    for j in range(n_obj):
        score = scores[j]
        if score < 0.5:
            continue
        cl_id = int(classes[j])
        label = image_classes[cl_id]
        box = boxes[j]
        print("  ", cl_id, label, score, box)

inp_file = 'dog.jpg'
inp, orig_res = get_input(inp_file, model_res)

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
    print_output(res, orig_res)

print("Avg: {:,} ms, Median {:,} ms (stddev: {:,})".format(mean(durations), median(durations), stdev(durations)))  
mem_usage()
