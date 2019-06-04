#!/usr/bin/env python3

import dlr
import numpy as np
import time
ms = lambda: int(round(time.time() * 1000))

voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# Open, resize and normalize image using cv2
def open_and_norm_image_cv2(f, target_res = (300, 300), mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    import cv2
    img = cv2.imread(f)
    # BGR to RGB
    img = img[..., ::-1]
    # GluonCV uses INTER_CUBIC interpolation method to resize the image
    img = cv2.resize(img, target_res, interpolation = cv2.INTER_CUBIC)
    img = np.asarray(img, "float32")
    img /= 255.0
    img -= np.array(mean)
    img /= np.array(std)
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, :]
    return img

# Open, resize and normalize image using PIL
def open_and_norm_image_pil(f, target_res = (300, 300), mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    from PIL import Image
    img = Image.open(f)
    # GluonCV uses INTER_CUBIC interpolation method to resize the image
    img = img.resize(target_res, Image.BICUBIC)
    img = np.asarray(img, "float32")
    img /= 255.0
    img -= np.array(mean)
    img /= np.array(std)
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, :]
    return img

def run(m, x):
    m_out = m.run(x)
    class_IDs, scores, bounding_boxes = m_out[0], m_out[1], m_out[2]
    return class_IDs, scores, bounding_boxes


######################################################################
# Load, resize and normalize demo image
x = open_and_norm_image_cv2('street_small.jpg')
#x = open_and_norm_image_pil('street_small.jpg')

model_path = "models/yolov3_darknet53"
print(model_path)

dshape = (1, 3, 300, 300)
dtype = "float32"

device = 'opencl'
m = dlr.DLRModel(model_path, device)

# warmup
print("Warm up....")
run(m, x)

# run
N = 10
print("Run")
for i in range(N):
  t1 = ms()
  class_IDs, scores, bboxes = run(m, x)
  t2 = ms()
  print("Inference time: {:,} ms".format(t2 - t1))

######################################################################
# Dump results
for i, cl in enumerate(scores[0]):
  prop = cl[0]
  if prop < 0.5:
    continue
  cl_id = int(class_IDs[0][i][0])
  bbox = bboxes[0][i]
  print(i, cl_id, voc_classes[cl_id], prop, bbox)


