import pdb
import PIL.Image
from IPython.display import Image
from dlr import DLRModel

import json
import numpy as np

file_name = "test.jpg"

# test image
Image(file_name)

image = PIL.Image.open(file_name)

image = np.asarray(image.resize((224, 224)))

# # Normalize
mean_vec = np.array([0.485, 0.456, 0.406])
stddev_vec = np.array([0.229, 0.224, 0.225])
image = (image/255 - mean_vec)/stddev_vec

# Transpose
if len(image.shape) == 2:  # for greyscale image
    image = np.expand_dims(image, axis=2)


image = np.rollaxis(image, axis=2, start=0)[np.newaxis, :]

print(image.shape)


input_data = {'data': image}
model_path = "./compiled_model"
model = DLRModel(model_path, 'cpu')

model.run(input_data)