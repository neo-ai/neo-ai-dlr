import json
import numpy as np
    
file_name = "test.jpg"

# test image
from IPython.display import Image
Image(file_name)

import PIL.Image
image = PIL.Image.open(file_name)

import pdb
image = np.asarray(image.resize((224, 224)))

# # Normalize
mean_vec = np.array([0.485, 0.456, 0.406])
stddev_vec = np.array([0.229, 0.224, 0.225])
image = (image/255- mean_vec)/stddev_vec

# Transpose
if len(image.shape) == 2:  # for greyscale image
    image = np.expand_dims(image, axis=2)


image = np.rollaxis(image, axis=2, start=0)[np.newaxis, :]

print(image.shape)

pdb.set_trace()

with open('test.npy', 'wb') as f:
    np.save(f, image);
    

