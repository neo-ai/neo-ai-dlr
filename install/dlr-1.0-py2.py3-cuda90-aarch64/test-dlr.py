#!/usr/bin/env python
import os
from dlr import DLRModel
import numpy as np
import time
import logging

logging.basicConfig(filename='test-dlr.log', level=logging.DEBUG)

current_milli_time = lambda: int(round(time.time() * 1000))

def run_inference():

    model_path = 'resnet50'
    batch_size = 1
    channels = 3
    height = width = 224
    input_shape = {'data': [batch_size, channels, height, width]}
    classes = 1000
    output_shape = [batch_size, classes]
    device = 'cpu'
    model = DLRModel(model_path, input_shape, output_shape, device)

    synset_path = os.path.join(model_path, 'synset.txt')
    with open(synset_path, 'r') as f:
        synset = eval(f.read())

    image = np.load('dog.npy').astype(np.float32)
    input_data = {'data': image}

    for rep in range(4):
        t1 = current_milli_time()
        out = model.run(input_data)
        t2 = current_milli_time()

        logging.debug('done m.run(), time (ms): {}'.format(t2 - t1))

        top1 = np.argmax(out[0])
        logging.debug('Inference result: {}, {}'.format(top1, synset[top1]))

    import resource
    logging.debug("peak memory usage (bytes on OS X, kilobytes on Linux) {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

    return {
        'synset_id': top1,
        'prediction': synset[top1],
        'time': t2 - t1
    }

if __name__ == '__main__':
    res = run_inference()
    cls_id = res['synset_id']
    exp_cls_id = 151
    assert cls_id == exp_cls_id, "Inference result class id {} is incorrect, expected class id is {}".format(cls_id, exp_cls_id)
    print("All tests PASSED!")
