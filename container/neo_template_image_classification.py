import time
import numpy as np
import io
import os
import logging
import dlr

import PIL.Image
import json
import glob

SAGEMAKER_ERROR_LOG_FILE = "/opt/ml/errors/errors.log"
SHAPES_FILE = 'model-shapes.json'
SUPPORTED_CONTENT_TYPE = ['image/jpeg', 'image/png', 'application/x-image']

def _read_input_shape(signature):
    shape = signature[-1]['shape']
    shape[0] = 1
    return shape

def _transform_image(image, shape_info):
    # Fetch image size
    input_shape = _read_input_shape(shape_info)

    # Perform color conversion
    if input_shape[-3] == 3:
        # training input expected is 3 channel RGB
        image = image.convert('RGB')
    elif input_shape[-3] == 1:
        # training input expected is grayscale
        image = image.convert('L')
    else:
        # shouldn't get here
        raise RuntimeError('Wrong number of channels in input shape')

    # Resize
    image = np.asarray(image.resize((input_shape[-2], input_shape[-1])))

    # Transpose
    if len(image.shape) == 2:  # for greyscale image
        image = np.expand_dims(image, axis=2)
    image = np.rollaxis(image, axis=2, start=0)[np.newaxis, :]
    return image

def _load_image(payload, shape_info):
    f = io.BytesIO(payload)
    return _transform_image(PIL.Image.open(f), shape_info)

class NeoImageClassificationPredictor():
    def __init__(self):
        self.model = None
        self._context = None
        self._batch_size = 0
        self.initialized = False

    def initialize(self, context):
        self._context = context
        self._batch_size = context.system_properties.get('batch_size')
        model_dir = context.system_properties.get('model_dir')
        print('Loading the model from directory {}'.format(model_dir))
        USE_GPU = os.getenv('USE_GPU', None)
        if USE_GPU == '1':
            self.model = dlr.DLRModel(model_dir, dev_type='gpu', error_log_file=SAGEMAKER_ERROR_LOG_FILE)
        else:
            self.model = dlr.DLRModel(model_dir, error_log_file=SAGEMAKER_ERROR_LOG_FILE)

        # Load shape info
        self.shape_info = None
        for aux_file in glob.glob(os.path.join(model_dir, '*.json')):
            if os.path.basename(aux_file) == SHAPES_FILE:
                try:
                    with open(aux_file, 'r') as f:
                        self.shape_info = json.load(f)
                except Exception as e:
                    raise Exception('Error parsing shape info')
        if self.shape_info is None:
            raise Exception('Shape info must be given as {}'.format(SHAPES_FILE))
        self.input_names = self.model.get_input_names()
        self.initialized = True

    def preprocess(self, batch_data):
        assert self._batch_size == len(batch_data), \
            'Invalid input batch size: expected {} but got {}'.format(self._batch_size,
                                                                      len(batch_data))
        processed_batch_data = []

        for k in range(len(batch_data)):
            req_body = batch_data[k]
            content_type = self._context.get_request_header(k, 'Content-type')
            if content_type is None:
                content_type = self._context.get_request_header(k, 'Content-Type')
                if content_type is None:
                    raise Exception('Content type could not be deduced')

            payload = batch_data[k].get('data')
            if payload is None:
                payload = batch_data[k].get('body')
            if payload is None:
                raise Exception('Nonexistent payload')

            if content_type in SUPPORTED_CONTENT_TYPE:
                try:
                    dtest = _load_image(payload, self.shape_info)
                    processed_batch_data.append(dtest)
                except Exception as e:
                    raise Exception('ClientError: Loading image data failed with exception:\n' +
                                    str(e))
            else:
                raise Exception('ClientError: Invalid content type. ' +
                                'Accepted content types are {}'.format(SUPPORTED_CONTENT_TYPE))

        return processed_batch_data

    def inference(self, batch_data):
        return [self.model.run({self.input_names[0]: x})[0] for x in batch_data]

    def postprocess(self, batch_preds):
        return [json.dumps(np.squeeze(x).tolist()) for x in batch_preds]

    def handle(self, data, context):
        start = time.time()
        try:
            model_input = self.preprocess(data)
            model_output = self.inference(model_input)
            response = self.postprocess(model_output)

            for k in range(len(data)):
                context.set_response_content_type(k, 'application/json')
            print('Inference time is {}'.format((time.time() - start) * 1000))

            return response
        except Exception as e:
            logging.error(e, exc_info=True)
            if str(e).startswith('ClientError:'):
                context.set_all_response_status(400, 'ClientError')
                return [str(e)] * len(data)
            else:
                context.set_all_response_status(500, 'InternalServerError')
                return ['Internal Server Error occured'] * len(data)


_service = NeoImageClassificationPredictor()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
