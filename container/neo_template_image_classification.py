import time
import numpy as np
import PIL.Image
import io
import os
import dlr
import json
import glob

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
        self.initialized = False

    def inference(self, data):
        return self.model.run({self.input_names[0]: data})

    def postprocess(self, preds):
        assert len(preds) == 1
        return [json.dumps(np.squeeze(preds[0]).tolist())]

    def initialize(self, context):
        manifest = context.manifest
        model_dir = context.system_properties.get("model_dir")
        print("Loading the model from directory {}".format(model_dir))

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

        USE_GPU = os.getenv('USE_GPU', None)
        if USE_GPU == '1':
            self.model = dlr.DLRModel(model_dir, dev_type='gpu')
        else:
            self.model = dlr.DLRModel(model_dir)
        self.input_names = self.model.get_input_names()
        self.initialized = True

    def preprocess(self, context, data):
        headers = context.request_processor._request_header
        content_type = None
        req_ids = context.request_ids

        assert len(req_ids) == len(data)

        batch_size = len(req_ids)

        if batch_size != 1:
            raise Exception('Batch prediction not yet supported')

        for k in range(batch_size):
            req_id = req_ids[k]
            req_body = data[k]
            content_type = headers[req_id].get('Content-type')
            if content_type is None:
                content_type = headers[req_id].get('Content-Type')
                if content_type is None:
                    raise Exception('Content type could not be deduced')

            if content_type in SUPPORTED_CONTENT_TYPE:
                print('content_type = {}'.format(content_type))
                try:
                    payload = req_body['body']
                    dtest = _load_image(payload, self.shape_info)
                    return dtest
                except Exception as e:
                    raise Exception('Loading image data failed with exception:\n{}'.format(str(e)))
            else:
                raise Exception('Invalid content type. Accepted content types are {}'.format(SUPPORTED_CONTENT_TYPE))


model_obj = NeoImageClassificationPredictor()


def predict(data, context):
    if model_obj is None:
        print("Model not loaded")
        return

    if not model_obj.initialized:
        model_obj.initialize(context)

    if data is None:
        return

    start = time.time()
    data = model_obj.preprocess(context, data)
    data = model_obj.inference(data)
    data = model_obj.postprocess(data)

    context.set_response_content_type(context.request_ids[0], "application/json")
    print("Inference time is {}".format((time.time() - start) * 1000))
    return data
