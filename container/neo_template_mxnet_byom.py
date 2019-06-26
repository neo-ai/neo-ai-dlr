import time
from six.moves.urllib.parse import urlparse
import importlib
import numpy as np
import boto3
import tarfile
import tempfile
import os
import sys
import dlr
import json
import glob

# Import user module, ignoring non-existent modules
# This way, we don't introduce MXNet/TF dependencies
def import_user_module(path, script_name):
    try:
        import builtins
    except ImportError:
        import __builtin__ as builtins
    from types import ModuleType

    class DummyModule(ModuleType):
        def __getattr__(self, key):
            return None
        __all__ = []   # support wildcard imports

    def tryimport(name, globals={}, locals={}, fromlist=[], level=0):
        try:
            return realimport(name, globals, locals, fromlist, level)
        except ImportError:
            return DummyModule(name)
        except KeyError:
            return DummyModule(name)

    # Intercept import semantics, to ignore ImportError
    #realimport, builtins.__import__ = builtins.__import__, tryimport

    sys.path.insert(0, path)
    user_module = importlib.import_module(script_name)

    # Restore import semantics
    #builtins.__import__ = realimport

    return user_module

def parse_s3_url(url):
    parsed_url = urlparse(url)
    if parsed_url.scheme != 's3':
        raise ValueError('Expecting \'s3\' scheme, got: {} in {}'\
                         .format(parsed_url.scheme, url))
    return parsed_url.netloc, parsed_url.path.lstrip('/')

def download_s3_resource(source, target):
    print('Downloading {} to {}'.format(source, target))
    s3 = boto3.resource('s3')

    script_bucket_name, script_key_name = parse_s3_url(source)
    script_bucket = s3.Bucket(script_bucket_name)
    script_bucket.download_file(script_key_name, target)

    return target

class NeoBYOMPredictor():
    def __init__(self):
        self.model = None
        self.initialized = False

    def inference(self, data):
        return self.model.run(data)

    def postprocess(self, preds):
        if len(preds) == 1:
            return_data, content_type = self.user_module.neo_postprocess(preds[0])
        else:
            return_data, content_type = self.user_module.neo_postprocess(preds)
        return [return_data], content_type

    def initialize(self, context):
        manifest = context.manifest
        model_dir = context.system_properties.get("model_dir")
        print("Loading the model from directory {}".format(model_dir))

        # Load user module
        SAGEMAKER_SUBMIT_DIRECTORY = os.getenv('SAGEMAKER_SUBMIT_DIRECTORY', None)
        tempdir = tempfile.gettempdir()
        source_tar = os.path.join(tempdir, 'script.tar.gz')
        download_s3_resource(SAGEMAKER_SUBMIT_DIRECTORY, source_tar)
        script_name = None
        with tarfile.open(source_tar, 'r:*') as tar:
            for member_info in tar.getmembers():
                if member_info.name.endswith('.py'):
                    if script_name is not None:
                        raise RuntimeError('{} contains more than one *.py file'\
                                           .format(source_tar))
                    print('Importing user module from {}...'.format(member_info.name))
                    tar.extract(member_info, path=tempdir)
                    script_name = member_info.name
        if script_name is None:
            raise RuntimeError('{} contains no *.py file'.format(source_tar))
        cur_dir = tempdir
        script_path = script_name[:-3]
        if '/' in script_path:
            file_depth = len(script_path.split('/')) - 1
            for i in range(file_depth):
                cur_dir = os.path.join(cur_dir, script_name[:-3].split('/')[i])
            script_path = script_path.split('/')[file_depth]
        self.user_module = import_user_module(cur_dir, script_path)

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

            # For BYOM, any content type is allowed
            print('content_type = {}'.format(content_type))
            try:
                payload = req_body['body']
                # User is responsible for parsing payload into input(s)
                input_values = self.user_module.neo_preprocess(payload, content_type)
            except Exception as e:
                raise Exception('User-defined pre-processing function failed:\n{}'.format(str(e)))

            # Validate parsed input(s)
            if isinstance(input_values, (np.ndarray, np.generic)):
                # Single input
                if len(self.input_names) != 1:
                    raise Exception('User-defined pre-processing function returns a single input, ' +\
                                    'but the model has multiple inputs.')
                input_values = {self.input_names[0]: input_values}
            elif isinstance(input_values, dict):
                # Multiple inputs
                given_names = set(input_values.keys())
                expected_names = set(self.input_names)
                if given_names != expected_names:  # Input name(s) mismatch
                    given_missing = expected_names - given_names
                    expected_missing = given_names - expected_names
                    msg = 'Input name(s) mismatch: {0} {1}'
                    if given_missing:
                        msg += ('\nExpected ' + ', '.join(str(s) for s in given_missing) + \
                                ' in input data')
                    if expected_missing:
                        msg += ('\nThe model does not accept the following inputs: ' + \
                                ', '.join(str(s) for s in expected_missing))
                    msg = msg.format(given_names, expected_names)
                    raise Exception(msg)
            else:
                raise Exception('User-defined pre-processing function must return either ' + \
                                'dict type or np.ndarray')

            return input_values


model_obj = NeoBYOMPredictor()


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
    data, content_type = model_obj.postprocess(data)

    context.set_response_content_type(context.request_ids[0], content_type)
    print("Inference time is {}".format((time.time() - start) * 1000))
    return data
