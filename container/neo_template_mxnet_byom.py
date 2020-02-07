import time
import numpy as np
import io
import os
import logging
import dlr

from six.moves.urllib.parse import urlparse
import importlib
import sys
import boto3
import tarfile
import tempfile
import json
import glob

SAGEMAKER_ERROR_LOG_FILE = "/opt/ml/errors/errors.log"

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

            # For BYOM, any content type is allowed
            print('content_type = {}'.format(content_type))
            try:
                # User is responsible for parsing payload into input(s)
                input_values = self.user_module.neo_preprocess(payload, content_type)
            except Exception as e:
                raise Exception('ClientError: User-defined pre-processing function failed:\n'
                                + str(e))

            # Validate parsed input(s)
            if isinstance(input_values, (np.ndarray, np.generic)):
                # Single input
                if len(self.input_names) != 1:
                    raise Exception('ClientError: User-defined pre-processing function returns ' +
                                    'a single input, but the model has multiple inputs.')
                input_values = {self.input_names[0]: input_values}
            elif isinstance(input_values, dict):
                # Multiple inputs
                given_names = set(input_values.keys())
                expected_names = set(self.input_names)
                if given_names != expected_names:  # Input name(s) mismatch
                    given_missing = expected_names - given_names
                    expected_missing = given_names - expected_names
                    msg = 'ClientError: Input name(s) mismatch: {0} {1}'
                    if given_missing:
                        msg += ('\nExpected ' + ', '.join(str(s) for s in given_missing) + \
                                ' in input data')
                    if expected_missing:
                        msg += ('\nThe model does not accept the following inputs: ' + \
                                ', '.join(str(s) for s in expected_missing))
                    msg = msg.format(given_names, expected_names)
                    raise Exception(msg)
            else:
                raise Exception('ClientError: User-defined pre-processing function must return ' +
                                'either dict type or np.ndarray')

            processed_batch_data.append(input_values)

        return processed_batch_data

    def inference(self, batch_data):
        return [self.model.run(x) for x in batch_data]

    def postprocess(self, batch_preds):
        # returns two lists, one for data and another for content_type
        response = []
        content_type = []
        for preds in batch_preds:
            if len(preds) == 1:
                r, c = self.user_module.neo_postprocess(preds[0])
            else:
                r, c = self.user_module.neo_postprocess(preds)
            response.append(r)
            content_type.append(c)

        return response, content_type

    def handle(self, data, context):
        start = time.time()
        try:
            model_input = self.preprocess(data)
            model_output = self.inference(model_input)
            response, content_type = self.postprocess(model_output)

            for k in range(len(data)):
                context.set_response_content_type(k, content_type[k])
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


_service = NeoBYOMPredictor()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
