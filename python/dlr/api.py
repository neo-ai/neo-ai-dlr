# coding: utf-8
from __future__ import absolute_import as _abs

import abc
import glob
import os
from .neologger import create_logger

# Interface
class IDLRModel:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_input_names(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_input(self, name, shape=None):
        raise NotImplementedError

    @abc.abstractmethod
    def get_output_names(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_version(self):
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, input_data):
        raise NotImplementedError


def _find_model_file(model_path, ext):
    if os.path.isfile(model_path) and model_path.endswith(ext):
        return model_path
    model_files = glob.glob(os.path.abspath(os.path.join(model_path, '*' + ext)))
    if len(model_files) > 1:
        raise ValueError('Multiple {} files found under {}'.format(ext, model_path))
    elif len(model_files) == 1:
        return model_files[0]
    return None

def _is_module_found(name):
    try:
        __import__(name)
        return True
    except ModuleNotFoundError:
        return False

# Wrapper class
class DLRModel(IDLRModel):
    def __init__(self, model_path, dev_type=None, dev_id=None, error_log_file=None):
        self.neo_logger = create_logger(log_file=error_log_file)
        try:
            # Find correct runtime implementation for the model
            tf_model_path = _find_model_file(model_path, '.pb')
            tflite_model_path = _find_model_file(model_path, '.tflite')
            # Check if found both Tensorflow and TFLite files
            if tf_model_path is not None and tflite_model_path is not None:
                raise ValueError('Found both .pb and .tflite files under {}'.format(model_path))
            # Tensorflow Python
            if tf_model_path is not None and _is_module_found("tensorflow"):
                from .tf_model import TFModelImpl
                self._impl = TFModelImpl(tf_model_path, dev_type, dev_id)
                return
            # TFLite Python
            if tflite_model_path is not None and _is_module_found("tensorflow.lite"):
                if dev_type is not None:
                    self.neo_logger.warning("dev_type parameter is not supported")
                if dev_id is not None:
                    self.neo_logger.warning("dev_id parameter is not supported")
                from .tflite_model import TFLiteModelImpl
                self._impl = TFLiteModelImpl(tflite_model_path)
                return
            # Default to DLR C API (Python wrapper)
            from .dlr_model import DLRModelImpl
            if dev_type is None:
                dev_type = 'cpu'
            if dev_id is None:
                dev_id = 0
            self._impl = DLRModelImpl(model_path, dev_type, dev_id)
        except Exception as ex:
            self.neo_logger.exception("error in DLRModel instantiation {}".format(ex))
            raise ex

    def run(self, input_values):
        try:
            return self._impl.run(input_values)
        except Exception as ex:
            self.neo_logger.exception("error in running inference {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    def get_input_names(self):
        try:
            return self._impl.get_input_names()
        except Exception as ex:
            self.neo_logger.exception("error in getting input names {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    def get_input(self, name, shape=None):
        try:
            return self._impl.get_input(name, shape)
        except Exception as ex:
            self.neo_logger.exception("error in getting inputs {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    def get_output_names(self):
        try:
            return self._impl.get_output_names()
        except Exception as ex:
            self.neo_logger.exception("error in getting output names {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    def get_version(self):
        try:
            return self._impl.get_version()
        except Exception as ex:
            self.neo_logger.exception("error in getting version {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    def has_metadata(self):
        try:
            return self._impl.has_metadata()
        except Exception as ex:
            self.neo_logger.exception("error in checking for metadata file {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    def get_input_dtypes(self):
        return self._impl.get_input_dtypes()

    def get_output_dtypes(self):
        return self._impl.get_output_dtypes()

    def get_input_name(self, index):
        return self._impl.get_input_name(index)

    def get_output_name(self, index):
        return self._impl.get_output_name(index)

    def get_input_dtype(self, index):
        return self._impl.get_input_dtype(index)

    def get_output_dtype(self, index):
        return self._impl.get_ouput_dtype(index)

