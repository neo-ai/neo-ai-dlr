# coding: utf-8
from __future__ import absolute_import as _abs

import abc
import glob
import os
import logging
from .neologger import create_logger

neo_logger = None
try:
    neo_logger = create_logger()
except Exception as ex:
    print(str(ex))


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


# Wrapper class
class DLRModel(IDLRModel):
    def __init__(self, model_path, dev_type=None, dev_id=None):
        try:
            # Find correct runtime implementation for the model
            tf_model_path = _find_model_file(model_path, '.pb')
            tflite_model_path = _find_model_file(model_path, '.tflite')
            # Check if found both Tensorflow and TFLite files
            if tf_model_path is not None and tflite_model_path is not None:
                raise ValueError('Found both .pb and .tflite files under {}'.format(model_path))
            # Tensorflow
            if tf_model_path is not None:
                from .tf_model import TFModelImpl
                self._impl = TFModelImpl(tf_model_path, dev_type, dev_id)
                return
            # TFLite
            if tflite_model_path is not None:
                if dev_type is not None:
                    logging.warning("dev_type parameter is not supported")
                if dev_id is not None:
                    logging.warning("dev_id parameter is not supported")
                from .tflite_model import TFLiteModelImpl
                self._impl = TFLiteModelImpl(tflite_model_path)
                return
            # Default to TVM+Treelite
            from .dlr_model import DLRModelImpl
            if dev_type is None:
                dev_type = 'cpu'
            if dev_id is None:
                dev_id = 0
            self._impl = DLRModelImpl(model_path, dev_type, dev_id)
        except Exception as ex:
            if neo_logger is not None:
                neo_logger.exception("error in DLRModel instantiation {}".format(ex))
            raise ex

    def run(self, input_values):
        try:
            return self._impl.run(input_values)
        except Exception as ex:
            if neo_logger is not None:
                neo_logger.exception("error in running inference {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    def get_input_names(self):
        try:
            return self._impl.get_input_names()
        except Exception as ex:
            if neo_logger is not None:
                neo_logger.exception("error in getting input names {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    def get_input(self, name, shape=None):
        try:
            return self._impl.get_input(name, shape)
        except Exception as ex:
            if neo_logger is not None:
                neo_logger.exception("error in getting inputs {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    def get_output_names(self):
        try:
            return self._impl.get_output_names()
        except Exception as ex:
            if neo_logger is not None:
                neo_logger.exception("error in getting output names {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    def get_version(self):
        try:
            return self._impl.get_version()
        except Exception as ex:
            if neo_logger is not None:
                neo_logger.exception("error in getting version {} {}".format(self._impl.__class__.__name__, ex))
            raise ex
