# coding: utf-8
from __future__ import absolute_import as _abs

import abc
import glob
import os
import sys

# Interface
class IDLRModel:
    __metaclass__=abc.ABCMeta

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
    def run(self, input_data):
        raise NotImplementedError

# Wrapper class
class DLRModel(IDLRModel):
    def __init__(self, model_path, dev_type='cpu', dev_id=0):
        #Determine if 3rdparty package needed
        tf_model_path = None
        if os.path.isfile(model_path):    
            if model_path.endswith('.pb'):
                tf_model_path = model_path
        else:
            model_files = glob.glob(os.path.abspath(os.path.join(model_path, '*.pb')))
            if len(model_files) > 1:
                raise ValueError('Multiple .pb files found under ' + model_path)
            elif len(model_files) == 1:
                tf_model_path = model_files[0]
        # Default to DLR model
        if tf_model_path is not None:
            from .tf_model import TFModelImpl
            self._impl = TFModelImpl(tf_model_path, dev_type, dev_id)
        else:
            from .dlr_model import DLRModelImpl
            self._impl = DLRModelImpl(model_path, dev_type, dev_id) 

    def run(self, input_values):
        return self._impl.run(input_values)
    
    def get_input_names(self):
        return self._impl.get_input_names()

    def get_input(self, name, shape=None):
        return self._impl.get_input(name, shape)

    def get_output_names(self):
        return self._impl.get_output_names()
