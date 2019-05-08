# coding: utf-8
from __future__ import absolute_import as _abs

import abc
import sys
import os

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
    def run(self, input_data):
        raise NotImplementedError

# Wrapper class
class DLRModel(IDLRModel):
    def __init__(self, model_path, dev_type='cpu', dev_id=0):
        #Determine if 3rdparty package needed
        if os.path.isfile(model_path):    
            if model_path.endswith('.pb'):
                from .tf_model import TFModelImpl
                self._impl = TFModelImpl(model_path, dev_type, dev_id)
        else:
            for (dirpath, dirnames, filenames) in os.walk(model_path):
                for filename in filenames:
                    if filename.endswith('.pb'): 
                        from .tf_model import TFModelImpl
                        self._impl = TFModelImpl(model_path, dev_type, dev_id)
        # Default to DLR model
        if not hasattr(self, '_impl'):
            from .dlr_model import DLRModelImpl
            self._impl = DLRModelImpl(model_path, dev_type, dev_id) 

    def run(self, input_values):
        return self._impl.run(input_values)
    
    def get_input_names(self):
        return self._impl.get_input_names()

    def get_input(self, name, shape=None):
        return self._impl.get_input(name, shape)
