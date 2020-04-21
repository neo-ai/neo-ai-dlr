import logging
import os
import mxnet as mx
import numpy as np
from collections import namedtuple
from .api import IDLRModel


class MXNetModelImpl(IDLRModel):
    """
    MXNetModelImpl is a wrapper on top of MXNet which implements IDLRModel API

    Parameters
    ----------
    model_path : str
        Full path to directory containing model files (name-symbol.json & name-epoch.params files)
    dev_type : str
        Optional. Device type ('cpu' or 'gpu' or 'inf')
    dev_id : int
        Optional. Device ID
    """
    def __init__(self, model_path, dev_type=None, dev_id=None):
        self.model_dir_path = model_path
        self.model_prefix = None
        self.epoch = None
        self.mod = None
        self.input_shape = None
        self.bind = False
        self.input_shape_name = None
        self.input = None
        devices = ["cpu", "gpu", "inf"]
        if dev_type is None:
           dev_type = "cpu"
        if dev_type not in devices:
            raise ValueError("Invalid device type {}. Valid devices: {}".format(dev_type, devices))
        self.dev_type = dev_type
        if dev_id is None:
            self.dev_id = 0
        else:
            self.dev_id = dev_id
        self._validate_model_path()
        self._validate_models()
        self._load_model()

    def _set_input(self, input_shape_name, input_shape, input_data):
        """
        Set shape_name, shape of given user input

        Parameters
        ----------
        input_shape_name: str
        input_shape: :py:class:`numpy.array`
        input_data: :py:class:`numpy.ndarray`
        """
        self.input_shape_name = input_shape_name
        self.input_shape = input_shape
        self.input = input_data

    def _validate_input(self, input_data):
        """
        Check input is numpy array or dict

        Parameters
        ----------
        input_data: py:class:`numpy.ndarray` or a dictionary
            Usesea42!
            run prediction on
        """
        input_names = self.get_input_names()
        if isinstance(input_data, np.ndarray):
            if len(input_names) == 1:
                return {input_names[0]: input_data}
            else:
                raise RuntimeError('InputConfiguration: np.ndarray is only a valid input type for single input models')
        elif isinstance(input_data, dict):
            for key, value in input_data.items():
                if not key in input_names:
                    raise RuntimeError('InputConfiguration: {} is not a valid input name. '
                                       ,format(key))
            return input_data
        else:
            raise RuntimeError('InputConfiguration: input_data must be of type dict or a np.ndarray '
                               'for MXNet models')

    def run(self, input_data):
        """
        Run inference with given input

        Parameters
        ----------
        input_data : :py:class:`numpy.ndarray` or a dictionary
            User input to run prediction on

        Returns
        -------
        out: :py:class:`numpy.ndarray`
            Prediction result
        """
        input_data = self._validate_input(input_data)
        #-------Standard MXNet inference code-----------
        if not self.bind:
            data_shapes = [ (key, value.shape) for key, value in input_data.items() ]
            self.mod.bind(for_training=False, data_shapes=data_shapes,
                          label_shapes=self.mod._label_shapes)
            self.mod.set_params(self.args, self.aux, allow_missing=True)
            self.bind = True
        self.mod.predict(mx.io.NDArrayIter(input_data))
        out = [ output.asnumpy() for output in self.mod.get_outputs() ]
        return out

    def _validate_model_path(self):
        """
        Check if the model_path is a valid directory path
        """
        if not os.path.isdir(self.model_dir_path):
            raise RuntimeError('InputConfiguration: {} directory does not exist. '
                               'Expecting a directory containing the mxnet model files. '
                               'Please make sure the framework you select is correct.'.format(self.model_dir_path))

    def _load_model(self):
        """
        Load MXNet Model
        """
        #------- Compile and set context based on device -----------
        if self.dev_type == 'inf':
            ctx = mx.neuron()  #inferentia context
        elif self.dev_type == 'gpu':
            ctx = mx.gpu(self.dev_id)
        else:
            ctx = mx.cpu()
        #-------Standard MXNet checkpoint load -----------
        sym, self.args, self.aux = mx.model.load_checkpoint(os.path.join(self.model_dir_path,
                                                            self.model_prefix), int(self.epoch))

        #-------Standard MXNet Module instantiation code-----------
        self.mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        print(self.mod.data_names)

    def _validate_models(self):
        """
        Check if the model directory contains valid model files.
        Mxnet models have name-symbol.json and name-epoch.params files
        Set model prefix and epoch.
        """
        model_artifacts = os.listdir(self.model_dir_path)
        model_path = None
        model_name = None
        model_params_path = None
        model_params_name = None
        for model_filename in model_artifacts:
            model_file_path = os.path.join(self.model_dir_path, model_filename)
            if os.path.isfile(model_file_path):
                if model_file_path.endswith('-symbol.json'):
                    if model_path is not None:
                        raise RuntimeError('InputConfiguration: Exactly one -symbol.json file is allowed for Mxnet models.')
                    model_path = model_file_path
                    model_name = model_filename
                elif model_file_path.endswith('.params'):
                    if model_params_path is not None:
                        raise RuntimeError('InputConfiguration: Exactly one .params file is allowed for Mxnet models.')
                    model_params_path = model_file_path
                    model_params_name = model_filename
        if model_path is None:
            raise RuntimeError('InputConfiguration: No valid Mxnet model file -symbol.json found. '
                               'Please make sure the framework you select is correct.')
        if model_params_path is None:
            raise RuntimeError('InputConfiguration: No valid Mxnet model file .params found. '
                               'Please make sure the framework you select is correct.')
        self.model_prefix, symbol  = self._get_prefix(model_name)
        model_params_prefix, self.epoch = self._get_prefix(model_params_name)
        if self.model_prefix != model_params_prefix:
            raise RuntimeError('InputConfiguration: Invalid model file {}. '
                               'Files symbol.json and .params prefix does not match. '
                               'Only name-symbol.json and name-epoch.params file are allowed for Mxnet model. '
                               'Please make sure the framework you select is correct.'.format(model_file_path))


    def _get_prefix(self, model_input):
        """
        Split the model file name

        Parameters
        ----------
        model_input : str
            Full name of the model file

        Returns
        -------
        model_input_name_parts : list of :py:class:`str`
            List of the words in the model file name
        """
        model_input_name = model_input.split(".")[0]
        model_input_name_parts = model_input_name.split("-")
        if (len(model_input_name_parts) < 2):
            raise RuntimeError('InputConfiguration: Invalid model file {}. '
                                'Only name-symbol.json and name-epoch.params file are allowed for Mxnet models. '
                                'Please make sure the framework you select is correct.'.format(model_input))
        return model_input_name_parts

    def get_input_names(self):
        return self.mod.data_names

    def get_input(self, name=None, shape=None):
        """
        Get the current value of an input
        Parameters
        ----------
        name : str
            The name of an input.
        shape : :py:class:`numpy.array` (optional)
            If given, use as the shape of the returned array. Otherwise, the shape of
            the returned array will be inferred from the last call to set_input().

        Returns
        -------
        out : :py:class:`numpy.ndarray`
        """
        out = self.input
        if shape is not None:
           out = out.reshape(shape)
        return out

    def get_output_names(self):
        return self.mod.output_names

    def get_version(self):
        """
        Get DLR version

        Returns
        -------
        out : py:class:`int`
        """
        pass
