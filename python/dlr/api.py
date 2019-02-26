# coding: utf-8
from __future__ import absolute_import as _abs

import ctypes
from ctypes import cdll
from ctypes import c_void_p, c_int, c_float, c_char_p, byref, POINTER, c_longlong
import numpy as np
from functools import reduce
from operator import mul
from argparse import ArgumentParser
import sys
import os

from .libpath import find_lib_path

class DLRError(Exception):
    """Error thrown by DLR"""
    pass

def _load_lib():
    """Load DLR library."""
    lib_paths = find_lib_path()
    if len(lib_paths) == 0:
        return None
    try:
        pathBackup = os.environ['PATH'].split(os.pathsep)
    except KeyError:
        pathBackup = []
    lib_success = False
    os_error_list = []
    for lib_path in lib_paths:
        try:
            # needed when the lib is linked with non-system-available dependencies
            os.environ['PATH'] = os.pathsep.join(pathBackup + [os.path.dirname(lib_path)])
            lib = ctypes.cdll.LoadLibrary(lib_path)
            lib_success = True
        except OSError as e:
            os_error_list.append(str(e))
            continue
        finally:
            os.environ['PATH'] = os.pathsep.join(pathBackup)
    if not lib_success:
        libname = os.path.basename(lib_paths[0])
        raise DLRError(
            'DLR library ({}) could not be loaded.\n'.format(libname) +
            'Likely causes:\n' +
            '  * OpenMP runtime is not installed ' +
            '(vcomp140.dll or libgomp-1.dll for Windows, ' +
            'libgomp.so for UNIX-like OSes)\n' +
            '  * You are running 32-bit Python on a 64-bit OS\n' +
            'Error message(s): {}\n'.format(os_error_list))
    lib.DLRGetLastError.restype = ctypes.c_char_p
    return lib


# load the DLR library globally
_LIB = _load_lib()


def _check_call(ret):
    """
    Check the return value of C API call
    This function will raise exception when error occurs.
    Wrap every API call with this function

    Parameters
    ----------
    ret : int
        return value from API calls
    """
    if ret != 0:
        raise DLRError(_LIB.DLRGetLastError().decode('ascii'))

class DLRModel:
    """
    Load a Neo-compiled model

    Parameters
    ----------
    model_path : str
        Full path to the directory containing the compiled model
    dev_type : str
        Device type ('cpu', 'gpu', or 'opencl')
    dev_id : int
        Device ID
    """

    def _lazy_init_output_shape(self):
        self.output_shapes = []
        self.output_size_dim = []
        for i in range(self.num_outputs):
            shape = self._get_output_shape(i)
            self.output_shapes.append(shape)

    def _parse_backend(self):
        backend = c_char_p()
        _check_call(_LIB.GetDLRBackend(byref(self.handle),
                                       byref(backend)))
        return backend.value.decode('ascii')

    def __init__(self, model_path, dev_type='cpu', dev_id=0):
        if not os.path.exists(model_path):
            raise ValueError("model_path %s doesn't exist" % model_path)

        self.handle = c_void_p()
        device_table = {
            'cpu': 1,
            'gpu': 2,
            'opencl': 4,
        }

        _check_call(_LIB.CreateDLRModel(byref(self.handle),
                                        c_char_p(model_path.encode()),
                                        c_int(device_table[dev_type]),
                                        c_int(dev_id)))

        self.backend = self._parse_backend()

        self.num_inputs = self._get_num_inputs()
        self.input_names = []
        for i in range(self.num_inputs):
            self.input_names.append(self._get_input_name(i))

        self.num_outputs = self._get_num_outputs()
        self._lazy_init_output_shape()

    def __del__(self):
        if getattr(self, "handle", None) is not None and self.handle is not None:
            if getattr(self, "lib", None) is not None:
                _check_call(_LIB.DeleteDLRModel(byref(self.handle)))
            self.handle = None

    def _get_num_inputs(self):
        """Get the number of inputs of a network"""
        num_inputs = c_int()
        _check_call(_LIB.GetDLRNumInputs(byref(self.handle),
                                         byref(num_inputs)))
        return num_inputs.value

    def get_input_names(self):
        """
        Get all input names

        Returns
        -------
        out : list of :py:class:`str`
        """
        return self.input_names

    def _get_input_name(self, index):
        name = ctypes.c_char_p()
        _check_call(_LIB.GetDLRInputName(byref(self.handle),
                                         c_int(index), byref(name)))
        return name.value.decode("utf-8")

    def _set_input(self, name, data):
        """Set the input using the input name with data

        Parameters
        __________
        name : str
            The name of an input.
        data : list of numbers
            The data to be set.
        """
        in_data = np.ascontiguousarray(data, dtype=np.float32)
        shape = np.array(in_data.shape, dtype=np.int64)
        _check_call(_LIB.SetDLRInput(byref(self.handle),
                                     c_char_p(name.encode('utf-8')),
                                     shape.ctypes.data_as(POINTER(c_longlong)),
                                     in_data.ctypes.data_as(POINTER(c_float)),
                                     c_int(in_data.ndim)))
        if self.backend == 'treelite':
            self._lazy_init_output_shape()

    def _run(self):
        """A light wrapper to call run in the DLR backend."""
        _check_call(_LIB.RunDLRModel(byref(self.handle)))

    def _get_num_outputs(self):
        """Get the number of outputs of a network"""
        num_outputs = c_int()
        _check_call(_LIB.GetDLRNumOutputs(byref(self.handle),
                                          byref(num_outputs)))
        return num_outputs.value

    def _get_output_size_dim(self, index):
        """Get the size and the dimenson of the index-th output.

        Parameters
        __________
        index : int
            The index of the output.

        Returns
        _______
        size : int
            The size of the index-th output.
        dim : int
            The dimension of the index-th output.
        """
        idx = ctypes.c_int(index)
        size = ctypes.c_longlong()
        dim = ctypes.c_int()
        _check_call(_LIB.GetDLROutputSizeDim(byref(self.handle), idx,
                                                      byref(size), byref(dim)))
        return size.value, dim.value

    def _get_output_shape(self, index):
        """Get the shape for the index-th output.

        Parameters
        __________
        index : int
            The index of the output.

        Returns
        _______
        shape : list
            The shape of the index-th output.
        """
        size, dim = self._get_output_size_dim(index)
        if not self.output_size_dim:
            self.output_size_dim = [(0, 0)] * self._get_num_outputs()
        self.output_size_dim[index] = (size, dim)
        shape = np.zeros(dim, dtype=np.int64)
        _check_call(_LIB.GetDLROutputShape(byref(self.handle),
                                                    c_int(index),
                    shape.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))))
        return shape

    def _get_output(self, index):
        """Get the index-th output

        Parameters
        __________
        index : int
            The index of the output.

        Returns
        _______
        out : np.array
            A numpy array contains the values of the index-th output
        """
        if index >= len(self.output_shapes) or index < 0:
            raise ValueError("index is expected between 0 and "
                             "len(output_shapes)-1, but got %d" % index)

        output = np.zeros(self.output_size_dim[index][0], dtype=np.float32)
        _check_call(_LIB.GetDLROutput(byref(self.handle), c_int(index),
                    output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
        out = output.reshape(self.output_shapes[index])
        return out

    def run(self, input_values):
        """
        Run inference with given input(s)

        Parameters
        ----------
        input_values : a single :py:class:`numpy.ndarray` or a dictionary
            For decision tree models, provide a single :py:class:`numpy.ndarray`
            to indicate a single input, as decision trees always accept only one
            input.

            For deep learning models, provide a dictionary where keys are input
            names (of type :py:class:`str`) and values are input tensors (of type
            :py:class:`numpy.ndarray`). Deep learning models allow more than one
            input, so each input must have a unique name.

        Returns
        -------
        out : :py:class:`numpy.ndarray`
            Prediction result
        """
        out = []
        # set input(s)
        if isinstance(input_values, (np.ndarray, np.generic)):
            # Treelite model or single input tvm/treelite model.
            # Treelite has a dummy input name 'data'.
            if self.input_names:
                self._set_input(self.input_names[0], input_values)
        elif isinstance(input_values, dict):
            # TVM model
            for key, value in input_values.items():
                if self.input_names and key not in self.input_names:
                    raise ValueError("%s is not a valid input name." % key)
                self._set_input(key, value)
        else:
            raise ValueError("input_values must be of type dict (tvm model) " +
                             "or a np.ndarray/generic (representing treelite models)")
        # run model
        self._run()
        # get output
        for i in range(self.num_outputs):
            ith_out = self._get_output(i)
            out.append(ith_out)
        return out
