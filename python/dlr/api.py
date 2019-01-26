import ctypes
from ctypes import cdll
from ctypes import c_void_p, c_int, c_float, c_char_p, byref, POINTER, c_longlong
import numpy as np
from functools import reduce
from operator import mul
from argparse import ArgumentParser
import sys
import os


class DLRError(Exception):
    """DLR exception class"""
    pass


class DLRModel:

    def _check_call(self, ret):
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
            raise DLRError(self.lib.DLRGetLastError().decode('ascii'))

    def __init__(self, tar_path, dev_type='cpu', dev_id=0):
        if not os.path.exists(tar_path):
            raise ValueError("tar_path %s doesn't exist" % tar_path)

        self.handle = c_void_p()
        libpath = os.path.join(os.path.dirname(
            os.path.abspath(os.path.expanduser(__file__))), 'libdlr.so')
        self.lib = cdll.LoadLibrary(libpath)
        self.lib.DLRGetLastError.restype = ctypes.c_char_p
        device_table = {
            'cpu': 1,
            'gpu': 2,
            'opencl': 4,
        }

        self._check_call(self.lib.CreateDLRModel(byref(self.handle),
                                                 c_char_p(tar_path.encode()),
                                                 c_int(device_table[dev_type]),
                                                 c_int(dev_id)))

        self.num_inputs = self._get_num_inputs()
        self.input_names = []
        for i in range(self.num_inputs):
            self.input_names.append(self._get_input_name(i))

        self.num_outputs = self._get_num_outputs()
        self.output_shapes = []
        self.output_size_dim = []
        for i in range(self.num_outputs):
            shape = self._get_output_shape(i)
            self.output_shapes.append(shape)

    def __del__(self):
        if getattr(self, "handle", None) is not None and self.handle is not None:
            if getattr(self, "lib", None) is not None:
                self._check_call(self.lib.DeleteDLRModel(byref(self.handle)))
            self.handle = None

    def _get_num_inputs(self):
        """Get the number of inputs of a network"""
        num_inputs = c_int()
        self._check_call(self.lib.GetDLRNumInputs(byref(self.handle),
                                                  byref(num_inputs)))
        return num_inputs.value

    def get_input_names(self):
        """Get all input_names"""
        return self.input_names

    def _get_input_name(self, index):
        name = ctypes.c_char_p()
        self._check_call(self.lib.GetDLRInputName(byref(self.handle),
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
        self._check_call(self.lib.SetDLRInput(byref(self.handle),
                                              c_char_p(name.encode('utf-8')),
                                              shape.ctypes.data_as(POINTER(c_longlong)),
                                              in_data.ctypes.data_as(POINTER(c_float)),
                                              c_int(in_data.ndim)))

    def _run(self):
        """A light wrapper to call run in the DLR backend."""
        self._check_call(self.lib.RunDLRModel(byref(self.handle)))

    def _get_num_outputs(self):
        """Get the number of outputs of a network"""
        num_outputs = c_int()
        self._check_call(self.lib.GetDLRNumOutputs(byref(self.handle),
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
        idx = ctypes.c_int()
        size = ctypes.c_longlong()
        dim = ctypes.c_int()
        self._check_call(self.lib.GetDLROutputSizeDim(byref(self.handle), idx,
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
        self.output_size_dim.append((size, dim))
        shape = np.zeros(dim, dtype=np.int64)
        self._check_call(self.lib.GetDLROutputShape(byref(self.handle),
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
        self._check_call(self.lib.GetDLROutput(byref(self.handle), c_int(index),
                         output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
        out = output.reshape(self.output_shapes[index])
        return out

    def run(self, input_values):
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
