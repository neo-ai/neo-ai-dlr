# coding: utf-8
import ctypes
from ctypes import c_void_p, c_int, c_char_p, byref, POINTER, c_longlong
import json
import numpy as np
import os
import sys
from pathlib import Path

from .api import IDLRModel
from .libpath import find_lib_path
from .neologger import create_logger

# Map from dtype string to ctype type.
# Equivalent to np.ctypeslib.as_ctypes_type which requires numpy>=1.16.1
DTYPE_TO_CTYPE = {
    "float32": ctypes.c_float,
    "float64": ctypes.c_double,
    "uint8": ctypes.c_ubyte,
    "uint32": ctypes.c_uint,
    "uint64": ctypes.c_ulong,
    "int8": ctypes.c_byte,
    "int32": ctypes.c_int,
    "int64": ctypes.c_long,
}

def _get_ctype_from_dtype(dtype):
    """
    Convert type string to ctype type.

    Parameters
    ----------
    dtype: str
        Type as a string, e.g. "float32".
    """
    if dtype not in DTYPE_TO_CTYPE:
        raise ValueError("Model has input or output datatype {} which is not supported.".format(dtype))
    return DTYPE_TO_CTYPE[dtype]

class DLRError(Exception):
    """Error thrown by DLR"""
    pass

def _load_lib(lib_path):
    """Load DLR library."""
    try:
        pathBackup = os.environ['PATH'].split(os.pathsep)
    except KeyError:
        pathBackup = []
    
    try:
        # needed when the lib is linked with non-system-available dependencies
        os.environ['PATH'] = os.pathsep.join(pathBackup + [os.path.dirname(lib_path)])
        lib = ctypes.cdll.LoadLibrary(lib_path)
    except Exception as e:
        libname = os.path.basename(lib_path)
        raise DLRError(
            'DLR library ({}) could not be loaded.\n'.format(libname) +
            'Likely causes:\n' +
            '  * OpenMP runtime is not installed ' +
            '(vcomp140.dll or libgomp-1.dll for Windows, ' +
            'libgomp.so for UNIX-like OSes)\n' +
            '  * You are running 32-bit Python on a 64-bit OS\n' +
            'Error message(s): {}\n'.format(e))
    finally:
        os.environ['PATH'] = os.pathsep.join(pathBackup)
    
    lib.DLRGetLastError.restype = ctypes.c_char_p
    return lib

class DLRModelImpl(IDLRModel):
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
    

    def __init__(self, model_path, dev_type='cpu', dev_id=0, error_log_file=None, use_default_dlr=False):
        self.logger = create_logger(log_file=error_log_file)
        
        if not os.path.exists(model_path):
            raise ValueError("model_path %s doesn't exist" % model_path)
        for file_name in os.listdir(model_path):
            if file_name.endswith(".tensorrt"):
                raise Exception("This model requires DLR release-1.1.0 to run.")
        self.handle = c_void_p()
        device_table = {
            'cpu': 1,
            'gpu': 2,
            'opencl': 4,
        }
        self.model_path = model_path
        self.use_default_dlr = use_default_dlr
        self._lib = None
        self._init_libdlr()
        self._check_call(self._lib.CreateDLRModel(byref(self.handle),
                                        c_char_p(model_path.encode()),
                                        c_int(device_table[dev_type]),
                                        c_int(dev_id)))

        self.backend = self._parse_backend()
        self.version = self._get_version()

        self.num_inputs = self._get_num_inputs()
        self.num_weights = self._get_num_weights()
        self.input_names = []
        self.input_name_to_index = {}
        self.output_names = []
        self.weight_names = []
        self.input_shapes = {}   # Remember shape used in _set_input()
        self.input_dtypes = []
        self.output_dtypes = []
        
        for i in range(self.num_weights):
            self.weight_names.append(self._get_weight_name(i))

        self.num_outputs = self._get_num_outputs()
        if self.backend != "relayvm":
            self._lazy_init_output_shape()
        self._fetch_input_names()
        self._fetch_input_dtypes()
        self._fetch_output_dtypes()

    def __del__(self):
        if getattr(self, "handle", None) is not None and self.handle is not None:
            if getattr(self, "_lib", None) is not None:
                self._check_call(self._lib.DeleteDLRModel(byref(self.handle)))
            self.handle = None

    def _lazy_init_output_shape(self):
        self.output_shapes = []
        self.output_size_dim = []
        for i in range(self.num_outputs):
            shape = self._get_output_shape(i)
            self.output_shapes.append(shape)

    def _parse_backend(self):
        backend = c_char_p()
        self._check_call(self._lib.GetDLRBackend(byref(self.handle),
                                       byref(backend)))
        return backend.value.decode('ascii')

    def _get_version(self):
        version = c_char_p()
        self._check_call(self._lib.GetDLRVersion(byref(version)))
        return version.value.decode('ascii')

    def _init_libdlr(self):
        self._lib = _load_lib(find_lib_path(self.model_path, self.use_default_dlr, self.logger))

    def _get_num_inputs(self):
        """Get the number of inputs of a network"""
        num_inputs = c_int()
        self._check_call(self._lib.GetDLRNumInputs(byref(self.handle),
                                         byref(num_inputs)))
        return num_inputs.value

    def _get_num_weights(self):
        """Get the number of weights of a network"""
        num_weights = c_int()
        self._check_call(self._lib.GetDLRNumWeights(byref(self.handle),
                                          byref(num_weights)))
        return num_weights.value

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
            raise DLRError(self._lib.DLRGetLastError().decode('ascii'))

    def get_input_names(self):
        """
        Get all input names

        Returns
        -------
        out : list of :py:class:`str`
        """
        return self.input_names

    def has_metadata(self) -> bool:
        flag = ctypes.c_bool()
        self._check_call(self._lib.GetDLRHasMetadata(byref(self.handle), byref(flag)))
        return flag.value
    
    def _fetch_output_names(self):
        self.output_names = []
        try:
            for i in range(self.num_outputs):
                name = c_char_p()
                self._check_call(self._lib.GetDLROutputName(byref(self.handle), i, byref(name)))
                self.output_names.append(name.value.decode('utf-8'))
        except Exception:
            """
                currently only tvm, tf_lite and treelite support this. For the backends that don't
                support this we throw the NotImplementedError in get_output_names method
            """
            pass

    def _fetch_input_names(self):
        for i in range(self.num_inputs):
            name = self._get_input_name(i)
            self.input_names.append(name)
            self.input_name_to_index[name] = i
        
    def _fetch_input_dtypes(self):
        self.input_dtypes = []
        try:
            for i in range(self.num_inputs):
                dtype = c_char_p()
                self._check_call(self._lib.GetDLRInputType(byref(self.handle), i, byref(dtype)))
                self.input_dtypes.append(dtype.value.decode('utf-8'))
        except Exception:
            """
                currently only tvm, tf_lite and treelite support this. For the backends that don't
                support this we throw the NotImplementedError in get_input_dtypes method
            """
            pass
        
    def _fetch_output_dtypes(self):
        self.output_dtypes = []
        try:
            for i in range(self.num_outputs):
                dtype = c_char_p()
                self._check_call(self._lib.GetDLROutputType(byref(self.handle), i, byref(dtype)))
                self.output_dtypes.append(dtype.value.decode('utf-8'))
        except Exception:
            """
                currently only tvm, tf_lite and treelite support this. For the backends that don't
                support this we throw the NotImplementedError in get_output_dtypes method
            """
            pass

    def get_output_names(self):
        if not self.output_names:
            self._fetch_output_names()
        return self.output_names

    def get_input_dtypes(self):
        if not self.input_dtypes:
            raise NotImplementedError
        return self.input_dtypes

    def get_output_dtypes(self):
        if not self.output_dtypes:
            raise NotImplementedError
        return self.output_dtypes

    def get_input_name(self, index):
        if not (0 <= index < self.num_outputs):
            raise Exception("Index cannot be greater than {}".format(self.num_inputs - 1))
        return self.get_input_names()[index]

    def get_output_name(self, index):
        if not (0 <= index < self.num_outputs):
            raise Exception("Index cannot be greater than {}".format(self.num_outputs - 1))
        return self.get_output_names()[index]

    def get_input_dtype(self, index):
        if not (0 <= index < self.num_outputs):
            raise Exception("Index cannot be greater than {}".format(self.num_inputs - 1))
        return self.get_input_dtypes()[index]

    def get_output_dtype(self, index):
        if not (0 <= index < self.num_outputs):
            raise Exception("Index cannot be greater than {}".format(self.num_outputs - 1))
        return self.get_output_dtypes()[index]

    def get_version(self):
        """
        Get DLR version

        Returns
        -------
        out : py:class:`int`
        """
        return self.version

    def _get_input_name(self, index):
        name = ctypes.c_char_p()
        self._check_call(self._lib.GetDLRInputName(byref(self.handle),
                                         c_int(index), byref(name)))
        return name.value.decode("utf-8")

    def _get_input_index(self, name) -> int:
        index = self.input_name_to_index.get(name)
        if index is None:
            raise ValueError("{} is not a valid input name.".format(name))
        return index

    def _get_weight_name(self, index):
        name = ctypes.c_char_p()
        self._check_call(self._lib.GetDLRWeightName(byref(self.handle),
                                          c_int(index), byref(name)))
        return name.value.decode("utf-8")

    def _get_input_or_weight_dtype_by_name(self, name):
        if name in self.weight_names:
            return "float32"
        return self.get_input_dtype(self._get_input_index(name))

    def _set_input(self, name, data):
        """Set the input using the input name with data

        Parameters
        __________
        name : str
            The name of an input.
        data : list of numbers
            The data to be set.
        """
        input_dtype = self._get_input_or_weight_dtype_by_name(name)
        if input_dtype == "json":
            # Special case for DataTransformed inputs. DLR will expect input as a serialized json
            # string.
            in_data = json.dumps(data.tolist())
            in_data_pointer = c_char_p(in_data.encode('utf-8'))
            shape = np.array([len(in_data)], dtype=np.int64)
        else:
            input_ctype = _get_ctype_from_dtype(input_dtype)
            # float32 inputs can accept any data (backward compatibility).
            if input_dtype == "float32":
                type_match = True
            else:
                type_match = (data.dtype.name == input_dtype)
            if not type_match:
                raise ValueError("input data with name {} should have dtype {} but {} is provided".
                                format(name, input_dtype, data.dtype.name))
            in_data = np.ascontiguousarray(data, dtype=input_dtype)
            in_data_pointer = in_data.ctypes.data_as(POINTER(input_ctype))
            shape = np.array(in_data.shape, dtype=np.int64)
        self.input_shapes[name] = shape
        self._check_call(self._lib.SetDLRInput(byref(self.handle),
                                     c_char_p(name.encode('utf-8')),
                                     shape.ctypes.data_as(POINTER(c_longlong)),
                                     in_data_pointer,
                                     c_int(len(shape))))
        if self.backend == 'treelite':
            self._lazy_init_output_shape()

    def _run(self):
        """A light wrapper to call run in the DLR backend."""
        self._check_call(self._lib.RunDLRModel(byref(self.handle)))
        if self.backend == "relayvm":
            self._lazy_init_output_shape()

    def _get_num_outputs(self):
        """Get the number of outputs of a network"""
        num_outputs = c_int()
        self._check_call(self._lib.GetDLRNumOutputs(byref(self.handle),
                                          byref(num_outputs)))
        return num_outputs.value

    def _get_output_size_dim(self, index):
        """Get the size and the dimension of the index-th output.

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
        self._check_call(self._lib.GetDLROutputSizeDim(byref(self.handle), idx,
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
        self._check_call(self._lib.GetDLROutputShape(byref(self.handle),
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
        output_dtype = self.get_output_dtype(index)
        output_ctype = _get_ctype_from_dtype(output_dtype)
        output = np.zeros(self.output_size_dim[index][0], dtype=output_dtype)
        self._check_call(self._lib.GetDLROutput(byref(self.handle), c_int(index),
                    output.ctypes.data_as(ctypes.POINTER(output_ctype))))
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
                if (self.input_names and key not in self.input_names) and \
                   (self.weight_names and key not in self.weight_names):
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

    def get_input(self, name, shape=None):
        """
        Get the current value of an input

        Parameters
        ----------
        name : str
            The name of an input
        shape : np.array (optional)
            If given, use as the shape of the returned array. Otherwise, the shape of
            the returned array will be inferred from the last call to set_input().
        """
        if name not in self.input_shapes and shape is None:
            raise ValueError('Since set_input() was never called with ' +
                             'input {}, we cannot infer its shape. '.format(name) +
                             'Shape parameter should be explicitly specified')
        input_dtype = self._get_input_or_weight_dtype_by_name(name)
        input_ctype = _get_ctype_from_dtype(input_dtype)
        if shape is None:
            shape = self.input_shapes[name]
        shape = np.array(shape)
        out = np.zeros(shape.prod(), dtype=input_dtype)
        self._check_call(self._lib.GetDLRInput(byref(self.handle),
                                     c_char_p(name.encode('utf-8')),
                                     out.ctypes.data_as(ctypes.POINTER(input_ctype))))
        out = out.reshape(shape)
        return out
