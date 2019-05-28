import logging
import os
from collections import OrderedDict
import tensorflow.lite as lite
from .api import IDLRModel


def _get_input_and_output_names(intrp):
    """
    Get input and output tensor names

    Parameters
    ----------
    intrp : :py:class:`str`tf.lite.Interpreter`
        TFLite Interpreter

    Returns
    -------
    input_tensor_dict : Input tensor Dict which maps tensor_name to tensor_id
    output_tensor_dict : Output tensor Dict which maps tensor_name to tensor_id
    """
    # Use OrderedDict to keep tensor names order
    input_tensor_dict = OrderedDict()
    output_tensor_dict = OrderedDict()
    for d in intrp.get_input_details():
        input_tensor_dict[d["name"]] = d["index"]
    for d in intrp.get_output_details():
        output_tensor_dict[d["name"]] = d["index"]
    return input_tensor_dict, output_tensor_dict


class TFLiteModelImpl(IDLRModel):
    """
    TFLiteModelImpl is a wrapper on top of Tensorflow Lite which implements IDLRModel API

    Parameters
    ----------
    tflite_file : :py:class:`str`
        Full path to TFLite file (.tflite file)
    """
    def __init__(self, tflite_file):
        if not tflite_file.endswith(".tflite"):
            raise ValueError("Not a TFLite file: {}".format(tflite_file))
        if not os.path.exists(tflite_file):
            raise ValueError("TFLite file {} doesn't exist".format(tflite_file))

        logging.info("Loading TFLite file: {}".format(tflite_file))
        self._intrp = lite.Interpreter(model_path=tflite_file)
        self._intrp.allocate_tensors()
        self._input_tensor_dict, self._output_tensor_dict = _get_input_and_output_names(self._intrp)

    def _validate_input_name(self, name):
        if name not in self._input_tensor_dict:
            raise ValueError(
                "Invalid input tensor name '{}'. List of input tensor names: {}".format(name, self.get_input_names()))

    def _validate_input(self, input_values):
        if isinstance(input_values, dict):
            for k in input_values.keys():
                if not isinstance(k, str):
                    raise ValueError("input key must be string")
                self._validate_input_name(k)
        else:
            raise ValueError("input_values must be of type dict")

    def get_input_names(self):
        """
        Get all input names

        Returns
        -------
        out : list of :py:class:`str`
        """
        return list(self._input_tensor_dict.keys())

    def get_output_names(self):
        """
        Get all output names

        Returns
        -------
        out : list of :py:class:`str`
        """
        return list(self._output_tensor_dict.keys())

    def get_input(self, name, shape=None):
        """
        Get the current value of an input

        Parameters
        ----------
        name : :py:class:`str`
            The name of an input
        shape : :py:class:`np.array` (optional)
            If given, use as the shape of the returned array. Otherwise, the shape of
            the returned array will be inferred from the last call to set_input().
        """
        self._validate_input_name(name)
        tensor_index = self._input_tensor_dict[name]
        out = self._intrp.get_tensor(tensor_index)
        if shape is not None:
            out = out.reshape(shape)
        return out

    def run(self, input_values):
        """
        Run inference with given input(s)

        Parameters
        ----------
        input_values : a dictionary where keys are input
            names (of type :py:class:`str`) and values are input tensors (of any type).
            Multiple inputs are allowed.

        Returns
        -------
        out : :py:class:`list`
            Prediction result. Multiple outputs are possible.
        """
        self._validate_input(input_values)
        for k, v in input_values.items():
            tensor_index = self._input_tensor_dict[k]
            self._intrp.set_tensor(tensor_index, v)
        self._intrp.invoke()
        return [self._intrp.get_tensor(v) for v in self._output_tensor_dict.values()]
