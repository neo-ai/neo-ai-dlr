# coding: utf-8
from __future__ import absolute_import as _abs

import abc
import glob
import os
from .neologger import create_logger
from .counter import call_phone_home


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
    
    @call_phone_home
    def __init__(self, model_path, dev_type=None, dev_id=None, error_log_file=None, use_default_dlr=False):
        """
        Load a Neo-compiled model.

        Parameters
        ----------
        model_path : str
            Full path to the directory containing the compiled model artifacts (.so, .params, .json)
        dev_type : str
            Device type ('cpu', 'gpu', or 'opencl')
        dev_id : int (optional)
            Device ID. Default is 0.
        error_log_file: str (optional)
            File to log errors to.
        use_default_dlr: bool
            DLR will load libdlr.so from the compiled model artifacts if it is available. This
            setting can override that behavior to use the system installed DLR when use_default_dlr
            is True.
        """
        self.neo_logger = create_logger(log_file=error_log_file)
        try:
            # Find correct runtime implementation for the model
            self._model = model_path
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
            self._impl = DLRModelImpl(model_path, dev_type, dev_id, error_log_file, use_default_dlr)
        except Exception as ex:
            self.neo_logger.exception("error in DLRModel instantiation {}".format(ex))
            raise ex

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
        try:
            return self._impl.run(input_values)
        except Exception as ex:
            self.neo_logger.exception("error in running inference {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    def get_input_names(self):
        """
        Get all input names

        Returns
        -------
        out : list of :py:class:`str`
        """
        try:
            return self._impl.get_input_names()
        except Exception as ex:
            self.neo_logger.exception("error in getting input names {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    def get_input(self, name, shape=None):
        """
        Get the current value of an input.

        Parameters
        ----------
        name : str
            The name of an input
        shape : np.array (optional)
            If given, use as the shape of the returned array. Otherwise, the shape of
            the returned array will be inferred from the last call to set_input().
        """
        try:
            return self._impl.get_input(name, shape)
        except Exception as ex:
            self.neo_logger.exception("error in getting inputs {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    def get_output_names(self):
        """
        Get all output names. Only valid when the model has a metadata file.

        Returns
        -------
        names : list of :py:class:`str`
        """
        try:
            return self._impl.get_output_names()
        except Exception as ex:
            self.neo_logger.exception("error in getting output names {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    
    def get_version(self):
        """
        Get version of loaded DLR library.

        Returns
        -------
        version : str "{major}.{minor}.{patch}"
        """
        try:
            return self._impl.get_version()
        except Exception as ex:
            self.neo_logger.exception("error in getting version {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    def has_metadata(self):
        """
        Whether the model has a metadata file which provides additional information such as output names.

        Returns
        -------
        has_metadata : bool
        """
        try:
            return self._impl.has_metadata()
        except Exception as ex:
            self.neo_logger.exception("error in checking for metadata file {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    def get_input_dtypes(self):
        """
        Get datatype of all inputs.

        Returns
        -------
        dtypes : list of :py:class:`str`
        """
        try:
            return self._impl.get_input_dtypes()
        except Exception as ex:
            self.neo_logger.exception("error in getting input data types {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    def get_output_dtypes(self):
        """
        Get datatype of all outputs.

        Returns
        -------
        dtypes : list of :py:class:`str`
        """
        try:
            return self._impl.get_output_dtypes()
        except Exception as ex:
            self.neo_logger.exception("error in getting output data types {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    def get_input_name(self, index):
        """
        Get the name of the input at the given index.

        Parameters
        ----------
        index : int
            Index of the input

        Returns
        -------
        name : str
        """
        try:
            return self._impl.get_input_name(index)
        except Exception as ex:
            self.neo_logger.exception("error in getting input name {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    def get_output_name(self, index):
        """
        Get the name of the output at the given index. Only valid when the model has a metadata file.

        Parameters
        ----------
        index : int
            Index of the input

        Returns
        -------
        name : str
        """
        try:
            return self._impl.get_output_name(index)
        except Exception as ex:
            self.neo_logger.exception("error in getting output name {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    def get_input_dtype(self, index):
        """
        Get the type of the input at the given index.

        Parameters
        ----------
        index : int
            Index of the input

        Returns
        -------
        type : str
        """
        try:
            return self._impl.get_input_dtype(index)
        except Exception as ex:
            self.neo_logger.exception("error in getting input data type {} {}".format(self._impl.__class__.__name__, ex))
            raise ex

    def get_output_dtype(self, index):
        """
        Get the type of the output at the given index.

        Parameters
        ----------
        index : int
            Index of the input

        Returns
        -------
        type : str
        """
        try:
            return self._impl.get_output_dtype(index)
        except Exception as ex:
            self.neo_logger.exception("error in getting output data type {} {}".format(self._impl.__class__.__name__, ex))
            raise ex
