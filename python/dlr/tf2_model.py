import tensorflow as tf
from tensorflow.python.tools import saved_model_utils
from packaging import version

from .api import IDLRModel
from .neologger import create_logger
from .metadata import VERSION, MIN_TENSORFLOW_VERSION

class TF2ModelImpl(IDLRModel):
    """
    TF2ModelImpl is a wrapper on top of tensorflow which implements DLRModel API
    Parameters
    ----------
    model_path : str
        Full path to the saved model directory
    dev_type : str
        Device type ('cpu' or 'gpu')
    dev_id : int
        Device ID
    """

    def __init__(self, model_path, dev_type=None, dev_id=None, error_log_file=None, use_default_dlr=False):
        # check for Tensorflow 2.x
        assert version.parse(tf.__version__) >= version.parse(MIN_TENSORFLOW_VERSION), \
            "Minimem Tensorflow supported version is {}, got {}".format(MIN_TENSORFLOW_VERSION, tf.__version__)
        self.model_path = model_path
        self.logger = create_logger(log_file=error_log_file)
        self.use_default_dlr = use_default_dlr
        if dev_type is not None or dev_id is not None:
            self.logger.warning("dev_type and dev_id are not supported for TF2 Models and the params are ignored.")
        physical_devices = tf.config.list_physical_devices('GPU')
        for physical_device in physical_devices:
            memory_growth = tf.config.experimental.get_memory_growth(physical_device)
            if not memory_growth:
                try:
                    tf.config.experimental.set_memory_growth(physical_device, True)
                    assert tf.config.experimental.get_memory_growth(physical_device)
                except:
                    self.logger.warning("tf.config.experimental.set_memory_growth failed.")
        self.version = VERSION
        self.saved_model = tf.saved_model.load(model_path)
        tag_set = 'serve'
        assert len(self.saved_model.signatures) > 0, "Found no signatures in the saved model."
        signature_def_key = 'serving_default'
        if signature_def_key not in self.saved_model.signatures:
            signature_def_key = list(self.saved_model.signatures.keys())[0]
        self.func = self.saved_model.signatures[signature_def_key]
        meta_graph_def = saved_model_utils.get_meta_graph_def(model_path, tag_set)
        inputs_tensor_info = meta_graph_def.signature_def[signature_def_key].inputs
        outputs_tensor_info = meta_graph_def.signature_def[signature_def_key].outputs
        self.input_tensor_names = list(inputs_tensor_info.keys())
        self.output_tensor_names = list(outputs_tensor_info.keys())

    def get_input_names(self):
        """
        Get all input names
        Returns
        -------
        out : list of :py:class:`str`
        """
        return self.input_tensor_names

    def get_output_names(self):
        """
        Get all output names
        Returns
        -------
        out : list of :py:class:`str`
        """
        return self.output_tensor_names

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
        self._validate_input_name(name)
        if name not in self.input_values:
            return None
        out = self.input_values[name]
        if shape is not None:
            out = out.reshape(shape)
        return out

    def _validate_input_name(self, name):
        if name not in self.input_tensor_names:
            raise ValueError(
                "Invalid input name '{}'. List of input names: {}".format(name, self.input_tensor_names))

    def _validate_input(self, input_values):
        if isinstance(input_values, dict):
            for k in input_values.keys():
                if not isinstance(k, str):
                    raise ValueError("input key must be string")
                self._validate_input_name(k)
            self.input_values = input_values
        elif isinstance(input_values, (list, tuple)):
            l_names, l_input_values = len(
                self.input_tensor_names), len(input_values)
            assert l_names == l_input_values, "Wrong number of inputs, expected {}, actual {}".format(
                l_names, l_input_values)
            self.input_values = dict(
                zip(self.input_tensor_names, input_values))
        else:
            self.input_values = {self.input_tensor_names[0]: input_values}

    def get_version(self):
        """
        Get DLR version

        Returns
        -------
        out : py:class:`int`
        """
        return self.version

    def run(self, input_values):
        """
        Run inference with given input(s)
        Parameters
        ----------
        input_values :
            Input tensor, or dict/list/tuple of input tensors (of any type).

        Returns
        -------
        out :
            Prediction result.
        """

        self._validate_input(input_values)
        if isinstance(self.input_values, dict):
            out = self.func(**self.input_values)
        else:
            out = self.func(input_values)

        return out
