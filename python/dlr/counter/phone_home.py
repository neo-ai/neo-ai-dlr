import json
import platform
import logging
import os

from .config import (
    CALL_HOME_USR_NOTIFICATION,
    CALL_HOME_USER_CONFIG_FILE,
    CALL_HOME_REQ_STOP_MAX_COUNT,
    CALL_HOME_MODEL_RUN_COUNT_TIME_SECS,
)
from .utils.helper import get_hash_string
from .utils import resturlutils
from .system import Factory

DEVICE_INFO_CONFIG = "enable_device_info"
ENABLE_PHONE_HOME_CONFIG = "enable_phone_home"


def exception_handler(func):
    def callback(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception:
            logging.exception("phone home failure, disable feature")
            PhoneHome.disable_feature()

    return callback


class PhoneHome:
    """
    Phone home client
    """

    instance = None
    _enable_feature = None
    _resp_err_count = 0
    RUNTIME_LOAD = 1
    MODEL_LOAD = 2
    MODEL_RUN = 3

    @staticmethod
    def get_config_path():
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)), CALL_HOME_USER_CONFIG_FILE
        )

    @staticmethod
    def is_enabled():
        feature_enb = False
        try:
            config_path = PhoneHome.get_config_path()
            if os.path.isfile(config_path):
                config_file = open(config_path, "r")
                data = json.loads(config_file.read())
                feature_enb = data.get(ENABLE_PHONE_HOME_CONFIG, False)
                config_file.close()
        except Exception:
            logging.exception("error reading config file")

        return feature_enb

    @staticmethod
    def get_config():
        config = {}
        config_path = PhoneHome.get_config_path()
        if os.path.isfile(config_path):
            with open(config_path, "r") as config_file:
                config = json.loads(config_file.read())
        return config

    @staticmethod
    def disable_feature():
        config = PhoneHome.get_config()
        config_path = PhoneHome.get_config_path()
        with open(config_path, "w") as config_file:
            config[ENABLE_PHONE_HOME_CONFIG] = False
            config_file.write(json.dumps(config))
        PhoneHome.instance = None
        PhoneHome._enable_feature = False

    @staticmethod
    def enable_feature():
        config_path = PhoneHome.get_config_path()
        config = PhoneHome.get_config()
        with open(config_path, "w") as config_file:
            config[ENABLE_PHONE_HOME_CONFIG] = True
            config_file.write(json.dumps(config))
        PhoneHome.instance = PhoneHome()
        PhoneHome._enable_feature = True

    @staticmethod
    def should_enable():
        config_path = PhoneHome.get_config_path()
        if os.path.isfile(config_path):
            return PhoneHome.is_enabled()
        else:
            return True

    @staticmethod
    def get_instance():
        try:
            if not PhoneHome.instance and PhoneHome.should_enable():
                PhoneHome.enable_feature()
            elif PhoneHome.instance and not PhoneHome.should_enable():
                PhoneHome.disable_feature()
        except Exception:
            logging.debug("unsupported system for call home feature")
        finally:
            return PhoneHome.instance

    def __init__(self):
        try:
            self.client = resturlutils.RestUrlUtils()

            machine_type = platform.machine()
            os_name = platform.system()
            os_supt = "{}_{}".format(os_name, machine_type)

            self.system = Factory.get_system(os_supt)
            if self.system is None:
                raise Exception("system is not supported")

            self.send_runtime_loaded()
            print(CALL_HOME_USR_NOTIFICATION)
        
        except Exception as ex:
            logging.debug("phone init error")

    def _is_device_info_sent(self):
        """ check if device information send only single time in DLR installation life time.
            create a dummy binary file in DLR installation path to record process state.

        Returns:
            bool: if device info is already sent
        """
        is_sent = False
        try:
            config_path = PhoneHome.get_config_path()
            config = PhoneHome.get_config()

            device_info = config.get(DEVICE_INFO_CONFIG, False)
            if not device_info:
                with open(config_path, "w") as config_file:
                    config[DEVICE_INFO_CONFIG] = True
                    config_file.write(json.dumps(config))
                    is_sent = True

        except Exception:
            logging.debug("error when reading device info sent")
        finally:
            return is_sent

    @exception_handler
    def send_runtime_loaded(self):
        """ Push device information on DLR library load
        """
        if not self._is_device_info_sent():
            msg = {"record_type": self.RUNTIME_LOAD}
            if self.system:
                msg.update(self.system.get_device_info())
            self.client.send(json.dumps(msg))

    @exception_handler
    def send_model_loaded(self, model):
        model_name = self.get_model_hash(model)
        data = {"record_type": self.MODEL_LOAD, "model": model_name}
        if self.system:
            data["uuid"] = self.system.get_device_uuid()
        self.client.send(json.dumps(data))

    def get_model_hash(self, model):
        """ Get hashsed model name

        Args:
            model str: name of model

        Returns:
            str: hased str of model
        """
        hashed = get_hash_string(model.encode())
        name = str(hashed.hexdigest())
        return name


MGR = PhoneHome.get_instance()


def call_phone_home(func):
    def callback(*args, **kwargs):
        resp = func(*args, **kwargs)
        try:
            if not MGR:
                return resp
            elif func.__name__ == "__init__":
                model = str(args[1])
                MGR.send_model_loaded(model)
        except Exception:
            logging.debug("phone home error")
        finally:
            return resp

    return callback
