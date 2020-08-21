"""call home feature"""
import json
import platform
import logging
import os

from .config import (
    CALL_HOME_USR_NOTIFICATION,
    CALL_HOME_USER_CONFIG_FILE,
    CALL_HOME_REQ_STOP_MAX_COUNT,
    CALL_HOME_RECORD_FILE,
    CALL_HOME_MODEL_RUN_COUNT_TIME_SECS,
)
from .utils.helper import get_hash_string
from .utils import resturlutils
from .system import Factory

DEVICE_INFO_CONFIG = "device_info"
PHONE_HOME_CONFIG = "phone_home"


def exception_handler(func):
    def callback(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Excpetion:
            logger.debug("phone home failure, disable feature")
            PhoneHome.disable_feature()
    return callback


class PhoneHome:
    """
    Phone home client
    """

    __instance = None
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
        if PhoneHome._enable_feature is not None:
            return PhoneHome._enable_feature

        feature_enb = False
        try:
            config_path = PhoneHome.get_config_path()
            if os.path.isfile(config_path):
                with open(config_path, "r") as config_file:
                    data = json.load(config_file)
                    feature_enb = data.get(PHONE_HOME_CONFIG, False)
        except Exception:
            logging.exception("error reading config file")

        PhoneHome._enable_feature = feature_enb
        return feature_enb

    @staticmethod
    def disable_feature() -> None:
        if PhoneHome.is_enabled():
            config_path = PhoneHome.get_config_path()
            os.remove(config_path)
            PhoneHome.__instance = None

    @staticmethod
    def enable_feature():
        if not PhoneHome.is_enabled():
            config_path = PhoneHome.get_config_path()
            with open(config_path, "w") as config_file:
                config = {PHONE_HOME_CONFIG: False}
                config_file.write(json.dumps(config))
            PhoneHome()

    @staticmethod
    def get_instance():
        """
        singleton instance getter function
        """
        try:
            if not PhoneHome.__instance:
                PhoneHome.enable_feature()
        except Exception:
            logging.debug("unsupported system for call home feature")
        finally:
            return PhoneHome.__instance

    def __init__(self):
        try:
            self.client = resturlutils.RestUrlUtils()
            self.enable_feature = None

            machine_type = platform.machine()
            os_name = platform.system()
            os_supt = f"{os_name}_{machine_type}"

            self.system = Factory.get_system(os_supt)
            if self.system is None:
                raise Exception("system is not supported")

            self.send_runtime_loaded()
        except Exception as ex:
            logging.debug("phone init error")

    def _is_device_info_sent(self):
        """ check if device information send only single time in DLR installation life time.
            create a dummy binary file in DLR installation path to record process state.

        Returns:
            [type]: [description]
        """
        is_sent = False
        try:
            config_file = PhoneHome.get_config_path()
            with open(config_path, "wr") as config_file:
                config = json.loads(config_file.read())
                device_info = config.get(DEVICE_INFO_CONFIG, False)
                if not device_info:
                    config[DEVICE_INFO_CONFIG] = True
                config_file.write(json.dumps(config))
                is_sent = True
        except Exception:
            logging.debug("error when reading device info sent")
        finally:
            return is_sent

    # @self.exception_handler
    def send_runtime_loaded(self):
        """ Push device information on DLR library load
        """
        if not self._is_device_info_sent():
            msg = {"record_type": self.RUNTIME_LOAD}
            if self.system:
                msg.update(self.system.get_device_info())
            self._send_message(msg)

    @exception_handler
    def send_model_loaded(self, model):
        model_name = self.get_model_hash(model)
        data = {"record_type": self.MODEL_LOAD, "model": model_name}
        if self.system:
            data["uuid"] = self.system.get_device_uuid()
        self.msgs.append(json.dumps(data))

    def get_model_hash(self, model):
        """get hash string of model"""
        hashed = get_hash_string(model.encode())
        name = str(hashed.hexdigest())
        return name


MGR = PhoneHome.get_instance()


def phone_home(func):
    def callback(*args, **kwargs):
        resp = func(*args, **kwargs)
        try:
            if not MGR:
                return resp
            elif func.__name__ == "__init__":
                MGR.send_model_loaded()
        except Exception:
            logger.debug("phone home error")
        finally:
            return resp

    return callback
