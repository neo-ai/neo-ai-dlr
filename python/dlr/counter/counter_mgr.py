import platform
import atexit
import json
import hashlib
import os

from .publisher import MsgPublisher
from .system import Factory
from .utils.dlrlogger import logger


class CallCounterMgr(object):
    RUNTIME_LOAD = 1
    MODEL_LOAD = 2
    MODEL_RUN = 3
    CCM_CONFIG_FILE = 'ccm.json'
    _instance = None

    @staticmethod
    def get_instance():
        """return single instance of class"""
        if CallCounterMgr._instance is None:
            if CallCounterMgr.is_feature_enabled():
                CallCounterMgr._instance = CallCounterMgr()
                atexit.register(CallCounterMgr._instance.stop)
            else:
                logger.warning("CCM feature disabled")
        return CallCounterMgr._instance

    def __init__(self):
        try:
            self.msg_publisher = MsgPublisher()
            self.os_name = platform.system()
            self.system = Factory.get_system(self.os_name)
        except Exception as e:
            logger.exception("Exception in Counter Mgr init", exc_info=True)

    @staticmethod
    def is_feature_enabled():
        """Load the configuration file, check if ccm.json file present in root folder,
        config like to disable ccm feature "ccm": "false" """
        try:
            if os.path.isfile(CallCounterMgr.CCM_CONFIG_FILE):
                with open(CallCounterMgr.CCM_CONFIG_FILE, "r") as ccm_json_file:
                    data = json.load(ccm_json_file)
                    if 'false' == str(data['ccm']).lower():
                        return False
                    else:
                        return True
            else:
                return True
        except Exception as e:
            logger.exception("Excpetion in reading ccm config file")

    def runtime_loaded(self):
        """push device information at event AI.DLR library loaded"""
        try:
            pub_data = {'record_type': CallCounterMgr.RUNTIME_LOAD}
            pub_data.update(self.system.get_info())
            self._push(pub_data)
        except Exception as e:
            logger.exception("Exception in runtime load", exc_info=True)
        return True

    def model_loaded(self, model):
        """push model load information at event ML/DL model loaded"""
        try:
            _md5model = hashlib.md5(model.encode())
            _md5model = str(_md5model.hexdigest())
            pub_data = {'record_type': CallCounterMgr.MODEL_LOAD}
            pub_data.update({'uuid': self.system.get_info()['uuid']})
            pub_data.update({'model': _md5model})
            self._push(pub_data)
        except Exception as e:
            logger.exception("Exception in model load push", exc_info=True)

        return True

    def model_executed(self, model):
        """push model run information at event ML/DL model run"""
        try:
            _md5model = hashlib.md5(model.encode())
            _md5model = str(_md5model.hexdigest())
            pub_data = {'record_type': CallCounterMgr.MODEL_RUN}
            pub_data.update({'uuid': self.system.get_info()['uuid']})
            pub_data.update({'model': _md5model})
            self._push(pub_data)
        except Exception as e:
            logger.exception("Exception in model execution", exc_info=True)
        return True

    def _push(self, data):
        """publish information to AWS Server"""
        self.msg_publisher.send(json.dumps(data))

    def stop(self):
        self.msg_publisher.stop()

    def __del__(self):
        self.stop()
