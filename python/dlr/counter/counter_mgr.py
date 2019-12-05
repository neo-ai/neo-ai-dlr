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
    _ccm_cfg_read = False
    _ccm_flag = True

    @staticmethod
    def get_instance():
        """return single instance of class"""
        if CallCounterMgr._instance is None:
            if not CallCounterMgr._ccm_cfg_read:
                CallCounterMgr.check_feature_config()
            if CallCounterMgr._ccm_flag:
                CallCounterMgr._instance = CallCounterMgr()
                atexit.register(CallCounterMgr._instance.stop)
            else:
                logger.info("CCM feature disabled!")
        return CallCounterMgr._instance

    def __init__(self):
        try:
            self.msg_publisher = MsgPublisher()
            self.os_name = platform.system()
            self.system = Factory.get_system(self.os_name)
            self._uuid = {}
            self._moddict = {}
        except Exception as e:
            logger.warning("Exception in Counter Mgr init!", exc_info=True)

    @staticmethod
    def check_feature_config():
        """Load the configuration file, check if ccm.json file present in root folder,
        config like to disable ccm feature "ccm": "false" """
        try:
            if os.path.isfile(CallCounterMgr.CCM_CONFIG_FILE):
                with open(CallCounterMgr.CCM_CONFIG_FILE, "r") as ccm_json_file:
                    data = json.load(ccm_json_file)
                    if 'false' == str(data['ccm']).lower():
                        CallCounterMgr._ccm_flag = False
                    else:
                        CallCounterMgr._ccm_flag = True
            else:
                CallCounterMgr._ccm_flag = True

            CallCounterMgr._ccm_cfg_read = True
        except Exception as e:
            logger.warning("Excpetion in reading ccm config file!")
            CallCounterMgr._ccm_cfg_read = True
            CallCounterMgr._ccm_flag = True

    def runtime_loaded(self):
        """push device information at event AI.DLR library loaded"""
        try:
            pub_data = {'record_type': CallCounterMgr.RUNTIME_LOAD}
            pub_data.update(self.system.get_info())
            self._uuid = {'uuid': pub_data['uuid']}
            self._push(pub_data)
        except Exception as e:
            logger.warning("Exception in runtime load!", exc_info=True)

        return True

    def model_loaded(self, model, oid):
        """push model load information at event ML/DL model loaded"""
        try:
            _md5model = hashlib.md5(model.encode())
            _md5model = str(_md5model.hexdigest())
            pub_data = {'record_type': CallCounterMgr.MODEL_LOAD}
            pub_data.update(self._uuid)
            pub_data.update({'model': _md5model})
            self._moddict = {oid: _md5model}
            self._push(pub_data)
        except Exception as e:
            logger.warning("Exception in model load push!", exc_info=True)

        return True

    def model_executed(self, oid):
        """push model run information at event ML/DL model run"""
        try:
            pub_data = {'record_type': CallCounterMgr.MODEL_RUN}
            pub_data.update(self._uuid)
            pub_data.update({'model': self._moddict[oid]})
            self._push(pub_data)
        except Exception as e:
            logger.warning("Exception in model execution!", exc_info=True)

        return True

    def _push(self, data):
        """publish information to AWS Server"""
        self.msg_publisher.send(json.dumps(data))

    def stop(self):
        self.msg_publisher.stop()

    def __del__(self):
        self.stop()
