import platform
import atexit
import json
import hashlib
import os

from .publisher import MsgPublisher
from .system import Factory
from .utils.dlrlogger import logger
from .config import *


class CallCounterMgr(object):
    RUNTIME_LOAD = 1
    MODEL_LOAD = 2
    MODEL_RUN = 3
    _instance = None

    @staticmethod
    def get_instance():
        """return single instance of class"""
        if CallCounterMgr._instance is None:
            if CallCounterMgr.is_feature_enabled():
                CallCounterMgr._instance = CallCounterMgr()
                atexit.register(CallCounterMgr._instance.stop)
            else:
                logger.warning("call home feature disabled")
        return CallCounterMgr._instance

    def __init__(self):
        try:
            self.msg_publisher = MsgPublisher.get_instance()
            self.os_name = platform.system()
            self.system = Factory.get_system(self.os_name)
        except Exception as e:
            logger.exception("while in counter mgr init", exc_info=True)

    @staticmethod
    def is_feature_enabled():
        """Load the configuration file, check if ccm.json file present in root folder,
        config like to disable ccm feature "ccm": "false" """
        try:
            if os.path.isfile(call_home_user_config_file):
                with open(call_home_user_config_file, "r") as ccm_json_file:
                    data = json.load(ccm_json_file)
                    if 'false' == str(data['ccm']).lower():
                        return False
                    else:
                        return True
            else:
                return True
        except Exception as e:
            logger.exception("while in reading ccm config file")
        return True

    def is_device_info_published(self):
        """check if device information send only one time in DLR installation life time.
            Create dummy binary file <DLR installation path>/ccm_record.txt to store flag
        """
        flag = False
        try:
            ccm_rec_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             call_home_record_file)
            if os.path.exists(ccm_rec_data_path):
                fp = open(ccm_rec_data_path, "rb")
                ccm_rec_flag = fp.read()
                if int.from_bytes(ccm_rec_flag, byteorder='big') == CallCounterMgr.RUNTIME_LOAD:
                    flag = True
                else:
                    flag = False
                fp.close()
            else:
                # write a file if not exist, content runtime_load as a flag
                fp = open(ccm_rec_data_path, "wb")
                fp.write(CallCounterMgr.RUNTIME_LOAD.to_bytes(1, byteorder='big'))
                fp.close()
        except IOError as e:
            logger.exception("while reading ccm event file")
        except Exception as e:
            logger.exception("while reading ccm event file")
        return flag

    def runtime_loaded(self):
        """push device information at event AI.DLR library loaded"""
        try:
            if not self.is_device_info_published():
                pub_data = {'record_type': CallCounterMgr.RUNTIME_LOAD}
                pub_data.update(self.system.get_device_info())
                self._push(pub_data)
        except Exception as e:
            logger.exception("while dlr runtime load", exc_info=True)
        return True

    def model_count(self, count_type, model):
        """push model load information at event ML/DL model loaded"""
        try:
            _md5model = hashlib.md5(model.encode())
            _md5model = str(_md5model.hexdigest())
            pub_data = {'record_type': count_type}
            pub_data.update({'uuid': self.system.get_device_uuid()})
            pub_data.update({'model': _md5model})
            self._push(pub_data)
        except Exception as e:
            logger.exception("unable to complete model count", exc_info=True)
        return True

    def _push(self, data):
        """publish information to AWS Server"""
        self.msg_publisher.send(json.dumps(data))

    def stop(self):
        self.msg_publisher.stop()

    def __del__(self):
        self.stop()
