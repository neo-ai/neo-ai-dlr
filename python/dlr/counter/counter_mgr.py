import platform
import atexit
import json
import hashlib
import os
import logging
import threading
import time
import concurrent.futures

from .publisher import MsgPublisher
from .system import Factory
from .model_exec_counter import ModelExecCounter
from . import config


def call_home(func):
    def wrapped_call_home(*args, **kwargs):
        call_counter = CallCounterMgr.get_instance()
        if call_counter:
            if func.__name__ == "init_call_home":
                func(*args, **kwargs)
                call_counter.runtime_loaded()
            elif func.__name__ == "__init__":
                func(*args, **kwargs)
                obj = args[0]
                call_counter.model_loaded(obj.get_model_name())
            else:
                res = func(*args, **kwargs)
                obj = args[0]
                call_counter.model_run(obj.get_model_name())
                return res

    return wrapped_call_home


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
                print(config.CALL_HOME_USR_NOTIFICATION)
                CallCounterMgr._instance = CallCounterMgr()
                atexit.register(CallCounterMgr._instance.stop)
                CallCounterMgr._instance.executor.submit(CallCounterMgr._instance.push_model_metric)
            else:
                logging.warning("call home feature disabled")
        return CallCounterMgr._instance

    def __init__(self):
        try:
            self.msg_publisher = MsgPublisher.get_instance()
            machine_typ = platform.machine()
            os_name = platform.system()
            os_supt = "{0}_{1}".format(os_name, machine_typ)
            self.system = Factory.get_system(os_supt)
            self.event = threading.Event()
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.CALL_HOME_MAX_WORKERS_THREADS)
            self._pub_model_metric = True
        except Exception as e:
            logging.exception("while in counter mgr init", exc_info=True)

    @staticmethod
    def is_feature_enabled():
        """check feature disabled config. ccm_config.json file present in root folder"""
        try:
            if os.path.isfile(config.CALL_HOME_USER_CONFIG_FILE):
                with open(config.CALL_HOME_USER_CONFIG_FILE, "r") as ccm_json_file:
                    data = json.load(ccm_json_file)
                    if 'false' == str(data['ccm']).lower():
                        return False
                    else:
                        return True
            else:
                return True
        except Exception as e:
            logging.exception("while in reading ccm config file")
        return True

    def is_device_info_published(self):
        """check if device information send only single time in DLR installation life time.
            create a dummy binary file in DLR installation path to record process state.
        """
        flag = False
        try:
            ccm_rec_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             config.CALL_HOME_RECORD_FILE)
            if os.path.exists(ccm_rec_data_path):
                flag = True
            else:
                # write runtime_load as a flag in a file
                with open(ccm_rec_data_path, "wb") as fp:
                    fp.write(CallCounterMgr.RUNTIME_LOAD.to_bytes(1, byteorder='big'))
        except IOError as e:
            logging.exception("while reading ccm publish data record file")
        except Exception as e:
            logging.exception("while reading ccm publish data record file")
        return flag

    def runtime_loaded(self):
        """push device information on AI.DLR library load"""
        try:
            if not self.is_device_info_published():
                pub_data = {'record_type': CallCounterMgr.RUNTIME_LOAD}
                if self.system:
                    pub_data.update(self.system.get_device_info())
                self.push(pub_data)
        except Exception as e:
            logging.exception("while dlr runtime load", exc_info=True)

    def model_loaded(self, model):
        self.model_info_publish(CallCounterMgr.MODEL_LOAD, model)

    def model_run(self, model):
        ModelExecCounter.add_model_run_count(model)

    def model_info_publish(self, model_event_type, model, count=0):
        """push model load information at time model load time"""
        try:
            _md5model = hashlib.md5(model.encode())
            _md5model = str(_md5model.hexdigest())
            pub_data = {'record_type': model_event_type, 'model': _md5model}
            if self.system:
                pub_data['uuid'] = self.system.get_device_uuid()
            if model_event_type == CallCounterMgr.MODEL_RUN:
                pub_data['run_count'] = count
            self.push(pub_data)
        except Exception as e:
            logging.exception("unable to complete model count", exc_info=True)

    def push(self, data):
        """publish information to Server"""
        if self.msg_publisher:
            self.msg_publisher.send(json.dumps(data))

    def push_model_metric(self):
        """publishing model run metric"""
        while self._pub_model_metric:
            time.sleep(config.CALL_HOME_MODEL_RUN_COUNT_TIME)
            for key, val in ModelExecCounter.get_model_counts_dict().items():
                self.model_info_publish(CallCounterMgr.MODEL_RUN, key, val)
            ModelExecCounter.clear_model_counts()

    def stop(self):
        self._pub_model_metric = False
        for key, val in ModelExecCounter.get_model_counts_dict().items():
            self.model_info_publish(CallCounterMgr.MODEL_RUN, key, val)
        ModelExecCounter.clear_model_counts()
        if self.msg_publisher:
            self.msg_publisher.stop()

    def __del__(self):
        self.stop()
