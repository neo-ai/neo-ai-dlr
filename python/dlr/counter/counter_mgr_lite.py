"""call home feature"""
import json
import atexit
import platform
from threading import Thread, Event
from collections import deque
import logging
import os

from .config import CALL_HOME_USR_NOTIFICATION, CALL_HOME_USER_CONFIG_FILE, \
    CALL_HOME_REQ_STOP_MAX_COUNT, CALL_HOME_RECORD_FILE, CALL_HOME_MODEL_RUN_COUNT_TIME_SECS
from .utils.helper import get_hash_string
from .utils import resturlutils
from .system import Factory


def call_home_lite(func):
    """function wrapper"""

    def wrapper(*args, **kwargs):
        global MGR

        if func.__name__ == "init_call_home":
            print(CALL_HOME_USR_NOTIFICATION)
            if not MGR:
                MGR = CounterMgrLite.get_instances()
            MGR.add_runtime_loaded()

        resp = func(*args, **kwargs)
        if func.__name__ == '__init__':
            model = args[0]
            if MGR:
                MGR.add_model_loaded(model.get_model_name())
        elif func.__name__ == 'run':
            model = args[0]
            if MGR:
                MGR.add_model_run(model.get_model_name())

        return resp

    return wrapper


class CounterMgrLite:
    """Lighter call home manager class"""
    _instance = None
    _enable_feature = None
    _resp_err_count = 0
    RUNTIME_LOAD = 1
    MODEL_LOAD = 2
    MODEL_RUN = 3

    @staticmethod
    def has_instance():
        """instance check function"""
        return CounterMgrLite._instance is not None

    @staticmethod
    def get_instances():
        """singleton instance getter function"""
        if not CounterMgrLite._instance and CounterMgrLite.is_feature_enabled():
            try:
                CounterMgrLite._instance = CounterMgrLite()
            except Exception:
                CounterMgrLite._instance = None
                logging.exception("unsupported system for call home feature", exc_info=False)
            else:
                atexit.register(CounterMgrLite._instance.clean_up)
        return CounterMgrLite._instance

    @staticmethod
    def is_feature_enabled():
        """feature enable or disable checker function"""
        if CounterMgrLite._enable_feature is not None:
            return CounterMgrLite._enable_feature

        feature_enb = False
        try:
            if os.path.isfile(CALL_HOME_USER_CONFIG_FILE):
                with open(CALL_HOME_USER_CONFIG_FILE, "r") as ccm_json_file:
                    data = json.load(ccm_json_file)
                    if str(data['ccm']).lower() == 'false':
                        feature_enb = False
                    else:
                        feature_enb = True
            else:
                feature_enb = True
        except Exception:
            logging.exception("while in reading ccm config file", exc_info=False)

        CounterMgrLite._enable_feature = feature_enb
        return feature_enb

    def __init__(self):
        try:
            # thread-safe
            self.msgs = deque([])
            self.metrics = {}

            self.client = resturlutils.RestUrlUtils()
            self.stop_evt = None
            self.worker = None

            self.enable_feature = None
            machine_typ = platform.machine()
            os_name = platform.system()
            os_supt = "{0}_{1}".format(os_name, machine_typ)
            self.system = Factory.get_system(os_supt)
            if self.system is None:
                raise Exception("unsupported system")
            self.create_thread()
        except Exception as ex:
            logging.exception("while in counter mgr init", exc_info=False)
            raise ex

    def is_device_info_published(self):
        """check if device information send only single time in DLR installation life time.
           create a dummy binary file in DLR installation path to record process state.
        """
        flag = False
        try:
            ccm_rec_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             CALL_HOME_RECORD_FILE)
            if os.path.exists(ccm_rec_data_path):
                flag = True
            else:
                # write runtime_load as a flag in a file
                with open(ccm_rec_data_path, "wb") as rec_file:
                    rec_file.write(CounterMgrLite.RUNTIME_LOAD.to_bytes(1, byteorder='big'))
        except IOError:
            logging.exception("while reading ccm publish data record file", exc_info=False)
        except Exception:
            logging.exception("while reading ccm publish data record file", exc_info=False)
        return flag

    def add_runtime_loaded(self):
        """push device information on DLR library load"""
        try:
            if not self.is_device_info_published():
                data = {'record_type': self.RUNTIME_LOAD}
                if self.system:
                    data.update(self.system.get_device_info())
                    self.msgs.append(json.dumps(data))
        except Exception:
            logging.exception("while dlr runtime load", exc_info=False)

    def add_model_loaded(self, model: str):
        """record model load event"""
        model_name = self.get_model_hash(model)
        data = {'record_type': self.MODEL_LOAD, 'model': model_name}
        if self.system:
            data['uuid'] = self.system.get_device_uuid()
        self.msgs.append(json.dumps(data))

    def add_model_run(self, model: str):
        """record and count model run event"""
        model_name = self.get_model_hash(model)
        if self.metrics.get(model_name) is not None:
            self.metrics[model_name] += 1
        else:
            self.metrics[model_name] = 1

    def get_model_hash(self, model):
        """get hash string of model"""
        hashed = get_hash_string(model.encode())
        name = str(hashed.hexdigest())
        return name

    def create_thread(self):
        """create a thread"""
        try:
            self.stop_evt = Event()
            self.worker = Worker(self.send_msg, self.stop_evt)
            self.worker.start()
        except RuntimeError:
            logging.exception("runtime error while thread start", exc_info=False)
        except Exception:
            logging.exception("generic exception while creating thread", exc_info=False)

    def send_msg(self):
        """dispatch messages to rest client"""
        for k in list(self.metrics):
            pub_data = {'record_type': self.MODEL_RUN, 'uuid': self.system.get_device_uuid(),
                        'model': k, 'run_count': self.metrics[k]}
            self.msgs.append(json.dumps(pub_data))
        self.metrics.clear()

        while len(self.msgs) != 0:
            msg = self.msgs.popleft()
            if CounterMgrLite._resp_err_count < CALL_HOME_REQ_STOP_MAX_COUNT:
                resp_code = self.client.send(msg)
                if resp_code != 200:
                    CounterMgrLite._resp_err_count += 1

    def clean_up(self):
        """clean up resources"""
        self.stop_evt.set()
        self.send_msg()


class Worker(Thread):
    """Worker thread class"""

    def __init__(self, func, event: Event):
        Thread.__init__(self, daemon=True)
        self.func = func
        self.stop_evt = event

    def run(self):
        while not self.stop_evt.wait(CALL_HOME_MODEL_RUN_COUNT_TIME_SECS):
            self.func()
