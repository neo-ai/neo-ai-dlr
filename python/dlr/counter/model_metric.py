import concurrent.futures
import logging
import time
import json
import threading

from .utils import resturlutils
from . import config
from .model_exec_counter import ModelExecCounter
from .utils.helper import *
from .publisher import MsgPublisher

class ModelMetric(object):
    _pub_model_metric = True
    _instance = None
    MODEL_RUN = 3
    resp_cnt = 0

    @staticmethod
    def get_instance(uuid):
        """return single instance of class"""
        if ModelMetric._instance is None:
            ModelMetric._instance = ModelMetric(uuid)
        return ModelMetric._instance

    def __init__(self, uuid=0):
        try:
            self.uuid = uuid
            self.publisher = MsgPublisher.get_instance()
            # start loop
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.CALL_HOME_MAX_WORKERS_THREADS)
            self.f = []
            self.f.append(self.executor.submit(self.push_model_metric))
            logging.info("model metric thread pool execution started")
        except Exception as e:
            logging.exception("model metric thread pool not started", exc_info=True)

    def push_model_metric(self):
        """publishing model run metric"""
        while ModelMetric._pub_model_metric:
            time.sleep(config.CALL_HOME_MODEL_RUN_COUNT_TIME_SECS)
            mod_dict = ModelExecCounter.get_dict()
            if mod_dict:
                for key, val in mod_dict.items():
                    self.model_run_info_publish(ModelMetric.MODEL_RUN, key, val)
                ModelExecCounter.clear_models_counts()

    def model_run_info_publish(self, model_event_type, model, count=0):
        """push model load information at time model load time"""
        try:
            _md5model = get_hash_string(model.encode())
            _md5model = str(_md5model.hexdigest())
            pub_data = {'record_type': model_event_type, 'model': _md5model, 'uuid': self.uuid, 'run_count': count}
            self.push(pub_data)
        except Exception as e:
            logging.exception("unable to complete model count", exc_info=True)

    def push(self, data):
        """publish information to Server"""
        if ModelMetric.resp_cnt < config.CALL_HOME_REQ_STOP_MAX_COUNT:
            resp_code = self.publisher.send(json.dumps(data))
            if resp_code != 200:
                ModelMetric.resp_cnt += 1

    def stop(self):
        ModelMetric._pub_model_metric = False
        self.executor.shutdown(False)
        mod_dict = ModelExecCounter.get_dict()
        if mod_dict:
            for key, val in mod_dict.items():
                self.model_run_info_publish(ModelMetric.MODEL_RUN, key, val)
        ModelExecCounter.clear_models_counts()

    def __del__(self):
        self.stop()
