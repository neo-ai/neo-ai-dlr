from .publisher import MsgPublisher
from .system import Factory
from .utils.dlrlogger import logger
from .config import feature
import platform
import atexit
import json
import hashlib

class CallCounterMgr(object):
    RUNTIME_LOAD = 1
    MODEL_LOAD = 2
    MODEL_RUN = 3
    _instance = None

    @staticmethod
    def get_instance():
        """return single instance of class"""
        if 'true' in feature:
            if CallCounterMgr._instance is None:
                CallCounterMgr._instance = CallCounterMgr()
                atexit.register(CallCounterMgr._instance.stop)
        else:
            print("feature disabled")
        return CallCounterMgr._instance

    def __init__(self):
        try:
            self.msg_publisher = MsgPublisher()
            self.os_name = platform.system()
            self.system = Factory.get_system(self.os_name)
        except Exception as e:
            logger.warning("Exception in Counter Mgr init!", exc_info=True)

    def runtime_loaded(self):
        self._push(CallCounterMgr.RUNTIME_LOAD)
        return True

    def model_loaded(self, model):
        _md5model = hashlib.md5(model.encode())
        _md5model = str(_md5model.hexdigest())
        self._push(CallCounterMgr.MODEL_LOAD, _md5model)
        return True

    def model_executed(self):
        self._push(CallCounterMgr.MODEL_RUN)
        return True

    def _push(self, record_type, model=''):
        pub_data = {}
        if record_type == CallCounterMgr.RUNTIME_LOAD:

            pub_data = self.system.get_info()
            pub_data.update({'record_type': record_type})
        elif record_type == CallCounterMgr.MODEL_LOAD:
            pub_data.update({'model': model})
            pub_data.update({'UUID': self.system.get_device().uuid})
            pub_data.update({'record_type': record_type})
        else:
            #pub_data.update({'model': model})
            pub_data.update({'UUID': self.system.get_device().uuid})
            pub_data.update({'record_type': record_type})

        self.msg_publisher.send(json.dumps(pub_data))

    def stop(self):
        self.msg_publisher.stop()

    def __del__(self):
        self.stop()
