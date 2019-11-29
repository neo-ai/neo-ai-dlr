from .publisher import MsgPublisher
from .system import Factory
import json
import platform

class CallCounterMgr(object):
    RUNTIME_LOAD = 1
    MODEL_LOAD = 2
    MODEL_RUN = 3
    _instance = None

    @staticmethod
    def get_instance():
        """return single instance of class"""
        if CallCounterMgr._instance is None:
            CallCounterMgr._instance = CallCounterMgr()
        return CallCounterMgr._instance

    def __init__(self):
        self.msg_publisher = MsgPublisher()
        self.os_name = platform.system()
        self.system = Factory.get_system(self.os_name)

    def runtime_loaded(self):
        self._push(CallCounterMgr.RUNTIME_LOAD)

    def model_loaded(self):
        self._push(CallCounterMgr.MODEL_LOAD)

    def model_executed(self):
        self._push(CallCounterMgr.MODEL_RUN)

    def _push(self, record_type):
        dev_info = self.system.get_info()
        dev_info.update({'record_type': record_type})
        self.msg_publisher.send(json.dumps(dev_info))

    def stop(self):
        self.msg_publisher.stop()

    def __del__(self):
        self.stop()


# ccm = CallCounterMgr.get_instance()
# ccm.runtime_loaded()
# ccm.model_loaded()
# ccm.model_executed()
# ccm.stop()
