from .publisher import MsgPublisher
from .system import Factory
import json


class CallCounterMgr(object):
    RUNTIME_LOAD = 1
    MODEL_LOAD = 2
    MODEL_RUN = 3
    _instance = None

    @staticmethod
    def get_instance():
        """return unique instance of class"""
        if CallCounterMgr._instance is None:
            CallCounterMgr()
        return CallCounterMgr._instance

    def __init__(self):
        self.msg_publisher = MsgPublisher()
        self.system = Factory.get_system('Linux')

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


# ccm = CallCounterMgr()
# ccm.runtime_loaded()
# ccm.model_loaded()
# ccm.model_executed()
# ccm.stop()
