class CallCounterMgr(object):
    RUNTIME_LOAD = 1
    MODEL_LOAD = 2
    MODEL_RUN = 3

    def __init__(self):
        self.boto_client = None
        self.system = None

    def runtime_loaded(self):
        self.push(CallCounterMgr.RUNTIME_LOAD)

    def model_loaded(self):
        self.push(CallCounterMgr.MODEL_LOAD)

    def model_executed(self):
        self.push(CallCounterMgr.MODEL_RUN)

    def push(self, record_type):
        dev_info = self.system.getDevInfo()
        self.boto_client.send(dev_info.update({'record_type':record_type}))

    def stop(self):
        self.boto_client.stop()