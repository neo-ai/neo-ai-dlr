import platform

from .system import System
from .system import Factory 

# singleton class
class CounterMgr:
    _instance = None

    @staticmethod
    def get_instance():
        """return unique instance of class"""
        if CounterMgr._instance is None:
            CounterMgr()
        return CounterMgr._instance

    def __init__(self):
        """init only one instance"""
        if CounterMgr._instance is not None:
            pass
        else: CounterMgr._instance = self
        self._os = platform.system()

    def push(self, type):
        """push information to aws services using boto client"""
        # get device info
        if type == 1:
            system = Factory.get_system(self._os)
            system.retrieve_info()
            lst = system.get_info()
            #for field in lst:
            #   print("lst :", field)
        elif type == 2:
            # push model load count  
            pass
        else: 
            # push model run count 
            pass      
