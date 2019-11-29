from __future__ import print_function
# from dlr.counter.system import Factory, ARM
from dlr.counter.counter_mgr import CallCounterMgr
import numpy as np
import sys
import os
import platform


def test_countermgr():
    os_type = platform.system()
    if os_type == 'Linux':
        # runtime loaded check
        ccm = CallCounterMgr.get_instance()
        ret = ccm.runtime_loaded()
        assert ret == True
        # check device info
        data = ccm.system.get_info()
        assert data['OS'] == "Linux"
        # model loaded push check
        ret = ccm.model_loaded()
        assert ret == True
        # model run push check
        ret = ccm.model_executed()
        assert ret == True
        ccm.stop()
