from __future__ import print_function

import platform

from dlr.counter.counter_mgr import CallCounterMgr


def test_counter_mgr():
    os_type = platform.system()
    if os_type == 'Linux':
        # runtime loaded check
        ccm = CallCounterMgr.get_instance()
        ret = ccm.runtime_loaded()
        assert ret == True
        # check device info
        data = ccm.system.get_info()
        assert data['os'] == "Linux"
        # model loaded push check, pass fake oid and model hash
        ret = ccm.model_loaded("123456789")
        assert ret == True
        # model run push check
        ret = ccm.model_executed("123456789")
        assert ret == True
        ccm.stop()
