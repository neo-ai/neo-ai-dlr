from __future__ import print_function

import platform

from dlr.counter.counter_mgr import CallCounterMgr

LINUX_X86 = 'Linux_x86'
LINUX_ARM = 'Linux_arm'


def test_counter_mgr_x86():
    machine_typ = platform.machine()
    os_name = platform.system()
    os_support = os_name
    os_support += "_" + machine_typ

    if LINUX_X86 in os_support:
        # runtime loaded check
        ccm = CallCounterMgr.get_instance()
        ret = ccm.runtime_loaded()
        assert ret == True
        # check device info
        data = ccm.system.get_device_info()
        assert data['os'].lower() == os_name.lower()
        assert data['machine'].lower() == machine_typ.lower()
        # model loaded push check, pass fake oid and model hash
        ret = ccm.model_count(CallCounterMgr.MODEL_LOAD, "123456789")
        assert ret == True
        # model run push check
        ret = ccm.model_count(CallCounterMgr.MODEL_RUN, "123456789")
        assert ret == True
        ccm.stop()


def test_counter_mgr_linux_arm():
    machine_typ = platform.machine()
    os_name = platform.system()
    os_support = os_name
    os_support += "_" + machine_typ

    if LINUX_ARM in os_support:
        # runtime loaded check
        ccm = CallCounterMgr.get_instance()
        ret = ccm.runtime_loaded()
        assert ret == True
        # check device info
        data = ccm.system.get_device_info()
        assert data['os'].lower() == os_name.lower()
        assert data['machine'].lower() == machine_typ.lower()
        # model loaded push check, pass fake oid and model hash
        ret = ccm.model_count(CallCounterMgr.MODEL_LOAD, "123456789")
        assert ret == True
        # model run push check
        ret = ccm.model_count(CallCounterMgr.MODEL_RUN, "123456789")
        assert ret == True
        ccm.stop()
