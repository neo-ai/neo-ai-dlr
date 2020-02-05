from __future__ import print_function
import platform
import time
from unittest import mock

from dlr.counter.counter_mgr import CallCounterMgr

LINUX_X86 = 'Linux_x86'
LINUX_ARM = 'Linux_arm'
SLEEP_SECS = 10
MODEL_HASH = "123456789"


def mock_push(data):
    if data:
        if data['record_type'] == 3:
            assert data['run_count'] == 1
            assert data['model']
            assert data['uuid']
        if data['record_type'] == 2:
            assert data['model']
            assert data['uuid']


@mock.patch('dlr.counter.counter_mgr.CallCounterMgr.push', side_effect=mock_push)
def test_non_support_system(push):
    machine_typ = platform.machine()
    os_name = platform.system()
    os_support = "{0}_{1}".format(os_name, machine_typ)
    if LINUX_ARM not in os_support and LINUX_X86 not in os_support:
        # runtime loaded check
        ccm = CallCounterMgr.get_instance()
        assert ccm is None 
    else:
        ccm = CallCounterMgr.get_instance()
        assert ccm is not None
        ccm.runtime_loaded()
        # check device info
        data = ccm.system.get_device_info()
        assert data['os'].lower() == os_name.lower()
        assert data['machine'].lower() == machine_typ.lower()
        # model loaded push check, pass fake model hash
        ccm.model_loaded(MODEL_HASH)
        # model run push check
        ccm.model_run(MODEL_HASH)
        
