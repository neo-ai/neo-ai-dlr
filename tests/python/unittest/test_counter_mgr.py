"""call home feature complete test case"""
from __future__ import print_function
import platform
from unittest import mock

from dlr.counter.counter_mgr_lite import CounterMgrLite

LINUX_X86 = 'Linux_x86'
LINUX_ARM = 'Linux_arm'
SLEEP_SECS = 10
MODEL_HASH = "123456789"


def mock_push(data):
    """test result validator"""
    if data:
        if data['record_type'] == 3:
            assert data['run_count'] == 1
            assert data['model']
            assert data['uuid']
        if data['record_type'] == 2:
            assert data['model']
            assert data['uuid']


@mock.patch('dlr.counter.counter_mgr_lite.CounterMgrLite.send_msg', side_effect=mock_push)
def test_counter_mgr_x86(send_msg):
    """test on linux x86 system"""
    machine_typ = platform.machine()
    os_name = platform.system()
    os_support = "{0}_{1}".format(os_name, machine_typ)

    if LINUX_X86 in os_support:
        # runtime loaded check
        ccm = CounterMgrLite.get_instances()
        ccm.add_runtime_loaded()
        # check device info
        data = ccm.system.get_device_info()
        assert data['os'].lower() == os_name.lower()
        assert data['machine'].lower() == machine_typ.lower()
        # model loaded push check, pass fake model hash
        ccm.add_model_loaded(MODEL_HASH)
        # model run push check
        ccm.add_model_run(MODEL_HASH)


@mock.patch('dlr.counter.counter_mgr_lite.CounterMgrLite.send_msg', side_effect=mock_push)
def test_counter_mgr_linux_arm(send_msg):
    """test on linux arm system"""
    machine_typ = platform.machine()
    os_name = platform.system()
    os_support = "{0}_{1}".format(os_name, machine_typ)

    if LINUX_ARM in os_support:
        # runtime loaded check
        ccm = CounterMgrLite.get_instances()
        ccm.add_runtime_loaded()
        # check device info
        data = ccm.system.get_device_info()
        assert data['os'].lower() == os_name.lower()
        assert data['machine'].lower() == machine_typ.lower()
        # model loaded push check, pass model hash
        ccm.add_model_loaded(MODEL_HASH)
        # model run push check
        ccm.add_model_run(MODEL_HASH)
