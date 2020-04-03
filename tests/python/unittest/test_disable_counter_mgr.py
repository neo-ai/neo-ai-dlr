"""Call Home feature disable test case"""
from __future__ import print_function
import os
import pytest

CCM_CONFIG_FILE = 'ccm_config.json'


@pytest.fixture(scope='module')
def resource_a_setup(request):
    usr_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CCM_CONFIG_FILE)

    """create a config file, feature disabled configuration"""
    with open(usr_config_path, 'w') as fp:
        fp.write('{\n    "ccm" : "false"\n}')

    def resource_a_teardown():
        """remove the config file if present"""
        if os.path.exists(usr_config_path):
            os.remove(usr_config_path)

    request.addfinalizer(resource_a_teardown)


def test_disable_counter_mgr(resource_a_setup):
    """import dlr module, imported here for test purpose"""
    from dlr.counter.counter_mgr_lite import CounterMgrLite
    # runtime loaded check
    ccm = CounterMgrLite.get_instances()
    assert ccm is None
