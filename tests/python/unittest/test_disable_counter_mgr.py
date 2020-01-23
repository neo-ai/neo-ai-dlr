from __future__ import print_function
import os
import pytest

CCM_CONFIG_FILE = 'ccm_config.json'


@pytest.fixture(scope='module')
def resource_a_setup(request):
    # create a config file, feature disabled configuration
    with open(CCM_CONFIG_FILE, 'w') as fp:
        fp.write('{\n    "ccm" : "false"\n}')

    def resource_a_teardown():
        # remove the config file if present
        if os.path.exists(CCM_CONFIG_FILE):
            os.remove(CCM_CONFIG_FILE)
    request.addfinalizer(resource_a_teardown)


def test_disable_counter_mgr(resource_a_setup):
    # import dlr module, imported here for test purpose
    from dlr.counter.counter_mgr import CallCounterMgr
    # runtime loaded check
    ccm = CallCounterMgr.get_instance()
    assert ccm is None
