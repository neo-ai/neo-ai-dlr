"""Call Home feature disable test case"""
from __future__ import print_function
import os
import pytest
import pkgutil

CCM_CONFIG_FILE = 'counter/ccm_config.json'


def get_dlr_path():
    """ get dlr module path """
    pkg = pkgutil.get_loader('dlr')
    pkg_path = pkg.get_filename().split("/")[:-1]
    return os.path.join("/", *pkg_path)


@pytest.fixture(scope='module')
def resource_a_setup(request):
    dlr_path = get_dlr_path()
    usr_config_path = os.path.join(dlr_path, CCM_CONFIG_FILE)

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
