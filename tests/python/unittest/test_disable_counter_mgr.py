from __future__ import print_function

import platform
import os


def create_ccm_config():
    # create ccm.json file, feature disabled configuration
    fp = open('ccm_config.json', 'w')
    fp.write('{\n    "ccm" : "false"\n}')
    fp.flush()
    fp.close()
    assert fp.closed


def remove_ccm_config():
    # remove config file if present
    if os.path.exists('ccm_config.json'):
        os.remove('ccm_config.json')


# write a ccm config file
create_ccm_config()


def test_disable_counter_mgr():
    # import dlr module, imported here for test purpose
    from dlr.counter.counter_mgr import CallCounterMgr
    # test the ccm feature disable
    os_type = platform.system()
    if os_type == 'Linux':
        # runtime loaded check
        ccm = CallCounterMgr.get_instance()
        # remove ccm file
        remove_ccm_config()
        assert ccm is None
    else:
        remove_ccm_config()