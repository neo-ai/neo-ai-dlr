from __future__ import print_function
import os

CCM_CONFIG_FILE = 'ccm_config.json'


def create_ccm_config():
    # create ccm.json file, feature disabled configuration
    with open(CCM_CONFIG_FILE, 'w') as fp:
        fp.write('{\n    "ccm" : "false"\n}')
        fp.flush()
        fp.close()


def remove_ccm_config():
    # remove config file if present
    if os.path.exists(CCM_CONFIG_FILE):
        os.remove(CCM_CONFIG_FILE)


# write a ccm config file
create_ccm_config()


def test_disable_counter_mgr():
    # import dlr module, imported here for test purpose
    from dlr.counter.counter_mgr import CallCounterMgr
    # test the ccm feature disable
    # runtime loaded check
    ccm = CallCounterMgr.get_instance()
    # remove ccm file
    remove_ccm_config()
    assert ccm is None
