from __future__ import print_function
import platform


def create_ccm_config():
    # create ccm.json file, feature disabled configuration
    fp = open('ccm.json', 'w')
    fp.write('{\n    "ccm" : "false"\n}')
    fp.flush()
    fp.close()
    assert fp.closed


# write a ccm config file
create_ccm_config()


def test_disable_counter_mgr():
    # import dlr module
    from dlr.counter.counter_mgr import CallCounterMgr
    # test the ccm feature disable
    os_type = platform.system()
    if os_type == 'Linux':
        # runtime loaded check
        ccm = CallCounterMgr.get_instance()
        assert ccm is None
