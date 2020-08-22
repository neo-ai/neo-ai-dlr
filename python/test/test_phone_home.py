import time
import os
import json

from dlr.counter.phone_home import call_phone_home, PhoneHome, ENABLE_PHONE_HOME_CONFIG

import unittest
from unittest.mock import MagicMock, patch, mock_open


class TestPhoneHome(unittest.TestCase):
    # def tearDown(self):
    #     self.clean_config()

    def clean_config(self):
        config_path = PhoneHome.get_config_path()
        if os.path.isfile(config_path):
            os.remove(config_path)

    def mock_func(self):
        func = MagicMock(return_vaule=MagicMock())
        return func

    def check_enable_by_default(self):
        func = self.mock_func()
        call_phone_home(func)

        config_path = PhoneHome.get_config_path()
        config_file = open(config_path, "r")
        data = json.loads(config_file.read())
        config_file.close()

        assert data[ENABLE_PHONE_HOME_CONFIG] == True
        self.clean_config()

    def check_disable_by_default(self):
        config = {ENABLE_PHONE_HOME_CONFIG: False}

        config_path = PhoneHome.get_config_path()
        config_file = open(config_path, "w")
        config_file.write(json.dumps(config))
        config_file.close()

        func = self.mock_func()
        call_phone_home(func)
        assert PhoneHome.get_instance() == None
        self.clean_config()

    def check_is_enable_false(self):
        PhoneHome.disable_feature()
        func = self.mock_func()
        call_phone_home(func)
        assert PhoneHome.get_instance() == None

        config_path = PhoneHome.get_config_path()
        config_file = open(config_path, "r")
        data = json.loads(config_file.read())
        assert data[ENABLE_PHONE_HOME_CONFIG] == False
        config_file.close()
        
        self.clean_config()

    def check_is_enable_true(self):
        PhoneHome.enable_feature()
        func = self.mock_func()
        call_phone_home(func)
        assert PhoneHome.get_instance() == None

        self.clean_config()

    def test_sync(self):
        self.check_enable_by_default()
        self.check_disable_by_default()
        self.check_is_enable_false()
        self.check_is_enable_true()

