import os
import json
import pdb
from dlr.counter.phone_home import phone_home, PhoneHome

import unittest
from unittest.mock import MagicMock


class TestPhoneHome(unittest.TestCase):
    def tearDown(self):
        config_path = PhoneHome.get_config_path()
        os.remove(config_path)

    def mock_func(self):
        func = MagicMock()
        func.return_value = MagicMock()
        return func

    def test_phone_home_enable(self):
        func = self.mock_func()
        phone_home(func)
        

        config_path = PhoneHome.get_config_path()
        with open(config_path, "r") as config_file:
            data = json.load(config_file)
            assert data.get("phone_home") == False


    # def test_phone_home_disable():
    #     # phone_home.callback()
    #     pass

    # def test_phone_home_disable():
    #     # phone_home.callback()
    #     pass

