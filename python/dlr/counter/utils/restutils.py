from .. import config
import requests
import logging


class RestHandler(object):
    def send(self, message):
        # post dlr device info on web server
        resp = requests.post(config.call_home_url, json=message)

        if resp.status_code != 200:
            logging.warning("Rest api status code: {}".format(resp.status_code))
