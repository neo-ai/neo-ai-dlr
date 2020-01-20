import logging
from urllib import request, parse, error

from .. import config


class RestUrlUtils(object):
    def send(self, message):
        """send data to AWS Rest server"""
        resp_code = 0
        try:
            headers = {'Content-Type': 'application/x-amz-json-1.1'}
            en_data = message.encode('utf-8')
            req = request.Request(config.CALL_HOME_URL, en_data, headers=headers)
            resp = request.urlopen(req)
            resp_data = resp.read()
            logging.info("rest api response: {}".format(resp_data))
            resp_code = resp.getcode()
        except error.HTTPError as e:
            logging.exception("rest api error!")
            resp_code = -1
        except Exception as e:
            logging.exception("rest api miscellaneous error")
            resp_code = -1
        return resp_code


