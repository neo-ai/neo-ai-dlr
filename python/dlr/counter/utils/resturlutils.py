from urllib import request, parse, error

from .. import config
from .dlrlogger import logger


class RestUrlUtils(object):
    def send(self, message):
        """send data to AWS Rest server"""
        try:
            headers = {'Content-Type': 'application/x-amz-json-1.1'}
            en_data = message.encode('utf-8')
            req = request.Request(config.call_home_url, en_data, headers=headers)
            resp = request.urlopen(req)
            resp_data = resp.read()
            logger.info("rest api response: {}".format(resp_data))
        except error.HTTPError as e:
            logger.exception("rest api error!", exc_info=True)
        except Exception as e:
            logger.exception("rest api miscellaneous error", exc_info=True)


