from .. import config
from urllib import request, parse, error
from .dlrlogger import logger


class RestUrlUtils():
    def send(self, message):
        try:
            headers = {'Content-Type': 'application/json'}
            en_data = message.encode('utf-8')
            req = request.Request(config.rest_post_api_url, en_data, headers=headers)
            resp = request.urlopen(req)
            resp_data = resp.read()
            logger.info("Rest api response: {}".format(resp_data))
        except error.HTTPError as e:
            logger.warning("Rest api error code: {}".format(e.code()), exc_info=True)
        except Exception as e:
            logger.warning("Rest api miscellaneous exception error", exc_info=True)


