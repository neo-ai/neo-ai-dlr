from .. import config
from urllib import request, parse, error
from .dlrlogger import logger


class RestUrlUtils():
    def send(self, message):
        try:
            en_data = parse.urlencode(message)
            en_data = en_data.encode('utf-8')
            req = request.Request(config.rest_post_api_url, en_data)
            resp = request.urlopen(req)
            resp_data = resp.read()
            logger.info("Rest api response: {}".format(resp_data))
        except error.HTTPError as e:
            logger.warning("Rest api error code: {}".format(e.code()), exc_info=True)
        except:
            logger.warning("Rest api miscellaneous exception error", exc_info=True)


