"""Rest client module"""
import logging
import urllib3
import certifi
from .. import config


class RestUrlUtils(object):
    """Rest client used to push messages"""

    def send(self, message):
        """send data to AWS Rest server
        Parameters
        ----------
        self:
        message: str

        Returns
        -------
        int
            resp_code a response code
        """
        resp_code = 0
        req = None
        try:
            header = {"Content-Type": "application/x-amz-json-1.1"}
            data = message.encode("utf-8")
            req = urllib3.PoolManager(
                cert_reqs="CERT_REQUIRED", ca_certs=certifi.where()
            )
            resp = req.request("POST", config.CALL_HOME_URL, headers=header, body=data)
            resp_code = resp.status
        except urllib3.exceptions.HTTPError:
            logging.exception("rest url http error")
            resp_code = -1
        except Exception:
            logging.exception("rest api miscellaneous error")
            resp_code = -1
        finally:
            if req is not None:
                req.clear()
        return resp_code
