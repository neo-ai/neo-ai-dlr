"""Rest client module"""
import logging
import requests 

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
        status_code = 0
        try:
            headers = {"Content-Type": "application/x-amz-json-1.1"}
            data = message.encode("utf-8")
            
            resp = requests.request("POST", config.CALL_HOME_URL, headers=headers, data=data)
            status_code = resp.status_code
            
            return status_code
        except Exception:
            logging.exception("rest api miscellaneous error")
            status_code = -1
        return status_code
