"""Rest client module"""
import logging
import http.client

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
        conn = None
        try:
            headers = {"Content-Type": "application/x-amz-json-1.1"}
            data = message.encode("utf-8")
            conn = http.client.HTTPSConnection("api.neo-dlr.amazonaws.com")
            conn.request("POST", "", body=data, headers=headers)
            resp = conn.getresponse()
            status_code = resp.status
            return status_code
        except Exception:
            logging.exception("rest api miscellaneous error")
            status_code = -1
        finally:
            if conn is not None:
                conn.close()
        return status_code
