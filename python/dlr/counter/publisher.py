import concurrent.futures
import queue
import threading
import logging

from .utils import resturlutils
from . import config

# Singleton class
class MsgPublisher(object):
    _stop_processing = False
    _instance = None

    @staticmethod
    def get_instance():
        """return single instance of class"""
        if MsgPublisher._instance is None:
            MsgPublisher._instance = MsgPublisher()
        return MsgPublisher._instance

    def __init__(self):
        try:
            self.client = resturlutils.RestUrlUtils()
            self.record_queue = queue.Queue(maxsize=config.CALL_HOME_PUBLISH_MESSAGE_MAX_QUEUE_SIZE)
            # start loop
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.CALL_HOME_MAX_WORKERS_THREADS)
            self.executor.submit(self._process_queue)
            logging.info("msg publisher thread pool execution started")
        except Exception as e:
            logging.exception("msg publisher thread pool not started", exc_info=True)

    def send(self, data):
        try:
            self.record_queue.put(data)
        except queue.Full as e:
            logging.exception("ccm msg publisher queue full")
        except Exception as e:
            logging.exception("unable to record messages in queue", exc_info=True)

    def _process_queue(self):
        while not MsgPublisher._stop_processing:
            self.client.send(self.record_queue.get(block=True))
        logging.info("ccm msg publisher execution stopped")

    def stop(self):
        self.record_queue.task_done()
        MsgPublisher._stop_processing = True
        self.executor.shutdown(wait=False)

