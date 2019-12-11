import concurrent.futures
import queue
import threading

from .utils import resturlutils
from .utils.dlrlogger import logger
from .config import *


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
            self.record_queue = queue.Queue(maxsize=call_home_publish_message_max_queue_size)
            self.event = threading.Event()
            # start loop
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=call_home_max_workers_threads)
            executor.submit(self._process_queue)
            logger.info("msg publisher thread pool execution started")
        except Exception as e:
            logger.exception("msg publisher thread pool not started", exc_info=True)

    def send(self, data):
        try:
            self.record_queue.put(data)
            self.event.set()
        except queue.Full as e:
            logger.exception("ccm msg publisher queue full")
        except Exception as e:
            logger.exception("unable to record messages in queue", exc_info=True)

    def _process_queue(self):
        while self.event.wait() and not MsgPublisher._stop_processing:
            self.client.send(self.record_queue.get(block=True))
        logger.info("ccm msg publisher execution stopped")

    def stop(self):
        while not self.record_queue.empty():
            pass
        MsgPublisher._stop_processing = True

    def __del__(self):
        self.stop()
