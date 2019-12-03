import concurrent.futures
import queue
import threading
import time
from .utils import resturlutils
from .utils.dlrlogger import logger
import time


class MsgPublisher(object):
    def __init__(self):
        try:
            self.client = resturlutils.RestUrlUtils()
            self.record_queue = queue.Queue(maxsize=100)
            self.event = threading.Event()
            # start loop
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
            executor.submit(self._process_queue)
            logger.info("Thread pool execution started")
        except Exception as e:
            logger.warning("Thread pool not started due to exception!", exc_info=True)

    def send(self, data):
        try:
            self.record_queue.put(data)
        except queue.Full as e:
            logger.warning("Queue full !")
        except Exception as e:
            logger.warning("Unable to record messages in queue !", exc_info=True)

    def _process_queue(self):
        while not self.event.is_set():
            time.sleep(1)
            while not self.record_queue.empty():
                self.client.send(self.record_queue.get())
        logger.info("Thread pool execution stopped")

    def stop(self):
        while not self.record_queue.empty():
            pass
        self.event.set()

    def __del__(self):
        self.stop()
