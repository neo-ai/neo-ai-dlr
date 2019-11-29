import concurrent.futures
import logging
import queue
import threading
import time
from .utils import botoutils
from .utils import restutils

import time


class MsgPublisher(object):
    def __init__(self):
        self.client = restutils.RestHandler()
        self.record_queue = queue.Queue(maxsize=100)
        self.event = threading.Event()
        # start loop
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        executor.submit(self._process_queue)
        logging.info("Thread pool execution started")

    def send(self, data):
        self.record_queue.put(data)

    def _process_queue(self):
        while not self.event.is_set():
            time.sleep(10)
            while not self.record_queue.empty():
                self.client.send(self.record_queue.get())
        logging.info("Thread pool execution stopped")

    def stop(self):
        self.event.set()

    def __del__(self):
        self.stop()
