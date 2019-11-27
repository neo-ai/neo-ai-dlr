import concurrent.futures
import logging
import queue
import threading
import time
from .utils import botoutils
import time

class MsgPublisher(object):
    def __init__(self):
        self.client = botoutils.SNS()
        self.record_queue = queue.Queue(maxsize=100)
        self.event = threading.Event()
        #start loop
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        executor.submit(self._process_queue)

    def send(self, data):
        self.record_queue.put(data)

    def _process_queue(self):
        while not self.event.is_set():
            time.sleep(10)
            while not self.record_queue.empty():
                self.client.send(self.record_queue.get())

    def stop(self):
        self.event.set()