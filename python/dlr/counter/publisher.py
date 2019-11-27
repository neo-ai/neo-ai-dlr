import concurrent.futures
import logging
import queue
import threading
import time

class MsgPublisher(object):
    def __init__(self):
        self.client = None
        self.record_queue = queue.Queue(maxsize=100)
        self.event = threading.Event()
        #start loop
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            executor.submit(self.__process_queue)

    def send(self, data):
        self.record_queue.put(data)

    def __process_queue(self, event):
        while not self.event.is_set():
            while not self.record_queue.empty():
                self.client.send(self.record_queue.get())

    def stop(self):
        self.event.set()