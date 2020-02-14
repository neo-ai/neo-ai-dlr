import concurrent.futures
import queue
import logging

from .utils import resturlutils
from . import config


# Singleton class
class MsgPublisher(object):
    _stop_processing = False
    _instance = None
    resp_cnt = 0

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
            logging.exception("msg publisher thread pool not started", exc_info=False)

    def send(self, data):
        try:
            self.record_queue.put(data, block=False)
        except queue.Full as e:
            logging.exception("ccm msg publisher queue full", exc_info=False)
        except Exception as e:
            logging.exception("unable to record messages in queue", exc_info=False)

    def _process_queue(self):
        while not MsgPublisher._stop_processing:
            while not self.record_queue.empty():
                try:
                    if MsgPublisher.resp_cnt < config.CALL_HOME_REQ_STOP_MAX_COUNT:
                        resp_code = self.client.send(self.record_queue.get(block=True, timeout=2))
                        if resp_code != 200:
                            MsgPublisher.resp_cnt += 1
                    else:
                        self.record_queue.get(block=False)
                except queue.Empty as e:
                    logging.exception("Queue is empty!", exc_info=False)
        logging.info("ccm msg publisher execution stopped")

    def stop(self):
        if MsgPublisher._instance:
            try:
                self.record_queue.task_done()
            except ValueError as e:
                logging.exception("called task_done more than required times on ccm queue", exc_info=False)
            MsgPublisher._stop_processing = True
            self.executor.shutdown(wait=False)
            MsgPublisher._instance = None
      
