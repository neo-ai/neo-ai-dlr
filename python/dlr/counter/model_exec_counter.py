"""Model inference count holder module"""
import queue
import logging


class ModelExecCounter(object):
    """Holding model inference counts in a queue"""
    _model_queue = queue.Queue()

    @staticmethod
    def update_model_run_count(model):
        """keep counting model inference in queue"""
        try:
            item = {model: 1}
            ModelExecCounter._model_queue.put(item, block=False)
        except queue.Full:
            logging.exception("inference count queue full", exc_info=False)
        except Exception:
            logging.exception("unable to record inference count in queue", exc_info=False)

    @staticmethod
    def get_dict():
        """prepare a dictionary with aggregate model inference counts"""
        model_dict = {}
        while not ModelExecCounter._model_queue.empty():
            try:
                dict_t = {}
                dict_t.update(ModelExecCounter._model_queue.get(block=False))
                key = list(dict_t)[0]
                cnt = model_dict.get(key)
                if cnt:
                    model_dict[key] += 1
                else:
                    model_dict[key] = 1
            except queue.Empty:
                logging.exception("Queue is empty!", exc_info=False)
            except Exception:
                logging.exception("exception while get dictionary", exc_info=False)
        return model_dict
