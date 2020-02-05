import threading
import copy

lock = threading.Lock()

class ModelExecCounter(object):
    model_dict = {}
    intr_dict = {}
    APPEND = 1 
    GETINTDICT = 2 

    @staticmethod
    def update_model_run_count(model):
        """keep counting model inference in dictionary"""
        ModelExecCounter.handle_dict(ModelExecCounter.APPEND, model)

    @staticmethod
    def handle_dict(op, model=None):
        with lock:
            if op == ModelExecCounter.APPEND:
                cnt = ModelExecCounter.model_dict.get(model)
                if cnt:
                    cnt += 1
                    ModelExecCounter.model_dict[model] = cnt 
                else:
                    ModelExecCounter.model_dict[model] = 1
            elif op == ModelExecCounter.GETINTDICT:
                ModelExecCounter.intr_dict = copy.deepcopy(ModelExecCounter.model_dict)
                ModelExecCounter.model_dict.clear()
 
    @staticmethod
    def get_dict():
        ModelExecCounter.handle_dict(ModelExecCounter.GETINTDICT)
        return ModelExecCounter.intr_dict

