import threading

lock = threading.Lock()

class ModelExecCounter(object):
    model_dict = {}
    INIT = 1
    INCREMENT = 2
    CLEAR = 3

    @staticmethod
    def update_model_run_count(model):
        """keep counting model inference in dictionary"""
        cnt = ModelExecCounter.model_dict.get(model)
        if cnt:
            cnt += 1
            ModelExecCounter.update_dict(ModelExecCounter.INCREMENT, model, cnt)
        else:
            ModelExecCounter.update_dict(ModelExecCounter.INIT, model)

    @staticmethod
    def update_dict(op, model=None, cnt=1):
        with lock:
            if op == ModelExecCounter.INIT:
                ModelExecCounter.model_dict[str(model)] = 1
            elif op == ModelExecCounter.INCREMENT:
                ModelExecCounter.model_dict[str(model)] = cnt
            elif op == ModelExecCounter.CLEAR:
                ModelExecCounter.model_dict.clear()
 
    @staticmethod
    def get_dict():
        return ModelExecCounter.model_dict

    @staticmethod
    def clear_models_counts():
        ModelExecCounter.update_dict(ModelExecCounter.CLEAR)
