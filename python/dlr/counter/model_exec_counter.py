class ModelExecCounter(object):
    model_dict = {}

    @staticmethod
    def add_model_run_count(model):
        """keep counting model inference in dictionary"""
        cnt = ModelExecCounter.model_dict.get(model)
        if cnt:
            cnt += 1
            ModelExecCounter.update_model_map('increment', model, cnt)
        else:
            ModelExecCounter.update_model_map('init', model)

    @staticmethod
    def update_model_map(op, model=None, cnt=1):
        if op is 'init':
            ModelExecCounter.model_dict[str(model)] = 1
        elif op is 'increment':
            ModelExecCounter.model_dict[str(model)] = cnt
        elif op is 'clear':
           ModelExecCounter.clear_model_counts()
 
    @staticmethod
    def get_model_counts_dict():
        return ModelExecCounter.model_dict

    @staticmethod
    def clear_model_counts():
        ModelExecCounter.model_dict.clear()
