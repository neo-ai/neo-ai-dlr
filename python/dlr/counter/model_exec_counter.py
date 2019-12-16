class ModelExecCounter(object):
    model_dict = {}

    @staticmethod
    def add_model_run_count(model):
        """keep counting model inference in dictionary"""
        cnt = ModelExecCounter.model_dict.get(model)
        if cnt:
            cnt += 1
            ModelExecCounter.model_dict.update({str(model): cnt})
        else:
            ModelExecCounter.model_dict.update({str(model): 1})

    @staticmethod
    def get_model_counts_dict():
        return ModelExecCounter.model_dict

    @staticmethod
    def clear_model_counts():
        ModelExecCounter.model_dict.clear()
