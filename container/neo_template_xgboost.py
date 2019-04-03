import time
import pickle as pkl
import csv
import numpy as np
from scipy.sparse import csr_matrix
import os
import dlr

def _sparse_to_dense(csr_matrix):
    out = np.full(shape=csr_matrix.shape, fill_value=np.nan, dtype=np.float32)
    rowind = np.repeat(np.arange(csr_matrix.shape[0]), np.diff(csr_matrix.indptr))
    out[rowind, csr_matrix.indices] = csr_matrix.data
    return out

class NeoXGBoostPredictor():
    def __init__(self):
        self.model = None
        self.initialized = False

    def inference(self, data):
        return self.model.run(data)

    def postprocess(self, preds):
        assert len(preds) == 1
        ret = [','.join([str(x) for x in row]) for row in preds[0].tolist()]
        return ret

    def initialize(self, context):
        manifest = context.manifest
        model_dir = context.system_properties.get("model_dir")
        print("Loading the model from directroy {}".format(model_dir))
        self.model = dlr.DLRModel(model_dir)
        self.initialized = True

    def preprocess(self, context, data):
        headers = context.request_processor._request_header
        content_type = None
        req_ids = context.request_ids

        assert len(req_ids) == len(data)

        batch_size = len(req_ids)

        if batch_size != 1:
            raise Exception('Batch prediction not yet supported')

        for k in range(batch_size):
            req_id = req_ids[k]
            req_body = data[k]
            content_type = headers[req_id].get('Content-type')
            if content_type is None:
                content_type = headers[req_id].get('Content-Type')
                if content_type is None:
                    raise Exception('Content type could not be deduced')

            if content_type == 'text/libsvm' or content_type == 'text/x-libsvm':
                row = []
                col = []
                entries = []
                payload = req_body['body'].rstrip().split('\n')
                colon = ':'
                for row_idx, line in enumerate(payload):
                    for entry in line.split(' '):
                        if colon in entry:
                            token = entry.split(colon)
                            col_idx, val = token[0], token[1]
                            row.append(row_idx)
                            col.append(col_idx)
                            entries.append(val)
                row = np.array(row)
                col = np.array(col).astype(np.int)
                entries = np.array(entries).astype(np.float)
                mat = csr_matrix((entries, (row, col)))
                return _sparse_to_dense(mat)
            elif content_type == 'text/csv':
                payload = req_body['body'].rstrip().split('\n')
                delimiter = csv.Sniffer().sniff(payload[0]).delimiter
                batch = list(map(lambda x: x.split(delimiter), payload))
                return np.array(batch).astype(np.float)
            else:
                raise Exception("Invalid content type. Accepted content types are \"text/libsvm\" and \"text/libsvm\" "
                                "received {}".format(content_type))


model_obj = NeoXGBoostPredictor()


def predict(data, context):
    if model_obj is None:
        print("Model not loaded")
        return

    if not model_obj.initialized:
        model_obj.initialize(context)

    if data is None:
        return

    start = time.time()
    data = model_obj.preprocess(context, data)
    data = model_obj.inference(data)
    data = model_obj.postprocess(data)

    context.set_response_content_type(context.request_ids[0], "text/csv")
    print("Inference time is {}".format((time.time() - start) * 1000))
    return data
