import time
import numpy as np
import io
import os
import logging
import dlr

from scipy.sparse import csr_matrix
import csv

def _sparse_to_dense(csr_matrix):
    out = np.full(shape=csr_matrix.shape, fill_value=np.nan, dtype=np.float32)
    rowind = np.repeat(np.arange(csr_matrix.shape[0]), np.diff(csr_matrix.indptr))
    out[rowind, csr_matrix.indices] = csr_matrix.data
    return out

class NeoXGBoostPredictor():
    def __init__(self):
        self.model = None
        self._context = None
        self._batch_size = 0
        self.initialized = False

    def initialize(self, context):
        self._context = context
        self._batch_size = context.system_properties.get('batch_size')
        model_dir = context.system_properties.get('model_dir')
        print('Loading the model from directory {}'.format(model_dir))
        self.model = dlr.DLRModel(model_dir)
        self.initialized = True

    def preprocess(self, batch_data):
        assert self._batch_size == len(batch_data), \
            'Invalid input batch size: expected {} but got {}'.format(self._batch_size,
                                                                      len(batch_data))
        processed_batch_data = []

        for k in range(len(batch_data)):
            req_body = batch_data[k]
            content_type = self._context.get_request_header(k, 'Content-type')
            if content_type is None:
                content_type = self._context.get_request_header(k, 'Content-Type')
                if content_type is None:
                    raise Exception('Content type could not be deduced')

            payload = batch_data[k].get('data')
            if payload is None:
                payload = batch_data[k].get('body')
            if payload is None:
                raise Exception('Nonexistent payload')

            if content_type == 'text/libsvm' or content_type == 'text/x-libsvm':
                row = []
                col = []
                entries = []
                payload = payload.rstrip().split('\n')
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
                processed_batch_data.append(_sparse_to_dense(mat))
            elif content_type == 'text/csv':
                payload = payload.rstrip().split('\n')
                delimiter = csv.Sniffer().sniff(payload[0]).delimiter
                batch = list(map(lambda x: x.split(delimiter), payload))
                processed_batch_data.append(np.array(batch).astype(np.float))
            else:
                raise Exception('ClientError: Invalid content type. Accepted content types are ' +
                                '"text/libsvm" and "text/csv". Received {}'.format(content_type))

        return processed_batch_data

    def inference(self, batch_data):
        return [self.model.run(x)[0] for x in batch_data]

    def postprocess(self, batch_preds):
        ret = []
        for preds in batch_preds:
            assert preds.ndim == 2
            if preds.shape[1] == 1:
                preds = preds.reshape((1, -1))
            with io.StringIO() as f:
                np.savetxt(f, preds, delimiter=',', newline='\n')
                ret.append(f.getvalue())

        return ret

    def handle(self, data, context):
        start = time.time()
        try:
            model_input = self.preprocess(data)
            model_output = self.inference(model_input)
            response = self.postprocess(model_output)

            for k in range(len(data)):
                context.set_response_content_type(k, 'text/csv')
            print('Inference time is {}'.format((time.time() - start) * 1000))

            return response
        except Exception as e:
            logging.error(e, exc_info=True)
            if str(e).startswith('ClientError:'):
                context.set_all_response_status(400, 'ClientError')
                return [str(e)] * len(data)
            else:
                context.set_all_response_status(500, 'InternalServerError')
                return ['Internal Server Error occured'] * len(data)


_service = NeoXGBoostPredictor()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
