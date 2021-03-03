'''
This file is to be named inference.py when hosting on sagemaker
'''
import grpc
import gzip
import os
import io
import tensorflow as tf
from PIL import Image
import numpy as np
from google.protobuf.json_format import MessageToJson
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

def input_handler(data, context):
    """Handle request.
    Args:
        data (obj): the request data
        context (Context): an object containing request and configuration details
    Returns:
        (proto): TFS PredictRequest protobuf object
    """
    '''
    stream = os.popen('/opt/aws/neuron/bin/neuron-cli list-model')
    output = stream.read()
    print(output)
    stream = os.popen('/opt/aws/neuron/bin/neuron-cli list-ncg')
    output = stream.read()
    print(output)
    '''
    f = data.read()
    f = io.BytesIO(f)
    image = Image.open(f).convert('RGB')
    batch_size = 1
    image = np.asarray(image.resize((224, 224)))
    image = np.concatenate([image[np.newaxis, :, :]] * batch_size)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'compiled_models'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['Placeholder:0'].CopyFrom(tf.compat.v1.make_tensor_proto(image, shape=image.shape, dtype=tf.float32))
    return request

def output_handler(data, context):
    """Post-process TensorFlow Serving gRPC output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving gRPC response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    jsonObj = MessageToJson(data)
    return jsonObj, 'application/json'