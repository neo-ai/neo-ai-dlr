'''
This file is to be named inference.py when hosting on sagemaker
'''
import grpc
import gzip
import os
import io
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from google.protobuf.json_format import MessageToJson

prediction_services = {}
compression_algo = gzip

def get_prediction_service(context):
    #global prediction_service
    if not context.grpc_port in prediction_services:
        print("Creating service for port %s" % context.grpc_port)
        channel = grpc.insecure_channel("localhost:{}".format(context.grpc_port))
        prediction_services[context.grpc_port] = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    return prediction_services[context.grpc_port]

def handler(data, context):
    """Handle request.
    Args:
        data (obj): the request data
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, (optional) response content type
    """
    '''
    stream = os.popen('/opt/aws/neuron/bin/neuron-cli list-model')
    output = stream.read()
    print(output)
    stream = os.popen('/opt/aws/neuron/bin/neuron-cli list-ncg')
    output = stream.read()
    print(output)
    '''
    #Preprocess input
    print("Received a request for grpc port: %s " % context.grpc_port)
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
    
    #Call Predict gRPC service
    result = get_prediction_service(context).Predict(request, 60.0)
    print("Returning the response for grpc port: %s" % (context.grpc_port))
    
    #Return response
    jsonObj = MessageToJson(result)
    return jsonObj, "application/json"