# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

from subprocess import CalledProcessError

from retrying import retry
from sagemaker_inference import model_server

from sagemaker_pytorch_inferentia_serving_container import handler_service

HANDLER_SERVICE = handler_service.__name__


def _retry_if_error(exception):
    return isinstance(exception, CalledProcessError)


@retry(stop_max_delay=1000 * 30,
       retry_on_exception=_retry_if_error)
def _start_model_server():
    import os
    stream = os.popen('lspci -d 1d0f:7064 | wc -l')
    output = stream.read()
    print("lspci {}".format(output))
    stream = os.popen('neuron-ls --json-output')
    output = stream.read()
    print("neuron ls {}".format(output))
    stream = os.popen('neuron-ls --json-output | jq length')
    output = stream.read()
    print("neuron ls length {}".format(output))
    print("environ {}".format(os.environ.get('NEURON_DEVICE_SIZES')))
    print("environ {}".format(os.environ.get('NEURONCORE_GROUP_SIZES')))
    # there's a race condition that causes the model server command to
    # sometimes fail with 'bad address'. more investigation needed
    # retry starting mms until it's ready
    model_server.start_model_server(handler_service=HANDLER_SERVICE)


def main():
<<<<<<< HEAD
    import os
    stream = os.popen('lspci -d 1d0f:7064 | wc -l')
    output = stream.read()
    print("lspci {}".format(output))
    stream = os.popen('neuron-ls --json-output')
    output = stream.read()
    print("neuron ls {}".format(output))
    stream = os.popen('neuron-ls --json-output | jq length')
    output = stream.read()
    print("neuron ls length {}".format(output))
    print("environ {}".format(os.environ.get('NEURON_DEVICE_SIZES')))
    print("environ {}".format(os.environ.get('NEURONCORE_GROUP_SIZES')))
=======
>>>>>>> init inf1 infeence container
    _start_model_server()